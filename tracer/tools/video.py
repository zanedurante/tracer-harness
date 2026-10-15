"""query_video_minute: let the agent LOOK at one minute of the actual video and
answer a free-form question about it (multimodal), instead of relying on the
precomputed atomic-action labels.

Clip resolution (per stay):
  - room       : stays.csv `room` column (tracer.data.room_for)
  - video base : stays.csv `video_dir` override, else <data root>/video
  - file       : <base>/<UTC_date>/<HH_MM>_<room>_<view>.mp4   (folders are UTC)
The agent passes a local-time minute; ZoneInfo handles DST -> UTC.

Perception model: by default the internal multimodal call uses the SAME
`model` passed by the agent loop (the model under test); if that model cannot
accept video the call returns an error dict (surfaced to the agent, never
silently swapped). Setting VIDEO_TOOL_MODEL overrides this explicitly, e.g.
VIDEO_TOOL_MODEL=gemini-3.1-flash-lite gives a text-only or weak-vision agent
a Gemini perceiver (the paper's "video tool served by Gemini 3.1 Flash Lite"
rows). The override is recorded per trial by the sweep driver.

Sampling (paper appendix, "Sampling and serving settings"): for Gemini the
original clip is sent with a 5 fps sampling hint (DEFAULT_FPS); for
OpenAI-compatible (vLLM) models the clip is re-encoded at 2 fps with its
longer side at 512 px and shipped inline as base64.
"""
from __future__ import annotations

import base64
import datetime
import hashlib
import os
import shutil
import string
import subprocess
import threading
import time
import uuid
import concurrent.futures as cf
from pathlib import Path
from zoneinfo import ZoneInfo

from google.genai import types

from .. import config, data
from ..backends import gemini as gem
from ..backends import openai_backend as oai
from .base import Tool

_UTC = ZoneInfo("UTC")

# Gemini samples video at 1 fps by default, which is too coarse for brief ICU
# motion (a 1-2 s cough). 5 fps caught the most movement in testing.
DEFAULT_FPS = int(os.environ.get("VIDEO_TOOL_GEMINI_FPS", "5"))

# Prepended before the agent's prompt on every perception call (verbatim in
# prompts/video_tool_preamble.txt and the paper appendix).
PERCEPTION_PREAMBLE = (
    "You are an expert ICU video analyst. Patients here are often sedated, so subtle "
    "patient movement (a brief head turn, eye flutter, blink, cough, swallow, finger/limb "
    "twitch) is clinically relevant. Review the one-minute clip carefully and list ONLY "
    "movements you can actually see, each with an approximate timestamp (mm:ss). Do NOT "
    "infer or assume movements that are not clearly visible -- if the patient appears still, "
    "say so explicitly. Then answer the custom prompt below."
)


def _pt_to_utc_parts(pt_str: str) -> tuple[str, str]:
    # tolerate a trailing zone abbreviation the model may append (e.g. "... 08:37 CET")
    s = pt_str.strip().rstrip(string.ascii_letters).strip()
    dt = None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            break
        except ValueError:
            continue
    if dt is None:
        raise ValueError(f"Could not parse local-time timestamp {pt_str!r} (use 'YYYY-MM-DD HH:MM').")
    u = dt.replace(tzinfo=config.local_tz()).astimezone(_UTC)
    return u.strftime("%Y_%m_%d"), u.strftime("%H_%M")


def clip_relpath(minute_pt: str, room: str, view: str) -> str:
    date_folder, hhmm = _pt_to_utc_parts(minute_pt)
    return f"{date_folder}/{hhmm}_{room}_{view}.mp4"


# ---------------------------------------------------------------------------
# Gemini path: Files API upload (API key) or inline bytes / GCS staging (Vertex)
# ---------------------------------------------------------------------------
_UPLOAD_CACHE: dict[str, str] = {}
_UPLOAD_LOCK = threading.Lock()


def _ensure_uploaded(client, path: Path, *, poll_s: float = 2.0, timeout_s: float = 120.0):
    """Upload `path` (or reuse a cached, still-ACTIVE handle) and return an
    ACTIVE File object."""
    key = str(path)
    with _UPLOAD_LOCK:
        cached = _UPLOAD_CACHE.get(key)
    if cached:
        try:
            f = client.files.get(name=cached)
            if "ACTIVE" in str(getattr(f, "state", "")).upper():
                return f
        except Exception:
            pass
    f = client.files.upload(file=key)
    waited = 0.0
    while True:
        st = str(getattr(f, "state", "")).upper()
        if "ACTIVE" in st:
            break
        if "FAILED" in st:
            raise RuntimeError("uploaded file processing FAILED")
        if waited >= timeout_s:
            raise RuntimeError("uploaded file did not become ACTIVE in time")
        time.sleep(poll_s)
        waited += poll_s
        f = client.files.get(name=f.name)
    with _UPLOAD_LOCK:
        _UPLOAD_CACHE[key] = f.name
    return f


def _delete_uploaded(client, f, key: str) -> None:
    """Uploaded clips persist for 48 h and count against the Files API storage
    quota, which a sweep exhausts within hours; delete right after inference."""
    with _UPLOAD_LOCK:
        _UPLOAD_CACHE.pop(key, None)
    try:
        client.files.delete(name=f.name)
    except Exception:
        pass


# Vertex AI has no Files API: clips are sent inline, and if the request is
# rejected (clip too large) staged in a GCS bucket of the billing project.
GCS_BUCKET = os.environ.get("GEMINI_VIDEO_BUCKET", "")
_GCS_CLIENT = None
_GCS_LOCK = threading.Lock()


def _gcs_bucket():
    global _GCS_CLIENT
    if not GCS_BUCKET:
        raise RuntimeError("set GEMINI_VIDEO_BUCKET to stage clips on the Vertex path")
    with _GCS_LOCK:
        if _GCS_CLIENT is None:
            from google.cloud import storage
            _GCS_CLIENT = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
    return _GCS_CLIENT.bucket(GCS_BUCKET)


def _gcs_stage(path: Path, rel: str):
    """Upload to gs://<bucket>/clips/<unique>/<rel>. The object name carries a
    per-call uuid so concurrent trials on the same minute never delete each
    other's object mid-request."""
    key = f"clips/{uuid.uuid4().hex}/{rel}"
    blob = _gcs_bucket().blob(key)
    blob.upload_from_filename(str(path), content_type="video/mp4")
    return f"gs://{GCS_BUCKET}/{key}", blob


VIDEO_CALL_TIMEOUT_S = float(os.environ.get("VIDEO_CALL_TIMEOUT_S", "150"))
VIDEO_TOOL_MODEL = os.environ.get("VIDEO_TOOL_MODEL", "").strip()


def _video_call(client, model: str, part, prompt: str) -> str:
    """One perception call with a hard wall-clock timeout. Not a `with` block:
    ThreadPoolExecutor.__exit__ would re-join the thread the timeout is
    abandoning; a stuck request thread is leaked instead."""
    ex = cf.ThreadPoolExecutor(max_workers=1)
    try:
        fut = ex.submit(lambda: client.models.generate_content(
            model=model, contents=[part, PERCEPTION_PREAMBLE + "\n\n" + prompt]))
        try:
            resp = fut.result(timeout=VIDEO_CALL_TIMEOUT_S)
        except cf.TimeoutError:
            raise TimeoutError(f"video call exceeded {VIDEO_CALL_TIMEOUT_S:.0f}s")
        return (resp.text or "").strip()
    finally:
        ex.shutdown(wait=False)


def query_video_minute(args: dict, pat_id: str | None, model: str) -> dict:
    if VIDEO_TOOL_MODEL:
        model = VIDEO_TOOL_MODEL
    pt = args.get("minute_pt") or args.get("time_pt") or args.get("timestamp")
    prompt = args.get("prompt") or args.get("question")
    if not pt or not prompt:
        return {"error": "Provide 'minute_pt' (local time 'YYYY-MM-DD HH:MM') and 'prompt'."}
    if not pat_id:
        return {"error": "No patient in context."}

    room = data.room_for(pat_id)
    base = data.video_base(pat_id)
    try:
        date_folder, hhmm = _pt_to_utc_parts(pt)
    except ValueError as e:
        return {"error": str(e)}

    if oai.is_openai_model(model):
        return _query_video_minute_openai(pt, prompt, room, base, date_folder, hhmm, model)

    client = gem.get_files_client()

    def _one_view(v):
        rel = f"{date_folder}/{hhmm}_{room}_{v}.mp4"
        path = base / rel
        if not path.exists():
            return v, {"clip": rel, "error": "no clip on disk for this view"}
        f = None
        blob = None
        try:
            if gem.is_adc():
                # Inline bytes first (fastest, no staging, no delete race);
                # fall back to GCS staging if the request is rejected.
                try:
                    return v, {"clip": rel, "answer": _video_call(
                        client, model, types.Part(
                            inline_data=types.Blob(data=path.read_bytes(), mime_type="video/mp4"),
                            video_metadata=types.VideoMetadata(fps=DEFAULT_FPS)), prompt)}
                except Exception as inline_err:
                    uri, blob = _gcs_stage(path, rel)
                    part = types.Part(
                        file_data=types.FileData(file_uri=uri, mime_type="video/mp4"),
                        video_metadata=types.VideoMetadata(fps=DEFAULT_FPS),
                    )
                    return v, {"clip": rel, "answer": _video_call(client, model, part, prompt),
                               "note": f"inline failed ({type(inline_err).__name__}), used gs:// staging"}
            else:
                f = _ensure_uploaded(client, path)
                part = types.Part(
                    file_data=types.FileData(file_uri=f.uri, mime_type=f.mime_type),
                    video_metadata=types.VideoMetadata(fps=DEFAULT_FPS),
                )
            resp = client.models.generate_content(
                model=model, contents=[part, PERCEPTION_PREAMBLE + "\n\n" + prompt])
            return v, {"clip": rel, "answer": (resp.text or "").strip()}
        except Exception as e:
            return v, {"clip": rel, "error": f"{type(e).__name__}: {e}"}
        finally:
            if blob is not None:
                try:
                    blob.delete()
                except Exception:
                    pass
            if f is not None:
                _delete_uploaded(client, f, str(path))

    # Both camera views in PARALLEL with the same prompt; both answers are
    # returned for the agent to reconcile. Collected with a deadline so one
    # hung view cannot freeze the caller.
    ex = cf.ThreadPoolExecutor(max_workers=2)
    futs = {v: ex.submit(_one_view, v) for v in ("lt", "rt")}
    ex.shutdown(wait=False)
    results = {}
    for v, fut in futs.items():
        try:
            results[v] = fut.result(timeout=VIDEO_CALL_TIMEOUT_S + 60)[1]
        except Exception as e:
            results[v] = {"clip": f"{date_folder}/{hhmm}_{room}_{v}.mp4",
                          "error": f"{type(e).__name__}: {e}"}
    return _assemble(pt, prompt, room, results)


def _assemble(pt, prompt, room, results) -> dict:
    if all("error" in r for r in results.values()):
        return {"error": f"No usable video for {pt} local time (room {room}).",
                "left_view": results["lt"], "right_view": results["rt"]}
    return {"minute_pt": pt, "prompt": prompt,
            "left_view": results["lt"], "right_view": results["rt"]}


# ---------------------------------------------------------------------------
# OpenAI-compatible path (vLLM-served Qwen3-VL etc.): re-encode locally to a
# small low-fps mp4 and ship inline as a base64 `video_url` part. Same
# preamble, same prompt, same parallel left/right query -> same contract.
# ---------------------------------------------------------------------------
OPENAI_VIDEO_FPS = float(os.environ.get("VIDEO_TOOL_FPS", "2"))
OPENAI_VIDEO_MAX_SIDE = int(os.environ.get("VIDEO_TOOL_MAX_SIDE", "512"))
OPENAI_VIDEO_RETRIES = int(os.environ.get("VIDEO_TOOL_RETRIES", "3"))
VIDEO_CONCURRENCY = int(os.environ.get("VIDEO_TOOL_CONCURRENCY", "3"))
_REENCODE_LOCK = threading.Lock()
_KEY_LOCKS: dict[str, threading.Lock] = {}
_VIDEO_ERR_LOCK = threading.Lock()


def _clip_cache_dir() -> Path:
    return config.cache_root() / "clip_cache"


class _video_file_lock:
    """Cross-process counting semaphore for video requests (several sweeps may
    share one vLLM server): at most VIDEO_CONCURRENCY video bodies in flight.
    Implemented with N flock files (POSIX only; a no-op elsewhere)."""
    def __enter__(self):
        self._fh = None
        try:
            import fcntl
            import random
            d = _clip_cache_dir()
            d.mkdir(parents=True, exist_ok=True)
            slots = list(range(VIDEO_CONCURRENCY))
            random.shuffle(slots)
            for i in slots:
                fh = open(d / f".video_request.{i}.lock", "w")
                try:
                    fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self._fh = fh
                    return self
                except OSError:
                    fh.close()
            fh = open(d / f".video_request.{slots[0]}.lock", "w")
            fcntl.flock(fh, fcntl.LOCK_EX)
            self._fh = fh
        except Exception:
            self._fh = None
        return self

    def __exit__(self, *exc):
        if self._fh is not None:
            try:
                import fcntl
                fcntl.flock(self._fh, fcntl.LOCK_UN)
            finally:
                self._fh.close()
        return False


def _log_video_error(rel: str, attempt: int, err: Exception) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{rel}\tattempt={attempt}\t{type(err).__name__}: {str(err)[:300]}\n"
    with _VIDEO_ERR_LOCK:
        d = _clip_cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "video_tool_errors.log", "a", encoding="utf-8") as fh:
            fh.write(line)


def _reencode_lock_for(key: str) -> threading.Lock:
    with _REENCODE_LOCK:
        lk = _KEY_LOCKS.get(key)
        if lk is None:
            lk = _KEY_LOCKS[key] = threading.Lock()
        return lk


def ffmpeg_bin() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def reencoded_clip(path: Path) -> Path:
    """Return a cached, downscaled, low-fps copy of `path` (re-encoding on miss)."""
    key = hashlib.sha1(f"{path}|{OPENAI_VIDEO_FPS}|{OPENAI_VIDEO_MAX_SIDE}".encode()).hexdigest()
    out = _clip_cache_dir() / key[:2] / f"{key}.mp4"
    if out.exists() and out.stat().st_size > 0:
        return out
    with _reencode_lock_for(key):
        if out.exists() and out.stat().st_size > 0:
            return out
        ffmpeg = ffmpeg_bin()
        if ffmpeg is None:
            raise RuntimeError("ffmpeg not on PATH (needed to re-encode clips for the OpenAI path)")
        out.parent.mkdir(parents=True, exist_ok=True)
        m = OPENAI_VIDEO_MAX_SIDE
        vf = (f"fps={OPENAI_VIDEO_FPS},"
              f"scale='if(gt(iw,ih),min({m},iw),-2)':'if(gt(iw,ih),-2,min({m},ih))',format=yuv420p")
        tmp = out.with_name(f"{key}.{os.getpid()}.{threading.get_ident()}.tmp.mp4")
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(path),
               "-vf", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
               "-movflags", "+faststart", "-y", str(tmp)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg re-encode failed: {res.stderr.strip()[:300]}")
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            nb = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                                 "-show_entries", "stream=nb_frames", "-of", "csv=p=0", str(tmp)],
                                capture_output=True, text=True).stdout.strip()
            if not nb.isdigit() or int(nb) < 2:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f"re-encoded clip has no decodable frames (nb_frames={nb!r})")
        tmp.replace(out)
        return out


def _query_video_minute_openai(pt, prompt, room, base, date_folder, hhmm, model) -> dict:
    def _one_view(v):
        rel = f"{date_folder}/{hhmm}_{room}_{v}.mp4"
        path = base / rel
        if not path.exists():
            return v, {"clip": rel, "error": "no clip on disk for this view"}
        if oai.is_mock_model(model):
            from ..backends import mock
            return v, {"clip": rel, "answer": mock.describe_clip(rel, prompt)}
        try:
            small = reencoded_clip(path)
            b64 = base64.b64encode(small.read_bytes()).decode()
            messages = [{"role": "user", "content": [
                {"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{b64}"}},
                {"type": "text", "text": PERCEPTION_PREAMBLE + "\n\n" + prompt},
            ]}]
            last_err = None
            for attempt in range(OPENAI_VIDEO_RETRIES):
                try:
                    with _video_file_lock():
                        turn = oai.chat(messages, model=model, tools=None, max_tokens=1024)
                    return v, {"clip": rel, "answer": (turn.text or "").strip()}
                except Exception as e:
                    last_err = e
                    _log_video_error(rel, attempt, e)
                    time.sleep(1.0 + attempt)
            raise last_err
        except Exception as e:
            return v, {"clip": rel, "error": f"{type(e).__name__}: {e}"}

    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        results = dict(ex.map(_one_view, ["lt", "rt"]))
    return _assemble(pt, prompt, room, results)


# --- tool DESCRIPTION variants (what the orchestration agent reads) ----------
# Selected per run via VIDEO_DESC_VARIANT (D1/D2/D3). D2 (hedged: evidence to
# weigh, not ground truth) is the default and the one every reported run used.
_COMMON_USAGE = (
    "This tool is useful for zooming in on a single minute of interest. You can call it on "
    "the SAME minute several times with DIFFERENT prompts to probe different details (e.g. "
    "first ask what the clinician is doing, then whether the patient moved). It returns BOTH "
    "camera angles (left and right view) answered separately.\n"
    "TIME: everything here is local time. Pass the minute exactly as it appears in the "
    "question, 'YYYY-MM-DD HH:MM' (24-hour, local time); the tool finds the right clip for you and you "
    "never convert timezones. The clip is one minute long, so if an event might straddle a "
    "boundary, also query the adjacent minute."
)
DESC_VARIANTS = {
    "D1": (
        "Look at the ACTUAL ICU surveillance video for ONE specific minute and answer a "
        "free-form question about what is visible -- patient movements/reactions, what a "
        "clinician is doing, devices/objects, body position, who is in the room. This is "
        "ground truth: prefer it over precomputed action labels, which can be unreliable "
        "for this patient.\n" + _COMMON_USAGE
    ),
    "D2": (
        "Look at the ICU surveillance video for ONE specific minute and answer a free-form "
        "question about what is visible -- patient movements/reactions, what a clinician is "
        "doing, devices/objects, body position, who is in the room.\n"
        "Treat its answer as EVIDENCE TO WEIGH, not ground truth: the video reader can miss "
        "brief events or misread motion, and the two camera views may disagree. When the "
        "answer matters, re-query (a different prompt or an adjacent minute) and cross-check "
        "against your other tools before committing.\n" + _COMMON_USAGE
    ),
    "D3": (
        "Look at the ICU surveillance video for one minute and answer a question about what "
        "is visible. Returns both camera views (left and right) answered separately.\n" + _COMMON_USAGE
    ),
}
DESC_VARIANT = os.environ.get("VIDEO_DESC_VARIANT", "D2")

VIDEO_DECL = types.FunctionDeclaration(
    name="query_video_minute",
    description=DESC_VARIANTS.get(DESC_VARIANT, DESC_VARIANTS["D2"]),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "minute_pt": types.Schema(
                type="STRING",
                description="The minute to view, local time, format 'YYYY-MM-DD HH:MM' (24-hour).",
            ),
            "prompt": types.Schema(
                type="STRING",
                description="The specific question to ask about what is visible in that minute of video.",
            ),
        },
        required=["minute_pt", "prompt"],
    ),
)

VIDEO_TOOL = Tool(name="query_video_minute", declaration=VIDEO_DECL, handler=query_video_minute, group="video")

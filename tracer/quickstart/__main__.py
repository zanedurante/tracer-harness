"""Run TRACER end to end on one continuous video, then open the viewer.

    export GEMINI_API_KEY=...
    python -m tracer.quickstart                       # sample video (downloaded if missing)
    python -m tracer.quickstart --video my_hour.mp4   # your own continuous recording

Steps (each is skipped when its outputs already exist, so the command can be
re-run or resumed):
  1. split the video into one-minute clips in the store layout
     (<out>/video/<YYYY_MM_DD>/<HH_MM>_<room>_lt.mp4), re-encoded to a size
     the Gemini inline-video path accepts;
  2. write stays.csv and copy the synthetic, non-PHI EHR tables and events
     from demo/data (the sample stay's clock, 2030-01-01 08:00 local, sits
     inside that synthetic record);
  3. minute-level perception with the paper's prompt (one Gemini call per
     clip; tracer.perception.run_perception);
  4. the minute-level timeline (tracer.perception.build_timeline);
  5. hourly reports (tracer.reports.generate_reports);
  6. the viewer (tracer.viewer) at http://127.0.0.1:<port>/.

The video is treated as one camera (the "lt" view); the tools and reports
work with one view, and the right-view slot simply reports no clip.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
SAMPLE_DIR = REPO / "demo" / "data"
DEFAULT_VIDEO = REPO / "demo" / "video" / "TRACER_SAMPLE_VIDEO.mp4"
SAMPLE_VIDEO_GDRIVE_ID = "15muTWlrIKeBEQpxo4loyLeIuFsifCZGU"

PAT_ID = "stay_A"          # matches the synthetic EHR tables in demo/data
ROOM = "r001"
START_UTC = "2026-10-16T08:00:00Z"       # the clock assigned to the first minute of the video
SAMPLE_START_UTC = "2030-01-01T08:00:00Z"  # the clock the synthetic record in demo/data is authored at
DEFAULT_MODEL = "gemini-3.1-flash-lite"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _runs(exe: str | None) -> bool:
    if not exe:
        return False
    try:
        return subprocess.run([exe, "-version"], capture_output=True, timeout=20).returncode == 0
    except Exception:
        return False


def _ffmpeg() -> str:
    """The first ffmpeg that actually runs: PATH, then the imageio-ffmpeg
    bundle. (A broken system install, e.g. a Homebrew ffmpeg whose shared
    libraries were upgraded away, is skipped rather than crashing mid-run.)"""
    candidates = [shutil.which("ffmpeg")]
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    for exe in candidates:
        if _runs(exe):
            return exe
    raise SystemExit("no working ffmpeg: install ffmpeg or `pip install imageio-ffmpeg`")


def _ffprobe_duration(ffmpeg: str, src: Path) -> float:
    """Duration in seconds from ffmpeg's own banner (ffprobe is not part of
    the imageio-ffmpeg bundle, and a broken system ffprobe must not matter)."""
    import re
    out = subprocess.run([ffmpeg, "-hide_banner", "-i", str(src)], capture_output=True, text=True)
    m = re.search(r"Duration: (\d+):(\d+):(\d+\.?\d*)", out.stderr)
    if not m:
        raise SystemExit(f"could not read the duration of {src}: {out.stderr.strip()[:300]}")
    h, mi, s = m.groups()
    return int(h) * 3600 + int(mi) * 60 + float(s)


def download_sample(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        import gdown
    except ImportError:
        raise SystemExit("the sample video is fetched with gdown: pip install gdown, or download it from "
                         f"https://drive.google.com/file/d/{SAMPLE_VIDEO_GDRIVE_ID}/view and pass --video")
    _log(f"Downloading the sample video (about 1.7 GB) to {dest} ...")
    gdown.download(id=SAMPLE_VIDEO_GDRIVE_ID, output=str(dest), quiet=False)
    if not dest.exists():
        raise SystemExit("download failed")
    return dest


def split_into_clips(ffmpeg: str, src: Path, out_video: Path, *, start_utc: str, room: str,
                     height: int, crf: int, max_minutes: int | None) -> list[str]:
    """Cut one-minute clips named by their UTC minute. Returns the clip ids."""
    import datetime as dt
    t0 = dt.datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
    duration = _ffprobe_duration(ffmpeg, src)
    n = int(duration // 60)
    if max_minutes:
        n = min(n, max_minutes)
    if n == 0:
        raise SystemExit(f"{src} is shorter than one minute ({duration:.1f}s)")
    ids: list[str] = []
    todo: list[tuple[int, Path]] = []
    for i in range(n):
        t = t0 + dt.timedelta(minutes=i)
        rel = f"{t.strftime('%Y_%m_%d')}/{t.strftime('%H_%M')}_{room}_lt.mp4"
        ids.append(rel)
        path = out_video / rel
        if not (path.exists() and path.stat().st_size > 0):
            todo.append((i, path))
    _log(f"  {n} minutes of video; {len(todo)} clips to encode ({n - len(todo)} already present)")
    t_start = time.time()
    for k, (i, path) in enumerate(todo, 1):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp.mp4")
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
               "-ss", str(i * 60), "-i", str(src), "-t", "60", "-an",
               "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)]
        subprocess.run(cmd, check=True)
        tmp.replace(path)
        if k % 10 == 0 or k == len(todo):
            _log(f"    {k}/{len(todo)} clips ({time.time() - t_start:.0f}s)")
    return ids


def _shift_iso(value, delta):
    """Shift an ISO-8601 timestamp string by `delta`, keeping its format
    (offset-aware stays aware; naive stays naive)."""
    import datetime as dt
    if not isinstance(value, str) or not value:
        return value
    try:
        t = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    t2 = t + delta
    if value.endswith("Z"):
        return t2.strftime("%Y-%m-%dT%H:%M:%S") + "Z"
    sep = "T" if "T" in value else " "
    s = t2.isoformat(sep=sep)
    if "." not in value and "." in s:
        s = s.split(".")[0] + (s[s.index("+"):] if "+" in s[s.index("."):] else "")
    return s


def write_store(out: Path, *, n_minutes: int, start_utc: str) -> dict:
    """stays.csv + the synthetic EHR/events for the stay, with every timestamp
    in the synthetic record shifted from the sample's clock to `start_utc`
    so medications, flowsheet rows and events line up with the video, and
    trimmed to the video's span: the synthetic record covers several hours,
    so rows before the first clip or after the last one are dropped and the
    encounter is closed at the end of the video.
    Returns the (shifted, trimmed) events, demographics and encounters."""
    import datetime as dt
    import pandas as pd
    t0 = dt.datetime.fromisoformat(start_utc.replace("Z", "+00:00"))
    delta = t0 - dt.datetime.fromisoformat(SAMPLE_START_UTC.replace("Z", "+00:00"))
    t_end = t0 + dt.timedelta(minutes=n_minutes - 1)      # start of the last clip
    t_stop = t0 + dt.timedelta(minutes=n_minutes)         # end of the video
    (out / "stays.csv").write_text(
        "pat_id,start_time_utc,end_time_utc,room,video_dir,timeline_dir,max_display_hours\n"
        f"{PAT_ID},{t0.strftime('%Y-%m-%dT%H:%M:%SZ')},{t_end.strftime('%Y-%m-%dT%H:%M:%SZ')},{ROOM},,,\n",
        encoding="utf-8")
    ehr = out / "ehr"
    ehr.mkdir(exist_ok=True)
    for name, col in (("medications.csv", "taken_time"), ("flowsheet.csv", "recorded_time")):
        df = pd.read_csv(SAMPLE_DIR / "ehr" / name)
        ts = pd.to_datetime(df[col]) + delta
        df = df[(ts >= t0.replace(tzinfo=None)) & (ts < t_stop.replace(tzinfo=None))].copy()
        df[col] = ts[df.index].dt.strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(ehr / name, index=False)
    (out / "timelines").mkdir(exist_ok=True)
    shutil.copy(REPO / "tracer" / "perception" / "taxonomy" / "activity_key.csv", out / "timelines" / "activity_key.csv")
    extra = {k: json.loads((SAMPLE_DIR / f"{k}.json").read_text(encoding="utf-8"))
             for k in ("events", "demographics", "encounters")}
    for ev in extra["events"]:
        ev["time"] = _shift_iso(ev.get("time"), delta)
    extra["events"] = [ev for ev in extra["events"]
                       if ev.get("time") and t0 <= dt.datetime.fromisoformat(ev["time"]) < t_stop]
    for enc in extra["encounters"]:
        for k in ("hosp_admsn_time", "icu_admsn_time"):
            if k in enc:
                enc[k] = _shift_iso(enc[k], delta)
        # the stay ends with the video
        for k in ("hosp_disch_time", "icu_disch_time"):
            if k in enc:
                enc[k] = t_stop.strftime("%Y-%m-%dT%H:%M:%S") + ("+00:00" if "+" in str(enc[k]) else "")
        if "icu_days" in enc:
            enc["icu_days"] = round(n_minutes / 1440, 3)
    return extra


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", default=str(DEFAULT_VIDEO), help="continuous video file (default: the sample)")
    ap.add_argument("--out", default=str(REPO / "demo" / "quickstart"), help="data store to build (TRACER_DATA_ROOT)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="perception VLM, report LM, agent and video tool (default: %(default)s)")
    ap.add_argument("--report-model", default=None, help="report LM (default: --model)")
    ap.add_argument("--max-minutes", type=int, default=None, help="only process the first N minutes (quick first run)")
    ap.add_argument("--workers", type=int, default=8, help="concurrent perception calls")
    ap.add_argument("--height", type=int, default=540, help="clip height in pixels (default 540; keeps clips under the inline limit)")
    ap.add_argument("--crf", type=int, default=26)
    ap.add_argument("--start-utc", default=START_UTC, help="UTC clock assigned to the first minute of the video")
    ap.add_argument("--local-tz", default="UTC", help="site-local zone shown to the user (TRACER_LOCAL_TZ)")
    ap.add_argument("--skip-perception", action="store_true", help="reuse existing perception outputs only")
    ap.add_argument("--no-serve", action="store_true", help="build the store but do not start the viewer")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--fresh", action="store_true", help="delete --out first")
    args = ap.parse_args(argv)

    out = Path(args.out).expanduser().resolve()
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["TRACER_DATA_ROOT"] = str(out)
    os.environ["TRACER_RESULTS_ROOT"] = str(out / "results")
    os.environ["TRACER_CACHE_ROOT"] = str(out / "cache")
    os.environ["TRACER_LOCAL_TZ"] = args.local_tz
    report_model = args.report_model or args.model

    from ..backends.gemini import get_gemini_api_key, is_adc

    def _have_key() -> bool:
        if is_adc():
            return True
        try:
            get_gemini_api_key()
            return True
        except RuntimeError:
            return False

    def _require_key(what: str) -> None:
        # The repo ships the perception outputs and hourly report for the
        # sample video, so a key is needed only for work that is not cached
        # (and, in the viewer, for the chat panel).
        if not _have_key():
            get_gemini_api_key()   # raises with the backend's own message
            raise SystemExit(f"a Gemini API key is needed to {what}")

    _log("=" * 72)
    _log("TRACER quickstart")
    _log(f"  video:  {args.video}\n  store:  {out}\n  models: {args.model} (reports: {report_model})")
    _log("=" * 72)

    # ---- 1. clips -------------------------------------------------------------
    src = Path(args.video).expanduser()
    if not src.exists():
        if src.resolve() == DEFAULT_VIDEO.resolve():
            download_sample(DEFAULT_VIDEO)
        else:
            raise SystemExit(f"video not found: {src}")
    _log("\n[1/6] Splitting the video into one-minute clips")
    ffmpeg = _ffmpeg()
    clip_ids = split_into_clips(ffmpeg, src, out / "video", start_utc=args.start_utc, room=ROOM,
                                height=args.height, crf=args.crf, max_minutes=args.max_minutes)

    # ---- 2. store -------------------------------------------------------------
    _log("\n[2/6] Writing stays.csv and the synthetic EHR tables")
    extra = write_store(out, n_minutes=len(clip_ids), start_utc=args.start_utc)
    from .. import data
    data.reset_caches()

    # ---- 3. perception --------------------------------------------------------
    from ..perception.save_results import get_run_dir
    run_dir = get_run_dir(args.model, "perception_prompt", 0, out / "perception")
    from ..perception.save_results import video_id_to_filename
    missing = [c for c in clip_ids if not (run_dir / video_id_to_filename(c)).exists()]
    if args.skip_perception:
        _log(f"\n[3/6] Skipping perception (reusing {run_dir})")
    elif not missing:
        _log(f"\n[3/6] Perception outputs for all {len(clip_ids)} clips are already in the store; nothing to run")
    else:
        _log(f"\n[3/6] Minute-level perception with {args.model} ({len(missing)} of {len(clip_ids)} clips, {args.workers} workers)")
        _require_key(f"run perception on {len(missing)} clip(s)")
        clips_txt = out / "clips.txt"
        clips_txt.write_text("\n".join(clip_ids) + "\n", encoding="utf-8")
        from ..perception import run_perception
        run_perception.main([args.model, "--videos", str(clips_txt), "--video-dir", str(out / "video"),
                             "--output-dir", str(out / "perception"), "--workers", str(args.workers),
                             "--skip-existing", "--run", "0"])
    n_json = len(list(run_dir.glob("*.json"))) if run_dir.exists() else 0
    if n_json == 0:
        raise SystemExit(f"no perception outputs in {run_dir}")
    _log(f"  {n_json} per-clip outputs in {run_dir.relative_to(out)}")

    # ---- 4. timeline ----------------------------------------------------------
    _log("\n[4/6] Building the minute-level timeline")
    from ..perception import build_timeline
    info = build_timeline.build(PAT_ID, run_dir, out / "timelines", events=extra["events"],
                                demographics=extra["demographics"], encounters=extra["encounters"])
    active = {k: v for k, v in info["per_label_minutes"].items() if v}
    _log(f"  {info['n_minutes']} minutes / {info['n_hours']} hours from {info['icu_start']} local")
    _log("  minutes active per label: " + (", ".join(f"{k.replace('patient_', '')}={v}" for k, v in active.items()) or "none"))

    # ---- 5. reports -----------------------------------------------------------
    # A report summarizes the perception outputs that existed when it was
    # written, so a report from a shorter earlier run (e.g. --max-minutes 5)
    # must be regenerated once more minutes have been processed. The marker
    # records how many per-clip outputs the existing reports were built from.
    _log(f"\n[5/6] Generating hourly reports with {report_model}")
    from ..tools import common as _common
    _common.reset_caches()
    from ..reports import generate_reports
    reports_dir = out / "reports" / PAT_ID
    marker = reports_dir / ".built_from"
    signature = f"{n_json} clips from {args.start_utc}, {len(extra['events'])} events"
    built_from = marker.read_text().strip() if marker.exists() else None
    existing = sorted(reports_dir.glob("hour_*.md")) if reports_dir.exists() else []
    if built_from != signature or not existing:
        _require_key("generate the hourly report(s)")
    if built_from != signature and existing:
        for f in existing:
            f.unlink()
        _log(f"  inputs changed ({built_from} -> {signature}): regenerating {len(existing)} report(s)")
    rinfo = generate_reports.generate_for_stay(PAT_ID, run_dir, model=report_model, events=extra["events"],
                                               skip_existing=True, log=lambda m: _log("  " + m))
    reports_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(signature)
    _log(f"  generated {rinfo['generated']}, kept {rinfo['skipped']} existing -> {Path(rinfo['reports_dir']).relative_to(out)}")

    # ---- 6. viewer ------------------------------------------------------------
    if args.no_serve:
        _log(f"\nDone. Start the viewer with:\n  TRACER_DATA_ROOT={out} TRACER_LOCAL_TZ={args.local_tz} "
             f"python -m tracer.viewer --port {args.port} --model {args.model}")
        return 0
    _log(f"\n[6/6] Viewer: http://{args.host}:{args.port}/   (Ctrl-C to stop)")
    if not _have_key():
        _log("  note: no Gemini API key is set, so the chat panel cannot answer questions until "
             "GEMINI_API_KEY is exported and the viewer restarted; the timeline and reports work without it")
    from ..viewer.app import create_app
    app = create_app(pat_id=PAT_ID, chat_model=args.model)
    app.run(host=args.host, port=args.port, threaded=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

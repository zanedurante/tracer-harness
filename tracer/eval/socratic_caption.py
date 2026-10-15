"""Batch-caption every one-minute clip of a stay with Gemini using the HourVideo
caption prompt (Socratic Models baseline, step 1).

Each clip is sent inline with prompts/hour_video_caption.yaml; Gemini samples
at its default 1 fps. Captions are written to
<TRACER_CACHE_ROOT>/caption_cache/<model>/<YYYY_MM_DD>/<clip_stem>.txt and
skipped on rerun (fully resumable).

Usage:
    python -m tracer.eval.socratic_caption --patients stay_A,stay_B --model gemini-3.1-flash-lite
"""
from __future__ import annotations

import argparse
import gc
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml
from google.genai import types

from .. import config
from .frame_sampler import enumerate_clips
from .runners_socratic import caption_cache_dir

MAX_RETRIES = 8
RETRY_BACKOFF_BASE = 10


def load_caption_prompt() -> str:
    with open(config.PROMPTS_DIR / "hour_video_caption.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["prompt"].strip()


def cache_path(model: str, clip_path: Path) -> Path:
    return caption_cache_dir() / model / clip_path.parent.name / f"{clip_path.stem}.txt"


def _is_retryable(exc: Exception) -> bool:
    err = str(exc)
    status = getattr(exc, "status_code", None)
    return (status in (429, 500, 502, 503) or "429" in err or "RESOURCE_EXHAUSTED" in err
            or "500" in err or "503" in err or "timeout" in err.lower()
            or isinstance(exc, (TimeoutError, ConnectionError, OSError)))


def caption_one(clip_path: Path, model: str, prompt: str, client) -> dict:
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            t0 = time.time()
            part_video = types.Part(inline_data=types.Blob(data=clip_path.read_bytes(), mime_type="video/mp4"))
            response = client.models.generate_content(model=model, contents=[types.Part(text=prompt), part_video])
            elapsed = time.time() - t0
            meta = response.usage_metadata
            out = cache_path(model, clip_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(response.text or "", encoding="utf-8")
            gc.collect()
            return {"clip": clip_path.name, "elapsed": elapsed,
                    "prompt_tokens": getattr(meta, "prompt_token_count", None) if meta else None,
                    "response_tokens": getattr(meta, "candidates_token_count", None) if meta else None}
        except FileNotFoundError:
            raise
        except Exception as e:
            last_exc = e
            if _is_retryable(e) and attempt < MAX_RETRIES:
                wait = min(RETRY_BACKOFF_BASE * (2 ** attempt), 300)
                print(f"  [{clip_path.name}] RETRY {attempt + 1}/{MAX_RETRIES} (wait {wait}s): {type(e).__name__}: {e}", flush=True)
                time.sleep(wait)
                continue
            raise last_exc


def main(argv=None):
    ap = argparse.ArgumentParser(description="Batch-caption minute clips for the Socratic baseline.")
    ap.add_argument("--patients", required=True, help="comma-separated stay ids")
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--video-dir", default=None, help="override the clip root")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-skip", action="store_true")
    args = ap.parse_args(argv)

    prompt = load_caption_prompt()
    video_dir = Path(args.video_dir).expanduser() if args.video_dir else None
    all_clips: list[Path] = []
    for pat in [p.strip() for p in args.patients.split(",") if p.strip()]:
        clips = [c for _, c in enumerate_clips(pat, base_override=video_dir)]
        print(f"  {pat}: {len(clips)} clips")
        all_clips.extend(clips)
    if not args.no_skip:
        before = len(all_clips)
        all_clips = [c for c in all_clips if not cache_path(args.model, c).exists()]
        if before - len(all_clips):
            print(f"Skipping {before - len(all_clips)} already-cached clips")
    if args.limit > 0:
        all_clips = all_clips[: args.limit]
    total = len(all_clips)
    if total == 0:
        print("Nothing to caption.")
        return

    from ..backends.gemini import get_gemini_client
    client = get_gemini_client()
    print(f"Model: {args.model} | clips: {total} | workers: {args.workers} | cache: {caption_cache_dir() / args.model}")

    ok = failed = 0
    t_start = time.time()
    sem = threading.Semaphore(args.workers + 4)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_clip: dict = {}

        def _submit_all():
            for clip in all_clips:
                sem.acquire()
                f = executor.submit(caption_one, clip, args.model, prompt, client)
                future_to_clip[f] = clip
                f.add_done_callback(lambda _: sem.release())

        submitter = threading.Thread(target=_submit_all, daemon=True)
        submitter.start()
        completed = 0
        while submitter.is_alive() or future_to_clip:
            done_futures = [f for f in list(future_to_clip) if f.done()]
            if not done_futures:
                time.sleep(0.1)
                continue
            for future in done_futures:
                clip = future_to_clip.pop(future)
                completed += 1
                try:
                    future.result()
                    ok += 1
                except Exception as e:
                    failed += 1
                    print(f"  FAILED: {clip.name}: {e}", flush=True)
                    traceback.print_exc()
                if completed % 50 == 0 or completed == total:
                    wall = time.time() - t_start
                    print(f"[{completed}/{total}] OK:{ok} Fail:{failed} | {completed / wall:.1f} clips/s", flush=True)
        submitter.join()
    print(f"Done | OK: {ok} | Failed: {failed}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

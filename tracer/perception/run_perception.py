"""Concurrent minute-level perception: run a Gemini model on one-minute clips
with the perception prompt (prompts/perception_prompt.yaml) and save one JSON
per clip (tracer.perception.save_results).

Every call is made with no generation configuration (API defaults) unless
--temperature is given; the five repeated runs behind the paper's perception
table differ only in --run.

Usage:
    python -m tracer.perception.run_perception gemini-3.1-flash-lite --videos clips.txt
    python -m tracer.perception.run_perception gemini-3.1-flash-lite --annotations labels.csv --workers 32
    python -m tracer.perception.run_perception gemini-3.1-flash-lite --videos clips.txt --skip-existing --run 0

`clips.txt` lists video ids (one per line) relative to <TRACER_DATA_ROOT>/video
(or --video-dir): <YYYY_MM_DD>/<HH_MM>_<room>_<lt|rt>.mp4. The annotations CSV
(clip benchmark) has a `video_id` column plus one YES/NO column per label.
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from google.genai import types

from .prompts import load_prompt
from .save_results import get_run_dir, next_run_id, save_result, video_id_to_filename
from .video import load_video

DEFAULT_PROMPT = "perception_prompt"
MAX_RETRIES = 8
RETRY_BACKOFF_BASE = 30


def _is_retryable(exc: Exception) -> bool:
    err_str = str(exc)
    status = getattr(exc, "status_code", None)
    return (status in (429, 500, 502, 503) or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            or "500" in err_str or "503" in err_str or "timeout" in err_str.lower()
            or isinstance(exc, (TimeoutError, ConnectionError, OSError)))


def run_one(video_id: str, model: str, prompt, client, run_id: int, temperature=None, fps=None,
            video_root: Path | None = None, output_root: Path | None = None) -> dict:
    """Load, infer, and save a single clip end-to-end with retry. A missing
    clip (FileNotFoundError) is never retried."""
    last_exc = None
    for attempt in range(MAX_RETRIES + 1):
        v = contents = response = None
        try:
            t0 = time.time()
            v = load_video(video_id, fps=fps, root=video_root)
            contents = types.Content(role="user", parts=[types.Part(text=prompt.prompt), v.part,
                                                         types.Part(text=prompt.user_prompt)])
            config = types.GenerateContentConfig(temperature=temperature) if temperature is not None else None
            response = client.models.generate_content(model=model, contents=contents, config=config)
            elapsed = time.time() - t0
            del contents
            contents = None
            meta = response.usage_metadata
            prompt_tokens = getattr(meta, "prompt_token_count", None) if meta else None
            response_tokens = getattr(meta, "candidates_token_count", None) if meta else None
            total_tokens = getattr(meta, "total_token_count", None) if meta else None
            response_text = response.text
            del response
            response = None
            save_result(video=v, prompt=prompt, model=model, raw_response=response_text,
                        elapsed_seconds=round(elapsed, 2), prompt_tokens=prompt_tokens,
                        response_tokens=response_tokens, total_tokens=total_tokens, run_id=run_id,
                        output_dir=get_run_dir(model, prompt.name, run_id, output_root) if output_root else None,
                        fps=fps or 1)
            del v
            v = None
            gc.collect()
            return {"video_id": video_id, "elapsed": elapsed, "prompt_tokens": prompt_tokens,
                    "response_tokens": response_tokens}
        except FileNotFoundError:
            raise
        except Exception as e:
            last_exc = e
            del v, contents, response
            v = contents = response = None
            if _is_retryable(e) and attempt < MAX_RETRIES:
                wait = min(RETRY_BACKOFF_BASE * (2 ** attempt), 300)
                print(f"  [{video_id}] RETRY ({attempt + 1}/{MAX_RETRIES}, waiting {wait}s): {type(e).__name__}: {e}", flush=True)
                time.sleep(wait)
                continue
            raise last_exc


def _fmt_duration(seconds: float) -> str:
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    h, remainder = divmod(s, 3600)
    return f"{h}h {remainder // 60}m"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run minute-level perception on one-minute clips (concurrent)")
    parser.add_argument("model", nargs="?", default="gemini-3.1-flash-lite")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help=f"prompt YAML stem in prompts/ (default: {DEFAULT_PROMPT})")
    parser.add_argument("--videos", default=None, metavar="FILE", help="text file with video ids, one per line")
    parser.add_argument("--annotations", default=None, metavar="CSV", help="clip-benchmark CSV with a video_id column")
    parser.add_argument("--video-dir", default=None, help="clip root (default: <TRACER_DATA_ROOT>/video)")
    parser.add_argument("--output-dir", default=None, help="perception output root (default: <TRACER_DATA_ROOT>/perception)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--run", type=int, default=None, metavar="N", help="run id (default: next available)")
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=None, help="sampling temperature (default: API default)")
    parser.add_argument("--output-name", default=None, help="override the output directory name")
    parser.add_argument("--fps", type=int, default=None, help="custom sampling fps (default: 1 fps)")
    args = parser.parse_args(argv)

    if not args.videos and not args.annotations:
        parser.error("pass --videos FILE or --annotations CSV")
    video_root = Path(args.video_dir).expanduser() if args.video_dir else None
    output_root = Path(args.output_dir).expanduser() if args.output_dir else None
    prompt = load_prompt(args.prompt)
    if args.output_name:
        prompt = prompt._replace(name=args.output_name)
    else:
        name = prompt.name
        if args.fps is not None:
            name += f"_fps{args.fps}"
        if args.temperature is not None:
            name += "_temp" + f"{args.temperature}".replace(".", "p")
        if name != prompt.name:
            prompt = prompt._replace(name=name)

    if args.videos:
        video_ids = [ln.strip() for ln in open(args.videos, encoding="utf-8") if ln.strip()]
    else:
        with open(args.annotations, encoding="utf-8") as f:
            video_ids = [row["video_id"] for row in csv.DictReader(f)]

    run_id = args.run if args.run is not None else (0 if args.videos else next_run_id(args.model, prompt.name, output_root))
    run_dir = get_run_dir(args.model, prompt.name, run_id, output_root)
    if args.skip_existing:
        before = len(video_ids)
        video_ids = [v for v in video_ids if not (run_dir / video_id_to_filename(v)).exists()]
        print(f"Skipping {before - len(video_ids)} already-processed clips")
    if args.limit > 0:
        video_ids = video_ids[: args.limit]

    from ..backends.gemini import get_gemini_client
    client = get_gemini_client()
    total = len(video_ids)
    print(f"Model: {args.model} | prompt: {prompt.name} | run: run_{run_id} ({run_dir}) | clips: {total} | workers: {args.workers}")
    if total == 0:
        return

    ok = failed = skipped = 0
    t_start = time.time()
    completed = 0
    sem = threading.Semaphore(args.workers + 4)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_to_vid: dict = {}

        def _submit_all():
            for vid in video_ids:
                sem.acquire()
                f = executor.submit(run_one, vid, args.model, prompt, client, run_id, args.temperature, args.fps,
                                    video_root, output_root)
                future_to_vid[f] = vid
                f.add_done_callback(lambda _: sem.release())

        submitter = threading.Thread(target=_submit_all, daemon=True)
        submitter.start()
        last_log = t_start
        while submitter.is_alive() or future_to_vid:
            done = [f for f in list(future_to_vid) if f.done()]
            if not done:
                time.sleep(0.1)
                continue
            for future in done:
                video_id = future_to_vid.pop(future)
                completed += 1
                try:
                    future.result()
                    ok += 1
                except FileNotFoundError:
                    skipped += 1
                    print(f"  SKIPPED (file not found): {video_id}", flush=True)
                except Exception as e:
                    failed += 1
                    print(f"  FAILED: {video_id}: {e}", flush=True)
                    traceback.print_exc()
                now = time.time()
                if completed == total or now - last_log >= 60:
                    wall = now - t_start
                    rate = completed / wall if wall > 0 else 0
                    print(f"[{completed:,}/{total:,}] OK:{ok:,} Fail:{failed:,} Skip:{skipped:,} | {rate:.1f} clips/s | "
                          f"Elapsed: {_fmt_duration(wall)} | ETA: {_fmt_duration((total - completed) / rate if rate else 0)}", flush=True)
                    last_log = now
        submitter.join()
    print(f"\nDone in {_fmt_duration(time.time() - t_start)} | OK: {ok:,} | Failed: {failed:,} | Skipped: {skipped:,}")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

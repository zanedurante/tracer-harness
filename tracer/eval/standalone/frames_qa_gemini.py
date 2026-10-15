#!/usr/bin/env python3
"""Long-context frames MCQ baseline with position rotation, Gemini via
Vertex AI (ADC) or an API key. Reads an unpacked bundle (tracer.eval.bundle:
manifest.json + frames/). Needs only google-genai.

Env: GOOGLE_CLOUD_PROJECT (+ optional GOOGLE_CLOUD_LOCATION, default "global")
for ADC; or GEMINI_API_KEY for the API-key path.

    python frames_qa_gemini.py --bundle-dir bundle_lite_n950 --model gemini-3.1-flash-lite --positions 5
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import LETTERS, Cache, extract_letter, fmt_duration, is_retryable, print_position_summary  # noqa: E402

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402
from google.genai.types import HttpOptions  # noqa: E402

import json  # noqa: E402

MAX_RETRIES = 8
RETRY_BACKOFF_BASE = 10

FRAMES_PREAMBLE = (
    "You are shown a large sequence of frames uniformly sampled from an ICU "
    "patient's surveillance video over their entire stay. The frames are in "
    "chronological order (earliest first). Use them to answer the following "
    "multiple-choice question. You do NOT have access to any other data or "
    "tools — reason from the frames alone. Choose exactly one option even if "
    "uncertain — never refuse. End your reply with a line in exactly this "
    "form:\nAnswer: <LETTER>"
)


def _extract_distractors(item):
    gold = item["gold_answer"]
    return [v for v in item["options"].values() if v != gold]


def _item_at_pos(item, pos):
    gold = item["gold_answer"]
    opts = list(_extract_distractors(item))
    opts.insert(pos, gold)
    letters = LETTERS[:len(opts)]
    options = dict(zip(letters, opts))
    opt_lines = "\n".join("%s) %s" % (L, options[L]) for L in letters)
    prompt = "%s\n\nQUESTION: %s\n\nOPTIONS:\n%s" % (FRAMES_PREAMBLE, item["question"], opt_lines)
    return options, letters[pos], prompt


def _n_positions(item):
    return len(_extract_distractors(item)) + 1


def get_client():
    if os.environ.get("GEMINI_API_KEY"):
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=HttpOptions(timeout=300_000))
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise SystemExit("set GEMINI_API_KEY, or GOOGLE_CLOUD_PROJECT for Vertex AI (ADC)")
    return genai.Client(vertexai=True, project=project, location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
                        http_options=HttpOptions(timeout=300_000))


def call_gemini(parts, model, client):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.models.generate_content(model=model, contents=parts).text or ""
        except Exception as e:
            if is_retryable(e) and attempt < MAX_RETRIES:
                wait = min(RETRY_BACKOFF_BASE * (2 ** attempt), 300)
                print("  RETRY %d/%d (wait %ds): %s: %s" % (attempt + 1, MAX_RETRIES, wait, type(e).__name__, e), flush=True)
                time.sleep(wait)
                continue
            raise


def main():
    ap = argparse.ArgumentParser(description="Frames MCQ baseline (Gemini)")
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--positions", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--patient", default=None)
    ap.add_argument("--sleep-between", type=float, default=2.0)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--frame-stride", type=int, default=1, help="sub-sample bundled frames (2 = 475 of 950)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    bundle_dir = Path(args.bundle_dir).resolve()
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    items, patients = manifest["items"], manifest["patients"]
    if args.patient:
        items = [it for it in items if it["patient_id"] == args.patient]
    if args.num_shards > 1:
        items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard]
    if args.limit > 0:
        items = items[:args.limit]
    results_dir = Path(args.results_dir) if args.results_dir else bundle_dir / "results"
    cache_path = results_dir / args.model / ("cache.jsonl" if args.num_shards == 1 else "cache.shard%d.jsonl" % args.shard)

    work = []
    for it in items:
        if args.positions <= 1:
            work.append((it, None))
        else:
            work.extend((it, pos) for pos in range(min(args.positions, _n_positions(it))))

    print("Preloading frame bytes per stay...", flush=True)
    pat_frame_parts = {}
    for pid in sorted(set(it["patient_id"] for it in items)):
        paths = patients[pid]["frame_paths"][::args.frame_stride]
        pat_frame_parts[pid] = [types.Part.from_bytes(data=(bundle_dir / rel).read_bytes(), mime_type="image/jpeg")
                                for rel in paths]
        print("  %s: %d frames" % (pid, len(pat_frame_parts[pid])), flush=True)

    client = get_client()
    cache = Cache(cache_path)
    todo = [(it, pos) for it, pos in work if cache.get("vlm_frames", args.model, it["item_id"], pos) is None]
    print("Model: %s | questions: %d | positions: %d | work: %d (cached %d) | cache: %s" % (
        args.model, len(items), args.positions, len(work), len(work) - len(todo), cache_path), flush=True)

    t_start = time.time()
    ok = errors = 0
    for idx, (item, pos) in enumerate(todo, start=1):
        pid = item["patient_id"]
        if pos is not None:
            options, correct_letter, prompt = _item_at_pos(item, pos)
        else:
            options, correct_letter, prompt = item["options"], item["correct_letter"], item["prompt"]
        valid = "".join(sorted(options.keys()))
        parts = list(pat_frame_parts[pid]) + [types.Part(text=prompt)]
        chosen, raw, error, method = None, "", None, ""
        try:
            raw = call_gemini(parts, args.model, client)
            chosen, method = extract_letter(raw, valid)
        except Exception as e:
            error = "%s: %s" % (type(e).__name__, str(e)[:200])
            errors += 1
        is_correct = (chosen == correct_letter) if chosen else False
        rec = {"system": "vlm_frames", "item_id": item["item_id"], "patient_id": pid, "pos": pos,
               "model": args.model, "chosen_letter": chosen, "correct_letter": correct_letter,
               "is_correct": is_correct, "raw_text": raw[:500], "extraction_method": method,
               "n_tool_calls": 0, "tool_calls": [], "error": error}
        if not error:
            cache.put(rec)
            ok += is_correct
        wall = time.time() - t_start
        rate = idx / wall if wall > 0 else 0
        print("(%d/%d) row %d pos=%s -> %s %s | %.1f/min | ETA: %s" % (
            idx, len(todo), item["item_id"], pos, chosen, "OK" if is_correct else ("ERR" if error else "X"),
            rate * 60, fmt_duration((len(todo) - idx) / rate if rate else 0)), flush=True)
        if args.sleep_between and not error:
            time.sleep(args.sleep_between)
        gc.collect()

    print("\nDone in %s | ran %d | correct %d | errors %d" % (fmt_duration(time.time() - t_start), len(todo), ok, errors))
    print_position_summary(cache, args.model, "vlm_frames")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()

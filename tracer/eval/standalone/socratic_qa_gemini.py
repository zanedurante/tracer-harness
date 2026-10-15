#!/usr/bin/env python3
"""Socratic QA (HourVideo protocol) with position rotation, Gemini answerer.
Needs google-genai, pyyaml.

Reads captions from <caption-dir>/<caption-model>/<YYYY_MM_DD>/<HH_MM>_<room>_<view>.txt
(written by tracer.eval.socratic_caption), subsamples them at a fixed linear
stride to fit the context window (286 tokens per caption in a 1M window), and
answers each MCQ with the HourVideo question prompt (prompts/hour_video_qa.yaml).

    python socratic_qa_gemini.py --stays stays.csv --csv questions.csv --caption-dir cache/caption_cache \
        --qa-prompt prompts/hour_video_qa.yaml --model gemini-3.1-flash-lite --positions 5
"""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import math
import os
import re
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (Cache, extract_socratic_answer, fmt_duration, is_retryable, item_with_gold_at,  # noqa: E402
                     load_mcqs, load_stays, n_positions, print_position_summary, render_options)

CLIP_RE = re.compile(r"^(?P<hh>\d{2})_(?P<mm>\d{2})_(?P<room>[a-zA-Z0-9]+)_(?P<view>lt|rt)$")
CONTEXT_TOKEN_LIMIT = 1_000_000
TOKENS_PER_CAPTION = 286
PROMPT_OVERHEAD_TOKENS = 5_000
MAX_RETRIES = 8
RETRY_BACKOFF_BASE = 10

STAYS: dict = {}


def load_captions(patient_id, caption_model, caption_dir):
    info = STAYS[patient_id]
    room, start, end = info["room"], info["start_utc"], info["end_utc"]
    model_dir = Path(caption_dir) / caption_model
    if not model_dir.is_dir():
        return []
    out = []
    d = start.date()
    while d <= end.date():
        folder = model_dir / d.strftime("%Y_%m_%d")
        d += dt.timedelta(days=1)
        if not folder.is_dir():
            continue
        for txt in sorted(folder.glob("*.txt")):
            m = CLIP_RE.match(txt.stem)
            if not m or m.group("room") != room:
                continue
            ts = dt.datetime.strptime(folder.name, "%Y_%m_%d").replace(hour=int(m.group("hh")), minute=int(m.group("mm")))
            if ts < start or ts > end:
                continue
            caption = txt.read_text(encoding="utf-8").strip()
            if caption:
                out.append((ts, caption))
    out.sort()
    return out


def compute_subsample_rate(n_captions):
    max_caps = (CONTEXT_TOKEN_LIMIT - PROMPT_OVERHEAD_TOKENS) // TOKENS_PER_CAPTION
    return max(1, math.ceil(n_captions / max_caps))


def format_captions(captions):
    return "\n\n".join("[%s UTC]\n%s" % (ts.strftime("%Y-%m-%d %H:%M"), text) for ts, text in captions)


def get_client():
    from google import genai
    from google.genai.types import HttpOptions
    if os.environ.get("GEMINI_API_KEY"):
        return genai.Client(api_key=os.environ["GEMINI_API_KEY"], http_options=HttpOptions(timeout=300_000))
    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        raise SystemExit("set GEMINI_API_KEY, or GOOGLE_CLOUD_PROJECT for Vertex AI (ADC)")
    return genai.Client(vertexai=True, project=project, location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
                        http_options=HttpOptions(timeout=300_000))


def call_qa(prompt, model, client):
    for attempt in range(MAX_RETRIES + 1):
        try:
            return client.models.generate_content(model=model, contents=prompt).text or ""
        except Exception as e:
            if is_retryable(e) and attempt < MAX_RETRIES:
                wait = min(RETRY_BACKOFF_BASE * (2 ** attempt), 300)
                print("  RETRY %d/%d (wait %ds): %s: %s" % (attempt + 1, MAX_RETRIES, wait, type(e).__name__, e), flush=True)
                time.sleep(wait)
                continue
            raise


def main():
    ap = argparse.ArgumentParser(description="Socratic QA (Gemini answerer)")
    ap.add_argument("--stays", required=True, help="stays.csv")
    ap.add_argument("--csv", required=True, help="question bank CSV")
    ap.add_argument("--caption-dir", required=True)
    ap.add_argument("--qa-prompt", required=True, help="prompts/hour_video_qa.yaml")
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--caption-model", default=None, help="caption store (default: --model)")
    ap.add_argument("--positions", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--patient", default=None)
    ap.add_argument("--sleep-between", type=float, default=5.0)
    ap.add_argument("--results-dir", default="results/socratic")
    args = ap.parse_args()

    STAYS.update(load_stays(args.stays))
    qa_data = yaml.safe_load(open(args.qa_prompt, encoding="utf-8"))
    qa_prompt, qa_prompt_last = qa_data["prompt"].strip(), qa_data["prompt_last"].strip()
    records = load_mcqs(args.csv)
    if args.patient:
        records = [r for r in records if r["patient_id"] == args.patient]
    if args.limit > 0:
        records = records[:args.limit]
    print("Loaded %d MCQs" % len(records), flush=True)

    patient_captions = {}
    for pat in sorted(set(r["patient_id"] for r in records)):
        caps = load_captions(pat, args.caption_model or args.model, args.caption_dir)
        rate = compute_subsample_rate(len(caps))
        sampled = caps[::rate]
        patient_captions[pat] = format_captions(sampled)
        print("  %s: %d captions, subsample x%d -> %d" % (pat, len(caps), rate, len(sampled)), flush=True)

    client = get_client()
    cache = Cache(Path(args.results_dir) / args.model / "cache.jsonl")
    total = sum(min(args.positions, n_positions(r)) for r in records)
    done = ok = errors = 0
    t_start = time.time()
    for rec in records:
        for pos in range(min(args.positions, n_positions(rec))):
            done += 1
            options, correct_letter = item_with_gold_at(rec, pos)
            cached = cache.get("socratic", args.model, rec["id"], pos)
            if cached is not None:
                ok += bool(cached["is_correct"])
                continue
            full_prompt = "%s\n\n%s\n\n%s\n\nTest #1:\n%s\n%s" % (
                qa_prompt, patient_captions[rec["patient_id"]], qa_prompt_last, rec["question"], render_options(options))
            chosen, raw, error = None, "", None
            try:
                raw = call_qa(full_prompt, args.model, client)
                chosen = extract_socratic_answer(raw, len(options))
            except Exception as e:
                error = "%s: %s" % (type(e).__name__, str(e)[:200])
                errors += 1
            is_correct = chosen == correct_letter if chosen else False
            result = {"system": "socratic", "model": args.model, "item_id": rec["id"], "patient_id": rec["patient_id"],
                      "pos": pos, "chosen_letter": chosen, "correct_letter": correct_letter,
                      "is_correct": is_correct, "raw_text": raw[:500], "error": error}
            if not error:
                cache.put(result)
                ok += is_correct
            wall = time.time() - t_start
            rate = done / wall if wall > 0 else 0
            print("(%d/%d) row %d pos=%d -> %s %s | ETA: %s" % (
                done, total, rec["id"], pos, chosen, "OK" if is_correct else ("ERR" if error else "X"),
                fmt_duration((total - done) / rate if rate else 0)), flush=True)
            if args.sleep_between and not error:
                time.sleep(args.sleep_between)
            gc.collect()
    print("\nDone in %s | OK: %d | Errors: %d" % (fmt_duration(time.time() - t_start), ok, errors))
    print_position_summary(cache, args.model, "socratic")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()

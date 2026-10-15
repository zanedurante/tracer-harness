#!/usr/bin/env python3
"""Socratic QA with an OpenAI-compatible (vLLM) ANSWERING model over Gemini
per-minute captions (paper appendix, Table "Socratic baselines with Gemini
3.1 Flash Lite captions").

Reuses socratic_qa_gemini.py end to end and swaps three things:
  * the caption store is always --caption-model, independent of the answerer;
  * the model call goes to $OPENAI_BASE_URL (vLLM) instead of Gemini;
  * the context budget is --ctx minus the answer budget, and tokens per caption
    are measured on the server's /tokenize endpoint over a caption sample, so
    the subsample stride is exact per model.

    OPENAI_BASE_URL=http://localhost:8000/v1 python socratic_qa_openai.py \
        --stays stays.csv --csv questions.csv --caption-dir cache/caption_cache \
        --qa-prompt prompts/hour_video_qa.yaml --model gpt-oss-20b --ctx 131072 \
        --results-dir results/socratic_gemcap
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import socratic_qa_gemini as svm  # noqa: E402
from _common import load_stays  # noqa: E402

ANSWER_TOKENS = 1024
SAFETY = 1.05


def _base_url() -> str:
    return os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")


def measure_tokens_per_caption(caption_dir: Path, caption_model: str, n_sample: int = 60) -> int:
    caps = []
    for pat in svm.STAYS:
        caps += svm.load_captions(pat, caption_model, caption_dir)
    random.Random(0).shuffle(caps)
    sample = caps[:n_sample]
    if not sample:
        raise RuntimeError("no captions found under %s/%s" % (caption_dir, caption_model))
    url = _base_url()[:-3] + "/tokenize"
    total = 0
    for ts, text in sample:
        body = json.dumps({"prompt": svm.format_captions([(ts, text)]) + "\n\n"}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            total += int(json.loads(r.read())["count"])
    return int(round(total / len(sample) * SAFETY))


def main():
    ap = argparse.ArgumentParser(description="Socratic QA: vLLM answerer over Gemini captions")
    ap.add_argument("--stays", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--caption-dir", required=True)
    ap.add_argument("--qa-prompt", required=True)
    ap.add_argument("--model", required=True, help="served model name on the vLLM endpoint")
    ap.add_argument("--ctx", type=int, required=True, help="served max_model_len (tokens)")
    ap.add_argument("--caption-model", default="gemini-3.1-flash-lite")
    ap.add_argument("--positions", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--patient", default=None)
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=ANSWER_TOKENS,
                    help="answer budget (8192 for gpt-oss-20b, whose reasoning channel otherwise consumes it)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    args = ap.parse_args()

    svm.STAYS.update(load_stays(args.stays))
    if args.num_shards > 1:
        _orig = svm.load_mcqs

        def load_shard(csv_path):
            return [r for i, r in enumerate(_orig(csv_path)) if i % args.num_shards == args.shard]
        svm.load_mcqs = load_shard

    from openai import OpenAI
    client = OpenAI(base_url=_base_url(), api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"))

    tpc = measure_tokens_per_caption(Path(args.caption_dir), args.caption_model)
    svm.CONTEXT_TOKEN_LIMIT = args.ctx - args.max_tokens
    svm.TOKENS_PER_CAPTION = tpc
    max_caps = (svm.CONTEXT_TOKEN_LIMIT - svm.PROMPT_OVERHEAD_TOKENS) // tpc
    print("Answerer: %s @ %s | captions: %s | ctx=%d - answer=%d ; tokens/caption=%d ; max captions=%d"
          % (args.model, _base_url(), args.caption_model, args.ctx, args.max_tokens, tpc, max_caps), flush=True)

    def call_openai(prompt, model, _client):
        resp = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                              max_tokens=args.max_tokens)
        return (resp.choices[0].message.content or "").strip()
    svm.call_qa = call_openai
    svm.get_client = lambda: client
    svm.MAX_RETRIES = 3

    sys.argv = [sys.argv[0], "--stays", args.stays, "--csv", args.csv, "--caption-dir", args.caption_dir,
                "--qa-prompt", args.qa_prompt, "--model", args.model, "--caption-model", args.caption_model,
                "--positions", str(args.positions), "--limit", str(args.limit), "--sleep-between", "0",
                "--results-dir", args.results_dir]
    if args.patient:
        sys.argv += ["--patient", args.patient]
    svm.main()


if __name__ == "__main__":
    main()

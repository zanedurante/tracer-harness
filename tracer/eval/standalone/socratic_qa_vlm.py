#!/usr/bin/env python3
"""Socratic QA (step 2) with a local HuggingFace model: text-only MCQ answering
over JSONL captions from socratic_caption_vlm.py, subsampled to the model's
context (tokens per caption measured with the model's tokenizer). Greedy
decoding in bfloat16, thinking disabled. Needs torch, transformers, pyyaml.

    python socratic_qa_vlm.py --stays stays.csv --captions captions.jsonl \
        --model /models/Qwen3-VL-8B-Instruct --csv questions.csv \
        --qa-prompt prompts/hour_video_qa.yaml --positions 5
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import gc  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (Cache, extract_socratic_answer, fmt_duration, item_with_gold_at, load_mcqs,  # noqa: E402
                     load_stays, n_positions, print_position_summary, render_options)

try:
    import torch
    import yaml
    from transformers import AutoConfig, AutoTokenizer
except ImportError as e:
    print(f"ERROR: required dependency missing: {e}", file=sys.stderr)
    sys.exit(2)

PROMPT_OVERHEAD_TOKENS = 1_500
MAX_NEW_TOKENS = 512


def resolve_caption_paths(paths):
    files = []
    for p in paths:
        p = Path(p)
        if p.is_dir():
            files.extend(sorted(p.glob("*.jsonl")))
        elif p.is_file():
            files.append(p)
    return files


def load_captions_jsonl(jsonl_paths, patient_id, meta):
    start, end = meta["start_utc"], meta["end_utc"]
    out = []
    for jp in jsonl_paths:
        with open(jp, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec["patient_id"] != patient_id or rec.get("error") or not rec.get("caption"):
                    continue
                ts = dt.datetime.fromisoformat(rec["minute_utc"])
                if ts < start or ts > end:
                    continue
                out.append((ts, rec["caption"]))
    out.sort()
    return out


def format_captions(captions):
    return "\n\n".join("[%s UTC]\n%s" % (ts.strftime("%Y-%m-%d %H:%M"), text) for ts, text in captions)


def estimate_tokens_per_caption(captions, tokenizer, n_sample=50):
    sample = captions[:n_sample]
    if not sample:
        return 286
    total = sum(len(tokenizer.encode(cap, add_special_tokens=False)) for _, cap in sample)
    return max(1, total // len(sample))


def compute_subsample_rate(n_captions, tokens_per_caption, context_limit):
    max_caps = (context_limit - PROMPT_OVERHEAD_TOKENS) // tokens_per_caption
    return n_captions if max_caps <= 0 else max(1, math.ceil(n_captions / max_caps))


def _pick_model_class(model_path: str):
    mt = getattr(AutoConfig.from_pretrained(model_path), "model_type", "")
    if mt.startswith("qwen3_vl"):
        from transformers import Qwen3VLForConditionalGeneration as MC
    elif mt.startswith("qwen2_5_vl") or mt.startswith("qwen2_vl"):
        from transformers import Qwen2_5_VLForConditionalGeneration as MC
    else:
        from transformers import AutoModelForCausalLM as MC
    return MC


def load_model(model_path, dtype_str, device_map):
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype_str]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    MC = _pick_model_class(model_path)
    kwargs = dict(torch_dtype=torch_dtype, trust_remote_code=True, low_cpu_mem_usage=True, attn_implementation="sdpa")
    model = MC.from_pretrained(model_path, **kwargs).to("cuda") if device_map == "cuda" else \
        MC.from_pretrained(model_path, device_map=device_map, **kwargs)
    model.eval()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    max_ctx = getattr(config, "max_position_embeddings",
                      getattr(getattr(config, "text_config", None), "max_position_embeddings", 32768))
    return tokenizer, model, max_ctx


def generate_text(tokenizer, model, prompt, max_new_tokens=MAX_NEW_TOKENS):
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(input_ids=inputs["input_ids"].to("cuda"), attention_mask=inputs["attention_mask"].to("cuda"),
                             max_new_tokens=max_new_tokens, do_sample=False, temperature=None, top_p=None)
    raw = tokenizer.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()


def main():
    ap = argparse.ArgumentParser(description="Socratic QA with a local HF model")
    ap.add_argument("--stays", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-label", default=None)
    ap.add_argument("--captions", required=True, nargs="+")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--qa-prompt", required=True)
    ap.add_argument("--results-dir", default="results/socratic_vlm")
    ap.add_argument("--positions", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--patient", default=None)
    ap.add_argument("--context-limit", type=int, default=0, help="0 = auto-detect from the model config")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="auto")
    args = ap.parse_args()

    stays = load_stays(args.stays)
    qa_data = yaml.safe_load(open(args.qa_prompt, encoding="utf-8"))
    qa_prompt, qa_prompt_last = qa_data["prompt"].strip(), qa_data["prompt_last"].strip()
    records = load_mcqs(args.csv)
    if args.patient:
        records = [r for r in records if r["patient_id"] == args.patient]
    if args.limit > 0:
        records = records[:args.limit]
    caption_files = resolve_caption_paths(args.captions)
    if not caption_files:
        sys.exit("no caption files found")
    raw_caps = {pat: load_captions_jsonl(caption_files, pat, stays[pat]) for pat in sorted(set(r["patient_id"] for r in records))}
    tokenizer, model, model_max_ctx = load_model(args.model, args.dtype, args.device_map)
    context_limit = args.context_limit or model_max_ctx
    tpc = estimate_tokens_per_caption([c for caps in raw_caps.values() for c in caps], tokenizer)
    patient_captions = {}
    for pat, caps in raw_caps.items():
        rate = compute_subsample_rate(len(caps), tpc, context_limit)
        patient_captions[pat] = format_captions(caps[::rate])
        print(f"  {pat}: {len(caps)} captions, subsample x{rate} -> {len(caps[::rate])}", flush=True)

    model_label = args.model_label or Path(args.model).name
    cache = Cache(Path(args.results_dir) / model_label / "cache.jsonl")
    total = sum(min(args.positions, n_positions(r)) for r in records)
    done = ok = errors = 0
    t_start = time.time()
    for rec in records:
        for pos in range(min(args.positions, n_positions(rec))):
            done += 1
            options, correct_letter = item_with_gold_at(rec, pos)
            if cache.get("socratic", model_label, rec["id"], pos) is not None:
                continue
            full_prompt = "%s\n\n%s\n\n%s\n\nTest #1:\n%s\n%s" % (
                qa_prompt, patient_captions[rec["patient_id"]], qa_prompt_last, rec["question"], render_options(options))
            chosen, raw, error = None, "", None
            try:
                raw = generate_text(tokenizer, model, full_prompt)
                chosen = extract_socratic_answer(raw, len(options))
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
                errors += 1
                traceback.print_exc()
            is_correct = chosen == correct_letter if chosen else False
            result = {"system": "socratic", "model": model_label, "item_id": rec["id"], "patient_id": rec["patient_id"],
                      "pos": pos, "chosen_letter": chosen, "correct_letter": correct_letter,
                      "is_correct": is_correct, "raw_text": raw[:500], "error": error}
            if not error:
                cache.put(result)
                ok += is_correct
            print(f"({done}/{total}) row {rec['id']} pos={pos} -> {chosen} {'OK' if is_correct else ('ERR' if error else 'X')}", flush=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(f"\nDone in {fmt_duration(time.time() - t_start)} | OK: {ok} | Errors: {errors}")
    print_position_summary(cache, model_label, "socratic")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()

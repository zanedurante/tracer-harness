#!/usr/bin/env python3
"""Long-context frames baseline with a local HuggingFace VLM, from a bundle
(tracer.eval.bundle). Greedy decoding in bfloat16; one MCQ per forward pass.
Output records mirror tracer.eval.runners.RunResult so the local scoring code
consumes them unchanged. Resumable: item/position pairs already in --out are
skipped. Needs torch, transformers, pillow (+ qwen_vl_utils for Qwen-VL).

    python run_local_vlm.py --bundle-dir bundle_n950 --model /models/Qwen3-VL-8B-Instruct \
        --model-label qwen3-vl-8b --out results/qwen3-vl-8b/results.jsonl --positions 5
    # 475 frames: --frame-stride 2
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import LETTERS, extract_letter  # noqa: E402

try:
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
except ImportError as e:
    print(f"ERROR: required dependency missing: {e}", file=sys.stderr)
    print("Install with: pip install torch transformers pillow accelerate", file=sys.stderr)
    sys.exit(2)

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
    opts = list(_extract_distractors(item))
    opts.insert(pos, item["gold_answer"])
    letters = LETTERS[:len(opts)]
    options = dict(zip(letters, opts))
    opt_lines = "\n".join("%s) %s" % (L, options[L]) for L in letters)
    return options, letters[pos], "%s\n\nQUESTION: %s\n\nOPTIONS:\n%s" % (FRAMES_PREAMBLE, item["question"], opt_lines)


def _n_positions(item):
    return len(_extract_distractors(item)) + 1


def _pick_model_class(model_path: str):
    mt = getattr(AutoConfig.from_pretrained(model_path), "model_type", "")
    print(f"[load] model_type={mt!r}", flush=True)
    if mt.startswith("qwen3_vl"):
        from transformers import Qwen3VLForConditionalGeneration as MC
    elif mt.startswith("gemma4"):
        from transformers import Gemma4ForConditionalGeneration as MC
    elif mt.startswith("qwen2_5_vl") or mt.startswith("qwen2_vl"):
        from transformers import Qwen2_5_VLForConditionalGeneration as MC
    else:
        from transformers import AutoModelForImageTextToText as MC
    return MC


def load_model(model_path, dtype, device_map, trust_remote_code):
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    MC = _pick_model_class(model_path)
    kwargs = dict(torch_dtype=torch_dtype, trust_remote_code=trust_remote_code, low_cpu_mem_usage=True,
                  attn_implementation="sdpa")
    model = MC.from_pretrained(model_path, **kwargs).to("cuda") if device_map == "cuda" else \
        MC.from_pretrained(model_path, device_map=device_map, **kwargs)
    model.eval()
    return processor, model


def _family(model) -> str:
    cls = type(model).__name__.lower()
    if any(k in cls for k in ("qwen3vl", "qwen3_vl", "qwen2_5_vl", "qwen2vl")):
        return "qwen_vl"
    if "gemma4" in cls or "gemma_4" in cls:
        return "gemma"
    return "auto"


def generate_one(processor, model, images, prompt, *, max_new_tokens, do_sample, family):
    if family == "qwen_vl":
        from qwen_vl_utils import process_vision_info
        messages = [{"role": "user", "content": [{"type": "image", "image": img} for img in images]
                     + [{"type": "text", "text": prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, _ = process_vision_info([messages], return_video_kwargs=True)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt").to("cuda")
    else:
        content = [{"type": "image", "image": img} for img in images] + [{"type": "text", "text": prompt}]
        inputs = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=True,
                                               return_tensors="pt", add_generation_prompt=True, return_dict=True)
        inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                             temperature=0.0 if not do_sample else 0.7)
    new_tokens = out[0][inputs["input_ids"].shape[-1]:]
    tok = getattr(processor, "tokenizer", None) or AutoTokenizer.from_pretrained(model.config._name_or_path, trust_remote_code=True)
    return tok.decode(new_tokens, skip_special_tokens=True)


def load_done(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    for line in out_path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if rec.get("error") is None and rec.get("chosen_letter") is not None:
            done.add((rec["item_id"], rec.get("pos")))
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run a manifest of MCQ+frames items through a local HF VLM.")
    ap.add_argument("--bundle-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-label", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--trust-remote-code", action="store_true", default=True)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--patient", default=None)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--positions", type=int, default=5)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    bundle_dir = Path(args.bundle_dir).resolve()
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    items, patients = manifest["items"], manifest["patients"]
    if args.patient:
        items = [it for it in items if it["patient_id"] == args.patient]
    if args.num_shards > 1:
        items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard]
    if args.limit:
        items = items[:args.limit]
    work = []
    for it in items:
        if args.positions <= 1:
            work.append((it, None))
        else:
            work.extend((it, pos) for pos in range(min(args.positions, _n_positions(it))))
    out_path = Path(args.out).resolve()
    done = load_done(out_path)
    todo = [(it, pos) for it, pos in work if (it["item_id"], pos) not in done]
    print(f"[bundle] {bundle_dir} items={len(items)} x positions={args.positions} = {len(work)} "
          f"(resume: {len(done)} done, {len(todo)} to run)")

    processor, model = load_model(args.model, args.dtype, args.device_map, args.trust_remote_code)
    family = _family(model)
    pat_images = {}
    for pid in patients:
        paths = patients[pid]["frame_paths"][:: args.frame_stride]
        pat_images[pid] = [Image.open(bundle_dir / p).convert("RGB") for p in paths]
        print(f"  {pid}: {len(pat_images[pid])} frames loaded")

    model_label = args.model_label or args.model
    t0 = time.time()
    for idx, (item, pos) in enumerate(todo, start=1):
        pid = item["patient_id"]
        if pos is not None:
            options, correct_letter, prompt = _item_at_pos(item, pos)
        else:
            options, correct_letter, prompt = item["options"], item["correct_letter"], item["prompt"]
        valid = "".join(sorted(options.keys()))
        try:
            reply = generate_one(processor, model, pat_images[pid], prompt, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, family=family)
            letter, method = extract_letter(reply, valid)
            error = None
        except Exception as e:
            reply, letter, method, error = "", None, "error", f"{type(e).__name__}: {e}"
            if args.verbose:
                traceback.print_exc()
        rec = {"system": "vlm_frames", "item_id": item["item_id"], "patient_id": pid, "pos": pos,
               "model": model_label, "chosen_letter": letter, "correct_letter": correct_letter,
               "is_correct": (letter == correct_letter), "raw_text": reply, "extraction_method": method,
               "n_tool_calls": 0, "tool_calls": [], "error": error}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"({idx}/{len(todo)}) row {item['item_id']} pos={pos} -> {letter} "
              f"[{'OK' if letter == correct_letter else ('ERR' if error else 'wrong')}]  {time.time() - t0:.0f}s cum")
    print(f"[done] wrote {out_path} (total items now: {len(load_done(out_path))})")


if __name__ == "__main__":
    main()

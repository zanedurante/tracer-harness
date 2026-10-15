#!/usr/bin/env python3
"""Socratic captioning (step 1) with a local HuggingFace VLM: 30 frames per
one-minute clip (every ~2 s, matching HourVideo) sent with the HourVideo
caption prompt; captions saved as JSONL. Needs torch, transformers, pillow,
pyyaml and a video decoder (av, or an older torchvision).

    python socratic_caption_vlm.py --stays stays.csv --video-dir /data/video \
        --model /models/Qwen3-VL-8B-Instruct --caption-prompt prompts/hour_video_caption.yaml \
        --out captions.jsonl --num-shards 8 --shard 0
"""
from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_stays  # noqa: E402

try:
    import torch
    import yaml
    from PIL import Image
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
except ImportError as e:
    print(f"ERROR: required dependency missing: {e}", file=sys.stderr)
    sys.exit(2)

try:
    import av as _av
except ImportError:
    _av = None
try:
    from torchvision.io import read_video as _read_video
except (ImportError, AttributeError):
    _read_video = None
if _av is None and _read_video is None:
    print("ERROR: no video decoder available (install 'av')", file=sys.stderr)
    sys.exit(2)

CLIP_RE = re.compile(r"^(?P<hh>\d{2})_(?P<mm>\d{2})_(?P<room>[a-zA-Z0-9]+)_(?P<view>lt|rt)\.mp4$")
FRAMES_PER_CLIP = 30
MAX_SIDE = 512


def enumerate_clips(video_dir: Path, patient_id: str, meta: dict) -> list[dict]:
    """Chronologically sorted clips for a stay under <video_dir>/<YYYY_MM_DD>/;
    prefers lt, falls back to rt; filters by room and window."""
    room, start_utc, end_utc = meta["room"], meta["start_utc"], meta["end_utc"]
    clips = []
    for date_dir in sorted(video_dir.iterdir()):
        if not date_dir.is_dir():
            continue
        try:
            folder_date = dt.datetime.strptime(date_dir.name, "%Y_%m_%d").date()
        except ValueError:
            continue
        if folder_date < start_utc.date() or folder_date > end_utc.date():
            continue
        by_minute: dict = {}
        for f in sorted(date_dir.iterdir()):
            m = CLIP_RE.match(f.name)
            if not m or m.group("room") != room:
                continue
            by_minute.setdefault((m.group("hh"), m.group("mm")), {})[m.group("view")] = f
        for (hh, mm), views in sorted(by_minute.items()):
            ts = dt.datetime.combine(folder_date, dt.time(int(hh), int(mm)))
            if ts < start_utc or ts > end_utc:
                continue
            clip_path = views.get("lt") or views.get("rt")
            if clip_path:
                clips.append({"path": clip_path, "patient_id": patient_id, "date": date_dir.name,
                              "clip_name": clip_path.name, "minute_utc": ts.isoformat()})
    clips.sort(key=lambda c: c["minute_utc"])
    return clips


def _resize(img, max_side):
    w, h = img.size
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS) if scale < 1.0 else img


def extract_frames(clip_path: Path, n_frames: int = FRAMES_PER_CLIP, max_side: int = MAX_SIDE):
    if _av is not None:
        with _av.open(str(clip_path)) as container:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            total = stream.frames or 0
            if total <= 0:
                dur = float(stream.duration * stream.time_base) if stream.duration else 0.0
                rate = float(stream.average_rate or 0)
                total = int(dur * rate) if dur and rate else 0
            if total > 0:
                k = min(n_frames, total)
                wanted = sorted({int(round(i * (total - 1) / max(1, k - 1))) for i in range(k)})
                want = set(wanted)
                out = []
                for i, frame in enumerate(container.decode(stream)):
                    if i in want:
                        out.append(_resize(frame.to_image(), max_side))
                        if len(out) == len(wanted):
                            break
                if out:
                    return out
            container.seek(0)
            frames = [_resize(f.to_image(), max_side) for f in container.decode(stream)]
        if not frames:
            return []
        k = min(n_frames, len(frames))
        idx = sorted({int(round(i * (len(frames) - 1) / max(1, k - 1))) for i in range(k)})
        return [frames[i] for i in idx]
    vframes, _, _ = _read_video(str(clip_path), pts_unit="sec")
    total = vframes.shape[0]
    if total == 0:
        return []
    indices = torch.linspace(0, total - 1, min(n_frames, total)).long()
    return [_resize(Image.fromarray(t.numpy()), max_side) for t in vframes[indices]]


def _pick_model_class(model_path: str):
    mt = getattr(AutoConfig.from_pretrained(model_path), "model_type", "")
    if mt.startswith("qwen3_vl"):
        from transformers import Qwen3VLForConditionalGeneration as MC
    elif mt.startswith("qwen2_5_vl") or mt.startswith("qwen2_vl"):
        from transformers import Qwen2_5_VLForConditionalGeneration as MC
    else:
        from transformers import AutoModelForImageTextToText as MC
    return MC


def load_model(model_path, dtype, device_map):
    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    MC = _pick_model_class(model_path)
    kwargs = dict(torch_dtype=torch_dtype, trust_remote_code=True, low_cpu_mem_usage=True, attn_implementation="sdpa")
    model = MC.from_pretrained(model_path, **kwargs).to("cuda") if device_map == "cuda" else \
        MC.from_pretrained(model_path, device_map=device_map, **kwargs)
    model.eval()
    return processor, model


def _family(model) -> str:
    cls = type(model).__name__.lower()
    return "qwen_vl" if any(k in cls for k in ("qwen3vl", "qwen3_vl", "qwen2_5_vl", "qwen2vl")) else "auto"


def generate_one(processor, model, images, prompt, *, max_new_tokens, family):
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
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)
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
        if rec.get("error") is None and rec.get("caption"):
            done.add((rec["patient_id"], rec["date"], rec["clip_name"]))
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(description="Per-minute captioning with a local VLM (Socratic step 1).")
    ap.add_argument("--stays", required=True)
    ap.add_argument("--video-dir", required=True, help="root containing <YYYY_MM_DD>/ clip folders")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-label", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--caption-prompt", required=True, help="prompts/hour_video_caption.yaml")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device-map", default="auto")
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--max-side", type=int, default=MAX_SIDE)
    ap.add_argument("--frames-per-clip", type=int, default=FRAMES_PER_CLIP)
    ap.add_argument("--patient", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    caption_prompt = yaml.safe_load(open(args.caption_prompt, encoding="utf-8"))["prompt"].strip()
    stays = load_stays(args.stays)
    video_dir = Path(args.video_dir).resolve()
    patients = [args.patient] if args.patient else list(stays)
    all_clips = []
    for pid in patients:
        clips = enumerate_clips(video_dir, pid, stays[pid])
        print(f"  {pid}: {len(clips)} minute clips", flush=True)
        all_clips.extend(clips)
    if args.num_shards > 1:
        all_clips = [c for i, c in enumerate(all_clips) if i % args.num_shards == args.shard]
    if args.limit:
        all_clips = all_clips[:args.limit]
    out_path = Path(args.out).resolve()
    done = load_done(out_path)
    todo = [c for c in all_clips if (c["patient_id"], c["date"], c["clip_name"]) not in done]
    print(f"[clips] total={len(all_clips)} resume={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        return
    processor, model = load_model(args.model, args.dtype, args.device_map)
    family = _family(model)
    model_label = args.model_label or args.model
    t0 = time.time()
    for idx, clip in enumerate(todo, start=1):
        try:
            images = extract_frames(clip["path"], n_frames=args.frames_per_clip, max_side=args.max_side)
            if not images:
                raise RuntimeError("no frames extracted")
            caption = generate_one(processor, model, images, caption_prompt,
                                   max_new_tokens=args.max_new_tokens, family=family)
            error = None
        except Exception as e:
            caption, error = "", f"{type(e).__name__}: {e}"
            if args.verbose:
                traceback.print_exc()
        rec = {"patient_id": clip["patient_id"], "date": clip["date"], "clip_name": clip["clip_name"],
               "minute_utc": clip["minute_utc"], "model": model_label, "caption": caption, "error": error}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"({idx}/{len(todo)}) {clip['patient_id']} {clip['clip_name']} [{'OK' if not error else 'ERR'}] "
              f"{time.time() - t0:.0f}s cum", flush=True)
    print(f"\n[done] {out_path} ({len(load_done(out_path))} captions)", flush=True)


if __name__ == "__main__":
    main()

"""Package questions + uniformly sampled frames into a self-contained bundle
for offline (cluster) VLM inference with tracer/eval/standalone/run_local_vlm.py
or frames_qa_gemini.py.

    bundle_<name>/
        manifest.json         every MCQ item + per-stay frame list
        frames/<pat_id>/0000.jpg ...

Manifest schema:
    {"config": {n_frames, frame_size, jpeg_q, csv, seed, ...},
     "patients": {"<pat_id>": {"frame_paths": [...], "frame_utc_times": [...]}},
     "items": [{"item_id", "patient_id", "question", "options", "correct_letter",
                "gold_answer", "prompt"}, ...]}

    python -m tracer.eval.bundle --csv questions.csv --name lite_n950 --out bundles --n-frames 950
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from .frame_extract import DEFAULT_JPEG_Q, DEFAULT_MAX_SIDE, extract_frames
from .frame_sampler import coverage_summary, sample_frames
from .ingest import load_records
from .mcq import DEFAULT_SEED, build_items
from .runners_frames import DEFAULT_N_FRAMES, build_frames_prompt


def build_bundle(*, csv: Path, name: str, out_dir: Path, n_frames: int = DEFAULT_N_FRAMES,
                 frame_size: int = DEFAULT_MAX_SIDE, jpeg_q: int = DEFAULT_JPEG_Q,
                 seed: int = DEFAULT_SEED, patient_filter=None, limit=None, log_fn=print) -> Path:
    recs, skipped = load_records(csv)
    items, degen = build_items(recs, seed=seed)
    if patient_filter:
        items = [it for it in items if it.patient_id == patient_filter]
    if limit:
        items = items[:limit]
    if not items:
        raise SystemExit(f"no items after filter (skipped={len(skipped)}, degen={len(degen)})")
    bundle_root = out_dir / f"bundle_{name}"
    if bundle_root.exists():
        shutil.rmtree(bundle_root)
    (bundle_root / "frames").mkdir(parents=True)

    patients = list(dict.fromkeys(it.patient_id for it in items))
    log_fn(f"bundle: {len(items)} items across {len(patients)} stays (n_frames={n_frames}, frame_size={frame_size})")
    manifest_patients: dict[str, dict] = {}
    for pat_id in patients:
        cov = coverage_summary(pat_id)
        log_fn(f"  {pat_id}: coverage {cov['n_clips']} clips ({cov['n_hours']}h); sampling {n_frames} frames")
        if cov["n_clips"] == 0:
            log_fn(f"    WARNING: no video coverage for {pat_id}; skipping")
            continue
        positions = sample_frames(pat_id, n_frames)
        jpeg_paths = extract_frames(positions, max_side=frame_size, jpeg_q=jpeg_q, log_fn=lambda m: log_fn(f"    {m}"))
        pat_dir = bundle_root / "frames" / pat_id
        pat_dir.mkdir(parents=True, exist_ok=True)
        rel_paths = []
        for i, src in enumerate(jpeg_paths):
            dst_rel = f"frames/{pat_id}/{i:04d}.jpg"
            shutil.copy(src, bundle_root / dst_rel)
            rel_paths.append(dst_rel)
        manifest_patients[pat_id] = {"frame_paths": rel_paths,
                                     "frame_utc_times": [p.utc_time.isoformat() for p in positions]}

    kept = [it for it in items if it.patient_id in manifest_patients]
    manifest = {
        "config": {"csv": str(csv), "n_frames": n_frames, "frame_size": frame_size, "jpeg_q": jpeg_q,
                   "seed": seed, "n_items": len(kept), "n_patients": len(manifest_patients)},
        "patients": manifest_patients,
        "items": [{"item_id": it.id, "patient_id": it.patient_id, "question": it.question,
                   "options": it.options, "correct_letter": it.correct_letter,
                   "gold_answer": it.gold_answer, "prompt": build_frames_prompt(it)} for it in kept],
    }
    (bundle_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log_fn(f"wrote {bundle_root / 'manifest.json'}")
    return bundle_root


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="bundles")
    ap.add_argument("--n-frames", type=int, default=DEFAULT_N_FRAMES)
    ap.add_argument("--frame-size", type=int, default=DEFAULT_MAX_SIDE)
    ap.add_argument("--patient", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    build_bundle(csv=Path(args.csv), name=args.name, out_dir=Path(args.out), n_frames=args.n_frames,
                 frame_size=args.frame_size, patient_filter=args.patient, limit=args.limit)


if __name__ == "__main__":
    main()

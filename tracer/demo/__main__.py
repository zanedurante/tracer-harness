"""Runnable end-to-end demo on the synthetic sample (no API key, no network).

    python -m tracer.demo [--out demo/out] [--positions 5] [--configs text_only,timeline_only,tracer]

Steps
  1. Expand demo/data into a data store at --out (stays.csv, EHR tables,
     clips, the hand-written report, questions) and expand labels.csv into
     per-clip perception outputs for both camera views (the production
     format written by tracer.perception.run_perception).
  2. Build the timeline (OR-merge of the two views, bit-packed npz +
     timeline JSON) with tracer.perception.build_timeline.
  3. Generate the hourly reports with tracer.reports.generate_reports using
     the offline template model (mock:report), skipping the hand-written one.
  4. Run the real agent loop, prompt assembly and tools with the scripted
     mock backend (mock:agent) over the three questions, five gold positions
     and three tool configurations, through tracer.harness.sweep.
  5. Score: per-configuration mean +/- std over gold positions and a paired
     McNemar test of text_only vs tracer; print one full agent transcript.

Set TRACER_DATA_ROOT / GEMINI_API_KEY and pass --model gemini-3.1-flash-lite
to run the same steps 4-5 with a real model (steps 2-3 still offline; the
report step also accepts --report-model).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
SAMPLE = REPO / "demo" / "data"
PAT_ID = "stay_A"


def _expand_perception(labels_csv: Path, run_dir: Path, room: str = "r001", date: str = "2030_01_01") -> int:
    """labels.csv -> one JSON per (minute, view) in the perception output format."""
    from ..perception.save_results import video_id_to_filename
    run_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(labels_csv, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    label_cols = [c for c in rows[0].keys() if c not in ("minute", "view", "utc_hhmm", "caption")]
    for r in rows:
        video_id = f"{date}/{r['utc_hhmm']}_{room}_{r['view']}.mp4"
        parsed = {"video_id": video_id, "labels": {c: r[c] for c in label_cols}, "summary": r["caption"]}
        rec = {"video_id": video_id, "model": "synthetic", "run_id": 0, "fps": 1,
               "prompt_name": "perception_prompt", "prompt": "(see prompts/perception_prompt.yaml)",
               "user_prompt": "Please find the video frames below:",
               "raw_response": "```json\n" + json.dumps(parsed, indent=2) + "\n```",
               "parsed_response": parsed, "elapsed_seconds": 0.0, "prompt_tokens": 0,
               "response_tokens": 0, "total_tokens": 0, "timestamp": "2030-01-01T00:00:00"}
        (run_dir / video_id_to_filename(video_id)).write_text(json.dumps(rec, indent=1), encoding="utf-8")
        n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "demo" / "out"))
    ap.add_argument("--model", default="mock:agent", help="agent model (mock:agent = offline scripted backend)")
    ap.add_argument("--report-model", default="mock:report")
    ap.add_argument("--configs", default="text_only,timeline_only,tracer")
    ap.add_argument("--positions", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--fresh", action="store_true", help="delete --out first")
    args = ap.parse_args(argv)

    t_start = time.time()
    out = Path(args.out).resolve()
    if args.fresh and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    # The data store must be configured before the tools are imported.
    os.environ["TRACER_DATA_ROOT"] = str(out)
    os.environ["TRACER_RESULTS_ROOT"] = str(out / "results")
    os.environ["TRACER_CACHE_ROOT"] = str(out / "cache")
    # The synthetic sample is authored in UTC (clip names and local times coincide).
    os.environ["TRACER_LOCAL_TZ"] = "UTC"

    print("=" * 72)
    print("TRACER demo on synthetic data")
    print(f"  sample: {SAMPLE}\n  store:  {out}\n  agent:  {args.model}")
    print("=" * 72)

    # ---- 1. expand the sample -------------------------------------------
    print("\n[1/5] Expanding the synthetic sample into a data store")
    shutil.copy(SAMPLE / "stays.csv", out / "stays.csv")
    for sub in ("ehr", "video", "reports"):
        if (out / sub).exists():
            shutil.rmtree(out / sub)
        shutil.copytree(SAMPLE / sub, out / sub)
    shutil.copy(SAMPLE / "questions.csv", out / "questions.csv")
    run_dir = out / "perception" / "synthetic" / "perception_prompt" / "run_0"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    n = _expand_perception(SAMPLE / "labels.csv", run_dir)
    print(f"  wrote {n} per-clip perception outputs (2 views x 180 minutes) -> {run_dir.relative_to(out)}")

    # ---- 2. timeline ------------------------------------------------------
    print("\n[2/5] Building the minute-level timeline (OR-merge of both views, bit-packed npz)")
    from .. import data
    data.reset_caches()
    from ..perception import build_timeline
    info = build_timeline.build(
        PAT_ID, run_dir, out / "timelines",
        events=json.loads((SAMPLE / "events.json").read_text(encoding="utf-8")),
        demographics=json.loads((SAMPLE / "demographics.json").read_text(encoding="utf-8")),
        encounters=json.loads((SAMPLE / "encounters.json").read_text(encoding="utf-8")))
    print(f"  {info['n_minutes']} minutes / {info['n_hours']} hours, icu_start {info['icu_start']} local")
    active = {k: v for k, v in info["per_label_minutes"].items() if v}
    print("  minutes active per label: " + ", ".join(f"{k.replace('patient_', '')}={v}" for k, v in active.items()))

    # ---- 3. reports -------------------------------------------------------
    print(f"\n[3/5] Generating hourly reports ({args.report_model}); the hand-written 08:00 report is kept")
    from ..tools import common as _common
    _common.reset_caches()
    from ..reports import generate_reports
    rinfo = generate_reports.generate_for_stay(PAT_ID, run_dir, model=args.report_model, skip_existing=True,
                                               log=lambda m: print("  " + m))
    print(f"  generated {rinfo['generated']}, kept {rinfo['skipped']} existing -> {Path(rinfo['reports_dir']).relative_to(out)}")

    # ---- 4. agent sweep ---------------------------------------------------
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    print(f"\n[4/5] Running the agent loop: {configs} x 3 questions x {args.positions} gold positions")
    from ..harness import sweep
    run_dir_res = sweep.run_sweep(str(out / "questions.csv"), configs, args.model, positions=args.positions,
                                  workers=args.workers, run_name="demo", quiet=True)
    summary = json.loads((run_dir_res / "summary.json").read_text(encoding="utf-8"))
    for name, s in summary["configs"].items():
        print(f"  {name:16s} tools={s['n_tools']:2d}  accuracy={s['acc'] * 100:5.1f}%  (n={s['n']} trials)")

    # ---- 5. scoring -------------------------------------------------------
    print("\n[5/5] Scoring: mean +/- std over gold positions, and a paired McNemar test")
    from ..eval import position_stats, compare_sweeps
    stats = position_stats.position_stats(position_stats.load_cache(run_dir_res / "cache.jsonl"))
    for name in configs:
        s = stats[name]
        pp = " ".join(f"pos{p}={a * 100:.0f}" for p, a in s["per_position"].items())
        print(f"  {name:16s} {s['mean'] * 100:5.1f} +/- {s['std'] * 100:4.1f}   [{pp}]")
    if "text_only" in configs and "tracer" in configs:
        m = compare_sweeps.mcnemar_counts(compare_sweeps.load_config("demo:text_only"),
                                          compare_sweeps.load_config("demo:tracer"),
                                          label_a="text_only", label_b="tracer")
        print(f"  McNemar text_only vs tracer: pairs={m['n_pairs']} tracer-only={m['a_wrong_b_correct']} "
              f"text_only-only={m['a_correct_b_wrong']} chi2(cc)={m['mcnemar_statistic']} "
              f"p={m['chi2_p_value']} exact p={m['exact_p_value']}")

    # one transcript, so the tool calls are visible
    print("\nOne full agent trial (config=tracer, question 1, gold at position 0):")
    from ..harness import agent as aa
    from ..harness.configs import resolve
    from ..harness.registry import build_registry
    from ..eval.ingest import load_records
    from ..eval.mcq import item_with_gold_at
    recs, _ = load_records(out / "questions.csv")
    item = item_with_gold_at(recs[0], 0)
    res = aa.run_ablated(aa.build_message(item), PAT_ID, resolve("tracer"), args.model, build_registry())
    for i, c in enumerate(res["tool_calls"], 1):
        resp = c["response"]
        brief = {k: resp[k] for k in list(resp)[:4]} if isinstance(resp, dict) else resp
        print(f"  call {i}: {c['name']}({json.dumps(c['args'])}) -> {json.dumps(brief, default=str)[:160]}...")
    print("  reply: " + res["reply"].replace("\n", " | "))
    print(f"  gold letter: {item.correct_letter}")

    elapsed = time.time() - t_start
    ok = True
    if args.model == "mock:agent" and "tracer" in configs:
        ok = abs(summary["configs"]["tracer"]["acc"] - 1.0) < 1e-9
    print("\n" + ("DEMO OK" if ok else "DEMO FAILED: the full harness did not answer every synthetic question")
          + f" ({elapsed:.1f}s). Outputs in {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

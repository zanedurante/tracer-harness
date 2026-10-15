"""Orchestrate the baseline evaluations: load -> build items -> run each system
on each item (with gold-position rotation) -> score -> write report.

STRICTLY ONE MODEL PER RUN: every question for every system is answered by
the single --model; transient errors are retried on the same model.

Modes:
  mcq       tool_agent (the full harness) and blind (single-turn text-only prior)
  frames    long-context VLM baseline (--n-frames, --frame-size)
  socratic  Socratic Models baseline over cached captions (tracer.eval.socratic_caption)

The harness ablation itself is driven by tracer.harness.sweep, which caches
by tool configuration; this orchestrator is for the baselines that do not use
tools. Results are cached by (system, model, item_id, pos) in a JSONL so reruns
resume; errored results are not cached.

  python -m tracer.eval.run_eval --csv questions.csv --mode frames --model gemini-3.1-flash-lite \
      --n-frames 950 --positions 5 --run-name frames950
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from .. import config as _cfg
from .ingest import load_records
from .mcq import build_items, item_with_gold_at, n_positions
from .runners import SYSTEMS, RunResult
from .runners_frames import DEFAULT_N_FRAMES, FRAMES_SYSTEMS
from .runners_socratic import SOCRATIC_SYSTEMS
from .frame_extract import DEFAULT_MAX_SIDE
from .retry import call_with_backoff
from . import score as scoring


class Cache:
    """JSONL cache keyed by (system, model, item_id, pos). Non-error results
    are final; errored ones are re-attempted on resume."""

    def __init__(self, path: Path):
        self.path = path
        self.done: dict[tuple, dict] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.done[(rec["system"], rec["primary_model"], rec["item_id"], rec.get("pos"))] = rec

    def get(self, system, model, item_id, pos=None):
        rec = self.done.get((system, model, item_id, pos))
        if rec and not rec["result"].get("error"):
            return rec["result"]
        return None

    def put(self, system, model, item_id, result_dict, pos=None):
        rec = {"system": system, "primary_model": model, "item_id": item_id, "pos": pos, "result": result_dict}
        self.done[(system, model, item_id, pos)] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def _bind(base_runner, extra: dict):
    if not extra:
        return base_runner

    def _r(item, model, _r=base_runner, _kw=extra):
        return _r(item, model, **_kw)
    return _r


def run_positions(records, systems, model, n_positions_requested, *, registry, cache=None,
                  runner_kwargs=None, sleep_between=0.0, patient_filter=None, limit=None,
                  sleep_fn=time.sleep, log_fn=print):
    """Run every (system, record, pos) with gold rotated through option slots.
    Returns {system: {pos: [RunResult, ...]}}."""
    recs = [r for r in records if not patient_filter or r.patient_id == patient_filter]
    if limit:
        recs = recs[:limit]
    results = {s: {p: [] for p in range(n_positions_requested)} for s in systems}
    total = len(recs) * len(systems) * n_positions_requested
    done = 0
    for rec in recs:
        for pos in range(min(n_positions_requested, n_positions(rec))):
            item = item_with_gold_at(rec, pos)
            for sysname in systems:
                done += 1
                cached = cache.get(sysname, model, item.id, pos=pos) if cache else None
                if cached is not None:
                    results[sysname][pos].append(RunResult(**cached))
                    log_fn(f"({done}/{total}) {sysname} row {item.id} pos={pos} [cached] -> {cached['chosen_letter']}")
                    continue
                runner = _bind(registry[sysname], runner_kwargs or {})
                res = call_with_backoff(runner, item, model, sleep_fn=sleep_fn, log_fn=log_fn)
                res.pos = pos
                results[sysname][pos].append(res)
                if cache and not res.error:
                    cache.put(sysname, model, item.id, asdict(res), pos=pos)
                mark = "ok" if res.is_correct else ("ERR" if res.error else "x")
                log_fn(f"({done}/{total}) {sysname} row {item.id} pos={pos} -> {res.chosen_letter} {mark}")
                if sleep_between:
                    sleep_fn(sleep_between)
    return results


def format_position_report(pos_results) -> str:
    lines = ["=" * 60, "MCQ EVAL SUMMARY (position-averaged)", "=" * 60]
    for sysname, by_pos in pos_results.items():
        flat = {p: [r.is_correct for r in rs] for p, rs in by_pos.items() if rs}
        mean, std, accs = scoring.position_mean_std(flat)
        total_n = sum(len(v) for v in flat.values())
        total_c = sum(sum(v) for v in flat.values())
        total_e = sum(1 for rs in by_pos.values() for r in rs if r.error)
        lines.append(f"\n[{sysname}]  mean={mean:.1%} +/- {std:.1%}  ({total_c}/{total_n})  errors={total_e}")
        for p in sorted(flat):
            lines.append(f"   pos={p}: {sum(flat[p])}/{len(flat[p])} = {sum(flat[p]) / len(flat[p]):.1%}")
        pat_pos = defaultdict(lambda: defaultdict(list))
        for p, rs in by_pos.items():
            for r in rs:
                pat_pos[r.patient_id][p].append(r.is_correct)
        parts = []
        for pat in sorted(pat_pos):
            m, s, _ = scoring.position_mean_std(dict(pat_pos[pat]))
            parts.append(f"{pat}={m:.0%}+/-{s:.0%}")
        lines.append("   by stay: " + ", ".join(parts))
    names = list(pos_results)
    if len(names) >= 2:
        def keyed(by_pos):
            return {(r.item_id, p): bool(r.is_correct) for p, rs in by_pos.items() for r in rs if not r.error}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                m = scoring.mcnemar_counts(keyed(pos_results[a]), keyed(pos_results[b]), label_a=a, label_b=b)
                lines.append(f"\n[McNemar {a} vs {b}] pairs={m['n_pairs']} a-only={m['a_correct_b_wrong']} "
                             f"b-only={m['a_wrong_b_correct']} chi2(cc)={m['mcnemar_statistic']} "
                             f"p={m['chi2_p_value']} exact p={m['exact_p_value']}")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run the baseline evaluations with gold-position rotation.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--mode", choices=["mcq", "frames", "socratic"], default="mcq")
    ap.add_argument("--n-frames", type=int, default=DEFAULT_N_FRAMES)
    ap.add_argument("--frame-size", type=int, default=DEFAULT_MAX_SIDE)
    ap.add_argument("--caption-model", default=None, help="--mode socratic: caption store to read (default: --model)")
    ap.add_argument("--positions", type=int, default=5)
    ap.add_argument("--systems", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--patient", default=None)
    ap.add_argument("--sleep-between", type=float, default=1.0)
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--run-name", default="run")
    args = ap.parse_args(argv)

    recs, skipped = load_records(args.csv)
    runner_kwargs: dict = {}
    if args.mode == "frames":
        registry, default_systems, cache_name = FRAMES_SYSTEMS, "vlm_frames", "cache_frames.jsonl"
        runner_kwargs = {"n_frames": args.n_frames, "max_side": args.frame_size}
    elif args.mode == "socratic":
        registry, default_systems, cache_name = SOCRATIC_SYSTEMS, "socratic", "cache_socratic.jsonl"
        if args.caption_model:
            runner_kwargs = {"caption_model": args.caption_model}
    else:
        registry, default_systems, cache_name = SYSTEMS, "tool_agent,blind", "cache.jsonl"
    systems = [s.strip() for s in (args.systems or default_systems).split(",") if s.strip()]
    unknown = [s for s in systems if s not in registry]
    if unknown:
        ap.error(f"unknown systems for --mode {args.mode}: {unknown}. Available: {list(registry)}")

    out_dir = _cfg.results_root() / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = None if args.no_cache else Cache(out_dir / cache_name)
    print(f"Loaded {len(recs)} records ({len(skipped)} skipped) | mode={args.mode} | model={args.model} "
          f"| systems={systems} | positions={args.positions}")
    pos_results = run_positions(recs, systems, args.model, args.positions, registry=registry, cache=cache,
                                runner_kwargs=runner_kwargs, sleep_between=args.sleep_between,
                                patient_filter=args.patient, limit=args.limit)
    flat = []
    for sysname, by_pos in pos_results.items():
        for pos, rs in by_pos.items():
            for r in rs:
                d = asdict(r)
                d["pos"] = pos
                flat.append(d)
    (out_dir / "results.json").write_text(json.dumps(flat, indent=2), encoding="utf-8")
    report = format_position_report(pos_results)
    (out_dir / "report.txt").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nWrote results to {out_dir}")


if __name__ == "__main__":
    main()

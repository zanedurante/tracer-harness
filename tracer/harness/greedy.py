"""Greedy forward selection of tools that maximize MCQ accuracy.

Start from the empty set (== text-only). Each round, evaluate adding each
not-yet-selected candidate tool, commit the one with the largest accuracy
gain, and stop when no candidate improves. Search on a stratified subsample
at a single arrangement, then validate the winning path on the full bank,
position-averaged (tracer.harness.sweep).

Usage:
  python -m tracer.harness.greedy --csv questions.csv --model gemini-3.1-flash-lite \
      [--per-patient 15] [--positions 1] [--workers 6] [--min-gain 0.0]
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
import concurrent.futures as cf
from collections import defaultdict

from .. import config as _cfg
from ..eval.ingest import load_records
from ..eval.extract import extract_choice
from ..eval.mcq import item_with_gold_at, n_positions
from ..eval.retry import is_transient
from ..tools import THE_ELEVEN
from .configs import config_sig
from .registry import build_registry
from . import agent as aa

CANDIDATES = list(THE_ELEVEN)


def stratified_subsample(recs, per_patient: int | None):
    by = defaultdict(list)
    for rec in recs:
        by[rec.patient_id].append(rec)
    chosen = []
    for pid, lst in by.items():
        k = min(per_patient or max(1, len(lst) // 3), len(lst))
        step = max(1, len(lst) // k)
        chosen.extend(lst[::step][:k])
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--all", action="store_true", help="search on ALL questions (no subsample)")
    ap.add_argument("--per-patient", type=int, default=None, help="questions per stay in the search subset")
    ap.add_argument("--positions", type=int, default=1)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-gain", type=float, default=0.0)
    ap.add_argument("--max-tools", type=int, default=len(CANDIDATES))
    ap.add_argument("--run-name", default="greedy")
    args = ap.parse_args(argv)

    out = _cfg.results_root() / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "cache.jsonl"
    recs, _ = load_records(args.csv)
    subset = list(recs) if args.all else stratified_subsample(recs, args.per_patient)
    registry = build_registry(include_video=True)
    patients = sorted({r.patient_id for r in subset})
    print(f"GREEDY forward selection | search subset = {len(subset)} Qs over {patients} "
          f"| pos={args.positions} samples={args.samples} | {args.model}")

    done = {}
    if cache.exists():
        for line in cache.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["sig"], r["qid"], r["pos"], r["sample"])] = r["correct"]
    lock = threading.Lock()

    def trial(tools, rec, pos, sample):
        sig = config_sig(tools)
        key = (sig, rec.id, pos, sample)
        if key in done:
            return
        item = item_with_gold_at(rec, pos)
        letter = None
        try:
            for attempt in range(4):
                o = aa.run_ablated(aa.build_message(item), rec.patient_id, tools, args.model, registry)
                err = o.get("error")
                if not err:
                    ex = extract_choice(o["reply"], item)
                    letter = ex.letter if ex else None
                if letter is not None:
                    break
                time.sleep(min(4 * (attempt + 1), 16) if (err and is_transient(err)) else 1.0)
        except Exception:
            letter = None
        if letter is None:
            return
        correct = (letter == item.correct_letter)
        with lock:
            with open(cache, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sig": sig, "qid": rec.id, "pos": pos, "sample": sample, "correct": correct}) + "\n")
            done[key] = correct

    def eval_config(tools):
        sig = config_sig(tools)
        triples = [(rec, p, s) for rec in subset
                   for p in range(min(args.positions, n_positions(rec))) for s in range(args.samples)]
        todo = [(tools, rec, p, s) for (rec, p, s) in triples if (sig, rec.id, p, s) not in done]
        if todo:
            with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
                list(ex.map(lambda t: trial(*t), todo))
        vals = [v for v in (done.get((sig, rec.id, p, s)) for (rec, p, s) in triples) if v is not None]
        pat = {}
        for pid in patients:
            pv = [v for v in (done.get((sig, rec.id, p, s)) for (rec, p, s) in triples if rec.patient_id == pid)
                  if v is not None]
            pat[pid] = (sum(pv) / len(pv)) if pv else None
        return (sum(vals) / len(vals) if vals else 0.0), pat

    selected: list[str] = []
    base, base_pat = eval_config(selected)
    path = [{"n_tools": 0, "added": "(text-only)", "acc": base, "per_patient": base_pat, "set": []}]
    cand_log = []
    print(f"round 0  (text-only)  acc={base:.1%}")
    remaining = list(CANDIDATES)
    while remaining and len(selected) < args.max_tools:
        scored = []
        for t in remaining:
            acc, pat = eval_config(selected + [t])
            scored.append((acc, t, pat))
            cand_log.append({"round": len(selected) + 1, "candidate": t, "acc": acc})
            print(f"    +{t:24s} -> {acc:.1%}")
        scored.sort(key=lambda x: -x[0])
        best_acc, best_t, best_pat = scored[0]
        if best_acc <= base + args.min_gain + 1e-9:
            print(f"  no candidate beats {base:.1%} (+{args.min_gain:.1%}); STOP.")
            break
        selected.append(best_t)
        remaining.remove(best_t)
        base, base_pat = best_acc, best_pat
        path.append({"n_tools": len(selected), "added": best_t, "acc": best_acc,
                     "per_patient": best_pat, "set": list(selected)})
        print(f"  ROUND {len(selected)}: + {best_t}  -> {best_acc:.1%}   set={selected}\n")

    print("\n==== GREEDY PATH (accuracy vs #tools) ====")
    for p in path:
        print(f"  k={p['n_tools']}  acc={p['acc']:.1%}  +{p['added']}")
    peak = max(path, key=lambda p: p["acc"])
    print(f"\n  PEAK: k={peak['n_tools']} tools, acc={peak['acc']:.1%}, set={peak['set']}")
    with open(out / "greedy_path.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["n_tools", "added", "acc", "tool_set"])
        for p in path:
            w.writerow([p["n_tools"], p["added"], round(p["acc"], 4), "|".join(p["set"])])
    with open(out / "greedy_candidates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["round", "candidate", "acc"])
        for c in cand_log:
            w.writerow([c["round"], c["candidate"], round(c["acc"], 4)])
    (out / "greedy_summary.json").write_text(json.dumps(
        {"model": args.model, "search_n": len(subset), "positions": args.positions,
         "samples": args.samples, "path": path, "peak": peak}, indent=2), encoding="utf-8")
    print(f"\nWrote {out}/greedy_path.csv, greedy_candidates.csv, greedy_summary.json")


if __name__ == "__main__":
    main()

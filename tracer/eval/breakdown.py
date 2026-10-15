"""Per-category / per-stay breakdown of result files, in the layout of the
paper's tables.

Inputs
  --categories CSV with columns idx (question row id), paper_category,
      patient, n_options. Categories in the paper: "Video Moments (<5 min)",
      "Video Windows (5-30 min)", "Long-form Video (>30 min)", "EHR",
      "Cross-modal". (The benchmark's own category file is not released with
      the questions; build one for your bank.)
  --results LABEL=PATH ...  each PATH is either a sweep cache
      (tracer.harness.sweep: keyed by qid/config) or a baseline JSONL of
      per-trial records with item_id + is_correct (tracer.eval.run_eval
      results.json entries, or the standalone drivers' cache.jsonl). For a
      sweep cache, use LABEL=PATH:CONFIG to pick one configuration.

Accuracy is the question mean (average the positions within a question, then
average questions). An unparseable answer counts as wrong. An analytic random
baseline (mean of 1/n_options) is printed first.

    python -m tracer.eval.breakdown --categories cats.csv \
        --results "TRACER=results/demo/cache.jsonl:tracer" "Frames=results/frames/results.json"
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

ORDER = ["Video Moments (<5 min)", "Video Windows (5-30 min)", "Long-form Video (>30 min)", "EHR", "Cross-modal"]
SHORT = {"Video Moments (<5 min)": "Moments", "Video Windows (5-30 min)": "Windows",
         "Long-form Video (>30 min)": "Long-form", "EHR": "EHR", "Cross-modal": "Cross-modal"}


def correctness(rec) -> bool:
    if rec.get("is_correct") is not None:
        return bool(rec["is_correct"])
    if rec.get("correct") is not None:
        return bool(rec["correct"])
    chosen = rec.get("chosen_letter") or rec.get("letter")
    gold = rec.get("correct_letter") or rec.get("gold")
    return bool(chosen) and chosen == gold


def load_results(path: Path, config: str | None) -> dict[int, list[bool]]:
    """{qid: [correct, ...]} from a JSONL/JSON of per-trial records."""
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        recs = json.loads(text)
        if isinstance(recs, dict):
            recs = recs.get("results", [])
    else:
        recs = [json.loads(l) for l in text.splitlines() if l.strip()]
    latest = {}
    for r in recs:
        if config and r.get("config") != config and r.get("sig") != config:
            continue
        if r.get("error"):
            continue
        qid = r.get("item_id", r.get("qid"))
        if qid is None:
            continue
        latest[(int(qid), r.get("pos"), r.get("sample", 0), r.get("system"))] = r
    per = collections.defaultdict(list)
    for (qid, _p, _s, _sys), r in latest.items():
        per[qid].append(correctness(r))
    return per


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", required=True)
    ap.add_argument("--results", nargs="+", required=True, help="LABEL=PATH[:CONFIG]")
    ap.add_argument("--write", default=None, help="write the table to this CSV")
    a = ap.parse_args(argv)
    bank = list(csv.DictReader(open(a.categories, encoding="utf-8")))
    cat = {int(r["idx"]): r["paper_category"] for r in bank}
    pat = {int(r["idx"]): r["patient"] for r in bank}
    nopt = {int(r["idx"]): int(r["n_options"]) for r in bank}

    def group_mean(per, ids):
        vals = [sum(per[q]) / len(per[q]) for q in ids if per.get(q)]
        return (sum(vals) / len(vals) * 100) if vals else float("nan")

    groups = [("All", sorted(cat))] + \
             [(SHORT[c], [q for q in sorted(cat) if cat[q] == c]) for c in ORDER if any(cat[q] == c for q in cat)] + \
             [(p, [q for q in sorted(cat) if pat[q] == p]) for p in sorted(set(pat.values()))]
    table = [("Random (analytic)", {g: sum(1 / nopt[q] for q in ids) / len(ids) * 100 for g, ids in groups if ids}, None)]
    for spec in a.results:
        label, _, rest = spec.partition("=")
        path_s, _, config = rest.rpartition(":") if ":" in rest and not rest[1:3] == ":\\" else (rest, "", "")
        if not path_s:
            path_s, config = rest, ""
        p = Path(path_s)
        if not p.exists():
            print(f"[skip] {label}: {p} not found")
            continue
        per = load_results(p, config or None)
        n_trials = sum(len(v) for v in per.values())
        missing = [q for q in cat if q not in per]
        table.append((label, {g: group_mean(per, ids) for g, ids in groups if ids}, (n_trials, len(per), missing)))

    names = [g for g, ids in groups if ids]
    print(f"{'row':28s} " + " ".join(f"{n:>11s}" for n in names))
    for label, vals, meta in table:
        print(f"{label:28s} " + " ".join(f"{vals[n]:10.1f}%" for n in names))
    print()
    for label, vals, meta in table[1:]:
        if meta:
            n_trials, n_q, missing = meta
            print(f"  {label}: {n_trials} trials over {n_q}/{len(cat)} questions"
                  + (f"; MISSING qids {missing[:8]}{'...' if len(missing) > 8 else ''}" if missing else ""))
    if a.write:
        with open(a.write, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["row"] + names)
            for label, vals, _ in table:
                w.writerow([label] + [f"{vals[n]:.1f}" for n in names])
        print(f"\nwrote {a.write}")


if __name__ == "__main__":
    main()

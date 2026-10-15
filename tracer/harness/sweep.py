"""Tool-configuration sweep over an MCQ bank with gold-position rotation.

A "config" is a named tool subset (tracer.harness.configs) or a comma list of
tool names. Each (config, question, position, sample) is one trial; the gold
answer is rotated through option positions (--positions 5 reproduces the
paper: every reported number is a mean over placements) and optionally
repeat-sampled. Results are cached in a JSONL keyed by
(config_sig, qid, pos, sample) so reruns resume and shared configs cost nothing.

Usage:
  python -m tracer.harness.sweep --csv questions.csv --configs text_only,plus_ehr,tracer \
      --model gemini-3.1-flash-lite --positions 5 --workers 6 --run-name my_run

Results: <TRACER_RESULTS_ROOT>/<run-name>/cache.jsonl, summary.json and one
per-question CSV per config. Aggregate with tracer.eval.position_stats and
compare configs with tracer.eval.compare_sweeps.
"""
from __future__ import annotations

import argparse
import csv
import json
import threading
import time
import concurrent.futures as cf
from collections import defaultdict
from pathlib import Path

from .. import config as _cfg
from ..eval.ingest import load_records
from ..eval.mcq import item_with_gold_at, n_positions
from ..eval.extract import extract_choice
from ..eval.retry import is_429, is_transient
from ..tools import video as _vt
from . import configs as cfgs
from .configs import config_sig
from .registry import build_registry
from . import agent as aa


def question_type(rec) -> str:
    return (getattr(rec, "question_type", "") or "unspecified")


def run_sweep(csv_path: str, config_names: list[str], model: str, *, positions: int = 5,
              samples: int = 1, workers: int = 6, patient: str | None = None,
              limit_per_patient: int | None = None, run_name: str = "sweep",
              qids: set[int] | None = None, out_root: Path | None = None,
              quiet: bool = False) -> Path:
    out_dir = (out_root or _cfg.results_root()) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "cache.jsonl"
    log = (lambda *a, **k: None) if quiet else print

    recs, _ = load_records(csv_path)
    if qids:
        recs = [r for r in recs if r.id in qids]
    if patient:
        recs = [r for r in recs if r.patient_id == patient]
    if limit_per_patient:
        seen = defaultdict(int)
        kept = []
        for r in recs:
            if seen[r.patient_id] < limit_per_patient:
                kept.append(r)
                seen[r.patient_id] += 1
        recs = kept
    by_id = {r.id: r for r in recs}

    prompt_variant = aa.prompt_variant_for(model)
    config_tools = {name: cfgs.resolve(name) for name in config_names}
    registry = build_registry(include_video=True)

    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["sig"], r["qid"], r["pos"], r["sample"])] = r["correct"]

    lock = threading.Lock()

    def trial(name, tools, rec, pos, sample):
        sig = config_sig(tools)
        key = (sig, rec.id, pos, sample)
        if key in done:
            return key, done[key]
        item = item_with_gold_at(rec, pos)
        # Retry so every config answers the SAME questions (equal n). Transient
        # API errors back off; an unparseable/empty reply gets a quick re-ask.
        letter = None
        n_calls = 0
        err = None
        out = {"tool_calls": [], "reply": ""}
        for attempt in range(4):
            out = aa.run_ablated(aa.build_message(item), rec.patient_id, tools, model, registry)
            n_calls = len(out["tool_calls"])
            err = out.get("error")
            if not err:
                ex = extract_choice(out["reply"], item)
                letter = ex.letter if ex else None
            if letter is not None:
                break
            if err and is_429(err):
                time.sleep(35.0 * (attempt + 1))
            else:
                time.sleep(min(4 * (attempt + 1), 16) if (err and is_transient(err)) else 0.0)
        if letter is None and err and ("connection" in err.lower() or is_transient(err)):
            raise RuntimeError(f"persistent backend failure on qid={rec.id} pos={pos}: {err}")
        correct = False if letter is None else (letter == item.correct_letter)
        n_tool_errors = sum(1 for c in out["tool_calls"]
                            if isinstance(c.get("response"), dict) and "error" in c["response"])
        with lock:
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sig": sig, "config": name, "qid": rec.id, "pos": pos,
                                    "sample": sample, "correct": correct,
                                    "csv": Path(csv_path).name,
                                    "n_tools": len(tools), "n_tool_calls": len(out["tool_calls"]),
                                    "letter": letter, "gold": item.correct_letter,
                                    "attempts": attempt + 1, "error": err,
                                    "forced_final": bool(out.get("forced_final")),
                                    "context_overflow": bool(out.get("context_overflow")),
                                    "n_tool_errors": n_tool_errors,
                                    "tools_used": [c["name"] for c in out["tool_calls"]],
                                    "prompt_variant": prompt_variant,
                                    "video_tool_model": _vt.VIDEO_TOOL_MODEL or None,
                                    "prompt_tokens": (out.get("tokens") or {}).get("prompt"),
                                    "completion_tokens": (out.get("tokens") or {}).get("completion"),
                                    "total_tokens": (out.get("tokens") or {}).get("total"),
                                    "n_model_turns": (out.get("tokens") or {}).get("turns"),
                                    "reply_tail": (out["reply"] or "")[-400:]}) + "\n")
            done[key] = correct
        return key, correct

    tasks = []
    for name in config_names:
        tools = config_tools[name]
        for rec in recs:
            for pos in range(min(positions, n_positions(rec))):
                for s in range(samples):
                    tasks.append((name, tools, rec, pos, s))
    todo = [t for t in tasks if (config_sig(t[1]), t[2].id, t[3], t[4]) not in done]
    log(f"Tool-config sweep: {len(recs)} Qs x {config_names} | {len(tasks)} trials "
        f"({len(todo)} to run) | pos={positions} samples={samples} | {model}")

    completed = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(trial, *t) for t in todo]
        for fut in cf.as_completed(futs):
            fut.result()
            completed += 1
            if completed % 20 == 0:
                log(f"  ...{completed}/{len(todo)}")

    # ---- aggregate -------------------------------------------------------
    def acc(pred):
        vals = [v for v in pred if v is not None]
        return (sum(vals) / len(vals)) if vals else None, len(vals)

    log("\n" + "=" * 72)
    log(f"TOOL-CONFIG ABLATION  ({model}; pos={positions}, samples={samples})")
    log("=" * 72)
    rows = []
    for name in config_names:
        tools = config_tools[name]
        sig = config_sig(tools)
        overall = [done.get((sig, r.id, p, s)) for r in recs
                   for p in range(min(positions, n_positions(r))) for s in range(samples)]
        a, n = acc(overall)
        per_pat = {}
        for pid in sorted({r.patient_id for r in recs}):
            vv = [done.get((sig, r.id, p, s)) for r in recs if r.patient_id == pid
                  for p in range(min(positions, n_positions(r))) for s in range(samples)]
            per_pat[pid] = acc(vv)
        rows.append((name, len(tools), a, n, per_pat))
    for name, nt, a, n, per_pat in rows:
        astr = f"{a:.1%}" if a is not None else "n/a"
        pp = "  ".join(f"{pid}={(pa[0]*100):.0f}%({pa[1]})" if pa[0] is not None else f"{pid}=n/a"
                       for pid, pa in per_pat.items())
        log(f"  {name:22s} tools={nt:2d}  acc={astr:>6} (n={n})   {pp}")

    for name in config_names:
        sig = config_sig(config_tools[name])
        with open(out_dir / f"{name}_per_question.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["idx", "patient", "n_options", f"{name}_correct", f"{name}_n", f"{name}_acc", "question"])
            for ri, rec in enumerate(recs):
                npos = min(positions, n_positions(rec))
                vals = [done.get((sig, rec.id, p, s)) for p in range(npos) for s in range(samples)]
                vals = [v for v in vals if v is not None]
                cc, nn = sum(vals), len(vals)
                q = rec.question.split("\n")[0].replace("Q:", "").strip()
                w.writerow([ri, rec.patient_id, n_positions(rec), cc, nn,
                            round(cc / nn, 3) if nn else "", q])

    # ---- fair comparison on the COMMON subset answered by every config ----
    sigs = {name: config_sig(config_tools[name]) for name in config_names}
    triples = [(r.id, p, s) for r in recs
               for p in range(min(positions, n_positions(r))) for s in range(samples)]
    common = [t for t in triples
              if all(done.get((sigs[name], t[0], t[1], t[2])) is not None for name in config_names)]
    common_acc = {}
    for name in config_names:
        sig = sigs[name]
        vals = [done[(sig, t[0], t[1], t[2])] for t in common]
        common_acc[name] = (sum(vals) / len(vals)) if vals else None

    (out_dir / "summary.json").write_text(json.dumps(
        {"model": model, "positions": positions, "samples": samples,
         "prompt_variant": prompt_variant, "csv": Path(csv_path).name,
         "configs": {name: {"n_tools": nt, "tools": config_tools[name], "acc": a, "n": n,
                            "per_patient": {k: v[0] for k, v in pp.items()}}
                     for name, nt, a, n, pp in rows},
         "common_subset": {"n": len(common), "acc": common_acc}}, indent=2), encoding="utf-8")
    log(f"\nWrote {out_dir / 'summary.json'}")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="question bank CSV (see docs/DATA_LAYOUT.md)")
    ap.add_argument("--configs", default="text_only,timeline_only,reports_only,ehr_only,video_only,"
                                         "w_timeline,plus_reports,plus_ehr,tracer")
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--positions", type=int, default=5, help="gold positions to average (paper: 5)")
    ap.add_argument("--samples", type=int, default=1, help="repeat samples per (q,pos)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--patient", default=None)
    ap.add_argument("--limit-per-patient", type=int, default=None)
    ap.add_argument("--run-name", default="sweep")
    ap.add_argument("--qids", default=None, help="comma-separated row ids to include")
    args = ap.parse_args(argv)
    qids = {int(q) for q in args.qids.split(",") if q.strip()} if args.qids else None
    run_sweep(args.csv, [c.strip() for c in args.configs.split(",") if c.strip()], args.model,
              positions=args.positions, samples=args.samples, workers=args.workers,
              patient=args.patient, limit_per_patient=args.limit_per_patient,
              run_name=args.run_name, qids=qids)


if __name__ == "__main__":
    main()

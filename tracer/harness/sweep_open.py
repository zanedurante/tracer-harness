"""Open-ended sweep: run tool configurations over the question bank WITHOUT
option lists, grade with the SimpleQA LLM judge (tracer.eval.judge), and
report accuracy.

Usage:
  python -m tracer.harness.sweep_open --csv questions.csv \
      --configs text_only,timeline_only,plus_reports,plus_ehr,tracer \
      --model gemini-2.5-flash --judge gemini-2.5-pro --run-name open_v1
"""
from __future__ import annotations

import argparse
import json
import threading
import concurrent.futures as cf
from collections import defaultdict
from pathlib import Path

from google.genai import types

from .. import config as _cfg
from ..backends.gemini import get_gemini_client
from ..eval.ingest import load_records
from ..eval.judge import DEFAULT_JUDGE_MODEL, grade_open
from . import configs as cfgs
from .configs import config_sig
from .registry import build_registry
from . import agent as aa

OPEN_PREAMBLE = (
    "Answer the following question about the patient currently on screen. Use "
    "your data tools to investigate before deciding. Keep the final answer "
    "short and direct: a few words or one sentence. If the data does not "
    "support an answer, say so honestly rather than guessing. End your reply "
    "with a line in exactly this form:\nAnswer: <your short answer>"
)

BLIND_OPEN_PREAMBLE = (
    "You are answering a question about an ICU patient. You do NOT have access "
    "to the patient's video, activity data, or medical records — answer as best "
    "you can from general reasoning. Keep the answer short and direct: a few "
    "words or one sentence. If you don't know, say so honestly."
)


def _extract_final_answer(reply: str) -> str:
    if not reply:
        return ""
    for line in reversed(reply.splitlines()):
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("answer:") or low.startswith("**answer:**") or low.startswith("final answer:"):
            return s.split(":", 1)[1].strip().strip("*").strip()
        return s
    return reply.strip()


def run_blind_open(question: str, model: str) -> str:
    client = get_gemini_client()
    prompt = f"{BLIND_OPEN_PREAMBLE}\n\nQUESTION: {question}\n\nANSWER:"
    config = types.GenerateContentConfig(
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    resp = client.models.generate_content(model=model, contents=[prompt], config=config)
    if resp.candidates:
        return "".join(p.text for p in (resp.candidates[0].content.parts or []) if getattr(p, "text", None))
    return ""


def run_agent_open(question: str, pat_id: str, tools: list[str], model: str, registry: dict) -> str:
    msg = f"{OPEN_PREAMBLE}\n\nQUESTION: {question}"
    out = aa.run_ablated(msg, pat_id, tools, model, registry)
    return out.get("reply", "")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--configs", default="text_only,timeline_only,plus_reports,plus_ehr,tracer")
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--judge", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patient", default=None)
    ap.add_argument("--limit-per-patient", type=int, default=None)
    ap.add_argument("--run-name", default="open_ablation")
    ap.add_argument("--qids", default=None)
    args = ap.parse_args(argv)

    out_dir = _cfg.results_root() / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "cache_open.jsonl"

    recs, _ = load_records(args.csv)
    if args.qids:
        qid_set = {int(q.strip()) for q in args.qids.split(",") if q.strip()}
        recs = [r for r in recs if r.id in qid_set]
    if args.patient:
        recs = [r for r in recs if r.patient_id == args.patient]
    if args.limit_per_patient:
        seen = defaultdict(int)
        kept = []
        for r in recs:
            if seen[r.patient_id] < args.limit_per_patient:
                kept.append(r)
                seen[r.patient_id] += 1
        recs = kept

    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    config_tools = {name: cfgs.resolve(name) for name in config_names}
    registry = build_registry(include_video=True)

    done = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                done[(r["sig"], r["qid"])] = r
    lock = threading.Lock()

    def trial(name, tools, rec):
        sig = config_sig(tools)
        key = (sig, rec.id)
        if key in done:
            return key, done[key]
        try:
            raw = run_blind_open(rec.question, args.model) if not tools else \
                run_agent_open(rec.question, rec.patient_id, tools, args.model, registry)
        except Exception:
            raw = ""
        prediction = _extract_final_answer(raw)
        try:
            grade = grade_open(rec.question, rec.gold_answer, prediction, judge_model=args.judge)
            letter, label = grade.letter, grade.label
        except Exception:
            letter, label = "C", "NOT_ATTEMPTED"
        result = {"sig": sig, "config": name, "qid": rec.id, "patient_id": rec.patient_id,
                  "grade": letter, "label": label, "correct": (letter == "A"),
                  "prediction": prediction[:500]}
        with lock:
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result) + "\n")
            done[key] = result
        return key, result

    tasks = [(name, config_tools[name], rec) for name in config_names for rec in recs]
    todo = [t for t in tasks if (config_sig(t[1]), t[2].id) not in done]
    print(f"Open-ended sweep: {len(recs)} Qs x {config_names} | {len(tasks)} trials "
          f"({len(todo)} to run) | model={args.model} judge={args.judge}")
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for i, fut in enumerate(cf.as_completed([ex.submit(trial, *t) for t in todo]), 1):
            fut.result()
            if i % 10 == 0:
                print(f"  ...{i}/{len(todo)}")

    print("\n" + "=" * 72)
    print(f"OPEN-ENDED ABLATION  ({args.model}; judge={args.judge})")
    print("=" * 72)
    summary = {"model": args.model, "judge": args.judge, "n_questions": len(recs)}
    for name in config_names:
        sig = config_sig(config_tools[name])
        results = [r for r in (done.get((sig, r.id)) for r in recs) if r is not None]
        n = len(results)
        correct = sum(1 for r in results if r["grade"] == "A")
        incorrect = sum(1 for r in results if r["grade"] == "B")
        not_attempted = sum(1 for r in results if r["grade"] == "C")
        acc = correct / n if n else 0
        print(f"  {name:22s} acc={acc:.1%} ({correct}/{n})  C={correct} I={incorrect} N={not_attempted}")
        summary[name] = {"acc": acc, "n": n, "correct": correct, "incorrect": incorrect,
                         "not_attempted": not_attempted}
    (out_dir / "summary_open.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote {out_dir / 'summary_open.json'}")


if __name__ == "__main__":
    main()

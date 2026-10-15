"""Evaluate perception outputs against human clip annotations.

Per category: accuracy, sensitivity (recall), specificity, precision, F1;
overall micro-average (each clip x category decision weighted equally, the
paper's headline accuracy / micro-F1) and macro-average. UNABLE TO IDENTIFY
predictions are counted separately ("UTI") and excluded from the confusion
counts, matching the paper's convention that unable-to-identify is treated as
negative for the released timelines but not scored as a decision here.

By default all 15 categories are scored, matching the paper's perception
table. --exclude-rare drops the two rarest (RARE_CATEGORIES), giving the 13
categories used for the per-category figure and the cohort analysis.

Usage:
    python -m tracer.perception.evaluate gemini-3.1-flash-lite --annotations labels.csv
    python -m tracer.perception.evaluate gemini-3.1-flash-lite --annotations labels.csv --aggregate
    python -m tracer.perception.evaluate --annotations labels.csv          # every model with outputs
    python -m tracer.perception.evaluate gemini-3.1-flash-lite --annotations labels.csv --runs 0 1 2 3 4 --csv

Annotations CSV: a `video_id` column plus one column per label in LABEL_KEYS
with values YES / NO. Outputs are read from <TRACER_DATA_ROOT>/perception
(or --outputs-dir).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from .prompts import load_prompt
from .save_results import MODEL_PRICING, detect_runs, get_run_dir, outputs_dir

LABEL_KEYS = [
    "patient_grabbing_medical_equipment",
    "patient_using_non_medical_objects",
    "patient_gesticulating",
    "patient_adjusting_bedding",
    "patient_other_arm_movements",
    "patient_position_change",
    "patient_other_torso_movement",
    "patient_looking_at_others",
    "patient_head_turning",
    "patient_leg_ADL_movement",
    "patient_leg_in_bed_movement",
    "clinician_patient_contact",
    "patient_communication",
    "clinician_patient_object_exchange",
    "patient_restrained",
]

# The two rarest categories. Excluding them (--exclude-rare) gives the 13
# categories of the paper's per-category figure and cohort analysis.
RARE_CATEGORIES = {
    "patient_other_torso_movement",
    "patient_restrained",
}

# Nothing is excluded by default: all 15 categories are scored, as in the
# paper's perception table. --exclude adds labels to this set.
DEFAULT_EXCLUDE: set[str] = set()

METRIC_KEYS = ["accuracy", "sensitivity", "specificity", "precision", "f1"]
DEFAULT_PROMPT = "perception_prompt"


def load_subset(subset_file: str) -> set[str]:
    with open(subset_file, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def load_annotations(path: str | Path, subset: set[str] | None = None) -> dict[str, dict[str, str]]:
    annots = {}
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row["video_id"]
            if subset is not None and vid not in subset:
                continue
            annots[vid] = {k: str(row[k]).strip().upper() for k in LABEL_KEYS}
    return annots


def _load_prompt_for_eval(prompt_name: str):
    """Prompt names in output dirs may carry _fps<N> / _temp<val> suffixes."""
    import re
    candidate = prompt_name
    while True:
        try:
            return load_prompt(candidate)
        except FileNotFoundError:
            pass
        stripped = candidate
        for pat in (re.compile(r"_temp[0-9p]+$"), re.compile(r"_fps\d+$")):
            new = pat.sub("", stripped)
            if new != stripped:
                stripped = new
                break
        if stripped == candidate:
            return None
        candidate = stripped


def load_predictions(model: str, prompt_name: str = DEFAULT_PROMPT, run_id: int = 0,
                     subset: set[str] | None = None, root: Path | None = None) -> dict[str, dict[str, str]]:
    run_dir = get_run_dir(model, prompt_name, run_id, root)
    if not run_dir.exists():
        return {}
    prompt_obj = _load_prompt_for_eval(prompt_name)
    alias_map: dict = (prompt_obj.label_aliases if prompt_obj else None) or {}
    bad = {v for v in alias_map.values() if v not in LABEL_KEYS}
    if bad:
        raise ValueError(f"Prompt {prompt_name!r} label_aliases maps to unknown label(s): {sorted(bad)}")
    preds = {}
    for jf in sorted(run_dir.glob("*.json")):
        with open(jf, "r", encoding="utf-8") as f:
            d = json.load(f)
        parsed = d.get("parsed_response")
        if parsed is None:
            continue
        video_id = d.get("video_id", "")
        if subset is not None and video_id not in subset:
            continue
        labels = parsed.get("labels", {})
        if alias_map:
            labels = {alias_map.get(k, k): v for k, v in labels.items()}
        normed = {}
        for k in LABEL_KEYS:
            raw = str(labels.get(k, "")).strip().upper()
            normed[k] = "YES" if raw.startswith("YES") else ("NO" if raw.startswith("NO") else "UNABLE TO IDENTIFY")
        preds[video_id] = normed
    return preds


def compute_metrics(gt: dict, pred: dict, exclude: set[str] | None = None) -> tuple[dict, list[str]]:
    active_labels = [k for k in LABEL_KEYS if k not in (exclude or set())]
    results = {}
    total_tp = total_fp = total_tn = total_fn = total_unable = 0
    common = sorted(set(gt) & set(pred))
    for label in active_labels:
        tp = fp = tn = fn = unable = 0
        for vid in common:
            truth = gt[vid][label]
            prediction = pred[vid].get(label, "")
            if prediction == "UNABLE TO IDENTIFY":
                unable += 1
                continue
            if truth == "YES" and prediction == "YES":
                tp += 1
            elif truth == "NO" and prediction == "YES":
                fp += 1
            elif truth == "NO" and prediction == "NO":
                tn += 1
            elif truth == "YES" and prediction == "NO":
                fn += 1
        n = tp + fp + tn + fn
        accuracy = (tp + tn) / n if n > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        results[label] = {"accuracy": accuracy, "sensitivity": sensitivity, "specificity": specificity,
                          "precision": precision, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn,
                          "n": n, "unable": unable}
        total_tp += tp
        total_fp += fp
        total_tn += tn
        total_fn += fn
        total_unable += unable

    total_n = total_tp + total_fp + total_tn + total_fn
    micro_sens = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    results["OVERALL (micro-avg)"] = {
        "accuracy": (total_tp + total_tn) / total_n if total_n > 0 else 0,
        "sensitivity": micro_sens,
        "specificity": total_tn / (total_tn + total_fp) if (total_tn + total_fp) > 0 else 0,
        "precision": micro_prec,
        "f1": (2 * micro_prec * micro_sens / (micro_prec + micro_sens)) if (micro_prec + micro_sens) > 0 else 0,
        "tp": total_tp, "fp": total_fp, "tn": total_tn, "fn": total_fn, "n": total_n, "unable": total_unable,
    }
    n_labels = len(active_labels)
    results["OVERALL (macro-avg)"] = {
        **{m: sum(results[l][m] for l in active_labels) / n_labels for m in METRIC_KEYS},
        "tp": total_tp, "fp": total_fp, "tn": total_tn, "fn": total_fn, "n": total_n, "unable": total_unable,
    }
    return results, active_labels


def print_results(model, results, active_labels, n_common, n_gt, n_pred, prompt_name="", run_id=None,
                  exclude=None, subset_file=None):
    print(f"\n{'=' * 90}")
    print(f"Model:  {model}")
    if prompt_name:
        print(f"Prompt: {prompt_name}")
    if run_id is not None:
        print(f"Run:    run_{run_id}")
    if subset_file:
        print(f"Subset: {subset_file}")
    if exclude:
        print(f"Excluded: {', '.join(sorted(exclude))}")
    print(f"Clips: {n_common} evaluated  ({n_gt} annotated, {n_pred} predicted)")
    print(f"{'=' * 90}")
    header = f"{'Category':<45} {'Acc':>6} {'Sens':>6} {'Spec':>6} {'Prec':>6} {'F1':>6} {'TP':>4} {'FP':>4} {'TN':>4} {'FN':>4} {'UTI':>4}"
    print(header)
    print("-" * len(header))
    for label in active_labels + ["OVERALL (micro-avg)", "OVERALL (macro-avg)"]:
        m = results[label]
        if label.startswith("OVERALL"):
            print("-" * len(header))
        print(f"{label[:44]:<45} {m['accuracy']:6.1%} {m['sensitivity']:6.1%} {m['specificity']:6.1%} "
              f"{m['precision']:6.1%} {m['f1']:6.1%} {m['tp']:4d} {m['fp']:4d} {m['tn']:4d} {m['fn']:4d} {m['unable']:4d}")


def save_csv_report(out_dir: Path, results, active_labels, basename="full_evaluation") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{basename}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["category"] + METRIC_KEYS + ["tp", "fp", "tn", "fn", "unable", "n"])
        w.writeheader()
        for label in active_labels + ["OVERALL (micro-avg)", "OVERALL (macro-avg)"]:
            m = results[label]
            w.writerow({"category": label, **{k: f"{m[k]:.4f}" for k in METRIC_KEYS},
                        **{k: m[k] for k in ("tp", "fp", "tn", "fn", "unable", "n")}})
    print(f"  CSV saved: {out_path}")
    return out_path


def compute_profiling(model, prompt_name, run_id=0, root=None) -> dict | None:
    run_dir = get_run_dir(model, prompt_name, run_id, root)
    if not run_dir.exists():
        return None
    times, costs = [], []
    input_price, output_price = MODEL_PRICING.get(model, (0, 0))
    for jf in sorted(run_dir.glob("*.json")):
        with open(jf, "r", encoding="utf-8") as f:
            d = json.load(f)
        if d.get("elapsed_seconds") is not None:
            times.append(d["elapsed_seconds"])
        cost = ((d.get("prompt_tokens") or 0) * input_price + (d.get("response_tokens") or 0) * output_price) / 1_000_000
        costs.append(cost * 100)
    if not times:
        return None
    n = len(times)
    avg = sum(times) / n
    return {"avg_time_sec": avg, "std_time_sec": math.sqrt(sum((t - avg) ** 2 for t in times) / n) if n > 1 else 0.0,
            "avg_cost_cents": sum(costs) / len(costs) if costs else 0.0, "n": n}


def _sample_std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def compute_aggregate_metrics(gt, model, prompt_name, run_ids, exclude=None, subset=None,
                              require_complete=False, root=None):
    """mean +/- sample std of every metric across runs."""
    per_run = []
    for rid in run_ids:
        preds = load_predictions(model, prompt_name, run_id=rid, subset=subset, root=root)
        if not preds:
            continue
        if require_complete and set(gt) - set(preds):
            print(f"  Skipped incomplete run_{rid} (missing {len(set(gt) - set(preds))})")
            continue
        per_run.append(compute_metrics(gt, preds, exclude=exclude))
    if not per_run:
        return {}, [], 0
    active_labels = per_run[0][1]
    agg = {}
    for key in active_labels + ["OVERALL (micro-avg)", "OVERALL (macro-avg)"]:
        agg[key] = {}
        for metric in METRIC_KEYS:
            values = [r[key][metric] for r, _ in per_run]
            agg[key][f"{metric}_mean"] = sum(values) / len(values)
            agg[key][f"{metric}_std"] = _sample_std(values)
    return agg, active_labels, len(per_run)


def print_aggregate_results(model, prompt_name, agg, active_labels, n_runs, exclude=None):
    print(f"\n{'=' * 100}\nModel:  {model}\nPrompt: {prompt_name}\nRuns:   {n_runs}")
    if exclude:
        print(f"Excluded: {', '.join(sorted(exclude))}")
    header = f"{'Category':<45} {'Acc':>13} {'Sens':>13} {'Spec':>13} {'Prec':>13} {'F1':>13}"
    print(header)
    print("-" * len(header))
    for label in active_labels + ["OVERALL (micro-avg)", "OVERALL (macro-avg)"]:
        m = agg[label]
        if label.startswith("OVERALL"):
            print("-" * len(header))
        fmt = lambda k: f"{m[f'{k}_mean']:5.1%} +/- {m[f'{k}_std']:4.1%}"
        print(f"{label[:44]:<45} {fmt('accuracy'):>13} {fmt('sensitivity'):>13} {fmt('specificity'):>13} "
              f"{fmt('precision'):>13} {fmt('f1'):>13}")


def get_available_runs(root: Path | None = None) -> list[tuple[str, str]]:
    root = root or outputs_dir()
    if not root.exists():
        return []
    runs = []
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name in ("evals", "profiling"):
            continue
        for prompt_dir in sorted(model_dir.iterdir()):
            if prompt_dir.is_dir() and any(rd.is_dir() and rd.name.startswith("run_") and list(rd.glob("*.json"))
                                           for rd in prompt_dir.iterdir()):
                runs.append((model_dir.name, prompt_dir.name))
    return runs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate perception outputs against clip annotations")
    parser.add_argument("model", nargs="*", default=None)
    parser.add_argument("--annotations", required=True, help="annotations CSV (video_id + one column per label)")
    parser.add_argument("--outputs-dir", default=None, help="perception output root (default: <TRACER_DATA_ROOT>/perception)")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--exclude", nargs="+", default=None, metavar="LABEL")
    parser.add_argument("--exclude-rare", action="store_true",
                        help="drop the two rarest categories (13-category evaluation)")
    parser.add_argument("--no-default-exclude", action="store_true",
                        help="kept for compatibility; all 15 categories are scored by default")
    parser.add_argument("--subset", default=None, metavar="FILE")
    parser.add_argument("--csv", action="store_true", help="also save CSV reports under <outputs>/evals")
    parser.add_argument("--require-complete", action="store_true")
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument("--run", type=int, default=None)
    run_group.add_argument("--aggregate", action="store_true")
    run_group.add_argument("--runs", type=int, nargs="+", default=None)
    args = parser.parse_args(argv)

    root = Path(args.outputs_dir).expanduser() if args.outputs_dir else None
    exclude = set(DEFAULT_EXCLUDE)
    if args.exclude_rare:
        exclude |= RARE_CATEGORIES
    if args.exclude:
        exclude |= set(args.exclude)
    bad = exclude - set(LABEL_KEYS)
    if bad:
        parser.error(f"Unknown label(s): {', '.join(sorted(bad))}")
    exclude = exclude or None
    subset = load_subset(args.subset) if args.subset else None
    gt = load_annotations(args.annotations, subset=subset)
    run_id = args.run if args.run is not None else 0

    if args.model:
        runs = [(m, args.prompt or DEFAULT_PROMPT) for m in args.model]
    else:
        runs = [(m, p) for m, p in get_available_runs(root) if not args.prompt or p == args.prompt]
        if not runs:
            print("No model outputs found. Run tracer.perception.run_perception first.")
            return

    if args.aggregate or args.runs:
        for model, prompt_name in runs:
            run_ids = args.runs if args.runs else detect_runs(model, prompt_name, root)
            if not run_ids:
                print(f"\nNo runs found for {model} / {prompt_name}")
                continue
            agg, active_labels, n_runs = compute_aggregate_metrics(gt, model, prompt_name, run_ids, exclude=exclude,
                                                                   subset=subset, require_complete=args.require_complete,
                                                                   root=root)
            if not agg:
                print(f"\nNo predictions found across runs for {model} / {prompt_name}")
                continue
            print_aggregate_results(model, prompt_name, agg, active_labels, n_runs, exclude=exclude)
            if args.csv:
                out_dir = (root or outputs_dir()) / "evals" / model / prompt_name / "aggregate"
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "full_evaluation.csv", "w", newline="", encoding="utf-8") as f:
                    fields = ["category", "n_runs"] + [f"{m}_{s}" for m in METRIC_KEYS for s in ("mean", "std")]
                    w = csv.DictWriter(f, fieldnames=fields)
                    w.writeheader()
                    for label in active_labels + ["OVERALL (micro-avg)", "OVERALL (macro-avg)"]:
                        w.writerow({"category": label, "n_runs": n_runs,
                                    **{k: f"{v:.4f}" for k, v in agg[label].items()}})
                print(f"  Aggregate CSV saved: {out_dir / 'full_evaluation.csv'}")
        return

    summary_rows = []
    for model, prompt_name in runs:
        preds = load_predictions(model, prompt_name, run_id=run_id, subset=subset, root=root)
        if not preds:
            print(f"\nNo predictions found for {model} / {prompt_name} / run_{run_id}")
            continue
        if args.require_complete and set(gt) - set(preds):
            print(f"\nSkipping {model} / {prompt_name} / run_{run_id}: incomplete")
            continue
        common = set(gt) & set(preds)
        results, active_labels = compute_metrics(gt, preds, exclude=exclude)
        print_results(model, results, active_labels, len(common), len(gt), len(preds), prompt_name=prompt_name,
                      run_id=run_id, exclude=exclude, subset_file=args.subset)
        profile = compute_profiling(model, prompt_name, run_id=run_id, root=root)
        if profile:
            print(f"  Speed:  {profile['avg_time_sec']:.2f}s avg, {profile['std_time_sec']:.2f}s std  (n={profile['n']})")
            print(f"  Cost:   {profile['avg_cost_cents']:.4f} cents/clip (list prices in MODEL_PRICING)")
        if args.csv:
            save_csv_report((root or outputs_dir()) / "evals" / model / prompt_name / f"run_{run_id}", results, active_labels)
        summary_rows.append((model, prompt_name, results["OVERALL (micro-avg)"]["accuracy"],
                             results["OVERALL (micro-avg)"]["f1"], results["OVERALL (macro-avg)"]["accuracy"],
                             results["OVERALL (macro-avg)"]["f1"]))
    if len(summary_rows) > 1:
        print(f"\n{'=' * 90}\nSUMMARY\n{'=' * 90}")
        print(f"{'Model':<28} {'Prompt':<24} {'Mi-Acc':>7} {'Mi-F1':>7} {'Ma-Acc':>7} {'Ma-F1':>7}")
        for model, prompt_name, mi_acc, mi_f1, ma_acc, ma_f1 in summary_rows:
            print(f"{model:<28} {prompt_name:<24} {mi_acc:7.1%} {mi_f1:7.1%} {ma_acc:7.1%} {ma_f1:7.1%}")


if __name__ == "__main__":
    main()

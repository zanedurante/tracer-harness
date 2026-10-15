"""Scoring + metrics over lists of RunResults, and paired McNemar tests.
Pure / offline.

Per system: overall and per-stay accuracy, error / unparseable rates,
position-bias diagnostics. Paired comparisons: McNemar discordant counts with
the continuity-corrected chi-square statistic (3.841 is p=0.05) and the exact
two-sided binomial p-value. SimpleQA metrics for open-ended results.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field


@dataclass
class SystemScore:
    system: str
    n: int
    n_correct: int
    accuracy: float
    n_error: int
    n_unparseable: int
    n_scored: int
    by_patient: dict = field(default_factory=dict)
    accuracy_by_correct_letter: dict = field(default_factory=dict)
    chosen_distribution: dict = field(default_factory=dict)


def _acc(correct: int, n: int) -> float:
    return round(correct / n, 4) if n else 0.0


def score_system(results: list) -> SystemScore:
    n = len(results)
    n_correct = sum(1 for r in results if r.is_correct)
    n_error = sum(1 for r in results if r.error)
    n_unparseable = sum(1 for r in results if not r.error and r.chosen_letter is None)
    by_patient = {}
    pat_groups = defaultdict(list)
    for r in results:
        pat_groups[r.patient_id].append(r)
    for pat, rs in sorted(pat_groups.items()):
        c = sum(1 for r in rs if r.is_correct)
        by_patient[pat] = {"n": len(rs), "correct": c, "accuracy": _acc(c, len(rs))}
    by_letter = {}
    letter_groups = defaultdict(list)
    for r in results:
        letter_groups[r.correct_letter].append(r)
    for letter, rs in sorted(letter_groups.items()):
        c = sum(1 for r in rs if r.is_correct)
        by_letter[letter] = _acc(c, len(rs))
    chosen = Counter(r.chosen_letter for r in results)
    chosen_dist = {(k if k is not None else "NONE"): v
                   for k, v in sorted(chosen.items(), key=lambda kv: (kv[0] is None, kv[0]))}
    sys_name = results[0].system if results else "?"
    return SystemScore(system=sys_name, n=n, n_correct=n_correct, accuracy=_acc(n_correct, n),
                       n_error=n_error, n_unparseable=n_unparseable, n_scored=n - n_error,
                       by_patient=by_patient, accuracy_by_correct_letter=by_letter,
                       chosen_distribution=chosen_dist)


def exact_two_sided_binom_p(k: int, n: int) -> float:
    """Two-sided exact binomial p-value for k successes in n trials, p=0.5."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def mcnemar_counts(a_correct: dict, b_correct: dict, *, label_a: str = "a", label_b: str = "b") -> dict:
    """Paired McNemar test from two {trial_key: bool} maps on their common keys.

    n10 = A correct & B wrong; n01 = A wrong & B correct. Reports the
    continuity-corrected chi-square ((|n01-n10|-1)^2 / (n01+n10); 3.841 is
    p=0.05 with one degree of freedom) and the exact two-sided p-value."""
    keys = sorted(set(a_correct) & set(b_correct))
    n10 = sum(1 for k in keys if a_correct[k] and not b_correct[k])
    n01 = sum(1 for k in keys if not a_correct[k] and b_correct[k])
    n11 = sum(1 for k in keys if a_correct[k] and b_correct[k])
    n00 = len(keys) - n10 - n01 - n11
    disc = n01 + n10
    stat = ((abs(n01 - n10) - 1) ** 2) / disc if disc else 0.0
    return {
        "system_a": label_a, "system_b": label_b, "n_pairs": len(keys),
        "acc_a": round(sum(a_correct[k] for k in keys) / len(keys), 4) if keys else None,
        "acc_b": round(sum(b_correct[k] for k in keys) / len(keys), 4) if keys else None,
        "a_correct_b_wrong": n10, "a_wrong_b_correct": n01,
        "both_correct": n11, "both_wrong": n00,
        "mcnemar_statistic": round(stat, 4),
        "chi2_p_value": round(_chi2_sf_1df(stat), 5) if disc else 1.0,
        "exact_p_value": round(exact_two_sided_binom_p(min(n01, n10), disc), 5),
    }


def _chi2_sf_1df(x: float) -> float:
    """Survival function of chi-square with 1 d.o.f. = erfc(sqrt(x/2))."""
    return math.erfc(math.sqrt(max(x, 0.0) / 2.0))


def mcnemar(results_a: list, results_b: list) -> dict:
    """Paired comparison of two systems on the same items (matched by item_id
    and pos when present). Only items where BOTH produced an answer count."""
    def keyed(rs):
        out = {}
        for r in rs:
            if r.error:
                continue
            out[(r.item_id, getattr(r, "pos", None))] = bool(r.is_correct)
        return out
    return mcnemar_counts(keyed(results_a), keyed(results_b),
                          label_a=results_a[0].system if results_a else "a",
                          label_b=results_b[0].system if results_b else "b")


def distractor_confusion(results: list, items_by_id: dict) -> list[dict]:
    out = []
    for r in results:
        if r.error or r.is_correct or r.chosen_letter is None:
            continue
        item = items_by_id.get(r.item_id)
        out.append({"item_id": r.item_id, "patient_id": r.patient_id,
                    "correct_letter": r.correct_letter, "chosen_letter": r.chosen_letter,
                    "chosen_text": item.options.get(r.chosen_letter) if item else None,
                    "gold_text": item.gold_answer if item else None})
    return out


def score_simpleqa(results: list) -> dict:
    """SimpleQA-style metrics (Wei et al. 2024) from the judge's trinary grade."""
    graded = [r for r in results if not r.error and r.grade_label]
    c = sum(1 for r in graded if r.grade_label == "CORRECT")
    i = sum(1 for r in graded if r.grade_label == "INCORRECT")
    n = sum(1 for r in graded if r.grade_label == "NOT_ATTEMPTED")
    unparse = sum(1 for r in graded if r.grade_label == "UNPARSEABLE")
    n_graded = c + i + n
    attempted = c + i
    accuracy_attempted = round(c / attempted, 4) if attempted else 0.0
    is_correct = round(c / n_graded, 4) if n_graded else 0.0
    f1 = round(2 * accuracy_attempted * is_correct / (accuracy_attempted + is_correct), 4) \
        if accuracy_attempted + is_correct > 0 else 0.0
    return {"n_correct": c, "n_incorrect": i, "n_not_attempted": n, "n_unparseable": unparse,
            "n_graded": n_graded, "accuracy_given_attempted": accuracy_attempted,
            "is_correct_rate": is_correct, "f1": f1}


def position_mean_std(by_pos: dict[int, list[bool]]) -> tuple[float, float, list[float]]:
    """Mean and sample std (n-1) of per-position accuracies."""
    accs = [sum(v) / len(v) for p, v in sorted(by_pos.items()) if v]
    if not accs:
        return 0.0, 0.0, []
    mean = sum(accs) / len(accs)
    std = math.sqrt(sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)) if len(accs) > 1 else 0.0
    return mean, std, accs


def summarize(results_by_system: dict, items_by_id: dict | None = None) -> dict:
    scores = {name: score_system(rs) for name, rs in results_by_system.items()}
    summary = {"systems": {name: asdict(s) for name, s in scores.items()}}
    names = list(results_by_system)
    if len(names) >= 2:
        summary["mcnemar"] = {f"{a}_vs_{b}": mcnemar(results_by_system[a], results_by_system[b])
                              for i, a in enumerate(names) for b in names[i + 1:]}
    simpleqa = {}
    for name, rs in results_by_system.items():
        if any(r.grade_label for r in rs if not r.error):
            simpleqa[name] = score_simpleqa(rs)
    if simpleqa:
        summary["simpleqa"] = simpleqa
    if items_by_id is not None:
        summary["distractor_confusion"] = {name: distractor_confusion(rs, items_by_id)
                                           for name, rs in results_by_system.items()}
    return summary


def format_report(summary: dict) -> str:
    lines = ["=" * 60, "MCQ EVAL SUMMARY", "=" * 60]
    for name, s in summary["systems"].items():
        lines.append(f"\n[{name}]  accuracy={s['accuracy']:.1%}  ({s['n_correct']}/{s['n']})")
        lines.append(f"   scored={s['n_scored']}  errors={s['n_error']}  unparseable={s['n_unparseable']}")
        lines.append("   by stay: " + ", ".join(
            f"{p}={v['accuracy']:.0%}({v['correct']}/{v['n']})" for p, v in s["by_patient"].items()))
        lines.append("   accuracy by gold letter: " + str(s["accuracy_by_correct_letter"]))
        lines.append("   chosen-letter dist: " + str(s["chosen_distribution"]))
    for label, m in (summary.get("mcnemar") or {}).items():
        lines.append(f"\n[McNemar: {label}]")
        lines.append(f"   pairs={m['n_pairs']}  a-only={m['a_correct_b_wrong']}  b-only={m['a_wrong_b_correct']}  "
                     f"both={m['both_correct']}  neither={m['both_wrong']}")
        lines.append(f"   chi2(cc)={m['mcnemar_statistic']}  p={m['chi2_p_value']}  exact p={m['exact_p_value']}")
    if "simpleqa" in summary:
        lines.append("\n[SimpleQA grader (CORRECT / INCORRECT / NOT_ATTEMPTED)]")
        for name, s in summary["simpleqa"].items():
            lines.append(f"   {name}: F1={s['f1']:.3f}  correct={s['n_correct']}  incorrect={s['n_incorrect']}  "
                         f"not_attempted={s['n_not_attempted']}  acc_given_attempted={s['accuracy_given_attempted']:.1%}")
    return "\n".join(lines)

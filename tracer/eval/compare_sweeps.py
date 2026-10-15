"""Paired McNemar comparison of two tool configurations (or two runs of the
same configuration) from sweep caches.

Pairs on identical (question, gold position) trials, so question difficulty
and position bias cancel. Reports accuracy on the shared trials, discordant
counts, the continuity-corrected chi-square (3.841 is p=0.05) and the exact
two-sided p-value. For two runs of the same config the discordant rate is the
sampling noise floor.

    python -m tracer.eval.compare_sweeps --a <run>:<config> --b <run>:<config>
    e.g. --a demo:text_only --b demo:tracer
"""
from __future__ import annotations

import argparse

from .position_stats import load_cache, resolve_run
from .score import mcnemar_counts


def load_config(spec: str) -> dict[tuple, bool]:
    run, _, cfg = spec.partition(":")
    entries = load_cache(resolve_run(run) / "cache.jsonl")
    out = {}
    for e in entries:
        if not cfg or e.get("config") == cfg or e.get("sig") == cfg:
            out[(e["qid"], e["pos"], e.get("sample", 0))] = bool(e["correct"])
    if not out:
        raise SystemExit(f"no trials for {spec}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="<run>:<config>")
    ap.add_argument("--b", required=True, help="<run>:<config>")
    args = ap.parse_args(argv)
    A, B = load_config(args.a), load_config(args.b)
    m = mcnemar_counts(A, B, label_a=args.a, label_b=args.b)
    verdict = "n.s." if m["mcnemar_statistic"] <= 3.841 else "SIGNIFICANT p<0.05"
    print(f"{'comparison':40s} {'n':>5s} {'A%':>6s} {'B%':>6s} {'delta':>7s} {'A-only':>7s} {'B-only':>7s} {'chi2':>8s} {'p':>8s} {'exact p':>8s}  verdict")
    print(f"{args.a + ' vs ' + args.b:40s} {m['n_pairs']:5d} {m['acc_a'] * 100:6.1f} {m['acc_b'] * 100:6.1f} "
          f"{(m['acc_b'] - m['acc_a']) * 100:+7.1f} {m['a_correct_b_wrong']:7d} {m['a_wrong_b_correct']:7d} "
          f"{m['mcnemar_statistic']:8.2f} {m['chi2_p_value']:8.4f} {m['exact_p_value']:8.4f}  {verdict}")
    return m


if __name__ == "__main__":
    main()

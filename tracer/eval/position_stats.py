"""Accuracy mean +/- std over gold positions from a sweep cache
(<run>/cache.jsonl written by tracer.harness.sweep), per config.

The paper reports, for each configuration, the mean over the five gold
placements of the per-placement accuracy (the "+/-" is the std across
placements); question-category columns are question means (average the
positions within a question, then average questions).

    python -m tracer.eval.position_stats --run <run-name-or-dir> [--configs a,b]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from .. import config as _cfg


def load_cache(path: Path) -> list[dict]:
    """Latest entry per (config, qid, pos, sample), errors dropped."""
    latest: dict[tuple, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            e = json.loads(line)
            latest[(e["sig"], e["qid"], e["pos"], e.get("sample", 0))] = e
    return [e for e in latest.values() if not e.get("error")]


def position_stats(entries: list[dict]) -> dict[str, dict]:
    by_cfg: dict[str, dict[int, list[bool]]] = defaultdict(lambda: defaultdict(list))
    names: dict[str, str] = {}
    for e in entries:
        by_cfg[e["sig"]][e["pos"]].append(bool(e["correct"]))
        names.setdefault(e["sig"], e.get("config", e["sig"]))
    out = {}
    for sig, by_pos in by_cfg.items():
        accs = [sum(v) / len(v) for p, v in sorted(by_pos.items()) if v]
        mean = sum(accs) / len(accs)
        std = math.sqrt(sum((a - mean) ** 2 for a in accs) / (len(accs) - 1)) if len(accs) > 1 else 0.0
        n = sum(len(v) for v in by_pos.values())
        out[names[sig]] = {"sig": sig, "mean": mean, "std": std, "n_trials": n,
                           "per_position": {p: sum(v) / len(v) for p, v in sorted(by_pos.items())}}
    return out


def resolve_run(run: str) -> Path:
    """A run name under TRACER_RESULTS_ROOT (preferred when it holds a cache),
    else a directory path."""
    cand = _cfg.results_root() / run
    if (cand / "cache.jsonl").exists():
        return cand
    p = Path(run)
    if p.is_dir():
        return p
    return cand


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run name under TRACER_RESULTS_ROOT, or a directory")
    ap.add_argument("--configs", default=None, help="comma-separated config names to show")
    args = ap.parse_args(argv)
    entries = load_cache(resolve_run(args.run) / "cache.jsonl")
    stats = position_stats(entries)
    want = [c.strip() for c in args.configs.split(",")] if args.configs else list(stats)
    print(f"{'config':32s} {'mean':>6s} {'std':>5s} {'n':>5s}  per-position")
    for name in want:
        if name not in stats:
            print(f"{name:32s} (not in cache)")
            continue
        s = stats[name]
        pp = " ".join(f"{p}:{a * 100:.1f}" for p, a in s["per_position"].items())
        print(f"{name:32s} {s['mean'] * 100:6.1f} {s['std'] * 100:5.1f} {s['n_trials']:5d}  {pp}")


if __name__ == "__main__":
    main()

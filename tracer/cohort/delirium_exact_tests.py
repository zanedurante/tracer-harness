"""Exact tests for the delirium vs non-delirium atomic-action comparison.

Companion to delirium_fr_sensitivity (same cohort construction, same
OR-merged per-minute bitmasks, same CAM-ICU label rule), adding inference the
bootstrap intervals cannot provide:

  (1) per action: two-sided Mann-Whitney U on per-patient rates (minutes with
      action / observed minutes), plus Holm and Benjamini-Hochberg adjustment
      across the 13 evaluated actions;
  (2) per action: two-sided permutation tests of the minute-weighted log2 FR
      and the equal-patient-weight log2 FR, permuting the group labels;
  (3) composites: per-patient fraction of minutes with ANY of a set of actions
      ("purposeful" = non-medical objects, object exchange, leg ADL,
      gesticulating; "seven" = the seven actions significant under both
      weightings), Mann-Whitney U + permutation on the pooled-rate difference.

Usage:
    python -m tracer.cohort.delirium_exact_tests [--data-dir DIR] [--perms 1_000_000] [--seed 20260723] [--out FILE]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

from . import delirium_fr_sensitivity as dfs

PURPOSEFUL = ["patient_using_non_medical_objects", "clinician_patient_object_exchange",
              "patient_leg_ADL_movement", "patient_gesticulating"]
SEVEN = PURPOSEFUL + ["patient_other_arm_movements", "patient_grabbing_medical_equipment", "patient_head_turning"]


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i]); adj[i] = min(1.0, running)
    return adj


def bh(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p)[::-1]; adj = np.empty(m)
    running = 1.0
    for rank, i in enumerate(order):
        running = min(running, m * p[i] / (m - rank)); adj[i] = running
    return adj


def perm_all(C, M, grp, B, rng, batch=4000):
    """Vectorised two-sided permutation tests, one label-permutation stream for
    every column of C. Returns observed statistics and counts for mw (log2
    minute-weighted ratio), ew (log2 equal-weight mean-rate ratio) and pd
    (difference of pooled rates)."""
    N, K = C.shape; n_del = int(grp.sum()); R = C / M[:, None]

    def stats_for(gmask):
        gm = gmask.astype(float); nm = 1.0 - gm
        cd, cn = gm @ C, nm @ C
        md, mn = gm @ M, nm @ M
        rd, rn = gm @ R / n_del, nm @ R / (N - n_del)
        with np.errstate(divide="ignore", invalid="ignore"):
            mw = np.log2((cd / md[:, None]) / (cn / mn[:, None]))
            ew = np.log2(rd / rn)
            pd = cd / md[:, None] - cn / mn[:, None]
        return mw, ew, pd
    obs_mw, obs_ew, obs_pd = (x[0] for x in stats_for(grp[None, :]))
    n_ge = np.zeros((3, K), int); done = 0
    while done < B:
        b = min(batch, B - done)
        perm = np.argsort(rng.random((b, N)), axis=1)
        gmask = np.zeros((b, N), bool); np.put_along_axis(gmask, perm[:, :n_del], True, axis=1)
        mw, ew, pd = stats_for(gmask)
        n_ge[0] += (np.abs(mw) >= np.abs(obs_mw) - 1e-12).sum(axis=0)
        n_ge[1] += (np.abs(ew) >= np.abs(obs_ew) - 1e-12).sum(axis=0)
        n_ge[2] += (np.abs(pd) >= np.abs(obs_pd) - 1e-12).sum(axis=0)
        done += b
    p = (n_ge + 1) / (B + 1)
    return dict(obs_mw=obs_mw, p_mw=p[0], n_ge_mw=n_ge[0], obs_ew=obs_ew, p_ew=p[1], n_ge_ew=n_ge[1],
                obs_pd=obs_pd, p_pd=p[2], n_ge_pd=n_ge[2])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--perms", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    data_dir = a.data_dir or dfs.default_data_dir()
    rng = np.random.default_rng(a.seed)

    labels = dfs.load_action_labels(data_dir)
    patients, excl0, missing = dfs.build_cohort(data_dir)
    counts = np.array([p["counts"] for p in patients], float)
    minutes = np.array([p["minutes"] for p in patients], float)
    grp = np.array([bool(p["delirium"]) for p in patients])
    rates = counts / minutes[:, None]
    n_del, n_non = int(grp.sum()), int((~grp).sum())
    print(f"cohort: {len(patients)} patients ({n_del} delirium / {n_non} non), excluded_zero_minutes={excl0}, missing={missing}")
    print(f"minutes: delirium={int(minutes[grp].sum())} non={int(minutes[~grp].sum())}; perms={a.perms}")

    idx = {l: i for i, l in enumerate(labels)}
    pat_ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(data_dir, "*.npz")))
    kept = [pid for pid in pat_ids if os.path.exists(os.path.join(data_dir, f"{pid}_timeline.json"))
            and dfs.load_matrix(data_dir, pid) is not None]
    assert len(kept) == len(patients), (len(kept), len(patients))

    def any_minutes(pid, cols):
        npz = np.load(os.path.join(data_dir, f"{pid}.npz"))
        hour_keys = sorted(k for k in npz.files if k.startswith("hour_"))
        m = np.vstack([np.unpackbits(npz[hk], axis=1)[:, :dfs.N_ACTIONS] for hk in hour_keys])
        return int(m[:, cols].any(axis=1).sum()), int(m.shape[0])

    COMPS = (("purposeful4", PURPOSEFUL), ("seven_significant", SEVEN))
    comp_cols = []
    for name, acts in COMPS:
        anym = np.array([any_minutes(pid, [idx[x] for x in acts]) for pid in kept], float)
        assert np.allclose(anym[:, 1], minutes), "minute totals differ from build_cohort"
        comp_cols.append(anym[:, 0])

    Cfull = np.column_stack([counts] + comp_cols)
    P = perm_all(Cfull, minutes, grp, a.perms, rng)

    rows = []
    for i, lab in enumerate(labels):
        rd, rn = rates[grp, i], rates[~grp, i]
        u = stats.mannwhitneyu(rd, rn, alternative="two-sided", method="asymptotic")
        rows.append(dict(label=lab, evaluated=lab not in dfs.NOT_IN_PUBLISHED_FIGURE,
                         median_del=float(np.median(rd)), median_non=float(np.median(rn)),
                         U=float(u.statistic), p_mwu=float(u.pvalue),
                         log2FR_mw=float(P["obs_mw"][i]), p_perm_mw=float(P["p_mw"][i]), n_ge_mw=int(P["n_ge_mw"][i]),
                         log2FR_ew=float(P["obs_ew"][i]), p_perm_ew=float(P["p_ew"][i]), n_ge_ew=int(P["n_ge_ew"][i])))
        print(f"  {lab:38s} MWU p={u.pvalue:.3g}  perm mw p={P['p_mw'][i]:.3g}  perm ew p={P['p_ew'][i]:.3g}", flush=True)

    ev_rows = [r for r in rows if r["evaluated"]]
    for r, h, b in zip(ev_rows, holm([r["p_mwu"] for r in ev_rows]), bh([r["p_mwu"] for r in ev_rows])):
        r["p_mwu_holm"] = float(h); r["p_mwu_bh"] = float(b)

    composites = {}
    for j, (name, acts) in enumerate(COMPS):
        k = dfs.N_ACTIONS + j
        frac = Cfull[:, k] / minutes; fd, fn = frac[grp], frac[~grp]
        u = stats.mannwhitneyu(fd, fn, alternative="two-sided", method="asymptotic")
        q = lambda x: [float(np.percentile(x, 25)), float(np.median(x)), float(np.percentile(x, 75))]
        composites[name] = dict(actions=acts, del_q25_med_q75=q(fd), non_q25_med_q75=q(fn),
                                pooled_rate_del=float(Cfull[grp, k].sum() / minutes[grp].sum()),
                                pooled_rate_non=float(Cfull[~grp, k].sum() / minutes[~grp].sum()),
                                U=float(u.statistic), p_mwu=float(u.pvalue),
                                pooled_diff=float(P["obs_pd"][k]), p_perm=float(P["p_pd"][k]), n_ge=int(P["n_ge_pd"][k]),
                                log2FR_mw=float(P["obs_mw"][k]), p_perm_mw=float(P["p_mw"][k]))
        print(f"  composite {name}: del median {np.median(fd):.4f} non {np.median(fn):.4f}  MWU p={u.pvalue:.3g}  perm p={P['p_pd'][k]:.3g}", flush=True)

    out = a.out
    if out is None:
        from .. import config
        config.results_root().mkdir(parents=True, exist_ok=True)
        out = str(config.results_root() / "delirium_exact_tests.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dict(seed=a.seed, perms=a.perms, n_delirium=n_del, n_non_delirium=n_non,
                       minutes_delirium=int(minutes[grp].sum()), minutes_non_delirium=int(minutes[~grp].sum()),
                       actions=rows, composites=composites), f, indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()

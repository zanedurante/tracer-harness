#!/usr/bin/env python3
"""Per-action delirium frequency ratios with sensitivity analyses.

Frequency ratio (paper Eq. freq-ratio): FR_a = f_a(del) / f_a(non-del), where
f_a(g) = sum_p m_a(p) / sum_p M(p) pools minutes over the patients of group g
(minute-weighted). This script adds
  (1) equal-patient-weight (mean and median) estimators,
  (2) patient-level bootstrap CIs for both estimators,
  (3) descriptive group characteristics (LOS, observed minutes, ventilation, RASS).

Delirium grouping: a patient is in the DELIRIUM group iff any
cam_icu_delirium_test event has cam_result == "Positive", at any point.

Reads only stored per-patient minute counts and timeline bundles
(<data-dir>/<pat_id>.npz, <pat_id>_timeline.json, activity_key.csv). Runs no
model inference. Emits aggregate statistics only.

Usage:
    python -m tracer.cohort.delirium_fr_sensitivity [--data-dir DIR] [--bootstrap 10000] [--seed 20260723] [--out FILE]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from datetime import datetime

import numpy as np

N_ACTIONS = 15
# Actions dropped from the published 13-bar rendering of the figure
NOT_IN_PUBLISHED_FIGURE = {"patient_restrained", "patient_other_torso_movement"}


def default_data_dir() -> str:
    from .. import config
    return str(config.data_root() / "timelines")


# ---------------------------------------------------------------------- parsing
def _parse_time(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        try:
            return datetime.strptime(s, "%Y-%m-%d %H:%M:%S")
        except Exception:
            return None


def _align(a, b):
    if a is None or b is None:
        return a, b
    if a.tzinfo is None and b.tzinfo is not None:
        a = a.replace(tzinfo=b.tzinfo)
    elif a.tzinfo is not None and b.tzinfo is None:
        b = b.replace(tzinfo=a.tzinfo)
    return a, b


def load_action_labels(data_dir: str):
    labels = {}
    with open(os.path.join(data_dir, "activity_key.csv"), encoding="utf-8") as f:
        next(f)
        for line in f:
            line = line.strip()
            if line:
                idx, label = line.split(",", 1)
                labels[int(idx)] = label
    return [labels[i] for i in range(N_ACTIONS)]


def load_matrix(data_dir: str, pat_id: str):
    """(col_sums[15], n_minutes) from the stored per-minute bitmask."""
    npz_path = os.path.join(data_dir, f"{pat_id}.npz")
    if not os.path.exists(npz_path):
        return None
    npz = np.load(npz_path)
    hour_keys = sorted(k for k in npz.files if k.startswith("hour_"))
    if not hour_keys:
        return None
    mat = np.vstack([np.unpackbits(npz[hk], axis=1)[:, :N_ACTIONS] for hk in hour_keys]).astype(np.uint8)
    if mat.shape[0] == 0:
        return None
    return mat.sum(axis=0).astype(np.float64), int(mat.shape[0])


def monitored_encounter(timeline):
    """The encounter whose admit..discharge window contains icu_start."""
    icu_start = _parse_time(timeline.get("icu_start"))
    if icu_start is None:
        return None, None, None
    for enc in timeline.get("encounters", []) or []:
        a = _parse_time(enc.get("hosp_admsn_time"))
        d = _parse_time(enc.get("hosp_disch_time"))
        if a is None:
            continue
        s, a2 = _align(icu_start, a)
        _, d2 = _align(icu_start, d) if d else (None, None)
        if a2 <= s and (d2 is None or s <= d2):
            return enc, a2, d2
    return None, None, None


def in_window(t, lo, hi):
    if t is None:
        return False
    t1, lo1 = _align(t, lo)
    if lo1 is not None and t1 < lo1:
        return False
    if hi is not None:
        t2, hi2 = _align(t, hi)
        if hi2 is not None and t2 > hi2:
            return False
    return True


# ---------------------------------------------------------------------- cohort
def build_cohort(data_dir: str):
    pat_ids = sorted(os.path.basename(f)[:-4] for f in glob.glob(os.path.join(data_dir, "*.npz")))
    patients, excluded_zero_minutes, missing_bundle = [], 0, 0
    for pid in pat_ids:
        tl_path = os.path.join(data_dir, f"{pid}_timeline.json")
        if not os.path.exists(tl_path):
            missing_bundle += 1
            continue
        m = load_matrix(data_dir, pid)
        if m is None:
            excluded_zero_minutes += 1
            continue
        col_sums, n_min = m
        with open(tl_path, encoding="utf-8") as f:
            tl = json.load(f)
        events = tl.get("events", []) or []
        delirium = any(e["type"] == "cam_icu_delirium_test" and (e.get("data") or {}).get("cam_result") == "Positive"
                       for e in events)
        n_cam = sum(1 for e in events if e["type"] == "cam_icu_delirium_test"
                    and (e.get("data") or {}).get("cam_result") in ("Positive", "Negative"))
        enc, adm, dis = monitored_encounter(tl)
        icu_days = None
        if enc is not None:
            try:
                icu_days = float(enc.get("icu_days")) if enc.get("icu_days") is not None else None
            except (TypeError, ValueError):
                icu_days = None
        vent_invasive = vent_any = False
        rass_scores = []
        for e in events:
            t = _parse_time(e.get("time"))
            if enc is not None and not in_window(t, adm, dis):
                continue
            if e["type"] == "ventilation":
                vt = (e.get("data") or {}).get("ventilation_type")
                vent_invasive |= vt == "Invasive"
                vent_any |= vt in ("Invasive", "NIV")
            elif e["type"] == "rass":
                try:
                    rass_scores.append(int((e.get("data") or {}).get("rass_score_int")))
                except (TypeError, ValueError):
                    pass
        patients.append({"counts": col_sums, "minutes": n_min, "delirium": delirium, "n_cam": n_cam,
                         "icu_days": icu_days, "vent_invasive": vent_invasive, "vent_any": vent_any,
                         "rass": rass_scores})
    return patients, excluded_zero_minutes, missing_bundle


# ---------------------------------------------------------------------- estimators
def minute_weighted(counts, minutes):
    return counts.sum(axis=0) / minutes.sum()


def equal_weight_mean(rates):
    return rates.mean(axis=0)


def equal_weight_median(rates):
    return np.median(rates, axis=0)


def safe_ratio(num, den):
    out = np.full_like(num, np.nan, dtype=np.float64)
    for i in range(len(num)):
        if den[i] > 0:
            out[i] = num[i] / den[i]
        elif num[i] > 0:
            out[i] = np.inf
    return out


def bootstrap(cd, md, rd, cn, mn, rn, B, seed, batch=1000):
    """Patient-level bootstrap: resample whole patients within each group.
    Returns log2 ratios for both estimators and the number of redrawn replicates."""
    rng = np.random.default_rng(seed)
    nd, nn = cd.shape[0], cn.shape[0]
    mw_out, ew_out = [], []
    redraws = kept = 0
    while kept < B:
        b = min(batch, (B - kept) * 2)
        i_d = rng.integers(0, nd, size=(b, nd))
        i_n = rng.integers(0, nn, size=(b, nn))
        f_d = cd[i_d].sum(axis=1) / md[i_d].sum(axis=1)[:, None]
        f_n = cn[i_n].sum(axis=1) / mn[i_n].sum(axis=1)[:, None]
        g_d = rd[i_d].mean(axis=1)
        g_n = rn[i_n].mean(axis=1)
        valid = (f_n > 0).all(axis=1) & (g_n > 0).all(axis=1)
        redraws += int((~valid).sum())
        f_d, f_n, g_d, g_n = f_d[valid], f_n[valid], g_d[valid], g_n[valid]
        take = min(f_d.shape[0], B - kept)
        if take > 0:
            with np.errstate(divide="ignore"):
                mw_out.append(np.log2(f_d[:take] / f_n[:take]))
                ew_out.append(np.log2(g_d[:take] / g_n[:take]))
            kept += take
    return np.vstack(mw_out), np.vstack(ew_out), redraws


def med_iqr(vals):
    v = np.asarray([x for x in vals if x is not None], dtype=np.float64)
    if v.size == 0:
        return None
    return (float(np.median(v)), float(np.percentile(v, 25)), float(np.percentile(v, 75)), int(v.size))


def fmt_iqr(t, dec=1):
    if t is None:
        return "n/a"
    m, q1, q3, _ = t
    return f"{m:.{dec}f} [{q1:.{dec}f}-{q3:.{dec}f}]"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="directory of <pat_id>.npz / <pat_id>_timeline.json / activity_key.csv")
    ap.add_argument("--bootstrap", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260723)
    ap.add_argument("--out", default=None, help="JSON output path (default: <results>/delirium_fr_sensitivity.json)")
    args = ap.parse_args(argv)
    data_dir = args.data_dir or default_data_dir()

    labels = load_action_labels(data_dir)
    patients, excl_zero, missing = build_cohort(data_dir)
    dele = [p for p in patients if p["delirium"]]
    nond = [p for p in patients if not p["delirium"]]
    if not dele or not nond:
        raise SystemExit(f"need both groups: delirium={len(dele)} non-delirium={len(nond)}")
    cd = np.array([p["counts"] for p in dele]); md = np.array([p["minutes"] for p in dele], dtype=np.float64)
    cn = np.array([p["counts"] for p in nond]); mn = np.array([p["minutes"] for p in nond], dtype=np.float64)
    rd = cd / md[:, None]
    rn = cn / mn[:, None]

    zero_non = [labels[i] for i in range(N_ACTIONS) if cn[:, i].sum() == 0]
    if zero_non:
        raise SystemExit("the non-delirium group has zero minutes for "
                         f"{zero_non}; the frequency ratio is undefined and the bootstrap cannot "
                         "converge (every replicate would be redrawn). Drop those labels or check the bundles.")
    mw = safe_ratio(minute_weighted(cd, md), minute_weighted(cn, mn))
    ewm = safe_ratio(equal_weight_mean(rd), equal_weight_mean(rn))
    ewmd = safe_ratio(equal_weight_median(rd), equal_weight_median(rn))
    mw_b, ew_b, redraws = bootstrap(cd, md, rd, cn, mn, rn, args.bootstrap, args.seed)
    mw_lo, mw_hi = np.percentile(mw_b, 2.5, axis=0), np.percentile(mw_b, 97.5, axis=0)
    ew_lo, ew_hi = np.percentile(ew_b, 2.5, axis=0), np.percentile(ew_b, 97.5, axis=0)
    frac_neg = (mw_b < 0).mean(axis=0)
    frac_neg_ew = (ew_b < 0).mean(axis=0)

    print(f"B = {args.bootstrap}, seed = {args.seed}")
    print(f"n_delirium = {len(dele)}, n_non_delirium = {len(nond)}, total = {len(patients)}")
    print(f"excluded (M=0): {excl_zero}, missing bundle: {missing}")
    print(f"total observed minutes: delirium = {int(md.sum())}, non-delirium = {int(mn.sum())}")
    print(f"bootstrap redraws (zero denominator): {redraws}\n")

    with np.errstate(divide="ignore", invalid="ignore"):
        print("## Frequency ratios")
        print("| Action | Minute-wt FR | Minute-wt log2FR | Equal-wt (mean) FR | Equal-wt (mean) log2FR | Equal-wt (median) FR |")
        print("|---|---|---|---|---|---|")
        for i, lab in enumerate(labels):
            star = " *" if lab in NOT_IN_PUBLISHED_FIGURE else ""
            emd = "undefined" if not np.isfinite(ewmd[i]) else f"{ewmd[i]:.2f}"
            print(f"| {lab}{star} | {mw[i]:.2f} | {np.log2(mw[i]):.2f} | {ewm[i]:.2f} | {np.log2(ewm[i]):.2f} | {emd} |")
        print("\n## Bootstrap intervals")
        print("| Action | Minute-wt log2FR [95% CI] | Equal-wt log2FR [95% CI] | Frac. replicates log2FR < 0 (mw / ew) |")
        print("|---|---|---|---|")
        for i, lab in enumerate(labels):
            star = " *" if lab in NOT_IN_PUBLISHED_FIGURE else ""
            print(f"| {lab}{star} | {np.log2(mw[i]):.2f} [{mw_lo[i]:.2f}, {mw_hi[i]:.2f}] | "
                  f"{np.log2(ewm[i]):.2f} [{ew_lo[i]:.2f}, {ew_hi[i]:.2f}] | {frac_neg[i]:.3f} / {frac_neg_ew[i]:.3f} |")

    print("\n## Group characteristics")
    for name, grp in (("DELIRIUM", dele), ("NON-DELIRIUM", nond)):
        los = med_iqr([p["icu_days"] for p in grp])
        mins = med_iqr([p["minutes"] for p in grp])
        vi = sum(p["vent_invasive"] for p in grp)
        va = sum(p["vent_any"] for p in grp)
        permed = med_iqr([float(np.median(p["rass"])) for p in grp if p["rass"]])
        allr = [s for p in grp for s in p["rass"]]
        deep = (sum(1 for s in allr if s <= -3) / len(allr) * 100) if allr else None
        print(f"  {name} (n={len(grp)})")
        print(f"    ICU LOS days           : {fmt_iqr(los, 2)}  (missing {sum(1 for p in grp if p['icu_days'] is None)})")
        print(f"    observed minutes       : {fmt_iqr(mins, 0)}")
        print(f"    invasive mech vent     : {vi} ({vi / len(grp) * 100:.1f}%)")
        print(f"    any vent (Invasive|NIV): {va} ({va / len(grp) * 100:.1f}%)")
        print(f"    per-patient median RASS: {fmt_iqr(permed, 2)}  (no RASS: {sum(1 for p in grp if not p['rass'])})")
        print(f"    RASS assessments       : {len(allr)}")
        print(f"    % RASS <= -3           : {deep:.1f}%" if deep is not None else "    % RASS <= -3: n/a")
        print(f"    patients w/ 0 CAM-ICU  : {sum(1 for p in grp if p['n_cam'] == 0)}")

    out = args.out
    if out is None:
        from .. import config
        config.results_root().mkdir(parents=True, exist_ok=True)
        out = str(config.results_root() / "delirium_fr_sensitivity.json")
    with np.errstate(divide="ignore", invalid="ignore"):
        payload = {
            "seed": args.seed, "B": args.bootstrap, "n_delirium": len(dele), "n_non_delirium": len(nond),
            "minutes_delirium": int(md.sum()), "minutes_non_delirium": int(mn.sum()), "redraws": redraws,
            "actions": [{"label": labels[i],
                         "minute_wt_FR": float(mw[i]), "minute_wt_log2FR": float(np.log2(mw[i])),
                         "equal_wt_mean_FR": float(ewm[i]), "equal_wt_mean_log2FR": float(np.log2(ewm[i])),
                         "equal_wt_median_FR": None if not np.isfinite(ewmd[i]) else float(ewmd[i]),
                         "mw_ci": [float(mw_lo[i]), float(mw_hi[i])], "ew_ci": [float(ew_lo[i]), float(ew_hi[i])],
                         "frac_neg_mw": float(frac_neg[i]), "frac_neg_ew": float(frac_neg_ew[i])}
                        for i in range(N_ACTIONS)],
        }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()

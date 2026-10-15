"""Timeline tools over the minute-level records: get_atomic_actions,
query_actions, find_first_occurrence, find_moments_of_interest.

All times are local-time-naive strings ("YYYY-MM-DD HH:MM") for consistency
with the timeline JSON's icu_start; video filenames are emitted in UTC.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from google.genai.types import FunctionDeclaration, Schema, Type
from scipy.signal import find_peaks

from .. import config, data
from .common import (icu_start_utc, load_action_matrix, load_timeline, npz_path,
                     video_minute_window, video_range_pt)
from .labels import ACTIVITY_LABELS, labels_inline


# ---------------------------------------------------------------------------
# get_atomic_actions
# ---------------------------------------------------------------------------
def get_atomic_actions(pat_id: str, hour_index: int) -> dict:
    """Return per-minute atomic action detections for a single hour.

    Returns 60 entries, each with the minute index (0-59) and a list of
    currently-active action labels (may be empty). Minutes outside the
    stay's video coverage window are flagged with outside_video_coverage=True
    and have an empty active_labels list.
    """
    path = npz_path(pat_id)
    if not path.exists():
        return {"error": f"No action data for patient {pat_id}"}
    data_ = np.load(path)
    key = f"hour_{int(hour_index):04d}"
    if key not in data_.files:
        return {"error": f"hour_index {hour_index} out of range"}
    arr = data_[key]
    unpacked = np.unpackbits(arr, axis=1)[:, : len(ACTIVITY_LABELS)]

    try:
        tl = load_timeline(pat_id)
        n_minutes_total = int(tl.get("n_minutes") or 0) or (data_["hour_0000"].shape[0] * len(data_.files))
    except Exception:
        tl, n_minutes_total = {}, 60 * len(data_.files)
    vmin_lo, vmin_hi = video_minute_window(pat_id, tl, n_minutes_total) if tl else (0, n_minutes_total)
    hour_start_abs = int(hour_index) * 60

    minutes = []
    for i in range(unpacked.shape[0]):
        abs_minute = hour_start_abs + i
        in_video = vmin_lo <= abs_minute < vmin_hi
        if in_video:
            row = unpacked[i]
            active = [ACTIVITY_LABELS[j] for j in range(len(ACTIVITY_LABELS)) if row[j]]
            minutes.append({"minute": i, "active_labels": active})
        else:
            minutes.append({"minute": i, "active_labels": [], "outside_video_coverage": True})
    return {
        "pat_id": pat_id,
        "hour_index": int(hour_index),
        "minutes": minutes,
        "video_minute_window": [vmin_lo - hour_start_abs, vmin_hi - hour_start_abs],
    }


GET_ATOMIC_ACTIONS_DECL = FunctionDeclaration(
    name="get_atomic_actions",
    description=(
        "Return minute-by-minute atomic action detections for one hour of the patient's stay. "
        "hour_index=0 is the first hour from ICU admission. Each minute lists currently-active "
        "action labels (from the atomic action taxonomy). Defaults pat_id to the current patient."
    ),
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "pat_id": Schema(type=Type.STRING, description="Patient ID. Optional."),
            "hour_index": Schema(type=Type.INTEGER, description="Zero-indexed hour since ICU start."),
        },
        required=["hour_index"],
    ),
)


# ---------------------------------------------------------------------------
# query_actions
# ---------------------------------------------------------------------------
def query_actions(pat_id: str, labels: list[str] | None = None,
                  start_hour: int | None = None, end_hour: int | None = None) -> dict:
    """Aggregate atomic action counts across a range of hours: per-label
    minutes active, per-hour totals, and per-label per-hour minutes."""
    try:
        full = load_action_matrix(pat_id)
    except (FileNotFoundError, RuntimeError) as e:
        return {"error": str(e)}
    tl = load_timeline(pat_id)
    n_hours = int(tl.get("n_hours", full.shape[0] // 60))

    s = 0 if start_hour is None else max(0, int(start_hour))
    e = n_hours if end_hour is None else min(n_hours, int(end_hour))
    if s >= e:
        return {"error": f"Empty hour range: start={s} end={e}"}

    slice_ = full[s * 60: e * 60].copy()

    n_minutes_total = full.shape[0]
    vmin_lo, vmin_hi = video_minute_window(pat_id, tl, n_minutes_total)
    rel_v_lo = max(0, vmin_lo - s * 60)
    rel_v_hi = min(slice_.shape[0], vmin_hi - s * 60)
    if rel_v_lo > 0:
        slice_[:rel_v_lo] = 0
    if rel_v_hi < slice_.shape[0]:
        slice_[rel_v_hi:] = 0

    if labels:
        missing = [l for l in labels if l not in ACTIVITY_LABELS]
        if missing:
            return {"error": f"Unknown labels: {missing}. Known: {ACTIVITY_LABELS}"}
        label_filter = [ACTIVITY_LABELS.index(l) for l in labels]
        slice_ = slice_[:, label_filter]
        label_names = labels
    else:
        label_names = ACTIVITY_LABELS

    per_label_minutes = slice_.sum(axis=0).tolist()
    per_hour_any = slice_.reshape(-1, 60, slice_.shape[1]).any(axis=2).sum(axis=1).tolist()
    per_hour_per_label = slice_.reshape(-1, 60, slice_.shape[1]).sum(axis=1)
    per_label_per_hour = {
        label_names[j]: [int(x) for x in per_hour_per_label[:, j]]
        for j in range(len(label_names))
    }
    return {
        "pat_id": pat_id,
        "start_hour": s,
        "end_hour": e,
        "total_minutes": int(slice_.shape[0]),
        "per_label_minutes_active": dict(zip(label_names, [int(x) for x in per_label_minutes])),
        "per_hour_minutes_any_label_active": [int(x) for x in per_hour_any],
        "per_label_per_hour_minutes_active": per_label_per_hour,
    }


QUERY_ACTIONS_DECL = FunctionDeclaration(
    name="query_actions",
    description=(
        "Aggregate atomic action counts across a range of hours. Returns: (a) total minutes each "
        "label was active over the window, (b) per-hour counts of minutes with any selected label "
        "active, and (c) per-label per-hour minutes active (dict[label] -> list over the hour range) "
        "-- use (c) to answer questions like 'which hour had the most X'. Filter by specific labels or "
        "omit to include all. start_hour/end_hour are zero-indexed from ICU start (end_hour is exclusive). "
        "Defaults pat_id to the current patient."
    ),
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "pat_id": Schema(type=Type.STRING, description="Patient ID. Optional."),
            "labels": Schema(
                type=Type.ARRAY, items=Schema(type=Type.STRING),
                description=("Atomic-action labels to include. Omit for all. Pick by what each label means:\n"
                             + labels_inline()),
            ),
            "start_hour": Schema(type=Type.INTEGER, description="Inclusive start hour (zero-indexed). Omit for 0."),
            "end_hour": Schema(type=Type.INTEGER, description="Exclusive end hour. Omit for all."),
        },
    ),
)


# ---------------------------------------------------------------------------
# find_first_occurrence
# ---------------------------------------------------------------------------
def find_first_occurrence(pat_id: str, labels: list[str],
                          start_hour: int | None = None, end_hour: int | None = None) -> dict:
    """Return the first minute where any of the given labels was active."""
    if not labels:
        return {"error": "Must pass at least one label."}
    missing = [l for l in labels if l not in ACTIVITY_LABELS]
    if missing:
        return {"error": f"Unknown labels: {missing}. Known: {ACTIVITY_LABELS}"}
    try:
        full = load_action_matrix(pat_id)
    except (FileNotFoundError, RuntimeError) as e:
        return {"error": str(e)}
    tl = load_timeline(pat_id)
    n_hours = int(tl.get("n_hours", full.shape[0] // 60))

    s = 0 if start_hour is None else max(0, int(start_hour))
    e = n_hours if end_hour is None else min(n_hours, int(end_hour))
    if s >= e:
        return {"error": f"Empty hour range: start={s} end={e}"}

    n_minutes_total = full.shape[0]
    vmin_lo, vmin_hi = video_minute_window(pat_id, tl, n_minutes_total)
    abs_lo = max(s * 60, vmin_lo)
    abs_hi = min(e * 60, vmin_hi)
    if abs_hi <= abs_lo:
        return {"pat_id": pat_id, "labels": labels, "found": False, "start_hour": s, "end_hour": e,
                "note": "Empty range after clamping to video coverage."}

    idxs = [ACTIVITY_LABELS.index(l) for l in labels]
    sub = full[abs_lo:abs_hi][:, idxs]
    any_active = sub.any(axis=1)
    if not any_active.any():
        return {"pat_id": pat_id, "labels": labels, "found": False, "start_hour": s, "end_hour": e}
    first_rel = int(np.argmax(any_active))
    abs_minute = abs_lo + first_rel
    hour_index = abs_minute // 60
    minute = abs_minute % 60
    row = sub[first_rel]
    active = [labels[j] for j in range(len(labels)) if row[j]]
    try:
        ts = (icu_start_utc(tl) + pd.Timedelta(minutes=int(abs_minute))).isoformat()
    except Exception:
        ts = None
    return {
        "pat_id": pat_id, "labels": labels, "found": True,
        "hour_index": int(hour_index), "minute": int(minute),
        "active_labels_at_that_minute": active, "utc_timestamp": ts,
        "start_hour": s, "end_hour": e,
    }


FIND_FIRST_OCCURRENCE_DECL = FunctionDeclaration(
    name="find_first_occurrence",
    description=(
        "Find the earliest minute where any of the given action labels was active. "
        "Returns hour_index, minute (0-59), and UTC timestamp. Prefer this over reading "
        "hourly reports or scanning atomic actions when the user asks 'when did X first happen'. "
        "Optionally restrict to a range via start_hour/end_hour (zero-indexed from ICU start, "
        "end_hour exclusive). Defaults pat_id to the current patient."
    ),
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "pat_id": Schema(type=Type.STRING, description="Patient ID. Optional."),
            "labels": Schema(
                type=Type.ARRAY, items=Schema(type=Type.STRING),
                description=("One or more atomic-action labels; the first minute where ANY of them is active "
                             "is returned. Pick by what each label means:\n" + labels_inline()),
            ),
            "start_hour": Schema(type=Type.INTEGER, description="Inclusive start hour. Optional."),
            "end_hour": Schema(type=Type.INTEGER, description="Exclusive end hour. Optional."),
        },
        required=["labels"],
    ),
)


# ---------------------------------------------------------------------------
# find_moments_of_interest
# ---------------------------------------------------------------------------
DEFAULT_MOMENT_LABELS = ["patient_leg_ADL_movement", "patient_using_non_medical_objects"]


def _load_patient_data(pat_id: str):
    tl = load_timeline(pat_id)
    full = load_action_matrix(pat_id)
    labels = list(ACTIVITY_LABELS)
    icu_start = pd.Timestamp(tl["icu_start"])
    # Only sedation MAR events (propofol, dexmedetomidine, opioids) are
    # cross-referenced; CNS adjuncts (anticonvulsants etc.) are excluded.
    meds = []
    for e in tl.get("events", []):
        if e.get("type") == "med_sedation":
            t = pd.Timestamp(e["time"])
            t = (t.tz_convert(config.local_tz_name()) if t.tzinfo else t).tz_localize(None)
            d = e.get("data") or {}
            meds.append({
                "time_pt": t,
                "display_name": d.get("display_name", "?"),
                "mar_action": d.get("mar_action", "?"),
                "route": d.get("route"),
                "infusion_rate": d.get("infusion_rate"),
                "infusion_rate_unit": d.get("mar_inf_rate_unit"),
            })
    return tl, full, labels, icu_start, meds


def _pt_to_utc_video(pt_timestamp: pd.Timestamp, room: str) -> str:
    utc = pt_timestamp.tz_localize(config.local_tz_name()).tz_convert("UTC")
    return f"{utc.strftime('%Y_%m_%d/%H_%M')}_{room}_rt.mp4"


def _format_moment_summary(t_start, t_peak, t_end, peak_pct, nearby_meds, signal_labels):
    start_s = t_start.strftime("%Y-%m-%d %H:%M") + " local time"
    peak_s = t_peak.strftime("%H:%M")
    end_s = t_end.strftime("%H:%M")
    duration = int((t_end - t_start).total_seconds() / 60)
    sig_str = " + ".join(signal_labels) if signal_labels else "(none)"
    base = (f"[Signal: {sig_str}] "
            f"Patient wasn't moving previously, then activity started at {start_s}, "
            f"peaked at {peak_s} ({peak_pct:.0f}% of 15-min window active), "
            f"and returned to baseline at {end_s} (duration {duration} min).")
    if nearby_meds:
        med_bits = []
        for m in nearby_meds:
            rate = ""
            if m.get("infusion_rate") and m.get("infusion_rate_unit"):
                rate = f" at {m['infusion_rate']} {m['infusion_rate_unit']}"
            med_bits.append(f"{m['name']} ({m['action']}{rate}) at {m['time_pt']}")
        base += " Medications given near this window: " + "; ".join(med_bits) + "."
    else:
        base += " No medication events recorded within this window."
    return base


def moments_of_interest(patient_id: str, start_pt=None, end_pt=None, target_labels=None,
                        min_rise_pct: float = 5.0, drug_window_min: int = 45,
                        video_start_pt=None, video_end_pt=None) -> dict:
    """Detect rises and falls ("blurps") in the 15-min rolling fraction of
    minutes where any target label fired, and cross-reference each with nearby
    medication events."""
    tl, full, labels, icu_start, meds = _load_patient_data(patient_id)
    room = data.room_for(patient_id)

    if target_labels is None:
        target_labels = list(DEFAULT_MOMENT_LABELS)
    L = {name: i for i, name in enumerate(labels)}
    target_cols = [L[x] for x in target_labels if x in L]

    raw_binary = full[:, target_cols].any(axis=1)
    sig = pd.Series(raw_binary.astype(float)).rolling(15, center=True, min_periods=1).mean().values * 100
    n = len(sig)

    icu_end = icu_start + pd.Timedelta(minutes=n)
    # The explicit video window (already clamped by the caller) is preferred;
    # fall back to stays.csv. Never fall back to icu_end, which is the admit
    # time + n_minutes rather than the actual video window.
    v_start = pd.Timestamp(video_start_pt) if video_start_pt else None
    v_end = pd.Timestamp(video_end_pt) if video_end_pt else None
    if v_start is None or v_end is None:
        vs, ve = video_range_pt(patient_id)
        v_start = v_start or vs
        v_end = v_end or ve
    if v_start is None or v_end is None:
        v_start, v_end = icu_start, icu_end
    coverage_start = max(icu_start, v_start)
    coverage_end = min(icu_end, v_end + pd.Timedelta(minutes=1))

    range_start = pd.Timestamp(start_pt) if start_pt else coverage_start
    range_end = pd.Timestamp(end_pt) if end_pt else coverage_end
    range_start = max(range_start, coverage_start)
    range_end = min(range_end, coverage_end)
    i_lo = int((range_start - icu_start).total_seconds() / 60)
    i_hi = int((range_end - icu_start).total_seconds() / 60)
    i_lo = max(0, min(i_lo, n - 1))
    i_hi = max(i_lo + 1, min(i_hi, n))

    peaks, _ = find_peaks(sig, prominence=5, distance=10)
    peaks = [p for p in peaks if i_lo <= p < i_hi]

    LOOKBACK = 30
    LOOKFORWARD = 30
    EPS = 1.0

    moments = []
    seen_starts = set()
    for p in peaks:
        left = max(i_lo, p - LOOKBACK)
        segL = sig[left:p + 1]
        baselineL = segL.min()
        candL = np.where(segL <= baselineL + EPS)[0]
        if len(candL) == 0:
            continue
        v_idx = left + int(candL[-1])
        if sig[p] - sig[v_idx] < min_rise_pct or v_idx >= p:
            continue
        right = min(i_hi, p + LOOKFORWARD)
        segR = sig[p:right]
        if len(segR) == 0:
            continue
        baselineR = segR.min()
        candR = np.where(segR <= baselineR + EPS)[0]
        if len(candR) == 0:
            continue
        e_idx = p + int(candR[0])
        if v_idx in seen_starts:
            continue
        seen_starts.add(v_idx)

        # The smoothed-signal valley sits about half a window early because of
        # the centered rolling mean; anchor start/end on the raw firing minutes.
        seg_raw = raw_binary[v_idx:e_idx + 1]
        if seg_raw.any():
            fire_indices = np.where(seg_raw)[0]
            anchored_start = v_idx + int(fire_indices[0])
            anchored_end = v_idx + int(fire_indices[-1])
        else:
            anchored_start = v_idx
            anchored_end = e_idx

        t_start = icu_start + pd.Timedelta(minutes=int(anchored_start))
        t_peak = icu_start + pd.Timedelta(minutes=int(p))
        t_end = icu_start + pd.Timedelta(minutes=int(anchored_end))

        nearby = []
        for m in meds:
            mt = m["time_pt"]
            if t_start - pd.Timedelta(minutes=drug_window_min) <= mt <= t_end + pd.Timedelta(minutes=drug_window_min):
                nearby.append({
                    "time_pt": mt.strftime("%Y-%m-%d %H:%M"),
                    "name": m["display_name"],
                    "action": m["mar_action"],
                    "route": m.get("route"),
                    "infusion_rate": m.get("infusion_rate"),
                    "infusion_rate_unit": m.get("infusion_rate_unit"),
                })

        summary = _format_moment_summary(t_start, t_peak, t_end, sig[p], nearby, target_labels)
        moments.append({
            "start_pt": t_start.strftime("%Y-%m-%d %H:%M"),
            "peak_pt": t_peak.strftime("%Y-%m-%d %H:%M"),
            "end_pt": t_end.strftime("%Y-%m-%d %H:%M"),
            "peak_pct": round(float(sig[p]), 1),
            "duration_min": int(anchored_end - anchored_start),
            "signal_labels": list(target_labels),
            "start_video": _pt_to_utc_video(t_start, room),
            "peak_video": _pt_to_utc_video(t_peak, room),
            "end_video": _pt_to_utc_video(t_end, room),
            "nearby_medications": nearby,
            "summary": summary,
        })

    return {
        "patient_id": patient_id,
        "signal_labels": target_labels,
        "query_range_pt": {"start": range_start.strftime("%Y-%m-%d %H:%M"),
                           "end": range_end.strftime("%Y-%m-%d %H:%M")},
        "window_minutes": 15,
        "total_moments": len(moments),
        "moments": moments,
    }


def find_moments_of_interest(pat_id: str, start_pt: str | None = None, end_pt: str | None = None,
                             target_labels: list[str] | None = None, min_rise_pct: float = 5.0,
                             drug_window_min: int = 45) -> dict:
    """Tool wrapper: validates labels and passes the clamped video window."""
    if target_labels:
        bad = [l for l in target_labels if l not in ACTIVITY_LABELS]
        if bad:
            return {"error": f"Unknown labels: {bad}. Known: {ACTIVITY_LABELS}"}
    v_start, v_end = video_range_pt(pat_id)
    return moments_of_interest(
        patient_id=pat_id, start_pt=start_pt, end_pt=end_pt, target_labels=target_labels,
        min_rise_pct=float(min_rise_pct), drug_window_min=int(drug_window_min),
        video_start_pt=v_start.strftime("%Y-%m-%d %H:%M") if v_start is not None else None,
        video_end_pt=v_end.strftime("%Y-%m-%d %H:%M") if v_end is not None else None,
    )


FIND_MOMENTS_OF_INTEREST_DECL = FunctionDeclaration(
    name="find_moments_of_interest",
    description=(
        "Detect 'moments of interest' -- rises and falls in a 15-minute rolling signal built "
        "from the atomic-action labels you select -- and cross-reference each detected blurp "
        "with medication events given nearby. The signal at minute t is the fraction of the "
        "previous 15 minutes in which any selected label fired. **You must choose target_labels "
        "to match the question.** Examples:\n"
        "  - 'when was the patient most active' -> broad set (head_turning, arm_movements, "
        "    position_change, leg_in_bed_movement, gesticulating, etc.)\n"
        "  - 'movement blurps' (default if you can't tell) -> leg_ADL_movement + "
        "    using_non_medical_objects\n"
        "  - 'when was the clinician interacting most' -> clinician_patient_contact + "
        "    clinician_patient_object_exchange\n"
        "  - 'when was the patient pulling at lines' -> grabbing_medical_equipment\n"
        "Returns a list of moments, each with start/peak/end timestamps (local time, "
        "'YYYY-MM-DD HH:MM'), peak intensity (% of the 15-min window active), duration, the "
        "atomic-action labels that defined the signal (signal_labels), the corresponding "
        "video filenames, and any medications administered within +/-drug_window_min "
        "(default 45) of the blurp's start/end. Optional start_pt/end_pt restrict to a "
        "local-time range ('YYYY-MM-DD HH:MM'); omit for the full ICU stay. Defaults pat_id "
        "to the current patient. **Always tell the user which atomic-action labels were "
        "used as the signal (the signal_labels field) so they know what behaviour the "
        "blurps are derived from.**\n\n"
        "Available atomic-action labels (pick from these for target_labels):\n"
        + labels_inline()
    ),
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "pat_id": Schema(type=Type.STRING, description="Patient ID. Optional."),
            "start_pt": Schema(type=Type.STRING, description="Inclusive start time, local time 'YYYY-MM-DD HH:MM'. Optional."),
            "end_pt": Schema(type=Type.STRING, description="Inclusive end time, local time 'YYYY-MM-DD HH:MM'. Optional."),
            "target_labels": Schema(
                type=Type.ARRAY, items=Schema(type=Type.STRING),
                description=("Atomic-action labels that define the signal -- a blurp = a rise+fall in "
                             "the 15-min rolling fraction of minutes where ANY of these labels fired. "
                             "Pick the labels that match the user's question. Omit to use the default "
                             "leg_ADL_movement + non_medical_objects. Available labels:\n" + labels_inline()),
            ),
            "min_rise_pct": Schema(type=Type.NUMBER,
                                   description="Minimum rise (peak - valley, in percentage points) to count as a blurp. Default 5."),
            "drug_window_min": Schema(type=Type.INTEGER,
                                      description=("+/- window in minutes around each blurp's start and end to look for "
                                                   "medication events (i.e. meds given between (start - drug_window_min) "
                                                   "and (end + drug_window_min) are reported). Default 45.")),
        },
    ),
)

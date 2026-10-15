"""query_medications: pandas-backed query over the medication administration
record (MAR) export so the agent can pull administration events directly.

Data:  <data root>/ehr/medications.csv with columns
       pat_id, display_name, taken_time, mar_action, sig, dose_unit, route,
       infusion_rate, mar_inf_rate_unit, pharm_class_name, thera_class_name
Times: `taken_time` is treated as local-time-naive (matches how the hourly
       reports and the video tool address time). No timezone conversion.

Handler signature matches the registry contract:
    query_medications(args, pat_id, model) -> dict
"""
from __future__ import annotations

import threading

import pandas as pd
from google.genai import types

from .. import data
from .base import Tool

_COLUMNS = ["pat_id", "display_name", "taken_time", "mar_action", "sig", "dose_unit", "route",
            "infusion_rate", "mar_inf_rate_unit", "pharm_class_name", "thera_class_name"]

_CACHE_LOCK = threading.RLock()
_MAR: dict[str, pd.DataFrame] = {}          # path -> full frame
_PER_PATIENT: dict[tuple[str, str], pd.DataFrame] = {}


def _load_mar() -> pd.DataFrame:
    path = data.medications_file()
    key = str(path)
    with _CACHE_LOCK:
        if key not in _MAR:
            df = pd.read_csv(path, low_memory=False)
            missing = [c for c in _COLUMNS if c not in df.columns]
            for c in missing:
                df[c] = pd.NA
            df = df[_COLUMNS]
            df["taken_time"] = pd.to_datetime(df["taken_time"], errors="coerce")
            df = df.dropna(subset=["taken_time"])
            _MAR[key] = df
        return _MAR[key]


def _for_patient(pat_id: str) -> pd.DataFrame:
    key = (str(data.medications_file()), pat_id)
    with _CACHE_LOCK:
        if key not in _PER_PATIENT:
            df = _load_mar()
            _PER_PATIENT[key] = (df[df["pat_id"].astype(str) == str(pat_id)]
                                 .sort_values("taken_time").reset_index(drop=True))
        return _PER_PATIENT[key]


def _s(v) -> str:
    """NaN-safe string: pandas NaN is truthy, so `v or ""` would leak NaN into
    the tool response, which is not valid JSON for a FunctionResponse."""
    return "" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)


def _format_dose(row) -> str | None:
    rate = row.get("infusion_rate")
    if pd.notna(rate) and float(rate) != 0.0:
        return f"{rate} {_s(row.get('mar_inf_rate_unit'))}".strip()
    sig = row.get("sig")
    if pd.notna(sig) and float(sig) != 0.0:
        return f"{sig} {_s(row.get('dose_unit'))}".strip()
    return None


def query_medications(args: dict, pat_id: str | None, model: str) -> dict:
    pid = args.get("pat_id") or pat_id
    if not pid:
        return {"error": "pat_id required (agent has no current patient)."}
    try:
        start = pd.Timestamp(args["start_pt"])
        end = pd.Timestamp(args["end_pt"])
    except Exception as e:
        return {"error": f"start_pt/end_pt must be 'YYYY-MM-DD HH:MM' local-time strings ({e})."}
    if end < start:
        return {"error": "end_pt is before start_pt."}

    df = _for_patient(pid)
    if df.empty:
        return {"pat_id": pid, "start_pt": args["start_pt"], "end_pt": args["end_pt"],
                "n_events_total": 0, "n_events_returned": 0, "events": [],
                "note": "no MAR rows for this patient."}

    m = (df["taken_time"] >= start) & (df["taken_time"] <= end)
    name_filter = (args.get("name_filter") or "").strip()
    if name_filter:
        m &= df["display_name"].fillna("").astype(str).str.contains(name_filter, case=False, na=False, regex=False)
    class_filter = (args.get("class_filter") or "").strip()
    if class_filter:
        m &= (df["pharm_class_name"].fillna("").astype(str).str.contains(class_filter, case=False, na=False, regex=False)
              | df["thera_class_name"].fillna("").astype(str).str.contains(class_filter, case=False, na=False, regex=False))

    hits = df[m]
    total = len(hits)
    max_rows = int(args.get("max_rows") or 50)
    hits = hits.head(max_rows)

    events = []
    for _, row in hits.iterrows():
        events.append({
            "taken_time_pt": row["taken_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "med_name": _s(row.get("display_name")),
            "mar_action": _s(row.get("mar_action")),
            "dose": _format_dose(row),
            "route": _s(row.get("route")),
            "pharm_class": _s(row.get("pharm_class_name")),
            "thera_class": _s(row.get("thera_class_name")),
        })
    out = {"pat_id": pid, "start_pt": args["start_pt"], "end_pt": args["end_pt"],
           "n_events_total": total, "n_events_returned": len(events), "events": events}
    if total > len(events):
        out["note"] = (f"Only the first {len(events)} of {total} events are shown; "
                       "narrow the time window or use name_filter/class_filter to see more.")
    return out


MED_DECL = types.FunctionDeclaration(
    name="query_medications",
    description=(
        "Return medication-administration-record (MAR) events for the current patient in a "
        "local-time window. Each event has taken_time_pt, med_name (display name), "
        "mar_action (Given / New Bag / Bolus / Stopped / etc.), dose (rate + unit or bolus + "
        "unit), route, and drug class (pharmacologic + therapeutic). Use this to answer "
        "questions like 'which meds were administered +/-45 min of X?', 'when was the last "
        "dose of oxycodone?', or 'what sedatives were running at 03:00?'. "
        "TIME: pass local-time strings 'YYYY-MM-DD HH:MM' (24-hour). Windows are inclusive on both ends. "
        "OPTIONAL FILTERS: name_filter (case-insensitive substring on the med name) and "
        "class_filter (substring on pharmacologic/therapeutic class, e.g. 'OPIOID', "
        "'SEDATI', 'ANTIBIOTIC'). Results are capped at max_rows (default 50)."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "pat_id": types.Schema(type="STRING", description="Patient ID. Optional (defaults to current)."),
            "start_pt": types.Schema(type="STRING", description="Window start, 'YYYY-MM-DD HH:MM' local time."),
            "end_pt": types.Schema(type="STRING", description="Window end, 'YYYY-MM-DD HH:MM' local time."),
            "name_filter": types.Schema(type="STRING", description="Optional case-insensitive substring on display_name."),
            "class_filter": types.Schema(type="STRING", description="Optional substring on pharmacologic/therapeutic class."),
            "max_rows": types.Schema(type="INTEGER", description="Cap on returned events (default 50)."),
        },
        required=["start_pt", "end_pt"],
    ),
)

MED_TOOL = Tool(name="query_medications", declaration=MED_DECL, handler=query_medications, group="ehr")

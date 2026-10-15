"""query_flowsheet: pandas-backed query over the nursing flowsheet export
(vitals, RASS, nursing assessments) for a stay.

Data:  <data root>/ehr/flowsheet.csv with columns
       pat_id, row_disp_name, recorded_time, meas_value, units
Times: `recorded_time` is treated as local-time-naive to match the med tool
       and the hourly reports.

Discovery mode: with no disp_name_filter, returns the list of available
measurement types (disp_name + count) for the stay so the model can find out
what is available. With a filter, returns the actual measurements.
"""
from __future__ import annotations

import threading

import pandas as pd
from google.genai import types

from .. import data
from .base import Tool

_COLUMNS = ["pat_id", "row_disp_name", "recorded_time", "meas_value", "units"]

_CACHE_LOCK = threading.RLock()
_DF: dict[str, pd.DataFrame] = {}
_PER_PATIENT: dict[tuple[str, str], pd.DataFrame] = {}


def _load() -> pd.DataFrame:
    path = data.flowsheet_file()
    key = str(path)
    with _CACHE_LOCK:
        if key not in _DF:
            if path.suffix in (".pkl", ".pickle"):
                df = pd.read_pickle(path)
            elif path.suffix == ".parquet":
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path, low_memory=False)
            if "PAT_ID" in df.columns and "pat_id" not in df.columns:
                df = df.rename(columns={"PAT_ID": "pat_id"})
            for c in _COLUMNS:
                if c not in df.columns:
                    df[c] = pd.NA
            df = df[_COLUMNS]
            df["recorded_time"] = pd.to_datetime(df["recorded_time"], errors="coerce")
            df = df.dropna(subset=["recorded_time"])
            _DF[key] = df
        return _DF[key]


def _for_patient(pat_id: str) -> pd.DataFrame:
    key = (str(data.flowsheet_file()), pat_id)
    with _CACHE_LOCK:
        if key not in _PER_PATIENT:
            df = _load()
            _PER_PATIENT[key] = (df[df["pat_id"].astype(str) == str(pat_id)]
                                 .sort_values("recorded_time").reset_index(drop=True))
        return _PER_PATIENT[key]


def query_flowsheet(args: dict, pat_id: str | None, model: str) -> dict:
    pid = args.get("pat_id") or pat_id
    if not pid:
        return {"error": "pat_id required (agent has no current patient)."}

    disp_filter = (args.get("disp_name_filter") or "").strip()
    df = _for_patient(pid)
    if df.empty:
        return {"pat_id": pid, "n_measurement_types": 0, "n_rows": 0,
                "note": "no flowsheet rows for this patient."}

    if not disp_filter:
        top = df["row_disp_name"].value_counts().head(80)
        return {"pat_id": pid,
                "n_measurement_types": int(df["row_disp_name"].nunique()),
                "n_rows": int(len(df)),
                "note": ("Pass disp_name_filter (case-insensitive substring, e.g. 'RASS', "
                         "'Heart Rate', 'SpO2', 'BP', 'Pain', 'Activity') to fetch "
                         "specific measurements. Top 80 measurement types shown; total unique count above."),
                "available_disp_names": [{"disp_name": k, "count": int(v)} for k, v in top.items()]}

    m = df["row_disp_name"].fillna("").astype(str).str.contains(disp_filter, case=False, na=False, regex=False)
    start_s = (args.get("start_pt") or "").strip()
    end_s = (args.get("end_pt") or "").strip()
    if start_s:
        try:
            m &= df["recorded_time"] >= pd.Timestamp(start_s)
        except Exception as e:
            return {"error": f"bad start_pt (need 'YYYY-MM-DD HH:MM' local time): {e}"}
    if end_s:
        try:
            m &= df["recorded_time"] <= pd.Timestamp(end_s)
        except Exception as e:
            return {"error": f"bad end_pt (need 'YYYY-MM-DD HH:MM' local time): {e}"}

    hits = df[m]
    total = len(hits)
    max_rows = int(args.get("max_rows") or 60)
    hits = hits.head(max_rows)

    rows = []
    for _, r in hits.iterrows():
        val = r.get("meas_value")
        if pd.isna(val):
            val = None
        elif not isinstance(val, str):
            val = str(val)
        unit = r.get("units")
        if pd.isna(unit) or not isinstance(unit, str) or not unit.strip():
            unit = None
        rows.append({
            "recorded_time_pt": r["recorded_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "disp_name": r.get("row_disp_name") or "",
            "value": val,
            "unit": unit,
        })
    out = {"pat_id": pid, "disp_name_filter": disp_filter,
           "n_rows_total": total, "n_rows_returned": len(rows), "rows": rows}
    if start_s:
        out["start_pt"] = start_s
    if end_s:
        out["end_pt"] = end_s
    if total > len(rows):
        out["note"] = f"Only first {len(rows)} of {total} rows shown; narrow window or use a more specific disp_name_filter."
    return out


FLOWSHEET_DECL = types.FunctionDeclaration(
    name="query_flowsheet",
    description=(
        "Query nursing-flowsheet measurements for the current patient: vitals "
        "(Heart Rate, Pulse, SpO2, Resp, Arterial Systolic BP, Mean Arterial "
        "Pressure, Arterial Diastolic BP, BP), sedation/agitation (Patient RASS "
        "Score, Anxiety Level, Pain Scale Type), and nursing assessments (Activity, "
        "ADLs, Communication, Ectopy, Cardiac Rhythm, etc.). Each row has "
        "recorded_time_pt, disp_name, value, and unit. "
        "DISCOVERY: call without disp_name_filter to see which measurement types "
        "exist for this patient (list of disp_name + count). "
        "QUERY: pass disp_name_filter (case-insensitive substring, e.g. 'RASS', "
        "'Heart Rate', 'SpO2') to fetch the actual measurements. Optionally scope "
        "with start_pt / end_pt (local-time 'YYYY-MM-DD HH:MM'). "
        "Use this for questions like 'when was the patient most deeply sedated?' "
        "(RASS), 'what was the peak heart rate?', 'when did BP swings occur?', "
        "'what was the anxiety/pain trend?'."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "pat_id": types.Schema(type="STRING", description="Patient ID (defaults to current)."),
            "disp_name_filter": types.Schema(type="STRING",
                                             description="Case-insensitive substring on measurement name. Omit for discovery mode."),
            "start_pt": types.Schema(type="STRING", description="Window start 'YYYY-MM-DD HH:MM' local time (optional)."),
            "end_pt": types.Schema(type="STRING", description="Window end 'YYYY-MM-DD HH:MM' local time (optional)."),
            "max_rows": types.Schema(type="INTEGER", description="Cap on rows returned (default 60)."),
        },
        required=[],
    ),
)

FLOWSHEET_TOOL = Tool(name="query_flowsheet", declaration=FLOWSHEET_DECL, handler=query_flowsheet, group="ehr")

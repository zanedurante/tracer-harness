"""Location-tolerant resolution of per-stay data files.

The tools, the baselines, and the cohort scripts all need to find, for a stay
identifier ``pat_id``:

  * ``<timelines>/<pat_id>_timeline.json``  stay metadata + EHR event list
  * ``<timelines>/<pat_id>.npz``            bit-packed minute-level action matrix
  * ``<timelines>/activity_key.csv``        the label order (idx,label)
  * ``<reports>/<pat_id>/hour_<date>_<HH>.md``  hourly reports
  * ``<video>/<YYYY_MM_DD>/<HH_MM>_<room>_<lt|rt>.mp4``  one-minute clips (UTC folders)
  * ``<ehr>/medications.csv``, ``<ehr>/flowsheet.csv``   EHR export tables

Every location is derived from ``TRACER_DATA_ROOT`` (tracer.config), with
optional per-stay overrides (``video_dir``, ``timeline_dir`` columns) in
``<data root>/stays.csv``. Resolution is per file, so a stay's files may live
in different places. See docs/DATA_LAYOUT.md.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import config

STAYS_COLUMNS = ["pat_id", "start_time_utc", "end_time_utc", "room"]


def stays_file() -> Path:
    return config.data_root() / "stays.csv"


def _coerce_override(value) -> str | None:
    """Stripped path string if `value` is a real path-like, else None.
    Treats NaN / '' / 'nan' as 'no override'."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    s = str(value).strip()
    if not s or s.lower() == "nan":
        return None
    return s


_STAYS_CACHE: dict[str, pd.DataFrame] = {}


def stays_df() -> pd.DataFrame:
    """The stays table (cached per path). Empty frame when the file is absent."""
    p = stays_file()
    key = str(p)
    if key not in _STAYS_CACHE:
        _STAYS_CACHE[key] = pd.read_csv(p) if p.exists() else pd.DataFrame()
    return _STAYS_CACHE[key]


def reset_caches() -> None:
    _STAYS_CACHE.clear()


def patient_row(pat_id: str):
    df = stays_df()
    if df.empty or "pat_id" not in df.columns:
        return None
    hits = df[df["pat_id"].astype(str) == str(pat_id)]
    return hits.iloc[0] if len(hits) else None


def _override_base(pat_id: str, column: str) -> Path | None:
    row = patient_row(pat_id)
    if row is None or column not in row.index:
        return None
    ov = _coerce_override(row.get(column))
    return Path(ov).expanduser() if ov else None


# ---------------------------------------------------------------------------
# Timelines (minute-level records)
# ---------------------------------------------------------------------------
def standard_timeline_bases() -> list[Path]:
    return [config.data_root() / "timelines"]


def candidate_bases(pat_id: str | None = None) -> list[Path]:
    """Ordered, de-duplicated list of base dirs to search for a stay's data:
    the stay's own timeline_dir override first, then the standard location."""
    bases: list[Path] = []
    if pat_id:
        ov = _override_base(pat_id, "timeline_dir")
        if ov is not None:
            bases.append(ov)
    bases.extend(standard_timeline_bases())
    seen: set[str] = set()
    out: list[Path] = []
    for b in bases:
        if str(b) not in seen:
            seen.add(str(b))
            out.append(b)
    return out


def _resolve(pat_id: str | None, filename: str) -> Path:
    """First candidate base that actually contains `filename`, else the
    highest-priority candidate (so error messages name a plausible path)."""
    bases = candidate_bases(pat_id)
    for base in bases:
        p = base / filename
        if p.exists():
            return p
    return bases[0] / filename


def timeline_json(pat_id: str) -> Path:
    return _resolve(pat_id, f"{pat_id}_timeline.json")


def npz(pat_id: str) -> Path:
    return _resolve(pat_id, f"{pat_id}.npz")


def activity_key_file() -> Path:
    """The label-order file. The data store's copy wins when present (it must
    match the packaged taxonomy); otherwise the packaged taxonomy is used."""
    search: list[Path] = list(standard_timeline_bases())
    df = stays_df()
    if not df.empty and "timeline_dir" in df.columns:
        for v in df["timeline_dir"]:
            ov = _coerce_override(v)
            if ov:
                search.append(Path(ov).expanduser())
    for base in search:
        p = base / "activity_key.csv"
        if p.exists():
            return p
    return config.TAXONOMY_FILE


def available_patients() -> list[str]:
    """Stay ids that have BOTH a timeline JSON and an npz somewhere on disk."""
    ids: set[str] = set()
    bases = list(standard_timeline_bases())
    df = stays_df()
    if not df.empty and "timeline_dir" in df.columns:
        for v in df["timeline_dir"]:
            ov = _coerce_override(v)
            if ov:
                bases.append(Path(ov).expanduser())
    seen_base: set[str] = set()
    for base in bases:
        if str(base) in seen_base or not base.exists():
            continue
        seen_base.add(str(base))
        for p in base.glob("*_timeline.json"):
            pid = p.name.removesuffix("_timeline.json")
            if npz(pid).exists():
                ids.add(pid)
    return sorted(ids)


# ---------------------------------------------------------------------------
# Stay metadata
# ---------------------------------------------------------------------------
def room_for(pat_id: str) -> str:
    """Room/camera identifier string for a stay from stays.csv, or ''."""
    row = patient_row(pat_id)
    if row is None or "room" not in row.index:
        return ""
    v = row.get("room")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def video_window_utc(pat_id: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(start, end) of the recorded video window as tz-aware UTC Timestamps."""
    row = patient_row(pat_id)
    if row is None:
        return None
    try:
        s = pd.Timestamp(row["start_time_utc"])
        e = pd.Timestamp(row["end_time_utc"])
    except Exception:
        return None
    if pd.isna(s) or pd.isna(e):
        return None
    s = s.tz_convert("UTC") if s.tzinfo else s.tz_localize("UTC")
    e = e.tz_convert("UTC") if e.tzinfo else e.tz_localize("UTC")
    return s, e


def max_display_hours(pat_id: str) -> float | None:
    """Optional per-stay clamp (hours after the video start) on the window the
    tools expose; absent or blank means no clamp."""
    row = patient_row(pat_id)
    if row is None or "max_display_hours" not in row.index:
        return None
    v = row.get("max_display_hours")
    try:
        if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Other stores
# ---------------------------------------------------------------------------
def reports_dir() -> Path:
    return config.data_root() / "reports"


def video_base(pat_id: str) -> Path:
    """Root of the clip tree for a stay (per-stay video_dir override, else
    <data root>/video)."""
    ov = _override_base(pat_id, "video_dir")
    return ov if ov is not None else config.data_root() / "video"


def medications_file() -> Path:
    return config.data_root() / "ehr" / "medications.csv"


def flowsheet_file() -> Path:
    return config.data_root() / "ehr" / "flowsheet.csv"


def perception_dir() -> Path:
    return config.data_root() / "perception"

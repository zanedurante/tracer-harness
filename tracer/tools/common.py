"""Shared loaders for the timeline, report and scaffold tools: the stay's
timeline JSON, the bit-packed action matrix, and the video-coverage window.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, data
from .labels import ACTIVITY_LABELS


def timeline_json_path(pat_id: str) -> Path:
    return data.timeline_json(pat_id)


def npz_path(pat_id: str) -> Path:
    return data.npz(pat_id)


def reports_path(pat_id: str) -> Path:
    return data.reports_dir() / pat_id


def load_timeline(pat_id: str) -> dict:
    path = timeline_json_path(pat_id)
    if not path.exists():
        raise FileNotFoundError(f"No timeline JSON for stay {pat_id}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_action_matrix(pat_id: str) -> np.ndarray:
    """Return the full (n_minutes, n_labels) binary action matrix for the stay.

    Unpacks the bit-packed npz: each hour is stored as (60, 2) uint8, unpacked
    big-endian to (60, 16), then sliced to the number of labels.
    """
    path = npz_path(pat_id)
    if not path.exists():
        raise FileNotFoundError(f"No action npz for stay {pat_id}")
    n_labels = len(ACTIVITY_LABELS)
    if n_labels == 0:
        raise RuntimeError("Activity labels not loaded; cannot decode actions")
    data_ = np.load(path)
    hour_keys = sorted(k for k in data_.files if k.startswith("hour_"))
    matrices = [np.unpackbits(data_[k], axis=1)[:, :n_labels] for k in hour_keys]
    return np.concatenate(matrices, axis=0) if matrices else np.zeros((0, n_labels), dtype=np.uint8)


_VIDEO_COVERAGE_CACHE: dict[str, dict] = {}


def video_coverage() -> dict[str, dict]:
    """Per-stay video coverage from stays.csv: {pat_id: {video_start_pt,
    video_end_pt, room}} with local-time-naive Timestamps. When the stay has a
    max_display_hours value, the end is clamped to start + that many hours so
    the tools never surface a timestamp outside the window an application
    would display."""
    key = str(data.stays_file())
    if key in _VIDEO_COVERAGE_CACHE:
        return _VIDEO_COVERAGE_CACHE[key]
    out: dict[str, dict] = {}
    df = data.stays_df()
    tz = config.local_tz_name()
    if not df.empty:
        for _, row in df.iterrows():
            pid = str(row["pat_id"])
            try:
                s_utc = pd.Timestamp(row["start_time_utc"])
                e_utc = pd.Timestamp(row["end_time_utc"])
                s_utc = s_utc if s_utc.tzinfo else s_utc.tz_localize("UTC")
                e_utc = e_utc if e_utc.tzinfo else e_utc.tz_localize("UTC")
                s_pt = s_utc.tz_convert(tz).tz_localize(None)
                e_pt = e_utc.tz_convert(tz).tz_localize(None)
            except Exception:
                continue
            cap_h = data.max_display_hours(pid)
            if cap_h is not None:
                cap = s_pt + pd.Timedelta(hours=cap_h)
                if e_pt > cap:
                    e_pt = cap
            out[pid] = {"video_start_pt": s_pt, "video_end_pt": e_pt,
                        "room": str(row.get("room", "") or "")}
    _VIDEO_COVERAGE_CACHE[key] = out
    return out


def reset_caches() -> None:
    _VIDEO_COVERAGE_CACHE.clear()
    data.reset_caches()


def video_range_pt(pat_id: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    info = video_coverage().get(pat_id)
    if not info:
        return None, None
    return info["video_start_pt"], info["video_end_pt"]


def video_minute_window(pat_id: str, tl: dict, n_minutes: int) -> tuple[int, int]:
    """Map the stay's video coverage into absolute-minute indices of the action
    matrix. Returns (lo, hi), inclusive/exclusive. Falls back to (0, n_minutes)
    if coverage is unknown."""
    v_start, v_end = video_range_pt(pat_id)
    if v_start is None or v_end is None:
        return 0, n_minutes
    icu_start = pd.Timestamp(tl["icu_start"])
    lo = max(0, int((v_start - icu_start).total_seconds() / 60))
    hi = min(n_minutes, int((v_end - icu_start).total_seconds() / 60) + 1)
    if hi <= lo:
        return 0, n_minutes
    return lo, hi


def icu_start_utc(tl: dict) -> pd.Timestamp:
    """Convert the local-time-naive icu_start in the timeline JSON to UTC."""
    return (
        pd.Timestamp(tl["icu_start"])
        .tz_localize(config.local_tz_name(), ambiguous=True, nonexistent="shift_forward")
        .tz_convert("UTC")
        .floor("min")
    )

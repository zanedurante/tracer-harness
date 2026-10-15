"""Report tools: list_hourly_reports and get_hourly_report over the hourly
markdown reports at <reports>/<pat_id>/hour_<YYYY-MM-DD>_<HH>.md.
"""
from __future__ import annotations

import re

import pandas as pd
from google.genai.types import FunctionDeclaration, Schema, Type

from .common import reports_path, video_range_pt

REPORT_FILENAME_RE = re.compile(r"^hour_(?P<date>\d{4}-\d{2}-\d{2})_(?P<hour>\d{2})\.md$")


def list_hourly_reports(pat_id: str) -> dict:
    """Sorted list of available hourly reports, filtered to those that overlap
    the stay's video coverage window."""
    d = reports_path(pat_id)
    if not d.exists():
        return {"pat_id": pat_id, "reports": []}
    v_start, v_end = video_range_pt(pat_id)
    entries = []
    for f in sorted(d.iterdir()):
        m = REPORT_FILENAME_RE.match(f.name)
        if not m:
            continue
        date, hour = m.group("date"), int(m.group("hour"))
        if v_start is not None and v_end is not None:
            try:
                hour_start = pd.Timestamp(f"{date} {hour:02d}:00:00")
                hour_end = hour_start + pd.Timedelta(hours=1)
                if hour_end <= v_start or hour_start >= v_end:
                    continue
            except Exception:
                pass
        entries.append({"date": date, "hour": hour})
    return {"pat_id": pat_id, "reports": entries}


LIST_HOURLY_REPORTS_DECL = FunctionDeclaration(
    name="list_hourly_reports",
    description=(
        "List all hourly markdown reports available for a patient. Each entry has a date "
        "(YYYY-MM-DD) and an hour (0-23). Defaults to the current patient if pat_id is omitted."
    ),
    parameters=Schema(
        type=Type.OBJECT,
        properties={"pat_id": Schema(type=Type.STRING, description="Patient ID. Optional.")},
    ),
)


def get_hourly_report(pat_id: str, date: str, hour: int) -> dict:
    """Raw markdown of one hourly report, with a header note when the hour
    overlaps the video-coverage boundary."""
    path = reports_path(pat_id) / f"hour_{date}_{int(hour):02d}.md"
    if not path.exists():
        return {"error": f"Report not found: {path.name}"}
    md = path.read_text(encoding="utf-8")

    try:
        hour_start = pd.Timestamp(f"{date} {int(hour):02d}:00:00")
        hour_end = hour_start + pd.Timedelta(hours=1)
        v_start, v_end = video_range_pt(pat_id)
        warning = None
        if v_start is not None and v_end is not None:
            if hour_end <= v_start or hour_start >= v_end:
                warning = (
                    "WARNING: this entire hour is OUTSIDE the patient's video coverage "
                    f"window ({v_start.strftime('%Y-%m-%d %H:%M')} -> {v_end.strftime('%Y-%m-%d %H:%M')} local time). "
                    "Do NOT cite any timestamps from this report -- none are clickable."
                )
            elif hour_start < v_start:
                missing = int((v_start - hour_start).total_seconds() / 60)
                warning = (
                    f"NOTE: the first {missing} minutes of this hour (before "
                    f"{v_start.strftime('%H:%M')} local time) are pre-video. Any minute the report "
                    f"mentions in min 0-{missing - 1} has no clickable video."
                )
            elif hour_end > v_end:
                covered = int((v_end - hour_start).total_seconds() / 60)
                warning = (
                    f"NOTE: only the first {covered} minutes of this hour (up to "
                    f"{v_end.strftime('%H:%M')} local time) have video coverage. Any minute the "
                    f"report mentions in min {covered}-59 has no clickable video."
                )
        if warning:
            md = f"> {warning}\n\n" + md
    except Exception:
        pass

    return {"pat_id": pat_id, "date": date, "hour": int(hour), "markdown": md}


GET_HOURLY_REPORT_DECL = FunctionDeclaration(
    name="get_hourly_report",
    description=(
        "Fetch the full markdown content of a single hourly report. Reports follow a fixed "
        "6-section structure: Patient Movement Summary, Patient-Clinician Interaction, "
        "Significant Moments, Medication Context, Activity-Medication Alignment, Brief Summary. "
        "Defaults pat_id to the current patient."
    ),
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "pat_id": Schema(type=Type.STRING, description="Patient ID. Optional."),
            "date": Schema(type=Type.STRING, description="Report date, YYYY-MM-DD."),
            "hour": Schema(type=Type.INTEGER, description="Hour of day, 0-23."),
        },
        required=["date", "hour"],
    ),
)

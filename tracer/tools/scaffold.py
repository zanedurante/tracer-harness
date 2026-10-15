"""Scaffold tools (in every tool configuration): get_patient_metadata and
convert_utc_local. list_patients is an application convenience that is
registered but not part of any reported configuration.
"""
from __future__ import annotations

import datetime
from zoneinfo import ZoneInfo

from google.genai.types import FunctionDeclaration, Schema, Type

from .. import config, data
from .common import load_timeline, video_coverage, video_range_pt


def convert_utc_local(dt_str, to_timezone="Local"):
    """Convert between UTC and the site's local time (TRACER_LOCAL_TZ).

    Args:
        dt_str (str): Datetime string "YYYY-MM-DD HH:MM:SS" in local or UTC time
        to_timezone (str): "Local" or "UTC"

    Returns:
        str: Converted datetime string with timezone abbreviation
    """
    local = config.local_tz()
    if to_timezone == "Local":
        dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=ZoneInfo("UTC"))
        return dt.astimezone(local).strftime("%Y-%m-%d %H:%M:%S %Z")
    elif to_timezone == "UTC":
        dt = datetime.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        dt = dt.replace(tzinfo=local)
        return dt.astimezone(ZoneInfo("UTC")).strftime("%Y-%m-%d %H:%M:%S %Z")
    else:
        raise ValueError("to_timezone must be 'Local' or 'UTC'")


CONVERT_UTC_LOCAL_DECL = FunctionDeclaration(
    name="convert_utc_local",
    description="Convert between UTC and local time.",
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "dt_str": Schema(type=Type.STRING,
                             description="Datetime string in format 'YYYY-MM-DD HH:MM:SS' in local or UTC time."),
            "to_timezone": Schema(type=Type.STRING,
                                  enum=["Local", "UTC"],
                                  description="Timezone to convert to: Local or UTC"),
        },
        required=["dt_str", "to_timezone"],
    ),
)


def list_patients() -> dict:
    """Return stays available (those with both timelines and action data)."""
    return {"patients": data.available_patients()}


LIST_PATIENTS_DECL = FunctionDeclaration(
    name="list_patients",
    description="List all patient IDs for which timeline metadata and atomic action data are available on disk.",
    parameters=Schema(type=Type.OBJECT, properties={}),
)


def get_patient_metadata(pat_id: str) -> dict:
    """Return demographics, ICU admission time, hour/minute counts, encounters,
    and video-coverage window for a stay."""
    tl = load_timeline(pat_id)
    v_start, v_end = video_range_pt(pat_id)
    info = video_coverage().get(pat_id, {})
    return {
        "pat_id": tl.get("pat_id"),
        "icu_start_admit_DO_NOT_CITE": tl.get("icu_start"),
        "n_hours": tl.get("n_hours"),
        "n_minutes": tl.get("n_minutes"),
        "demographics": tl.get("demographics"),
        "encounters": tl.get("encounters"),
        "video_start_pt": v_start.strftime("%Y-%m-%d %H:%M") if v_start is not None else None,
        "video_end_pt": v_end.strftime("%Y-%m-%d %H:%M") if v_end is not None else None,
        "video_room": info.get("room") or None,
        "note": (
            "icu_start_admit_DO_NOT_CITE is the recorded admit time and may pre-date video "
            "recording -- never display it as a timestamp. Only [video_start_pt, video_end_pt] "
            "have playable videos. Clamp every timestamp you cite to that window."
        ),
    }


GET_PATIENT_METADATA_DECL = FunctionDeclaration(
    name="get_patient_metadata",
    description=(
        "Get demographics, ICU admission time, observation window length, and encounter "
        "history for a patient. Defaults to the patient currently on screen if pat_id is omitted."
    ),
    parameters=Schema(
        type=Type.OBJECT,
        properties={
            "pat_id": Schema(type=Type.STRING, description="Patient ID. Optional -- defaults to current patient."),
        },
    ),
)

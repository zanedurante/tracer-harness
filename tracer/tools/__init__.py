"""The agent's tools.

Eleven tools, grouped by family (paper appendix, "Tool Interface"):

  scaffold : get_patient_metadata, convert_utc_local
  timeline : get_atomic_actions, query_actions, find_first_occurrence,
             find_moments_of_interest
  report   : list_hourly_reports, get_hourly_report
  ehr      : query_medications, query_flowsheet
  video    : query_video_minute

plus ``list_patients``, an application convenience that is registered but
not part of any reported configuration.

Each tool is a plain Python function returning a JSON-serializable dict, with
a google.genai FunctionDeclaration (converted for OpenAI-compatible backends
by tracer.backends.openai_backend). Tools that accept ``pat_id`` treat it as
optional: ``dispatch`` injects the current stay when the model omits it.
"""
from __future__ import annotations

from typing import Any

from google.genai.types import FunctionDeclaration

from .base import Tool
from .labels import ACTIVITY_LABELS, LABEL_DESCRIPTIONS, labels_inline
from .scaffold import (CONVERT_UTC_LOCAL_DECL, GET_PATIENT_METADATA_DECL, LIST_PATIENTS_DECL,
                       convert_utc_local, get_patient_metadata, list_patients)
from .timeline import (FIND_FIRST_OCCURRENCE_DECL, FIND_MOMENTS_OF_INTEREST_DECL,
                       GET_ATOMIC_ACTIONS_DECL, QUERY_ACTIONS_DECL, find_first_occurrence,
                       find_moments_of_interest, get_atomic_actions, query_actions)
from .report import (GET_HOURLY_REPORT_DECL, LIST_HOURLY_REPORTS_DECL, get_hourly_report,
                     list_hourly_reports)
from .ehr_med import MED_TOOL
from .ehr_flowsheet import FLOWSHEET_TOOL
from .video import VIDEO_TOOL

# Shared (positional-signature) tools: name -> (function, declaration).
SHARED_TOOLS: dict[str, tuple[Any, FunctionDeclaration]] = {
    "convert_utc_local": (convert_utc_local, CONVERT_UTC_LOCAL_DECL),
    "list_patients": (list_patients, LIST_PATIENTS_DECL),
    "get_patient_metadata": (get_patient_metadata, GET_PATIENT_METADATA_DECL),
    "list_hourly_reports": (list_hourly_reports, LIST_HOURLY_REPORTS_DECL),
    "get_hourly_report": (get_hourly_report, GET_HOURLY_REPORT_DECL),
    "get_atomic_actions": (get_atomic_actions, GET_ATOMIC_ACTIONS_DECL),
    "query_actions": (query_actions, QUERY_ACTIONS_DECL),
    "find_first_occurrence": (find_first_occurrence, FIND_FIRST_OCCURRENCE_DECL),
    "find_moments_of_interest": (find_moments_of_interest, FIND_MOMENTS_OF_INTEREST_DECL),
}

_PAT_ID_TOOLS = {
    name for name, (fn, _) in SHARED_TOOLS.items()
    if "pat_id" in fn.__code__.co_varnames[: fn.__code__.co_argcount]
}

GROUPS = {
    "scaffold": ["get_patient_metadata", "convert_utc_local"],
    "timeline": ["get_atomic_actions", "query_actions", "find_first_occurrence", "find_moments_of_interest"],
    "report": ["list_hourly_reports", "get_hourly_report"],
    "ehr": ["query_medications", "query_flowsheet"],
    "video": ["query_video_minute"],
}


def group_of(name: str) -> str:
    for g, names in GROUPS.items():
        if name in names:
            return g
    return "other"


def dispatch(name: str, args: dict, current_pat_id: str | None = None) -> dict:
    """Execute a shared tool by name. Auto-injects current_pat_id for tools
    that accept pat_id."""
    if name not in SHARED_TOOLS:
        return {"error": f"Unknown tool: {name}"}
    fn, _ = SHARED_TOOLS[name]
    args = dict(args or {})
    if name in _PAT_ID_TOOLS and not args.get("pat_id"):
        if not current_pat_id:
            return {"error": "No patient in context and no pat_id provided."}
        args["pat_id"] = current_pat_id
    try:
        return fn(**args)
    except TypeError as e:
        return {"error": f"Bad arguments to {name}: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _wrap_shared(name: str) -> Tool:
    _fn, decl = SHARED_TOOLS[name]

    def handler(args, pat_id, model):
        return dispatch(name, args, current_pat_id=pat_id)

    return Tool(name=name, declaration=decl, handler=handler, group=group_of(name))


def all_tools() -> dict[str, Tool]:
    """Every tool, keyed by name, with the uniform handler(args, pat_id, model)."""
    reg = {name: _wrap_shared(name) for name in SHARED_TOOLS}
    reg[MED_TOOL.name] = MED_TOOL
    reg[FLOWSHEET_TOOL.name] = FLOWSHEET_TOOL
    reg[VIDEO_TOOL.name] = VIDEO_TOOL
    return reg


THE_ELEVEN = [n for g in ("scaffold", "timeline", "report", "ehr", "video") for n in GROUPS[g]]

__all__ = ["Tool", "ACTIVITY_LABELS", "LABEL_DESCRIPTIONS", "labels_inline", "SHARED_TOOLS",
           "GROUPS", "THE_ELEVEN", "dispatch", "all_tools", "group_of"]

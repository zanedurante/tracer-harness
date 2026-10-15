"""Named tool configurations.

Tool families (tracer.tools.GROUPS):
  scaffold : get_patient_metadata, convert_utc_local        (every configuration)
  timeline : get_atomic_actions, query_actions, find_first_occurrence, find_moments_of_interest
  report   : list_hourly_reports, get_hourly_report
  ehr      : query_medications, query_flowsheet
  video    : query_video_minute

The paper's harness-ablation table (Table 3) has these rows; the config name
is the slug in the second column (configs/tool_configs.yaml lists the same
mapping). Historical names used by the sweep caches are kept as aliases and
resolve to the identical tool set, so cache signatures are unchanged.

  Table row            config           tools
  Text-only            text_only        (none)                       alias: blind
  Timeline only        timeline_only    scaffold + timeline          alias: structured, w_timeline
  Reports only         reports_only     scaffold + report            alias: narrative, stack_hour
  EHR only             ehr_only         scaffold + ehr
  Video only           video_only       scaffold + video
  w/ timeline          w_timeline       scaffold + timeline
  + reports            plus_reports     + report                     alias: stack_hour_minute
  + EHR                plus_ehr         + ehr                        alias: stack_hour_minute_ehr_v2
  + video (TRACER)     tracer           all eleven                   alias: plus_video, full,
                                                                     stack_hour_minute_ehr_v2_video
"""
from __future__ import annotations

from ..tools import GROUPS

SCAFFOLD = list(GROUPS["scaffold"])
TIMELINE = list(GROUPS["timeline"])
REPORT = list(GROUPS["report"])
EHR = list(GROUPS["ehr"])
VIDEO = list(GROUPS["video"])

ALL_ELEVEN = SCAFFOLD + TIMELINE + REPORT + EHR + VIDEO

# Paper row label -> config name (kept in sync with configs/tool_configs.yaml).
PAPER_ROWS = {
    "Text-only": "text_only",
    "Timeline only": "timeline_only",
    "Reports only": "reports_only",
    "EHR only": "ehr_only",
    "Video only": "video_only",
    "w/ timeline": "w_timeline",
    "+ reports": "plus_reports",
    "+ EHR": "plus_ehr",
    "+ video (TRACER)": "tracer",
}


def build_named_configs() -> dict[str, list[str]]:
    base = {
        # single-toolset rows
        "text_only":     [],
        "timeline_only": SCAFFOLD + TIMELINE,
        "reports_only":  SCAFFOLD + REPORT,
        "ehr_only":      SCAFFOLD + EHR,
        "video_only":    SCAFFOLD + VIDEO,
        # cumulative rows
        "w_timeline":    SCAFFOLD + TIMELINE,
        "plus_reports":  SCAFFOLD + TIMELINE + REPORT,
        "plus_ehr":      SCAFFOLD + TIMELINE + REPORT + EHR,
        "tracer":        SCAFFOLD + TIMELINE + REPORT + EHR + VIDEO,
        # other combinations used in the appendix / diagnostics
        "scaffold":      list(SCAFFOLD),
        "video_solo":    list(VIDEO),
        "visual_only":   SCAFFOLD + TIMELINE + VIDEO,       # strictly video-derived evidence
        "reports_video": SCAFFOLD + TIMELINE + REPORT + VIDEO,
        "reports_ehr":   SCAFFOLD + REPORT + EHR,
    }
    aliases = {
        "blind": "text_only",
        "structured": "timeline_only",
        "narrative": "reports_only",
        "stack_hour": "reports_only",
        "stack_hour_minute": "plus_reports",
        "stack_hour_minute_ehr_v2": "plus_ehr",
        "stack_hour_minute_ehr_v2_video": "tracer",
        "stack_hour_minute_video": "reports_video",
        "stack_hour_ehr_v2": "reports_ehr",
        "stack_hour_med": "reports_ehr",
        "plus_video": "tracer",
        "full": "tracer",
        "all_tools": "tracer",
    }
    out = dict(base)
    for a, target in aliases.items():
        out[a] = list(base[target])
    return out


def resolve(names_or_config: str) -> list[str]:
    """A comma-list of tool names, or a single named config -> tool list."""
    named = build_named_configs()
    if names_or_config in named:
        return named[names_or_config]
    if names_or_config in PAPER_ROWS:
        return named[PAPER_ROWS[names_or_config]]
    tools = [t.strip() for t in names_or_config.split(",") if t.strip()]
    unknown = [t for t in tools if t not in ALL_ELEVEN and t != "list_patients"]
    if unknown:
        raise ValueError(f"Unknown config or tool name(s): {unknown}. "
                         f"Named configs: {sorted(named)}; tools: {ALL_ELEVEN}")
    return tools


def config_sig(tools: list[str]) -> str:
    """Cache signature of a tool set (order-independent)."""
    return "BLIND" if not tools else "+".join(sorted(tools))

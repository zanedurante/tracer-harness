"""System-prompt assembly per tool configuration.

The prompt is built from ONLY the tools that are available in the current
configuration (each tool's own FunctionDeclaration.description), so the prompt
is honest for every ablation point. With zero tools the prompt degrades to the
text-only framing, which is what makes `tools = 0` the text-only baseline.

The exact text is also released verbatim in prompts/system_prompt.md.
"""
from __future__ import annotations

# Shared across ALL configs (text-only and every tool subset) so the only thing
# that varies between configs is the available tool list -- not the answer contract.
FORCE_CHOICE = (
    "You MUST select exactly one option as your single best guess, even if you are "
    "unsure or the data is incomplete. Never refuse and never say you cannot "
    "determine the answer."
)

BLIND_PROMPT = (
    "You are answering a multiple-choice question about an ICU patient. You do NOT "
    "have access to the patient's video, activity data, or medical records, and you "
    "have no tools. Answer as best you can from general reasoning. " + FORCE_CHOICE
)

_HEADER = (
    "You are an analytic assistant embedded in a clinical video-viewer app for ICU "
    "patients. Answer the multiple-choice question about the patient currently on "
    "screen. Investigate with your tools BEFORE deciding; base your answer on what "
    "the tools actually return, not on priors."
)

_FOOTER = (
    "If a tool returns an error or nothing useful, adjust and try another approach. "
    "Tools that take a patient id default to the current patient — omit pat_id. "
    + FORCE_CHOICE
)


def build_system_prompt(allowed_names, registry) -> str:
    allowed = list(allowed_names)
    if not allowed:
        return BLIND_PROMPT
    lines = [_HEADER, "", "Tools available to you:"]
    for n in allowed:
        desc = (getattr(registry[n].declaration, "description", "") or "").strip().split("\n")[0]
        lines.append(f"- {n}: {desc}")
    lines += ["", _FOOTER]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GUIDED addendum (the "guided" prompt variant used for every reported run).
#
# States the stay's data windows up front and gives explicit working rules.
# The answer contract and the tool set are unchanged; the text-only config
# never receives this text. Each rule is emitted only when the tools it names
# are in the current config, so the addendum never steers the model toward a
# tool that is not exposed.
# ---------------------------------------------------------------------------
GUIDED_CONTEXT = """
Patient context (from get_patient_metadata):
- Video and activity data ({activity_sources}) cover {start} to {end} local time ({n_hours} hours).
- hour_index counts whole hours from {start}: {hour_examples}. To convert a time to hour_index, subtract {start} and take the whole number of hours; write the subtraction down.
{ehr_line}Every date or time in a question refers to THIS stay in local time. Never guess a year or a date outside these windows.

Working rules (follow them strictly):
"""
GUIDED_EHR_LINE = ("- EHR data (query_medications, query_flowsheet) covers the whole hospital admission, "
                   "{ehr_start} to {ehr_end} local time, which extends beyond the video window.\n")

# Rule text: (required tool names, template)
GUIDED_RULES = [
    ({"query_medications", "query_flowsheet"},
     "For \"during the stay\", \"first\", \"last\", \"how many\", \"which day\" questions about medications or flowsheet data, query the ENTIRE admission in ONE call (start_pt = {ehr_start}, end_pt = {ehr_end}); for activity data use start_hour 0 to end_hour {n_hours}. Do not step through the stay day by day."),
    ({"query_actions"},
     "For \"during the stay\" / \"how many hours\" / \"what percentage\" questions about activity, query the ENTIRE stay in ONE call (start_hour 0 to end_hour {n_hours}). Do not step through the stay day by day."),
    (set(),
     "Zero rows or zero events never prove absence. Before concluding \"none\", re-check the dates against the patient context, remove filters, widen the window, and try a shorter substring. If a result carries a \"hint\" field, follow it."),
    ({"query_medications"},
     "Medications: use name_filter ALONE with a short substring of the drug name (e.g. 'norepi', 'oxycod', 'hepar'). Do not add class_filter unless the question is about a drug class, and never combine the two."),
    ({"query_flowsheet"},
     "Flowsheet measurement names must be discovered, not guessed: call query_flowsheet with NO disp_name_filter first to see the exact names, then filter with a short substring taken from that list (e.g. 'CAM' rather than 'CAM-ICU', 'Temp' rather than 'Temperature')."),
    (set(),
     "Outputs are capped. If n_events_returned < n_events_total, n_rows_returned < n_rows_total, or a \"_truncated\" note appears, narrow the filter or time window (or raise max_rows) before drawing any conclusion from a partial list."),
    (set(),
     "Prefer totals the tools already computed (per_label_minutes_active, n_events_total, n_rows_total) over adding long arrays by hand. If you must count or add over a long list, do it in blocks of 10 and keep a running total."),
    ({"get_atomic_actions"},
     "For what the patient was doing over a several-minute window, first compute the hour_index, then use get_atomic_actions for that hour (minute-level labels){hourly_clause}{video_clause}"),
    (set(),
     "Investigate with at least two different tools or queries before answering whenever the tools can bear on the question, and cross-check them against each other."),
    (set(),
     "Keep the reasoning short and end with the line: Answer: <LETTER>"),
]


def build_system_prompt_guided(allowed_names, registry, *, start: str, end: str, n_hours,
                               ehr_start: str, ehr_end: str, hour_examples: str) -> str:
    """Base prompt + the guided addendum, restricted to rules whose tools are
    in this config (tool configs only; text-only stays text-only)."""
    allowed = set(allowed_names)
    base = build_system_prompt(allowed_names, registry)
    if not allowed:
        return base
    has_ehr = bool(allowed & {"query_medications", "query_flowsheet"})
    sources = []
    if "get_hourly_report" in allowed:
        sources.append("hourly reports")
    if allowed & {"get_atomic_actions", "query_actions", "find_first_occurrence", "find_moments_of_interest"}:
        sources.append("atomic actions")
    if "query_video_minute" in allowed:
        sources.append("query_video_minute")
    ctx = GUIDED_CONTEXT.format(start=start, end=end, n_hours=n_hours, hour_examples=hour_examples,
                                activity_sources=", ".join(sources) or "the tools listed above",
                                ehr_line=GUIDED_EHR_LINE.format(ehr_start=ehr_start, ehr_end=ehr_end) if has_ehr else "")
    hourly_clause = " together with the hourly report" if "get_hourly_report" in allowed else ""
    video_clause = ("; use query_video_minute to check the specific minute (and the adjacent minute) when the "
                    "question is about a single moment." if "query_video_minute" in allowed else ".")
    rules = []
    for needs, text in GUIDED_RULES:
        if needs and not needs.issubset(allowed):
            continue
        if needs == {"query_actions"} and has_ehr:
            continue          # the EHR version of rule 1 already covers activity data
        rules.append(text.format(ehr_start=ehr_start, ehr_end=ehr_end, n_hours=n_hours,
                                 hourly_clause=hourly_clause, video_clause=video_clause))
    body = "\n".join(f"{i}. {r}" for i, r in enumerate(rules, 1))
    return base + "\n" + ctx + body + "\n"

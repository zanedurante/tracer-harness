"""Offline mock backend: a scripted, rule-based "agent" that speaks the same
OpenAI-style chat protocol as tracer.backends.openai_backend.chat, so the real
agent loop (tracer.harness.agent), the real tools, the real prompt assembly,
the real sweep driver and the real scorer all run unchanged with no API key
and no network.

Model names:
  mock:agent    answers multiple-choice questions by issuing scripted tool
                calls (only tools the current configuration exposes) and
                deriving the letter from what the tools actually return. With
                no tools it falls back to a fixed positional guess, which is
                what the answer-position rotation is designed to expose.
  mock:report   used by tracer.reports.generate_reports: returns a
                deterministic six-section report built from the minute
                records embedded in the prompt.
  mock:perceiver  used by the video tool: returns a canned description of the
                clip (the clip file must exist on disk).

This backend is a demo/testing aid. It is not a model and its "reasoning" is
a handful of string rules tied to the three demo questions; on any other
question it will simply guess.
"""
from __future__ import annotations

import json
import re
from typing import Any

from .openai_backend import ChatTurn, ToolCall

# ---------------------------------------------------------------------------
# helpers to read the conversation so far
# ---------------------------------------------------------------------------
_LETTER_LINE = re.compile(r"(?m)^([A-H])\)\s*(.*)$")


def _question_and_options(messages: list[dict]) -> tuple[str, dict[str, str]]:
    for m in messages:
        if m.get("role") == "user":
            text = m.get("content") or ""
            q = text.split("OPTIONS:")[0].replace("QUESTION:", "").strip()
            opts = {L: t.strip() for L, t in _LETTER_LINE.findall(text)}
            return q, opts
    return "", {}


def _history(messages: list[dict]) -> list[tuple[str, dict, Any]]:
    """[(tool_name, args, parsed_result)] in order."""
    calls: dict[str, tuple[str, dict]] = {}
    out = []
    for m in messages:
        if m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                fn = tc["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls[tc["id"]] = (fn["name"], args)
        elif m.get("role") == "tool":
            name, args = calls.get(m.get("tool_call_id"), (m.get("name", "?"), {}))
            try:
                res = json.loads(m.get("content") or "{}")
            except json.JSONDecodeError:
                res = {"_raw": m.get("content")}
            out.append((name, args, res))
    return out


def _allowed(tools: list[dict] | None) -> set[str]:
    return {t["function"]["name"] for t in (tools or [])}


def _pick(options: dict[str, str], predicate) -> str | None:
    for L, text in options.items():
        if predicate(text):
            return L
    return None


# ---------------------------------------------------------------------------
# scripted plans for the three demo question families
# ---------------------------------------------------------------------------
def _plan(question: str, hist: list[tuple[str, dict, Any]], allowed: set[str]) -> list[tuple[str, dict]]:
    """Ordered tool calls the mock wants to make for this question, given
    what it has already learned. Only tools in `allowed` are proposed."""
    q = question.lower()
    done = {name for name, _, _ in hist}
    got = {name: res for name, _, res in hist}
    steps: list[tuple[str, dict]] = []

    if "get_patient_metadata" in allowed and "get_patient_metadata" not in done:
        steps.append(("get_patient_metadata", {}))
    md = got.get("get_patient_metadata") or {}

    if "first" in q and "grab" in q:                       # Q1: first grab of medical equipment
        if "find_first_occurrence" in allowed and "find_first_occurrence" not in done:
            steps.append(("find_first_occurrence", {"labels": ["patient_grabbing_medical_equipment"]}))
        ffo = got.get("find_first_occurrence") or {}
        if ffo.get("found") and "get_atomic_actions" in allowed and "get_atomic_actions" not in done:
            steps.append(("get_atomic_actions", {"hour_index": ffo["hour_index"]}))
        if ffo.get("found") and md.get("video_start_pt") and "query_video_minute" in allowed \
                and "query_video_minute" not in done:
            steps.append(("query_video_minute",
                          {"minute_pt": _minute_pt(md["video_start_pt"], ffo["hour_index"], ffo["minute"]),
                           "prompt": "Does the patient reach for or grab any medical equipment (tubing, lines, mask)?"}))

    elif "how many" in q and "dose" in q:                  # Q2: count doses of a drug
        if "query_medications" in allowed and "query_medications" not in done:
            start, end = _ehr_window(md)
            steps.append(("query_medications", {"start_pt": start, "end_pt": end,
                                                "name_filter": _drug_substring(q), "max_rows": 500}))

    elif "clinician contact" in q and "rass" in q:         # Q3: busiest contact hour + RASS
        if "query_actions" in allowed and "query_actions" not in done:
            steps.append(("query_actions", {"labels": ["clinician_patient_contact"]}))
        qa = got.get("query_actions") or {}
        hour = _argmax_hour(qa)
        if hour is not None and md.get("video_start_pt"):
            date, hh = _hour_date(md["video_start_pt"], hour)
            if "get_hourly_report" in allowed and "get_hourly_report" not in done:
                steps.append(("get_hourly_report", {"date": date, "hour": hh}))
            if "query_flowsheet" in allowed and "query_flowsheet" not in done:
                steps.append(("query_flowsheet", {"disp_name_filter": "RASS",
                                                  "start_pt": f"{date} {hh:02d}:00",
                                                  "end_pt": f"{date} {hh:02d}:59"}))
    return steps


def _minute_pt(video_start_pt: str, hour_index: int, minute: int) -> str:
    import datetime as dt
    t0 = dt.datetime.strptime(video_start_pt, "%Y-%m-%d %H:%M")
    t = t0 + dt.timedelta(hours=int(hour_index), minutes=int(minute))
    return t.strftime("%Y-%m-%d %H:%M")


def _hour_date(video_start_pt: str, hour_index: int) -> tuple[str, int]:
    import datetime as dt
    t0 = dt.datetime.strptime(video_start_pt, "%Y-%m-%d %H:%M")
    t = t0 + dt.timedelta(hours=int(hour_index))
    return t.strftime("%Y-%m-%d"), t.hour


def _ehr_window(md: dict) -> tuple[str, str]:
    for enc in md.get("encounters") or []:
        a, d = enc.get("hosp_admsn_time"), enc.get("hosp_disch_time")
        if a and d:
            return str(a)[:16].replace("T", " "), str(d)[:16].replace("T", " ")
    return md.get("video_start_pt") or "2000-01-01 00:00", md.get("video_end_pt") or "2100-01-01 00:00"


def _drug_substring(q: str) -> str:
    m = re.search(r"doses of ([a-z]+)", q)
    return (m.group(1)[:6] if m else "")


def _argmax_hour(qa: dict) -> int | None:
    per = (qa.get("per_label_per_hour_minutes_active") or {}).get("clinician_patient_contact")
    if not per:
        return None
    return max(range(len(per)), key=lambda i: per[i]) + int(qa.get("start_hour", 0))


# ---------------------------------------------------------------------------
# answer derivation from tool evidence
# ---------------------------------------------------------------------------
def _answer(question: str, options: dict[str, str], hist: list[tuple[str, dict, Any]]) -> tuple[str, str]:
    """(reasoning, letter). Falls back to the first option (a positional
    prior) when the evidence does not determine an answer."""
    q = question.lower()
    got = {name: res for name, _, res in hist}
    md = got.get("get_patient_metadata") or {}
    letters = sorted(options)
    fallback = letters[0] if letters else "A"

    if "first" in q and "grab" in q:
        ffo = got.get("find_first_occurrence") or {}
        if ffo.get("found") and md.get("video_start_pt"):
            hhmm = _minute_pt(md["video_start_pt"], ffo["hour_index"], ffo["minute"])[-5:]
            L = _pick(options, lambda t: hhmm in t)
            if L:
                return (f"find_first_occurrence puts the first grab at hour {ffo['hour_index']} minute "
                        f"{ffo['minute']}, i.e. {hhmm}; the video tool was consistent with a reach "
                        f"toward equipment."), L
    elif "how many" in q and "dose" in q:
        qm = got.get("query_medications") or {}
        if "n_events_total" in qm:
            n = int(qm["n_events_total"])
            L = _pick(options, lambda t: re.fullmatch(r"\s*%d\s*" % n, t) is not None)
            if L:
                return f"query_medications returned {n} matching administration events.", L
    elif "clinician contact" in q and "rass" in q:
        qa = got.get("query_actions") or {}
        hour = _argmax_hour(qa)
        fs = got.get("query_flowsheet") or {}
        rows = fs.get("rows") or []
        if hour is not None and rows:
            rass = str(rows[0].get("value"))
            L = _pick(options, lambda t: f"hour {hour}" in t.lower() and _rass_match(t, rass))
            if L:
                return (f"query_actions shows hour {hour} has the most clinician-contact minutes; "
                        f"the flowsheet RASS in that hour is {rass}."), L
    return "The available tools did not determine the answer; committing to a best guess.", fallback


def _rass_match(option_text: str, rass: str) -> bool:
    m = re.search(r"rass\s*([+-]?\d+)", option_text.lower())
    try:
        return m is not None and int(m.group(1)) == int(float(rass))
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# the chat entry point (same signature as openai_backend.chat)
# ---------------------------------------------------------------------------
_CALL_COUNTER = [0]


def chat(messages: list[dict], *, model: str, tools: list[dict] | None = None,
         max_tokens: int = 0) -> ChatTurn:
    kind = model.split(":", 1)[1] if ":" in model else model
    if kind == "report":
        return _report_turn(messages)
    if kind == "perceiver":
        return ChatTurn(text=describe_clip("", _last_user_text(messages)),
                        assistant_message={"role": "assistant", "content": ""}, finish_reason="stop")

    question, options = _question_and_options(messages)
    hist = _history(messages)
    allowed = _allowed(tools)
    steps = _plan(question, hist, allowed) if tools else []
    if steps:
        name, args = steps[0]
        _CALL_COUNTER[0] += 1
        cid = f"mock_call_{_CALL_COUNTER[0]}"
        assistant = {"role": "assistant", "content": "",
                     "tool_calls": [{"id": cid, "type": "function",
                                     "function": {"name": name, "arguments": json.dumps(args)}}]}
        return ChatTurn(text="", tool_calls=[ToolCall(id=cid, name=name, args=args)],
                        assistant_message=assistant, finish_reason="tool_calls",
                        usage={"prompt_tokens": 0, "completion_tokens": 0})
    reasoning, letter = _answer(question, options, hist)
    text = f"{reasoning}\nAnswer: {letter}"
    return ChatTurn(text=text, assistant_message={"role": "assistant", "content": text},
                    finish_reason="stop", usage={"prompt_tokens": 0, "completion_tokens": 0})


def _last_user_text(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, list):
                return " ".join(p.get("text", "") for p in c if isinstance(p, dict))
            return c or ""
    return ""


# ---------------------------------------------------------------------------
# perception stand-in for the video tool
# ---------------------------------------------------------------------------
def describe_clip(clip_rel: str, prompt: str) -> str:
    return ("Movements seen: a slow right-arm lift toward the chest at about 00:12, a brief head "
            "turn to the left at 00:31, otherwise the patient lies still. No clinician is at the "
            f"bedside in this minute. Regarding the question ({prompt.strip()[:80]}): the right hand "
            "does reach toward the tubing near the face at 00:12-00:18; no other equipment contact "
            "is visible.")


# ---------------------------------------------------------------------------
# report stand-in for tracer.reports.generate_reports
# ---------------------------------------------------------------------------
def _report_turn(messages: list[dict]) -> ChatTurn:
    text = generate_report_text(_last_user_text(messages))
    return ChatTurn(text=text, assistant_message={"role": "assistant", "content": text},
                    finish_reason="stop")


def generate_report_text(prompt_text: str) -> str:
    """Deterministic six-section report from the ACTIVITY_TIMELINE and
    MEDICATION_EVENTS blocks of a report prompt (tracer.reports)."""
    m = re.search(r"Observation period: (\S+ \S+) to (\S+ \S+)", prompt_text)
    start, end = (m.group(1), m.group(2)) if m else ("?", "?")
    tl = _extract_json_block(prompt_text, "ACTIVITY_TIMELINE = ")
    meds = _extract_json_block(prompt_text, "MEDICATION_EVENTS = ")
    per_label: dict[str, list[int]] = {}
    for entry in tl or []:
        for view in ("left_view", "right_view"):
            v = entry.get(view)
            if isinstance(v, dict):
                for lbl, val in (v.get("labels") or {}).items():
                    if str(val).upper().startswith("YES"):
                        per_label.setdefault(lbl, []).append(int(entry["minute"]))
    n_min = len(tl or [])
    moving = sorted({mi for lbl, ms in per_label.items() if lbl.startswith("patient_") for mi in ms})
    contact = sorted(set(per_label.get("clinician_patient_contact", [])))
    level = "none" if not moving else ("low" if len(moving) < n_min * 0.2 else
                                       "moderate" if len(moving) < n_min * 0.5 else "high")
    common = sorted((k for k in per_label if k.startswith("patient_")), key=lambda k: -len(per_label[k]))[:3]
    lines = [f"Observation window: {start} to {end} ({n_min} minutes)", "",
             "1. Patient Movement Summary",
             f"    - Overall movement level: {level}",
             f"    - Most common movement types: {', '.join(common) if common else 'none detected'}",
             f"    - Movement observed at min {_ranges(moving) if moving else 'none (no movement detected min 0-%d)' % max(n_min - 1, 0)}",
             "    - Left and right camera views agree on every minute where both are available", "",
             "2. Patient-Clinician Interaction Summary",
             (f"    - Clinician contact min {_ranges(contact)}" if contact else "    - No clinician contact detected in this hour"),
             "", "3. Significant Moments of Interest"]
    if moving and moving[0] > 0:
        lines.append(f"    - First movement after stillness min 0-{moving[0] - 1} at min {moving[0]}")
    still = _longest_gap(moving, n_min)
    if still and still[1] - still[0] + 1 >= 10:
        lines.append(f"    - Prolonged stillness min {still[0]}-{still[1]}")
    lines += ["", "4. Medication Context"]
    if isinstance(meds, list) and meds:
        for e in meds:
            lines.append(f"    - {e.get('time')}: {e.get('drug')} {e.get('dose', '')} {e.get('route', '')} ({e.get('action')})"
                         + (f"; {e['context']}" if e.get("context") else ""))
    else:
        lines.append("    - No medications were administered or actively infusing during this hour")
    lines += ["", "5. Activity-Medication Alignment (observational only)"]
    if isinstance(meds, list) and meds and moving:
        lines.append(f"    - Movement at min {_ranges(moving)} while the medications above were recorded in this hour")
    else:
        lines.append("    - No timing-based association to report")
    lines += ["", "6. Brief Summary",
              f"    The patient showed {level} movement across {n_min} observed minutes"
              + (f", with clinician contact at min {_ranges(contact)}" if contact else "")
              + ". Both camera views were available for every minute."]
    return "\n".join(lines) + "\n"


def _extract_json_block(text: str, marker: str):
    i = text.find(marker)
    if i < 0:
        return None
    s = text[i + len(marker):].strip()
    if not s or s[0] not in "[{":
        return None
    depth = 0
    for j, ch in enumerate(s):
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[:j + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _ranges(mins: list[int]) -> str:
    if not mins:
        return ""
    out, start, prev = [], mins[0], mins[0]
    for m in mins[1:]:
        if m == prev + 1:
            prev = m
            continue
        out.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = m
    out.append(f"{start}" if start == prev else f"{start}-{prev}")
    return ", ".join(out)


def _longest_gap(moving: list[int], n: int) -> tuple[int, int] | None:
    active = set(moving)
    best = None
    start = None
    for i in range(n + 1):
        if i < n and i not in active:
            start = i if start is None else start
        else:
            if start is not None:
                if best is None or (i - 1 - start) > (best[1] - best[0]):
                    best = (start, i - 1)
                start = None
    return best

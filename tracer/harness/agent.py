"""The gated agent loop: exposes ONLY an allowed subset of tools and assembles
its system prompt from that subset.

Loop limits (paper appendix, "Sampling and serving settings"): at most 25
turns and 25 tool calls, after which the agent is forced to answer with no
tools exposed; up to three retries of a malformed call; on the
OpenAI-compatible path tool results are truncated to TOOL_RESULT_CHARS and the
transcript to TRANSCRIPT_CHARS.

Two backends implement the SAME loop:
  * Gemini (google-genai): system_instruction + function declarations, with
    optional context caching of the shared prefix.
  * OpenAI-compatible (vLLM) and the offline mock backend.
Both gate tool calls to `allowed_tools`, wrap any non-dict tool response in
{"result": value}, and record every call in `tool_calls`.
"""
from __future__ import annotations

import json
import os

from google.genai import types

from ..backends import openai_backend as oai
from ..eval.mcq import render_options
from . import prompt_cache
from .prompt import build_system_prompt, build_system_prompt_guided

MAX_TURNS = int(os.environ.get("AGENT_MAX_TURNS", "25"))
MAX_TOOL_CALLS = int(os.environ.get("AGENT_MAX_TOOL_CALLS", "25"))
MAX_MALFORMED_RETRIES = 3

FORCED_FINAL = ("You have used all available tool calls. Based on what you have found so far, "
                "commit to your single best-guess answer now. End with a line: Answer: <LETTER>")


def build_message(item) -> str:
    return (
        f"QUESTION: {item.question}\n\n"
        f"OPTIONS:\n{render_options(item)}\n\n"
        "You must choose exactly one option even if uncertain — never refuse.\n"
        "End your reply with a line in exactly this form:\nAnswer: <LETTER>"
    )


def _new_tokens() -> dict:
    return {"prompt": 0, "completion": 0, "total": 0, "turns": 0}


def _add_gemini_usage(tok: dict, resp) -> None:
    """Accumulate one Gemini turn's usage. AGENT LOOP ONLY: model calls made
    inside a tool (query_video_minute's perception call) are not visible here,
    so these totals under-count end-to-end cost for video configurations."""
    u = getattr(resp, "usage_metadata", None)
    if u is None:
        return
    pt = getattr(u, "prompt_token_count", None) or 0
    ct = (getattr(u, "candidates_token_count", None) or 0) + (getattr(u, "thoughts_token_count", None) or 0)
    tt = getattr(u, "total_token_count", None) or (pt + ct)
    tok["prompt"] += pt
    tok["completion"] += ct
    tok["total"] += tt
    tok["turns"] += 1


# ---------------------------------------------------------------------------
# Prompt variant: "guided" = base prompt + the guided addendum (the variant
# used for every reported run); "base" = header/tools/footer only.
# ---------------------------------------------------------------------------
def _canon_variant(v: str | None, default: str) -> str:
    v = (v or default).strip().lower()
    if v in ("open", "open-weight", "openweight", "guided"):
        return "guided"
    if v in ("gemini", "base", "plain"):
        return "base"
    return v


PROMPT_VARIANT = _canon_variant(os.environ.get("AGENT_PROMPT_VARIANT"), "guided")


def prompt_variant_for(model: str) -> str:
    """Which system prompt `model` gets. Default "guided" for every backend;
    AGENT_PROMPT_VARIANT=base selects the plain header/tools/footer prompt."""
    return PROMPT_VARIANT


def _stay_context(pat_id, registry, model) -> dict | None:
    """Stay window facts for the guided addendum, all taken from
    get_patient_metadata: the video/activity window, the hospital admission
    window (EHR coverage), and worked hour_index examples."""
    import datetime as _dt
    try:
        md = registry["get_patient_metadata"].handler({}, pat_id, model)
        if not (isinstance(md, dict) and md.get("video_start_pt") and md.get("video_end_pt")):
            return None
        start, end = md["video_start_pt"], md["video_end_pt"]
        ehr_start, ehr_end = start, end
        for enc in md.get("encounters") or []:
            a, d = enc.get("hosp_admsn_time"), enc.get("hosp_disch_time")
            if a and d:
                ehr_start = str(a)[:16].replace("T", " ")
                ehr_end = str(d)[:16].replace("T", " ")
                break
        t0 = _dt.datetime.strptime(start, "%Y-%m-%d %H:%M")
        ex = "; ".join(f"hour_index {k} = {(t0 + _dt.timedelta(hours=k)).strftime('%Y-%m-%d %H:%M')}"
                       for k in (0, 6, 24, 48))
        return {"start": start, "end": end, "n_hours": md.get("n_hours", "?"),
                "ehr_start": ehr_start, "ehr_end": ehr_end, "hour_examples": ex}
    except Exception:
        return None


def _system_prompt(allowed, pat_id, registry, model, variant) -> str:
    ctx = _stay_context(pat_id, registry, model) if (allowed and variant == "guided") else None
    if ctx:
        return build_system_prompt_guided(allowed, registry, **ctx)
    return build_system_prompt(allowed, registry)


def _zero_result_hint(name: str, args: dict, result: dict, pat_id, model, registry) -> dict:
    """Guided variant only: when a filtered EHR query returns nothing, attach
    the closest available names so the model can correct its filter instead of
    concluding absence. Pure lookup on the same tools; the data is unchanged."""
    import difflib
    try:
        if name == "query_flowsheet" and args.get("disp_name_filter") and result.get("n_rows_total") == 0:
            disc = registry["query_flowsheet"].handler({k: v for k, v in args.items() if k != "disp_name_filter"},
                                                       pat_id, model)
            avail = disc.get("available_disp_names") or []
            names = [a.get("disp_name") if isinstance(a, dict) else str(a) for a in avail]
            names = [n for n in names if n]
            f = args["disp_name_filter"].lower()
            close = [n for n in names if any(tok and tok in n.lower() for tok in f.replace("-", " ").split())]
            close = close[:8] or difflib.get_close_matches(args["disp_name_filter"], names, n=8, cutoff=0.3)
            result = dict(result)
            result["hint"] = (f"No measurement name contains '{args['disp_name_filter']}'. Available names that "
                              f"may match: {close}. Call again with one of these as disp_name_filter (a short "
                              f"substring is enough); call with no filter to see all {len(names)} names.")
        elif name == "query_medications" and result.get("n_events_total") == 0 and \
                (args.get("name_filter") or args.get("class_filter")):
            wide = registry["query_medications"].handler(
                {"start_pt": args.get("start_pt"), "end_pt": args.get("end_pt"), "max_rows": 5000}, pat_id, model)
            evs = wide.get("events") or []
            names = sorted({e.get("med_name", "") for e in evs if e.get("med_name")})
            f = (args.get("name_filter") or "").lower()
            close = [n for n in names if f and f[:5] in n.lower()][:8] or \
                difflib.get_close_matches(args.get("name_filter") or "", names, n=8, cutoff=0.3)
            result = dict(result)
            result["hint"] = (f"No events matched name_filter={args.get('name_filter')!r} "
                              f"class_filter={args.get('class_filter')!r} in this window ({wide.get('n_events_total')} "
                              f"events total in the window). Drop class_filter and use a short name substring. "
                              f"Medication names in this window that may match: {close}")
    except Exception:
        pass
    return result


def _call_tool(name, args, allowed, registry, pat_id, model, variant):
    if name in allowed and name in registry:
        try:
            result = registry[name].handler(args, pat_id, model)
        except Exception as e:
            result = {"error": f"{type(e).__name__}: {e}"}
    else:
        result = {"error": f"Tool '{name}' is not available in this configuration."}
    if not isinstance(result, dict):
        result = {"result": result}
    if variant == "guided":
        result = _zero_result_hint(name, args, result, pat_id, model, registry)
    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def run_ablated(message: str, pat_id, allowed_tools, model: str, registry,
                *, max_turns: int = MAX_TURNS) -> dict:
    """Run the MCQ message through the agent with only `allowed_tools` exposed.
    Returns {"reply": str, "tool_calls": list, "error": str | None, ...}.

    Backend is chosen by model name: Gemini model ids use the google-genai
    path; anything else (a vLLM-served model, or ``mock:agent``) uses the
    OpenAI-compatible path. Both implement the same loop."""
    if oai.is_openai_model(model):
        return _run_ablated_openai(message, pat_id, allowed_tools, model, registry, max_turns=max_turns)
    return _run_ablated_gemini(message, pat_id, allowed_tools, model, registry, max_turns=max_turns)


# ---------------------------------------------------------------------------
# Gemini path
# ---------------------------------------------------------------------------
def _run_ablated_gemini(message, pat_id, allowed_tools, model, registry, *, max_turns):
    from ..backends.gemini import get_gemini_client
    client = get_gemini_client()
    allowed = list(allowed_tools)
    variant = prompt_variant_for(model)
    system_prompt = _system_prompt(allowed, pat_id, registry, model, variant)
    tools_list = None
    if allowed:
        decls = [registry[n].declaration for n in allowed]
        tools_list = [types.Tool(function_declarations=decls)]

    def _config():
        cache_name = prompt_cache.get_or_create(client, model, system_prompt, tools_list)
        if cache_name:
            return types.GenerateContentConfig(
                cached_content=cache_name,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        kw = dict(system_instruction=system_prompt,
                  automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        if tools_list is not None:
            kw["tools"] = tools_list
        return types.GenerateContentConfig(**kw)

    config = _config()
    contents = [types.Content(role="user", parts=[types.Part(text=message)])]
    tool_calls_log: list[dict] = []
    n_malformed = 0
    tok = _new_tokens()

    for _turn in range(max_turns):
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            if prompt_cache.is_cache_missing_error(e):
                prompt_cache.invalidate(model, system_prompt, tools_list)
                config = _config()
                try:
                    resp = client.models.generate_content(model=model, contents=contents, config=config)
                except Exception as e2:
                    return {"reply": "", "tool_calls": tool_calls_log, "error": f"{type(e2).__name__}: {e2}"}
            else:
                return {"reply": "", "tool_calls": tool_calls_log, "error": f"{type(e).__name__}: {e}"}

        _add_gemini_usage(tok, resp)
        if not resp.candidates:
            return {"reply": "(no response from model)", "tool_calls": tool_calls_log,
                    "error": None, "tokens": tok}

        # MALFORMED_FUNCTION_CALL: the model tried to call a tool but emitted it
        # in the wrong syntax; the candidate carries no parts. Resample the turn
        # a few times instead of scoring an empty reply.
        if str(getattr(resp.candidates[0], "finish_reason", "")).endswith("MALFORMED_FUNCTION_CALL"):
            n_malformed += 1
            if n_malformed <= MAX_MALFORMED_RETRIES:
                continue
            return {"reply": "", "tool_calls": tool_calls_log, "n_malformed": n_malformed,
                    "error": f"MALFORMED_FUNCTION_CALL x{n_malformed} (model emitted an unparseable tool call)"}

        content = resp.candidates[0].content
        if content is None:
            fr = getattr(resp.candidates[0], "finish_reason", None)
            return {"reply": "", "tool_calls": tool_calls_log,
                    "error": f"empty candidate content (finish_reason={fr})"}
        contents.append(content)                      # raw Content preserves thought signatures

        calls = [p.function_call for p in (content.parts or [])
                 if getattr(p, "function_call", None) and p.function_call.name]
        if not calls:
            text = "".join(p.text for p in (content.parts or []) if getattr(p, "text", None))
            return {"reply": text, "tool_calls": tool_calls_log, "error": None,
                    "n_malformed": n_malformed, "tokens": tok}

        # One response turn per call turn, with ONE part per call (parallel
        # calls require the function-response part count to match).
        response_parts = []
        for fc in calls:
            args = dict(fc.args or {})
            result = _call_tool(fc.name, args, allowed, registry, pat_id, model, variant)
            response_parts.append(types.Part(
                function_response=types.FunctionResponse(id=fc.id, name=fc.name, response=result)))
            tool_calls_log.append({"name": fc.name, "args": args, "response": result})
        contents.append(types.Content(role="user", parts=response_parts))

        if len(tool_calls_log) >= MAX_TOOL_CALLS:
            break

    # Tool-call cap without a final answer: force one, with no tools exposed.
    contents.append(types.Content(role="user", parts=[types.Part(text=FORCED_FINAL)]))
    try:
        final_cfg = types.GenerateContentConfig(system_instruction=system_prompt)
        resp = client.models.generate_content(model=model, contents=contents, config=final_cfg)
        _add_gemini_usage(tok, resp)
        text = ("".join(p.text for p in (resp.candidates[0].content.parts or []) if getattr(p, "text", None))
                if resp.candidates else "")
        return {"reply": text, "tool_calls": tool_calls_log, "error": None,
                "forced_final": True, "tokens": tok}
    except Exception as e:
        return {"reply": "", "tool_calls": tool_calls_log, "tokens": tok,
                "error": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# OpenAI-compatible path (vLLM-served open-weight models) and the mock backend
# ---------------------------------------------------------------------------
# Context budgeting for limited-window models. Gemini (1M tokens) receives every
# tool payload in full; an open-weight model served at 64-128K cannot. Two
# deterministic rules keep the transcript inside the window without changing
# what the tools compute:
#   1. per-result cap: a dict whose serialization exceeds TOOL_RESULT_CHARS has
#      its longest list field cut to a prefix, with an explicit note.
#   2. transcript cap: when the running transcript exceeds TRANSCRIPT_CHARS, the
#      OLDEST tool results are replaced by a short stub until it fits; the
#      tool_calls log keeps the full responses.
TOOL_RESULT_CHARS = int(os.environ.get("TOOL_RESULT_CHARS", "16000"))
TRANSCRIPT_CHARS = int(os.environ.get("TRANSCRIPT_CHARS", "100000"))
_STUB_CHARS = 1500


def _tool_result_content(result, limit: int = TOOL_RESULT_CHARS) -> str:
    s = json.dumps(result, default=str)
    if len(s) <= limit:
        return s
    if isinstance(result, dict):
        sizes = {k: len(json.dumps(v, default=str)) for k, v in result.items() if isinstance(v, list)}
        if sizes:
            k = max(sizes, key=sizes.get)
            items = result[k]
            other = len(s) - sizes[k]
            kept = []
            used = other + 200
            for it in items:
                n = len(json.dumps(it, default=str)) + 2
                if used + n > limit:
                    break
                kept.append(it)
                used += n
            trimmed = dict(result)
            trimmed[k] = kept
            trimmed["_truncated"] = (
                f"'{k}' had {len(items)} items; only the first {len(kept)} are shown because the "
                f"full output does not fit the model context. Narrow the query (e.g. a start_pt/end_pt "
                f"window, fewer labels, or a smaller hour range) to see the rest.")
            return json.dumps(trimmed, default=str)
    return s[:limit] + f'... [truncated {len(s) - limit} chars: narrow the query to see the rest]'


def _compact_transcript(messages: list[dict]) -> int:
    """Stub the oldest tool results until the transcript fits TRANSCRIPT_CHARS."""
    total = sum(len(m.get("content") or "") for m in messages if isinstance(m.get("content"), str))
    n = 0
    for m in messages:
        if total <= TRANSCRIPT_CHARS:
            break
        c = m.get("content") or ""
        if m.get("role") == "tool" and isinstance(c, str) and len(c) > _STUB_CHARS + 200:
            stub = c[:_STUB_CHARS] + "... [earlier tool output elided to fit the model context; re-call the tool if you need it]"
            total -= len(c) - len(stub)
            m["content"] = stub
            n += 1
    return n


def _run_ablated_openai(message, pat_id, allowed_tools, model, registry, *, max_turns):
    allowed = list(allowed_tools)
    variant = prompt_variant_for(model)
    system_prompt = _system_prompt(allowed, pat_id, registry, model, variant)
    tools = oai.gemini_decls_to_openai_tools([registry[n].declaration for n in allowed]) if allowed else None

    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": message}]
    tool_calls_log: list[dict] = []
    tok = _new_tokens()
    turn_meta: list[dict] = []
    context_overflow = False

    def _out(reply, error, forced=False):
        return {"reply": reply, "tool_calls": tool_calls_log, "error": error, "tokens": tok,
                "messages": messages, "turn_meta": turn_meta,
                "forced_final": forced, "context_overflow": context_overflow}

    def _usage(turn):
        if turn.usage:
            tok["prompt"] += turn.usage.get("prompt_tokens") or 0
            tok["completion"] += turn.usage.get("completion_tokens") or 0
            tok["total"] = tok["prompt"] + tok["completion"]
        tok["turns"] += 1

    for _turn in range(max_turns):
        try:
            turn = oai.chat(messages, model=model, tools=tools)
        except Exception as e:
            if oai.is_context_length_error(e):
                context_overflow = True
                break
            return _out("", f"{type(e).__name__}: {e}")

        _usage(turn)
        messages.append(turn.assistant_message)
        turn_meta.append({"finish_reason": turn.finish_reason, "usage": turn.usage,
                          "n_calls": len(turn.tool_calls), "text_len": len(turn.text or "")})
        if not turn.tool_calls:
            return _out(turn.text, None)

        for tc in turn.tool_calls:
            args = dict(tc.args or {})
            result = _call_tool(tc.name, args, allowed, registry, pat_id, model, variant)
            messages.append({"role": "tool", "tool_call_id": tc.id, "name": tc.name,
                             "content": _tool_result_content(result)})
            tool_calls_log.append({"name": tc.name, "args": args, "response": result})

        n_stubbed = _compact_transcript(messages)
        if n_stubbed:
            turn_meta[-1]["stubbed_tool_msgs"] = n_stubbed

        if len(tool_calls_log) >= MAX_TOOL_CALLS:
            break

    # Forced final answer (tool cap hit, or the context filled up): no tools
    # exposed so the model must commit to a letter -- parity with the Gemini path.
    if context_overflow:
        for m in messages:
            if m.get("role") == "tool" and isinstance(m.get("content"), str) and len(m["content"]) > 2000:
                m["content"] = m["content"][:2000] + "... [truncated: context limit]"
    messages.append({"role": "user", "content": FORCED_FINAL})
    for attempt in range(2):
        try:
            turn = oai.chat(messages, model=model, tools=None)
            _usage(turn)
            messages.append(turn.assistant_message)
            turn_meta.append({"finish_reason": turn.finish_reason, "usage": turn.usage,
                              "n_calls": 0, "text_len": len(turn.text or ""), "forced": True})
            return _out(turn.text, None, forced=True)
        except Exception as e:
            if oai.is_context_length_error(e) and attempt == 0:
                messages[:] = [messages[0], messages[1], messages[-1]]
                continue
            return _out("", f"{type(e).__name__}: {e}", forced=True)
    return _out("", "unreachable", forced=True)

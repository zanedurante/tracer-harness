"""Multi-turn, streaming chat over the TRACER harness.

The MCQ agent in tracer/harness/agent.py answers one question per call and
must end with "Answer: <LETTER>". The viewer needs the same tools, handlers,
prompt pieces and tool-call budget, but with a conversation that persists
across turns, free-form answers, and events emitted as the agent works so the
UI can show tool calls live. This module provides that loop.

History is kept server-side as a list of google.genai Content objects (user
turns, raw model turns, and function-response turns). Keeping the raw model
Content preserves thought signatures and tool-call ids, which the Gemini 3
models require to be echoed back verbatim.

Events yielded by stream_chat:
    {"type": "tool_call",   "name": str, "args": dict}
    {"type": "tool_result", "name": str, "response": dict}
    {"type": "reply",       "text": str, "tool_calls": list, "history": list}
    {"type": "error",       "message": str}
"""
from __future__ import annotations

from google.genai import types

from ..backends import openai_backend as oai
from ..harness import agent as _agent
from ..harness import prompt as _prompt
from ..harness.configs import ALL_ELEVEN
from ..harness.registry import build_registry

MAX_TOOL_CALLS_PER_TURN = 25
MAX_MODEL_TURNS = 30

CHAT_HEADER = (
    "You are an analytic assistant embedded in a clinical video-viewer app for ICU "
    "patients. The user is watching one patient's stay and asks questions about it in "
    "a chat panel. Investigate with your tools BEFORE answering; base every claim on "
    "what the tools actually return, not on priors."
)

CHAT_FOOTER = (
    "If a tool returns an error or nothing useful, adjust and try another approach. "
    "Tools that take a patient id default to the current patient — omit pat_id. "
    "Be concise and cite the tool results you used. If the evidence is insufficient, "
    "say so rather than guessing.\n"
    "Write every timestamp as `YYYY-MM-DD HH:MM` (24-hour, local time, full date) as "
    "plain text, not inside links or code spans: the viewer turns each one into a "
    "clickable jump to that minute of video. Only cite timestamps inside the video "
    "window reported by get_patient_metadata."
)

FORCED_FINAL_CHAT = (
    "You have used all available tool calls for this turn. Answer now from the "
    "evidence gathered so far, and say what you could not verify."
)


def build_chat_system_prompt(allowed_names, registry, ctx: dict | None) -> str:
    """Chat header + one line per exposed tool + the harness's guided context
    and working rules (minus the multiple-choice answer rule) + chat footer."""
    allowed = list(allowed_names)
    lines = [CHAT_HEADER, "", "Tools available to you:"]
    for n in allowed:
        desc = (getattr(registry[n].declaration, "description", "") or "").strip().split("\n")[0]
        lines.append(f"- {n}: {desc}")
    lines += ["", CHAT_FOOTER]
    base = "\n".join(lines)
    if not ctx:
        return base
    # Reuse the guided addendum verbatim, then drop its last rule (the MCQ
    # answer contract) so the chat prompt is the paper's prompt minus the
    # multiple-choice framing.
    guided = _prompt.build_system_prompt_guided(allowed, registry, **ctx)
    addendum = guided[len(_prompt.build_system_prompt(allowed, registry)):]
    rules = [ln for ln in addendum.rstrip("\n").split("\n")]
    if rules and "Answer: <LETTER>" in rules[-1]:
        rules = rules[:-1]
    return base + "\n" + "\n".join(rules) + "\n"


def _text_of(content) -> str:
    return "".join(p.text for p in (getattr(content, "parts", None) or []) if getattr(p, "text", None))


def _calls_of(content):
    return [p.function_call for p in (getattr(content, "parts", None) or [])
            if getattr(p, "function_call", None) and p.function_call.name]


def stream_chat(message: str, history: list, pat_id: str, *, model: str,
                registry=None, allowed=None):
    """Run one user turn. `history` is the server-side list of Content objects
    from previous turns (not modified in place; the final event carries the
    new history to persist)."""
    registry = registry or build_registry()
    allowed = list(allowed) if allowed is not None else [n for n in ALL_ELEVEN if n in registry]

    if oai.is_openai_model(model):
        # vLLM-served or mock models: single-turn through the harness's
        # OpenAI-compatible loop (no cross-turn memory on this path).
        res = _agent.run_ablated(message, pat_id, allowed, model, registry)
        for c in res.get("tool_calls", []):
            yield {"type": "tool_call", "name": c["name"], "args": c["args"]}
            yield {"type": "tool_result", "name": c["name"], "response": c["response"]}
        if res.get("error"):
            yield {"type": "error", "message": res["error"]}
            return
        yield {"type": "reply", "text": res.get("reply", ""), "tool_calls": res.get("tool_calls", []),
               "history": list(history)}
        return

    from ..backends.gemini import get_gemini_client
    client = get_gemini_client()
    variant = _agent.prompt_variant_for(model)
    ctx = _agent._stay_context(pat_id, registry, model) if variant == "guided" else None
    system_prompt = build_chat_system_prompt(allowed, registry, ctx)
    decls = [registry[n].declaration for n in allowed]
    tools_list = [types.Tool(function_declarations=decls)] if decls else None
    kw = dict(system_instruction=system_prompt,
              automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
    if tools_list:
        kw["tools"] = tools_list
    config = types.GenerateContentConfig(**kw)

    contents = list(history) + [types.Content(role="user", parts=[types.Part(text=message)])]
    tool_calls_log: list[dict] = []
    n_calls = 0

    for _turn in range(MAX_MODEL_TURNS):
        try:
            resp = client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
            return
        if not resp.candidates or resp.candidates[0].content is None:
            fr = getattr(resp.candidates[0], "finish_reason", None) if resp.candidates else None
            yield {"type": "error", "message": f"empty response from the model (finish_reason={fr})"}
            return
        content = resp.candidates[0].content
        contents.append(content)
        calls = _calls_of(content)
        if not calls:
            yield {"type": "reply", "text": _text_of(content), "tool_calls": tool_calls_log,
                   "history": contents}
            return
        response_parts = []
        for fc in calls:
            args = dict(fc.args or {})
            yield {"type": "tool_call", "name": fc.name, "args": args}
            result = _agent._call_tool(fc.name, args, allowed, registry, pat_id, model, variant)
            tool_calls_log.append({"name": fc.name, "args": args, "response": result})
            n_calls += 1
            yield {"type": "tool_result", "name": fc.name, "response": result}
            response_parts.append(types.Part(
                function_response=types.FunctionResponse(id=fc.id, name=fc.name, response=result)))
        contents.append(types.Content(role="user", parts=response_parts))
        if n_calls >= MAX_TOOL_CALLS_PER_TURN:
            break

    contents.append(types.Content(role="user", parts=[types.Part(text=FORCED_FINAL_CHAT)]))
    try:
        final_cfg = types.GenerateContentConfig(system_instruction=system_prompt)
        resp = client.models.generate_content(model=model, contents=contents, config=final_cfg)
        content = resp.candidates[0].content if resp.candidates else None
        text = _text_of(content) if content is not None else ""
        if content is not None:
            contents.append(content)
        yield {"type": "reply", "text": text, "tool_calls": tool_calls_log, "history": contents}
    except Exception as e:
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}

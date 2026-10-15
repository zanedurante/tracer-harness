"""OpenAI-compatible chat backend (vLLM / any /v1/chat/completions server).

Lets the agent run on an open-weight model such as Qwen3-VL served by vLLM
with tool calling enabled. Selection is by model name: any name that is not
``gemini-*`` (or that carries the explicit "openai:" prefix) is routed here;
``mock:*`` names are routed to the offline mock backend.

Endpoint / auth come from the environment (same names the `openai` SDK reads):

    OPENAI_BASE_URL   e.g. http://localhost:8000/v1
    OPENAI_API_KEY    anything non-empty for vLLM (defaults to "EMPTY")

A typical vLLM launch that pairs with this backend (paper appendix,
"Sampling and serving settings"):

    vllm serve Qwen/Qwen3-VL-8B-Instruct --served-model-name qwen3-vl-8b \
        --dtype bfloat16 --max-model-len 131072 \
        --enable-auto-tool-choice --tool-call-parser hermes

Two helpers are exported:
  gemini_decls_to_openai_tools(decls)  -> list[dict]   (tool schema conversion)
  chat(...)                            -> ChatTurn      (one model turn)
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field

OPENAI_PREFIX = "openai:"
MOCK_PREFIX = "mock:"
DEFAULT_BASE_URL = "http://localhost:8000/v1"


def is_mock_model(model: str) -> bool:
    return model.startswith(MOCK_PREFIX)


def is_openai_model(model: str) -> bool:
    """True when `model` should be routed to the OpenAI-compatible loop (which
    the mock backend also implements). Anything named `gemini-*` is a Gemini
    model even if it is not in the registry, so a newly released Gemini model
    never silently routes to a local server that is not there."""
    if model.startswith(OPENAI_PREFIX) or is_mock_model(model):
        return True
    if model.startswith("gemini-"):
        return False
    from .gemini import GEMINI_MODELS
    return model not in GEMINI_MODELS


def served_name(model: str) -> str:
    return model[len(OPENAI_PREFIX):] if model.startswith(OPENAI_PREFIX) else model


_client = None
_client_lock = threading.Lock()


def get_openai_client():
    global _client
    with _client_lock:
        if _client is None:
            from openai import OpenAI
            _client = OpenAI(
                base_url=os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL),
                api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                timeout=float(os.environ.get("OPENAI_TIMEOUT_S", "600")),
                max_retries=int(os.environ.get("OPENAI_MAX_RETRIES", "2")),
            )
        return _client


# ---------------------------------------------------------------------------
# Tool schema conversion: google.genai FunctionDeclaration -> OpenAI tool dict
# ---------------------------------------------------------------------------
_TYPE_MAP = {
    "OBJECT": "object", "STRING": "string", "INTEGER": "integer",
    "NUMBER": "number", "BOOLEAN": "boolean", "ARRAY": "array", "NULL": "null",
}


def _schema_to_json(s) -> dict:
    if s is None:
        return {}
    out: dict = {}
    t = s.get("type")
    if t is not None:
        t = str(t).split(".")[-1].upper()
        out["type"] = _TYPE_MAP.get(t, t.lower())
    for k in ("description", "enum", "format", "minimum", "maximum", "min_items", "max_items"):
        if s.get(k) is not None:
            out[{"min_items": "minItems", "max_items": "maxItems"}.get(k, k)] = s[k]
    if s.get("nullable"):
        out["type"] = [out.get("type", "string"), "null"]
    if s.get("properties"):
        out["properties"] = {k: _schema_to_json(v) for k, v in s["properties"].items()}
    if s.get("required"):
        out["required"] = list(s["required"])
    if s.get("items"):
        out["items"] = _schema_to_json(s["items"])
    return out


def gemini_decl_to_openai_tool(decl) -> dict:
    d = decl.model_dump(mode="json", exclude_none=True) if hasattr(decl, "model_dump") else dict(decl)
    params = _schema_to_json(d.get("parameters")) or {"type": "object", "properties": {}}
    params.setdefault("type", "object")
    params.setdefault("properties", {})
    return {"type": "function",
            "function": {"name": d["name"],
                         "description": d.get("description", "") or "",
                         "parameters": params}}


def gemini_decls_to_openai_tools(decls) -> list[dict]:
    return [gemini_decl_to_openai_tool(d) for d in decls]


# ---------------------------------------------------------------------------
# One model turn
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ChatTurn:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    assistant_message: dict = field(default_factory=dict)   # appended to history verbatim
    finish_reason: str | None = None
    usage: dict | None = None


def _parse_args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else {"value": v}
    except json.JSONDecodeError:
        return {"_unparsed": raw}


# Open-weight agents are capped at 8,192 output tokens per turn (Gemini agents
# have no output cap); 2,048 cut off ~2% of Qwen turns mid-reasoning.
DEFAULT_MAX_TOKENS = int(os.environ.get("OPENAI_MAX_TOKENS", "8192"))


def chat(messages: list[dict], *, model: str, tools: list[dict] | None = None,
         tool_choice: str | None = None, max_tokens: int = DEFAULT_MAX_TOKENS,
         temperature: float | None = None) -> ChatTurn:
    """Send `messages` (OpenAI format) and return the assistant turn.

    `messages` should include the system message as the first entry. Tool
    results are ordinary {"role": "tool", "tool_call_id": ..., "content": ...}
    entries; the caller owns the history. ``mock:*`` models are served by
    tracer.backends.mock without any network access.
    """
    if is_mock_model(model):
        from . import mock
        return mock.chat(messages, model=model, tools=tools, max_tokens=max_tokens)
    client = get_openai_client()
    kwargs: dict = dict(model=served_name(model), messages=messages, max_tokens=max_tokens)
    if tools:
        kwargs["tools"] = tools
        if tool_choice:
            kwargs["tool_choice"] = tool_choice
    if temperature is not None:
        kwargs["temperature"] = temperature
    resp = _create_with_retry(client, kwargs)
    choice = resp.choices[0]
    msg = choice.message
    calls: list[ToolCall] = []
    for i, tc in enumerate(msg.tool_calls or []):
        fn = tc.function
        calls.append(ToolCall(id=tc.id or f"call_{i}", name=fn.name, args=_parse_args(fn.arguments)))
    assistant: dict = {"role": "assistant", "content": msg.content or ""}
    if calls:
        assistant["tool_calls"] = [
            {"id": c.id, "type": "function",
             "function": {"name": c.name, "arguments": json.dumps(c.args)}}
            for c in calls]
    usage = None
    if getattr(resp, "usage", None) is not None:
        usage = {"prompt_tokens": resp.usage.prompt_tokens,
                 "completion_tokens": resp.usage.completion_tokens}
    return ChatTurn(text=msg.content or "", tool_calls=calls, assistant_message=assistant,
                    finish_reason=choice.finish_reason, usage=usage)


CONNECTION_RETRIES = int(os.environ.get("OPENAI_CONN_RETRIES", "12"))


def is_connection_error(err: Exception) -> bool:
    name = type(err).__name__
    s = str(err).lower()
    return (name in ("APIConnectionError", "APITimeoutError", "InternalServerError")
            or "connection error" in s or "timed out" in s or "connection reset" in s
            or "503" in s or "502" in s)


def _create_with_retry(client, kwargs: dict):
    """Retry transport failures (dead tunnel, server restart) with backoff.
    A trial must never be scored on a transport failure; if the server stays
    unreachable for the whole retry window the exception propagates so the
    caller can abort loudly instead of recording a wrong answer."""
    import time
    delay = 5.0
    for attempt in range(CONNECTION_RETRIES + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            if not is_connection_error(e) or attempt == CONNECTION_RETRIES:
                raise
            time.sleep(delay)
            delay = min(delay * 1.6, 90.0)


def is_context_length_error(err: Exception) -> bool:
    s = str(err).lower()
    return ("maximum context length" in s or "context length" in s
            or "too many tokens" in s or "exceeds the model" in s)

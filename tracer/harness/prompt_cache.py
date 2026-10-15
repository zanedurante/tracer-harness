"""Gemini context-cache manager for the agent loop.

Every trial in a sweep sends the SAME system prompt + tool declarations. That
prefix can be uploaded once via the Gemini context-cache API (`cachedContents`)
and referenced by name for the rest of the sweep, at a fraction of the input
token cost. Response content and per-trial `contents` are unchanged, so
accuracy is unaffected.

Cache key: sha256(model || system_prompt || serialized tool declarations).
Falls back silently to `None` if the SDK/model rejects cache creation.
"""
from __future__ import annotations

import hashlib
import json
import threading

_LOCK = threading.Lock()
_CACHE: dict[str, tuple[str, float]] = {}
_FAILED: set[str] = set()

DEFAULT_TTL = "3600s"
RECREATE_AFTER_S = 3000.0   # recreate before the server-side TTL lapses


def _decl_dump(tools) -> str:
    out = []
    for t in tools or []:
        for d in (t.function_declarations or []):
            out.append(d.model_dump(mode="json", exclude_none=True))
    return json.dumps(out, sort_keys=True, default=str)


def _key(model: str, system_prompt: str, tools) -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x00")
    h.update(system_prompt.encode())
    h.update(b"\x00")
    h.update(_decl_dump(tools).encode())
    return h.hexdigest()


def get_or_create(client, model: str, system_prompt: str, tools, ttl: str = DEFAULT_TTL):
    """Return a cached-content resource name, or None if caching is unavailable."""
    import time
    k = _key(model, system_prompt, tools)
    with _LOCK:
        if k in _CACHE:
            name, created = _CACHE[k]
            if time.time() - created < RECREATE_AFTER_S:
                return name
            _CACHE.pop(k, None)
        if k in _FAILED:
            return None
    try:
        from google.genai import types
        cfg = types.CreateCachedContentConfig(
            system_instruction=system_prompt,
            tools=tools if tools else None,
            ttl=ttl,
        )
        cache = client.caches.create(model=model, config=cfg)
        name = cache.name
    except Exception:
        with _LOCK:
            _FAILED.add(k)
        return None
    with _LOCK:
        _CACHE[k] = (name, time.time())
    return name


def invalidate(model: str, system_prompt: str, tools) -> None:
    with _LOCK:
        _CACHE.pop(_key(model, system_prompt, tools), None)


def is_cache_missing_error(err: Exception) -> bool:
    return "CachedContent not found" in str(err) or ("cached_content" in str(err).lower() and "403" in str(err))


def stats() -> dict:
    with _LOCK:
        return {"n_caches": len(_CACHE), "n_failed": len(_FAILED)}

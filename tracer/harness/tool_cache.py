"""Cross-position response cache for deterministic tool handlers.

Within a sweep, positions 0..N-1 of one MCQ differ only in the option-letter
shuffle. Tool calls are grounded in the question text, so the model issues the
SAME args to the SAME tool across positions. For a tool whose handler is a
pure function of its args (pandas queries, file reads), the response is
bit-identical, so position 0's response can be reused for positions 1..N-1.

The video tool is NOT cached: query_video_minute invokes a model, which is
stochastic. Per-process, thread-safe. Optional; the agent loop does not use it
unless a caller wires it in.
"""
from __future__ import annotations

import json
import threading

_CACHEABLE = frozenset([
    "get_hourly_report", "list_hourly_reports", "get_atomic_actions", "query_actions",
    "find_first_occurrence", "find_moments_of_interest", "query_medications", "query_flowsheet",
    "get_patient_metadata", "list_patients", "convert_utc_local",
])

_CACHE: dict[tuple, dict] = {}
_LOCK = threading.Lock()
_HITS = 0
_MISSES = 0


def is_cacheable(tool_name: str) -> bool:
    return tool_name in _CACHEABLE


def make_key(qid: int, tool_name: str, args: dict) -> tuple:
    return (qid, tool_name, json.dumps(args, sort_keys=True, default=str))


def get(key: tuple):
    global _HITS
    with _LOCK:
        if key in _CACHE:
            _HITS += 1
            return _CACHE[key]
    return None


def put(key: tuple, response: dict) -> None:
    global _MISSES
    with _LOCK:
        _CACHE[key] = response
        _MISSES += 1


def stats() -> dict:
    with _LOCK:
        return {"cache_size": len(_CACHE), "hits": _HITS, "misses": _MISSES}


def reset() -> None:
    global _HITS, _MISSES
    with _LOCK:
        _CACHE.clear()
        _HITS = 0
        _MISSES = 0

"""Transient-error classification and single-model retry with backoff.

STRICTLY ONE MODEL PER RUN: transient 429/5xx conditions are retried on the
SAME model (honoring the API's retryDelay); if a question still fails it is
recorded as an error rather than answered by a different model.
"""
from __future__ import annotations

import re
import time

_RETRY_DELAY = re.compile(r"retry in ([\d.]+)s")
_RETRY_DELAY2 = re.compile(r"retryDelay'?:?\s*'?(\d+)s")


def is_429(err: str) -> bool:
    return bool(err) and ("429" in err or "RESOURCE_EXHAUSTED" in err)


def is_transient(err: str) -> bool:
    if not err:
        return False
    e = err.lower()
    return (is_429(err)
            or "503" in err or "unavailable" in e
            or "high traffic" in e or "overloaded" in e
            or "500" in err or "internal error" in e)


def parse_retry_delay(err: str, default: float) -> float:
    for pat in (_RETRY_DELAY, _RETRY_DELAY2):
        m = pat.search(err or "")
        if m:
            return float(m.group(1)) + 1.0
    return default


def call_with_backoff(runner, item, model: str, *, max_retries: int = 3, max_delay: float = 65.0,
                      base_backoff: float = 5.0, sleep_fn=time.sleep, log_fn=print):
    res = runner(item, model)
    attempts = 0
    while res.error and is_transient(res.error) and attempts < max_retries:
        attempts += 1
        delay = min(parse_retry_delay(res.error, base_backoff), max_delay)
        log_fn(f"  [transient error on {model}; retry {attempts}/{max_retries} after {delay:.0f}s]")
        sleep_fn(delay)
        res = runner(item, model)
    return res

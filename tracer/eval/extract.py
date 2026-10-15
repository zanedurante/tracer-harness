"""Extract the chosen letter (A-H) from a model's free-text reply.

Scoring must not penalize verbosity, so extraction is layered:
  1. regex_letter      explicit markers ("answer is C", "(C)", "**C**", a lone
                       letter line, a LaTeX \\boxed{C}, or a leading "C)").
  2. match_option_text the reply restated an option verbatim instead of its letter.
  3. llm_extract       optional last resort with the SAME model under test.

Returns Extraction(letter, method); letter is None when nothing could be
resolved (counts as wrong).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .mcq import MCQItem

_MARKER = re.compile(
    r"(?:final\s+answer|answer|choice|option|correct(?:\s+(?:answer|option))?|select(?:ed)?|chose|choose|pick)"
    r"\s*(?:is|are|:|=|->|\u2192)?\s*\(?\*{0,2}([A-Ha-h])\b",
    re.IGNORECASE,
)
_BOXED = re.compile(r"\\boxed\s*\{\s*\(?\*{0,2}([A-Ha-h])\*{0,2}\)?\s*\}")
_BRACKET = re.compile(r"\(\s*([A-Ha-h])\s*\)")
_BOLD = re.compile(r"\*\*\s*([A-Ha-h])\s*\*\*")
_LINE = re.compile(r"(?m)^\s*\(?\*{0,2}([A-Ha-h])\*{0,2}[\).:]?\s*$")
_LEAD = re.compile(r"^\s*\(?\*{0,2}([A-Ha-h])\*{0,2}[\).:]")


@dataclass
class Extraction:
    letter: str | None
    method: str  # 'regex' | 'option_text' | 'llm' | 'none'


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def regex_letter(reply: str, valid: str) -> str | None:
    """Markers/brackets/bold/line take the LAST occurrence (the model's
    conclusion); a leading letter takes the first."""
    valid_set = set(valid.upper())
    if not reply:
        return None
    for pat in (_MARKER, _BOXED, _BRACKET, _BOLD, _LINE):
        cands = [m.group(1).upper() for m in pat.finditer(reply)]
        cands = [c for c in cands if c in valid_set]
        if cands:
            return cands[-1]
    m = _LEAD.match(reply)
    if m and m.group(1).upper() in valid_set:
        return m.group(1).upper()
    return None


def match_option_text(reply: str, options: dict[str, str]) -> str | None:
    """If the reply contains an option's text verbatim, return that letter
    (longest option first)."""
    r = _norm(reply)
    if not r:
        return None
    for letter, text in sorted(options.items(), key=lambda kv: -len(kv[1])):
        t = _norm(text)
        if t and t in r:
            return letter
    return None


def _default_llm_extract(reply: str, item: MCQItem, model: str) -> str | None:
    from ..backends.gemini import call_gemini
    opts = "\n".join(f"{L}) {item.options[L]}" for L in sorted(item.options))
    prompt = (
        "A model was asked a multiple-choice question and gave the response below. "
        "Which option letter did it ultimately choose? Reply with ONLY the single "
        "letter, or the word NONE if no choice can be determined.\n\n"
        f"QUESTION: {item.question}\n\nOPTIONS:\n{opts}\n\nRESPONSE:\n{reply}\n\nLETTER:"
    )
    try:
        out = call_gemini(prompt, model=model)
    except Exception:
        return None
    return regex_letter(out, "".join(sorted(item.options)))


def extract_choice(reply: str, item: MCQItem, *, allow_llm: bool = False,
                   llm_model: str | None = None, llm_fn=None) -> Extraction:
    """Parse the chosen letter. Offline regex + option-text by default."""
    valid = "".join(sorted(item.options))
    letter = regex_letter(reply or "", valid)
    if letter:
        return Extraction(letter, "regex")
    letter = match_option_text(reply or "", item.options)
    if letter:
        return Extraction(letter, "option_text")
    if allow_llm:
        if llm_fn is None:
            if not llm_model:
                raise ValueError("allow_llm=True requires llm_model (the model under test)")
            llm_fn = lambda r, it: _default_llm_extract(r, it, llm_model)
        letter = llm_fn(reply, item)
        if letter and letter.upper() in item.options:
            return Extraction(letter.upper(), "llm")
    return Extraction(None, "none")

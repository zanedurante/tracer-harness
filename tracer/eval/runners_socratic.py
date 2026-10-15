"""Socratic Models baseline (HourVideo protocol).

  1. Per-minute captioning of every clip with the published HourVideo caption
     prompt (tracer.eval.socratic_caption; left camera only).
  2. Subsample captions at a fixed linear stride to fit the answering model's
     context window (286 tokens per caption inside a 1M window for Gemini).
  3. Text-only QA: captions + the published HourVideo question prompt.

Caption store: <TRACER_CACHE_ROOT>/caption_cache/<model>/<YYYY_MM_DD>/<clip_stem>.txt
"""
from __future__ import annotations

import datetime as dt
import json
import math
import re
from pathlib import Path

import yaml

from .. import config, data
from .extract import Extraction, extract_choice
from .mcq import MCQItem, render_options
from .runners import RunResult, _result

_CLIP_RE = re.compile(r"^(?P<hh>\d{2})_(?P<mm>\d{2})_(?P<room>[a-zA-Z0-9]+)_(?P<view>lt|rt)$")

CONTEXT_TOKEN_LIMIT = 1_000_000
TOKENS_PER_CAPTION = 286
PROMPT_OVERHEAD_TOKENS = 5_000


def caption_cache_dir() -> Path:
    return config.cache_root() / "caption_cache"


def load_qa_prompt() -> tuple[str, str]:
    with open(config.PROMPTS_DIR / "hour_video_qa.yaml", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return d["prompt"].strip(), d["prompt_last"].strip()


def load_captions(patient_id: str, caption_model: str, caption_dir: Path | None = None
                  ) -> list[tuple[dt.datetime, str]]:
    caption_dir = caption_dir or caption_cache_dir()
    win = data.video_window_utc(patient_id)
    if win is None:
        return []
    start_utc = win[0].to_pydatetime().replace(tzinfo=None)
    end_utc = win[1].to_pydatetime().replace(tzinfo=None)
    room = data.room_for(patient_id)
    model_dir = caption_dir / caption_model
    if not model_dir.is_dir():
        return []
    d = start_utc.date()
    date_strs = []
    while d <= end_utc.date():
        date_strs.append(d.strftime("%Y_%m_%d"))
        d += dt.timedelta(days=1)
    out: list[tuple[dt.datetime, str]] = []
    for ds in date_strs:
        folder = model_dir / ds
        if not folder.is_dir():
            continue
        for txt in sorted(folder.glob("*.txt")):
            m = _CLIP_RE.match(txt.stem)
            if not m or (room and m.group("room") != room):
                continue
            ts = dt.datetime.strptime(ds, "%Y_%m_%d").replace(hour=int(m.group("hh")), minute=int(m.group("mm")))
            if ts < start_utc or ts > end_utc:
                continue
            caption = txt.read_text(encoding="utf-8").strip()
            if caption:
                out.append((ts, caption))
    out.sort(key=lambda t: t[0])
    return out


def compute_subsample_rate(n_captions: int, tokens_per_caption: int = TOKENS_PER_CAPTION,
                           context_limit: int = CONTEXT_TOKEN_LIMIT) -> int:
    max_captions = (context_limit - PROMPT_OVERHEAD_TOKENS) // tokens_per_caption
    return max(1, math.ceil(n_captions / max_captions))


def format_captions(captions: list[tuple[dt.datetime, str]]) -> str:
    return "\n\n".join(f"[{ts.strftime('%Y-%m-%d %H:%M')} UTC]\n{text}" for ts, text in captions)


def format_mcq(item: MCQItem) -> str:
    return f"Test #1:\n{item.question}\n{render_options(item)}"


def parse_socratic_answer(raw: str) -> str | None:
    """The HourVideo prompt asks for a JSON list with YOUR_ANSWER."""
    try:
        text = raw.strip()
        start = text.find("[")
        end = text.rfind("]") + 1
        if start >= 0 and end > start:
            answers = json.loads(text[start:end])
            if answers and isinstance(answers, list):
                ans = answers[0].get("YOUR_ANSWER", "")
                letter = re.match(r"[A-H]", ans.strip())
                if letter:
                    return letter.group(0)
    except (json.JSONDecodeError, KeyError, IndexError, AttributeError):
        pass
    return None


def build_socratic_prompt(item: MCQItem, captions_text: str, qa_prompt: str, qa_prompt_last: str) -> str:
    return f"{qa_prompt}\n\n{captions_text}\n\n{qa_prompt_last}\n\n{format_mcq(item)}"


def run_socratic(item: MCQItem, model: str, *, caption_model: str | None = None,
                 caption_dir: Path | None = None, gemini_fn=None, extract_fn=None, log_fn=None) -> RunResult:
    _log = log_fn or (lambda *a: None)
    cap_model = caption_model or model
    captions = load_captions(item.patient_id, cap_model, caption_dir)
    if not captions:
        return _result("socratic", item, model, "", None,
                       error=f"no captions for {item.patient_id} (model={cap_model})")
    rate = compute_subsample_rate(len(captions))
    sampled = captions[::rate]
    _log(f"  socratic: {item.patient_id} {len(captions)} captions, subsample /{rate} -> {len(sampled)}")
    qa_prompt, qa_prompt_last = load_qa_prompt()
    prompt = build_socratic_prompt(item, format_captions(sampled), qa_prompt, qa_prompt_last)
    if gemini_fn is None:
        from ..backends.gemini import call_gemini as gemini_fn
    try:
        raw = gemini_fn(prompt, model=model)
    except Exception as e:
        return _result("socratic", item, model, "", None, error=f"{type(e).__name__}: {e}")
    letter = parse_socratic_answer(raw)
    if letter and letter in item.options:
        extraction = Extraction(letter=letter, method="socratic_json")
    else:
        extraction = (extract_fn or (lambda r: extract_choice(r, item)))(raw)
    return _result("socratic", item, model, raw, extraction)


SOCRATIC_SYSTEMS = {"socratic": run_socratic}

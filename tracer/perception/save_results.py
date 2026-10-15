"""Per-clip perception output files.

One JSON per (model, prompt, run, clip):

    <perception root>/<model>/<prompt_name>/run_<N>/<date>__<HH_MM>_<room>_<view>.json
    {
        "video_id": "2030_01_01/08_15_r001_lt.mp4",
        "model": "...", "run_id": 0, "fps": 1,
        "prompt_name": "...", "prompt": "...", "user_prompt": "...",
        "raw_response": "```json ... ```",
        "parsed_response": {"video_id": ..., "labels": {...}, "summary": "..."},
        "elapsed_seconds": 12.3, "prompt_tokens": 5266, "response_tokens": 180, "total_tokens": ...,
        "timestamp": "..."
    }

The perception root defaults to <TRACER_DATA_ROOT>/perception.
"""
from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from .. import data
from .prompts import Prompt
from .video import LoadedVideo

# Pricing per 1M tokens (input, output), list prices used for the cost figure.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash-lite":  (0.10, 0.40),
    "gemini-2.5-flash":       (0.30, 2.50),
    "gemini-2.5-pro":         (1.25, 10.00),
    "gemini-3-flash-preview": (0.50, 3.00),
    "gemini-3-pro-preview":   (2.00, 12.00),
    "gemini-3.5-flash":       (0.50, 3.00),
}


def outputs_dir() -> Path:
    return data.perception_dir()


def get_run_dir(model: str, prompt_name: str, run_id: int, root: Path | None = None) -> Path:
    return (root or outputs_dir()) / model / prompt_name / f"run_{run_id}"


def detect_runs(model: str, prompt_name: str, root: Path | None = None) -> list[int]:
    parent = (root or outputs_dir()) / model / prompt_name
    if not parent.exists():
        return []
    ids = []
    for d in parent.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            try:
                ids.append(int(d.name[4:]))
            except ValueError:
                continue
    return sorted(ids)


def next_run_id(model: str, prompt_name: str, root: Path | None = None) -> int:
    existing = detect_runs(model, prompt_name, root)
    return existing[-1] + 1 if existing else 0


def video_id_to_filename(video_id: str) -> str:
    """"2030_01_01/08_15_r001_lt.mp4" -> "2030_01_01__08_15_r001_lt.json"."""
    stem = video_id.replace("/", "__")
    stem = re.sub(r"\.\w+$", "", stem)
    return f"{stem}.json"


def try_parse_json(raw: str) -> dict | None:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


def save_result(video: LoadedVideo, prompt: Prompt, model: str, raw_response: str, *, fps: int = 1,
                elapsed_seconds=None, prompt_tokens=None, response_tokens=None, total_tokens=None,
                run_id: int | None = None, output_dir: Path | None = None) -> Path:
    if output_dir is None:
        output_dir = get_run_dir(model, prompt.name, run_id) if run_id is not None else outputs_dir() / model / prompt.name
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / video_id_to_filename(video.video_id)
    result: dict[str, Any] = {
        "video_id": video.video_id, "model": model, "run_id": run_id, "fps": fps,
        "prompt_name": prompt.name, "prompt": prompt.prompt, "user_prompt": prompt.user_prompt,
        "raw_response": raw_response, "parsed_response": try_parse_json(raw_response),
        "elapsed_seconds": elapsed_seconds, "prompt_tokens": prompt_tokens,
        "response_tokens": response_tokens, "total_tokens": total_tokens,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return out_path

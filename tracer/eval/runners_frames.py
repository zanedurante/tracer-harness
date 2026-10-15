"""Long-context VLM baseline: the model's context is filled with N uniformly
sampled frames from the stay before the MCQ question (paper appendix,
"Baseline Implementations": 475 or 950 frames, left camera, 512 px).

Sampling: tracer.eval.frame_sampler (uniform over every minute clip, left
view preferred). Extraction: tracer.eval.frame_extract (ffmpeg -> JPEG cache).
Delivery: one Gemini call with N inline image parts followed by the prompt.
No tools. Every external call is injectable for offline tests.
"""
from __future__ import annotations

from pathlib import Path

from .extract import extract_choice
from .frame_extract import DEFAULT_JPEG_Q, DEFAULT_MAX_SIDE, extract_frames, read_jpeg
from .frame_sampler import FramePosition, coverage_summary, sample_frames
from .mcq import MCQItem, render_options
from .runners import RunResult, _result

# 950 frames at 512 px is the largest round count that fits Gemini 3.1 Flash
# Lite's million-token window (1,120 tokens per frame).
DEFAULT_N_FRAMES = 950

FRAMES_PREAMBLE = (
    "You are shown a large sequence of frames uniformly sampled from an ICU "
    "patient's surveillance video over their entire stay. The frames are in "
    "chronological order (earliest first). Use them to answer the following "
    "multiple-choice question. You do NOT have access to any other data or "
    "tools — reason from the frames alone. Choose exactly one option even if "
    "uncertain — never refuse. End your reply with a line in exactly this "
    "form:\nAnswer: <LETTER>"
)


def build_frames_prompt(item: MCQItem) -> str:
    return f"{FRAMES_PREAMBLE}\n\nQUESTION: {item.question}\n\nOPTIONS:\n{render_options(item)}"


def _call_gemini_multimodal(prompt: str, jpeg_paths: list[Path], model: str) -> str:
    from google.genai import types
    from ..backends.gemini import get_gemini_client
    client = get_gemini_client()
    parts = [types.Part.from_bytes(data=read_jpeg(p), mime_type="image/jpeg") for p in jpeg_paths]
    parts.append(types.Part(text=prompt))
    resp = client.models.generate_content(model=model, contents=parts)
    return (resp.text or "").strip()


def run_vlm_frames(item: MCQItem, model: str, *, n_frames: int = DEFAULT_N_FRAMES,
                   max_side: int = DEFAULT_MAX_SIDE, jpeg_q: int = DEFAULT_JPEG_Q,
                   sampler_fn=None, extractor_fn=None, gemini_multimodal_fn=None,
                   extract_fn=None, log_fn=None) -> RunResult:
    _sample = sampler_fn or (lambda pid, n: sample_frames(pid, n))
    _extract_files = extractor_fn or (lambda positions: extract_frames(positions, max_side=max_side,
                                                                         jpeg_q=jpeg_q, log_fn=log_fn))
    _call = gemini_multimodal_fn or _call_gemini_multimodal
    _extract_letter = extract_fn or (lambda r: extract_choice(r, item))

    positions: list[FramePosition] = _sample(item.patient_id, n_frames)
    if not positions:
        summary = coverage_summary(item.patient_id) if sampler_fn is None else {"n_clips": 0}
        return _result("vlm_frames", item, model, "", None,
                       error=f"no video coverage for {item.patient_id} (summary={summary})")
    try:
        jpeg_paths = _extract_files(positions)
    except Exception as e:
        return _result("vlm_frames", item, model, "", None, error=f"frame_extract: {type(e).__name__}: {e}")
    try:
        raw = _call(build_frames_prompt(item), jpeg_paths, model)
    except Exception as e:
        return _result("vlm_frames", item, model, "", None, error=f"{type(e).__name__}: {e}")
    return _result("vlm_frames", item, model, raw, _extract_letter(raw))


FRAMES_SYSTEMS = {"vlm_frames": run_vlm_frames}

"""Load a one-minute clip as a Gemini inline video part.

Gemini samples video at 1 fps by default (60 frames per one-minute clip),
which is exactly what the perception prompt expects. ``video_metadata``
(custom fps) is not set on the perception path: the cameras record no audio
track and the API rejects the field on audio-less video.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

from google.genai import types

from .. import config

_INLINE_MAX_BYTES = 100 * 1024 * 1024


class LoadedVideo(NamedTuple):
    video_id: str      # e.g. "2030_01_01/08_15_r001_lt.mp4" (relative to the clip root)
    path: Path
    size_bytes: int
    part: types.Part


def video_root() -> Path:
    return config.data_root() / "video"


def resolve(video_id: str, root: Path | None = None) -> Path:
    full = (root or video_root()) / video_id
    if not full.exists():
        raise FileNotFoundError(f"Video not found: {full}")
    return full


def load_video(path: str | Path, *, fps: int | None = None, root: Path | None = None) -> LoadedVideo:
    p = Path(path)
    root = root or video_root()
    if not p.is_absolute():
        p = resolve(str(path), root)
    if not p.exists():
        raise FileNotFoundError(f"Video file not found: {p}")
    try:
        video_id = p.relative_to(root).as_posix()
    except ValueError:
        video_id = p.name
    video_bytes = p.read_bytes()
    size = len(video_bytes)
    if size >= _INLINE_MAX_BYTES:
        raise ValueError(f"Video is {size / 1e6:.1f} MB, above the inline limit.")
    suffix = p.suffix.lower()
    mime_type = f"video/{suffix.lstrip('.')}" if suffix else "video/mp4"
    kw = dict(inline_data=types.Blob(data=video_bytes, mime_type=mime_type))
    if fps is not None:
        kw["video_metadata"] = types.VideoMetadata(fps=fps)
    return LoadedVideo(video_id=video_id, path=p, size_bytes=size, part=types.Part(**kw))

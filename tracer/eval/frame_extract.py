"""Extract single JPEG frames from minute clips via ffmpeg, with a
content-addressed on-disk cache (<TRACER_CACHE_ROOT>/frame_cache).

Frames are scaled so the longer side is max_side (512 by default, giving
512x288 for 16:9 cameras), matching the paper's long-context baseline.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import shutil
import subprocess
from pathlib import Path

from .. import config
from .frame_sampler import FramePosition

DEFAULT_MAX_SIDE = 512
DEFAULT_JPEG_Q = 4          # ffmpeg -q:v scale is 2..31, lower is better
DEFAULT_WORKERS = 8


class FfmpegMissing(RuntimeError):
    """Raised when ffmpeg is not available."""


def default_cache_dir() -> Path:
    return config.cache_root() / "frame_cache"


def _ffmpeg_bin() -> str:
    exe = shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg
            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            exe = None
    if exe is None:
        raise FfmpegMissing("ffmpeg not found on PATH (or install imageio-ffmpeg).")
    return exe


def _cache_key(pos: FramePosition, max_side: int, jpeg_q: int) -> str:
    h = hashlib.sha1()
    h.update(str(pos.clip_path).encode("utf-8"))
    h.update(f"|{pos.sec_within:.3f}|{max_side}|{jpeg_q}".encode("utf-8"))
    return h.hexdigest()


def _cache_path(pos, max_side, jpeg_q, cache_dir: Path) -> Path:
    key = _cache_key(pos, max_side, jpeg_q)
    return cache_dir / key[:2] / f"{key}.jpg"


def _run_ffmpeg_once(clip: Path, sec: float, out: Path, max_side: int, jpeg_q: int, ffmpeg_bin: str):
    """OUTPUT-side seek (accurate). format=yuvj420p forces JPEG-range 4:2:0,
    which some non-standard-range source clips otherwise reject."""
    out.parent.mkdir(parents=True, exist_ok=True)
    vf = (f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))',"
          f"format=yuvj420p")
    cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-nostdin",
           "-i", str(clip), "-ss", f"{sec:.3f}", "-frames:v", "1",
           "-vf", vf, "-q:v", str(jpeg_q), "-y", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True)


def _run_ffmpeg(clip: Path, sec: float, out: Path, max_side: int, jpeg_q: int, ffmpeg_bin=None) -> None:
    """Clips are nominally 60 s but often run slightly shorter; if `sec`
    overshoots, fall back through earlier offsets inside the same minute."""
    bin_ = ffmpeg_bin or _ffmpeg_bin()
    attempts: list[float] = [sec]
    if sec > 55.0:
        attempts.append(55.0)
    if sec > 30.0:
        attempts.append(30.0)
    attempts.append(1.0)
    last = None
    for s in attempts:
        res = _run_ffmpeg_once(clip, s, out, max_side, jpeg_q, bin_)
        last = res
        if res.returncode == 0 and out.exists() and out.stat().st_size > 0:
            return
    raise RuntimeError(f"ffmpeg failed for {clip.name} @ {sec:.2f}s (tried {attempts}): "
                       f"rc={last.returncode if last else '?'} stderr={(last.stderr if last else '').strip()[:400]}")


def extract_frame(pos: FramePosition, *, max_side: int = DEFAULT_MAX_SIDE, jpeg_q: int = DEFAULT_JPEG_Q,
                  cache_dir: Path | None = None, ffmpeg_bin: str | None = None) -> Path:
    cache_dir = cache_dir or default_cache_dir()
    out = _cache_path(pos, max_side, jpeg_q, cache_dir)
    if out.exists() and out.stat().st_size > 0:
        return out
    _run_ffmpeg(pos.clip_path, pos.sec_within, out, max_side, jpeg_q, ffmpeg_bin)
    return out


def extract_frames(positions: list[FramePosition], *, max_side: int = DEFAULT_MAX_SIDE,
                   jpeg_q: int = DEFAULT_JPEG_Q, cache_dir: Path | None = None,
                   workers: int = DEFAULT_WORKERS, log_fn=None) -> list[Path]:
    """Extract every position (using the cache) in parallel; paths in order."""
    if not positions:
        return []
    cache_dir = cache_dir or default_cache_dir()
    ffmpeg_bin = _ffmpeg_bin()
    n = len(positions)
    out_paths: list[Path | None] = [None] * n

    def _one(i_pos):
        i, pos = i_pos
        return i, extract_frame(pos, max_side=max_side, jpeg_q=jpeg_q, cache_dir=cache_dir, ffmpeg_bin=ffmpeg_bin)

    done = 0
    with cf.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for i, path in pool.map(_one, enumerate(positions)):
            out_paths[i] = path
            done += 1
            if log_fn and (done == 1 or done % 100 == 0 or done == n):
                log_fn(f"    frames extracted: {done}/{n}")
    return [p for p in out_paths if p is not None]


def read_jpeg(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()

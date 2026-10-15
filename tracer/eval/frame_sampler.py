"""Enumerate a stay's minute clips and uniformly sample frame positions across
the entire stay (long-context VLM baseline).

  1. Read the stay's UTC time window from stays.csv.
  2. Walk every UTC date folder inside that window under the stay's video base
     and enumerate every minute clip in [start_utc, end_utc]. Prefer the LEFT
     view ('_lt.mp4'); fall back to the RIGHT view when only that side exists.
  3. Uniformly sample N frame positions across the CONCATENATED timeline of
     the surviving clips (each clip contributes ~60 s), taking bin midpoints.

Pure filesystem walking + arithmetic; no ffmpeg, no model, no network.
"""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from pathlib import Path

from .. import data

CLIP_RE = re.compile(r"^(?P<hh>\d{2})_(?P<mm>\d{2})_(?P<room>[a-zA-Z0-9]+)_(?P<view>lt|rt)\.mp4$")
CLIP_SECONDS = 60


@dataclass(frozen=True)
class FramePosition:
    clip_path: Path
    sec_within: float
    utc_time: dt.datetime


def _patient_window(pat_id: str) -> tuple[dt.datetime, dt.datetime] | None:
    win = data.video_window_utc(pat_id)
    if win is None:
        return None
    s, e = win
    return s.to_pydatetime().replace(tzinfo=None), e.to_pydatetime().replace(tzinfo=None)


def _date_folders(base: Path, start_utc: dt.datetime, end_utc: dt.datetime) -> list[Path]:
    if not base.exists():
        return []
    d = start_utc.date()
    end_d = end_utc.date()
    dates: list[str] = []
    while d <= end_d:
        dates.append(d.strftime("%Y_%m_%d"))
        d += dt.timedelta(days=1)
    return [base / name for name in dates if (base / name).is_dir()]


def enumerate_clips(pat_id: str, *, base_override: Path | None = None,
                    window_override: tuple[dt.datetime, dt.datetime] | None = None
                    ) -> list[tuple[dt.datetime, Path]]:
    """ORDERED list of (utc_minute, chosen_clip_path); prefers lt, falls back to rt."""
    base = base_override or data.video_base(pat_id)
    win = window_override or _patient_window(pat_id)
    if win is None:
        return []
    start_utc, end_utc = win
    room = data.room_for(pat_id)
    out: list[tuple[dt.datetime, Path]] = []
    for folder in _date_folders(base, start_utc, end_utc):
        try:
            names = sorted(p.name for p in folder.iterdir())
        except OSError:
            continue
        by_minute: dict[tuple[str, str], dict[str, str]] = {}
        for name in names:
            m = CLIP_RE.match(name)
            if not m or (room and m.group("room") != room):
                continue
            by_minute.setdefault((m.group("hh"), m.group("mm")), {})[m.group("view")] = name
        try:
            folder_date = dt.datetime.strptime(folder.name, "%Y_%m_%d")
        except ValueError:
            continue
        for (hh, mm), views in by_minute.items():
            ts = folder_date.replace(hour=int(hh), minute=int(mm))
            if ts < start_utc or ts > end_utc:
                continue
            fname = views.get("lt") or views.get("rt")
            if fname:
                out.append((ts, folder / fname))
    out.sort(key=lambda t: t[0])
    return out


def sample_frames(pat_id: str, n_frames: int, *, base_override: Path | None = None,
                  window_override=None) -> list[FramePosition]:
    """n_frames uniformly-spaced FramePositions across every available clip.
    Deterministic. Empty list when the stay has no clips."""
    if n_frames <= 0:
        return []
    clips = enumerate_clips(pat_id, base_override=base_override, window_override=window_override)
    if not clips:
        return []
    n_clips = len(clips)
    total_seconds = n_clips * CLIP_SECONDS
    positions: list[FramePosition] = []
    for i in range(n_frames):
        t = (i + 0.5) * total_seconds / n_frames
        clip_idx = min(int(t // CLIP_SECONDS), n_clips - 1)
        sec_within = t - clip_idx * CLIP_SECONDS
        sec_within = max(0.0, min(sec_within, CLIP_SECONDS - 0.5))
        minute_ts, clip_path = clips[clip_idx]
        positions.append(FramePosition(clip_path=clip_path, sec_within=sec_within,
                                       utc_time=minute_ts + dt.timedelta(seconds=sec_within)))
    return positions


def coverage_summary(pat_id: str, *, base_override: Path | None = None, window_override=None) -> dict:
    clips = enumerate_clips(pat_id, base_override=base_override, window_override=window_override)
    if not clips:
        return {"n_clips": 0, "n_hours": 0.0, "first_utc": None, "last_utc": None}
    first, last = clips[0][0], clips[-1][0]
    return {"n_clips": len(clips), "n_hours": round(len(clips) / 60.0, 2),
            "first_utc": first.strftime("%Y-%m-%d %H:%M"), "last_utc": last.strftime("%Y-%m-%d %H:%M")}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Preview uniform-frame sampling for a stay.")
    ap.add_argument("--pat", required=True)
    ap.add_argument("--n", type=int, default=950)
    args = ap.parse_args()
    print("coverage:", coverage_summary(args.pat))
    positions = sample_frames(args.pat, args.n)
    print(f"sampled {len(positions)} frames (of {args.n} requested)")
    for i in (0, len(positions) // 2, len(positions) - 1):
        if 0 <= i < len(positions):
            p = positions[i]
            print(f"  [{i}] {p.utc_time.isoformat()} <- {p.clip_path.name} @ {p.sec_within:.1f}s")

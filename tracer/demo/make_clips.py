"""Generate the placeholder one-minute clips for the synthetic stay with
ffmpeg (solid colour with a static rectangle, 320x180, 5 fps, ~10 KB each).

The two clips for the planted minute (08:37 local; the sample is authored in
UTC, so the clip name is 08_37) are committed
under demo/data/video/2030_01_01/; run this only to regenerate them or to add
clips for other minutes (--minutes "08_37,08_38"). Requires ffmpeg on PATH or
the imageio-ffmpeg package.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from ..tools.video import ffmpeg_bin


def make_clip(ffmpeg: str, out: Path, colour: str, box_x: int) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi",
           "-i", f"color=c={colour}:s=320x180:d=60:r=5",
           "-vf", f"drawbox=x={box_x}:y=60:w=120:h=70:color=0x9aa8b8@1:t=fill",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(out)]
    subprocess.run(cmd, check=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "demo" / "data" / "video"))
    ap.add_argument("--date", default="2030_01_01")
    ap.add_argument("--room", default="r001")
    ap.add_argument("--minutes", default="08_37", help="comma-separated HH_MM (UTC)")
    args = ap.parse_args(argv)
    ff = ffmpeg_bin()
    if ff is None:
        raise SystemExit("ffmpeg not found: install ffmpeg or `pip install imageio-ffmpeg`")
    for hhmm in args.minutes.split(","):
        for view, colour, x in (("lt", "0x2b3a4a", 40), ("rt", "0x3a2b4a", 160)):
            p = Path(args.out) / args.date / f"{hhmm}_{args.room}_{view}.mp4"
            make_clip(ff, p, colour, x)
            print("wrote", p)


if __name__ == "__main__":
    main()

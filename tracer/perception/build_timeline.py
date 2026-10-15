"""Build a stay's minute-level timeline from per-minute perception outputs.

Input: the per-clip JSON files written by tracer.perception.run_perception
for a stay (both camera views), plus the stay's row in stays.csv and an
optional EHR event list.

Merging rule (paper appendix, "Minute-Level Inference Details"): the two
camera views are merged by a logical OR over the 15 labels, with UNABLE TO
IDENTIFY treated as negative, giving one binary label vector per stay-minute.

Outputs, in the layout the tools read (tracer.data):
  <out>/<pat_id>.npz            one array per hour, key "hour_%04d", shape (60, 2)
                                uint8 = np.packbits of the (60, 15) binary matrix
                                (big-endian bits, padded to 16)
  <out>/<pat_id>_timeline.json  {pat_id, icu_start (local-time-naive
                                "YYYY-MM-DD HH:MM:SS"), n_hours, n_minutes,
                                demographics, encounters, events}
  <out>/activity_key.csv        the label order

`icu_start` is the local time of the first minute of the matrix. Minutes with
no perception output are all-zero rows. Events (medications, CAM-ICU, RASS,
ventilation) are copied from --events JSON if given; the timeline JSON format
of the events is {type, time (ISO, tz-aware), data}.

    python -m tracer.perception.build_timeline --pat-id stay_A \
        --perception-dir data/perception/<model>/<prompt>/run_0 --out data/timelines \
        [--events data/ehr/stay_A_events.json] [--demographics demo.json]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config, data
from ..tools.labels import ACTIVITY_LABELS

_FNAME_RE = re.compile(r"^(?P<date>\d{4}_\d{2}_\d{2})__(?P<hh>\d{2})_(?P<mm>\d{2})_(?P<room>[A-Za-z0-9]+)_(?P<view>lt|rt)\.json$")


def parse_output_filename(name: str):
    m = _FNAME_RE.match(name)
    if not m:
        return None
    y, mo, d = (int(x) for x in m.group("date").split("_"))
    utc = dt.datetime(y, mo, d, int(m.group("hh")), int(m.group("mm")), tzinfo=config.UTC)
    return {"utc": utc, "room": m.group("room"), "view": m.group("view")}


def load_minute_labels(perception_dir: Path, room: str | None = None) -> dict[dt.datetime, dict[str, np.ndarray]]:
    """{utc_minute: {view: binary vector over ACTIVITY_LABELS}}. UNABLE TO
    IDENTIFY (or anything not starting with YES) is negative."""
    out: dict[dt.datetime, dict[str, np.ndarray]] = {}
    for jf in sorted(Path(perception_dir).glob("*.json")):
        meta = parse_output_filename(jf.name)
        if meta is None or (room and meta["room"] != room):
            continue
        with open(jf, encoding="utf-8") as f:
            d = json.load(f)
        parsed = d.get("parsed_response")
        if parsed is None:
            continue
        labels = parsed.get("labels") or {}
        vec = np.array([1 if str(labels.get(l, "")).strip().upper().startswith("YES") else 0
                        for l in ACTIVITY_LABELS], dtype=np.uint8)
        out.setdefault(meta["utc"], {})[meta["view"]] = vec
    return out


def merge_views(per_minute: dict[dt.datetime, dict[str, np.ndarray]]) -> dict[dt.datetime, np.ndarray]:
    return {t: np.bitwise_or.reduce(list(views.values())) for t, views in per_minute.items()}


def pack_matrix(mat: np.ndarray) -> dict[str, np.ndarray]:
    """(n_minutes, n_labels) binary -> {"hour_%04d": (60, 2) uint8}."""
    n = mat.shape[0]
    n_hours = -(-n // 60)
    padded = np.zeros((n_hours * 60, mat.shape[1]), dtype=np.uint8)
    padded[:n] = mat
    return {f"hour_{h:04d}": np.packbits(padded[h * 60:(h + 1) * 60], axis=1) for h in range(n_hours)}


def build(pat_id: str, perception_dir: Path, out_dir: Path, *, events: list | None = None,
          demographics: dict | None = None, encounters: list | None = None, room: str | None = None,
          start_utc: dt.datetime | None = None, end_utc: dt.datetime | None = None) -> dict:
    room = room or data.room_for(pat_id) or None
    win = data.video_window_utc(pat_id)
    if start_utc is None and win is not None:
        start_utc = win[0].to_pydatetime()
    if end_utc is None and win is not None:
        end_utc = win[1].to_pydatetime()
    merged = merge_views(load_minute_labels(perception_dir, room))
    if not merged and start_utc is None:
        raise SystemExit(f"no perception outputs for {pat_id} under {perception_dir}")
    t0 = start_utc or min(merged)
    t1 = end_utc or max(merged)
    t0 = t0.replace(second=0, microsecond=0)
    n_minutes = int((t1 - t0).total_seconds() // 60) + 1
    mat = np.zeros((n_minutes, len(ACTIVITY_LABELS)), dtype=np.uint8)
    n_filled = 0
    for t, vec in merged.items():
        i = int((t - t0).total_seconds() // 60)
        if 0 <= i < n_minutes:
            mat[i] = vec
            n_filled += 1
    packed = pack_matrix(mat)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"{pat_id}.npz", **packed)
    icu_start_local = t0.astimezone(config.local_tz()).replace(tzinfo=None)
    tl = {
        "pat_id": pat_id,
        "icu_start": icu_start_local.strftime("%Y-%m-%d %H:%M:%S"),
        "n_hours": len(packed),
        "n_minutes": n_minutes,
        "demographics": demographics or {},
        "encounters": encounters or [],
        "events": events or [],
    }
    (out_dir / f"{pat_id}_timeline.json").write_text(json.dumps(tl, indent=2), encoding="utf-8")
    key_src = data.activity_key_file()
    key_dst = out_dir / "activity_key.csv"
    if key_src.resolve() != key_dst.resolve():
        shutil.copy(key_src, key_dst)
    return {"pat_id": pat_id, "n_minutes": n_minutes, "n_hours": len(packed), "minutes_with_output": n_filled,
            "icu_start": tl["icu_start"], "per_label_minutes": dict(zip(ACTIVITY_LABELS, mat.sum(axis=0).tolist()))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pat-id", required=True)
    ap.add_argument("--perception-dir", required=True, help="run directory with per-clip JSON outputs")
    ap.add_argument("--out", default=None, help="timelines directory (default: <TRACER_DATA_ROOT>/timelines)")
    ap.add_argument("--events", default=None, help="JSON list of EHR events to embed in the timeline")
    ap.add_argument("--demographics", default=None, help="JSON object")
    ap.add_argument("--encounters", default=None, help="JSON list")
    args = ap.parse_args(argv)
    events = json.loads(Path(args.events).read_text(encoding="utf-8")) if args.events else None
    demo = json.loads(Path(args.demographics).read_text(encoding="utf-8")) if args.demographics else None
    enc = json.loads(Path(args.encounters).read_text(encoding="utf-8")) if args.encounters else None
    out = Path(args.out) if args.out else config.data_root() / "timelines"
    info = build(args.pat_id, Path(args.perception_dir), out, events=events, demographics=demo, encounters=enc)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()

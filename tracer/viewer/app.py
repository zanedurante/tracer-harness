"""Flask app serving the viewer UI over a TRACER data store.

Ported from the internal hospital-video-viewer application (templates/index.html
and its JSON API), with every data source re-pointed at the store layout in
docs/DATA_LAYOUT.md through tracer.data / tracer.tools.common, and the chat
panel driven by tracer.viewer.chat (the harness's own agent loop and tools).

Times: clip folders and names are UTC; everything shown to the user is in the
site's local zone (TRACER_LOCAL_TZ). JSON keys named *_pacific are kept for
compatibility with the template and hold local-zone strings.
"""
from __future__ import annotations

import json
import re
import threading
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request, send_from_directory, stream_with_context

from .. import config, data
from ..harness.configs import ALL_ELEVEN
from ..harness.registry import build_registry
from ..tools import common
from ..tools.labels import ACTIVITY_LABELS
from .chat import stream_chat

CHAT_MODELS = {
    "gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
}
DEFAULT_CHAT_MODEL = "gemini-3.1-flash-lite"

_CLIP_RE = re.compile(r"^(?P<hh>\d{2})_(?P<mm>\d{2})_(?P<room>[A-Za-z0-9]+)_(?P<view>lt|rt)\.mp4$")
_REPORT_RE = re.compile(r"^hour_(?P<date>\d{4}-\d{2}-\d{2})_(?P<hour>\d{1,2})\.md$")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def clean_for_json(obj):
    """Recursively convert numpy / pandas scalars and NaN into JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if obj is pd.NaT:
        return None
    return obj


def _local_name() -> str:
    return config.local_tz_name()


def _to_utc(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def _local_naive_to_utc(ts) -> pd.Timestamp:
    """A local-time-naive timestamp (EHR tables, report names) -> UTC."""
    return pd.Timestamp(ts).tz_localize(_local_name(), ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")


def _local_str(ts_utc) -> str:
    return _to_utc(ts_utc).tz_convert(_local_name()).strftime("%Y-%m-%d %H:%M:%S %Z")


def _window_utc(pat_id: str):
    w = data.video_window_utc(pat_id)
    if not w:
        return None, None
    return _to_utc(w[0]).floor("min"), _to_utc(w[1]).floor("min")


def _stays() -> list[str]:
    df = data.stays_df()
    return [str(p) for p in df["pat_id"].tolist()] if not df.empty else []


def _clip_minute_utc(video: str) -> pd.Timestamp | None:
    try:
        date_folder, name = video.split("/", 1)
        m = _CLIP_RE.match(name)
        if not m:
            return None
        y, mo, d = date_folder.split("_")
        return pd.Timestamp(f"{y}-{mo}-{d} {m['hh']}:{m['mm']}:00", tz="UTC")
    except Exception:
        return None


def list_videos(pat_id: str) -> tuple[list[str], set[str]]:
    """Clips on disk for the stay's room inside its video window: one entry per
    minute (rt preferred, lt otherwise) plus the set of every clip path."""
    base = data.video_base(pat_id)
    room = data.room_for(pat_id)
    start, end = _window_utc(pat_id)
    found: list[str] = []
    if base.exists():
        for date_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            for f in sorted(p.name for p in date_dir.iterdir() if p.is_file() and p.suffix == ".mp4"):
                m = _CLIP_RE.match(f)
                if not m or (room and m["room"] != room):
                    continue
                rel = f"{date_dir.name}/{f}"
                t = _clip_minute_utc(rel)
                if t is None or (start is not None and t < start) or (end is not None and t > end):
                    continue
                found.append(rel)
    available = set(found)
    videos: list[str] = []
    seen: set[str] = set()
    for v in found:
        stem = v[:-len("_lt.mp4")] if v.endswith("_lt.mp4") else v[:-len("_rt.mp4")]
        if stem in seen:
            continue
        seen.add(stem)
        videos.append(f"{stem}_rt.mp4" if f"{stem}_rt.mp4" in available else f"{stem}_lt.mp4")
    return videos, available


def activity_heatmap(pat_id: str) -> tuple[list[dict], dict[str, list[str]]]:
    """[{utc, bits}] per minute of the video window, and {YYYY_MM_DD/HH_MM: [labels]}."""
    try:
        tl = common.load_timeline(pat_id)
        full = common.load_action_matrix(pat_id)
    except FileNotFoundError:
        return [], {}
    t0 = common.icu_start_utc(tl)
    start, end = _window_utc(pat_id)
    heat: list[dict] = []
    minute_to_labels: dict[str, list[str]] = {}
    n_labels = len(ACTIVITY_LABELS)
    for i in range(full.shape[0]):
        t = t0 + pd.Timedelta(minutes=i)
        if start is not None and t < start:
            continue
        if end is not None and t > end:
            break
        row = full[i]
        heat.append({"utc": t.isoformat(), "bits": row.tolist()})
        if row.any():
            minute_to_labels[t.strftime("%Y_%m_%d/%H_%M")] = [ACTIVITY_LABELS[j] for j in range(n_labels) if row[j]]
    return heat, minute_to_labels


def video_activities(videos: list[str], minute_to_labels: dict[str, list[str]]) -> dict[str, list[str]]:
    out = {}
    for v in videos:
        t = _clip_minute_utc(v)
        out[v] = minute_to_labels.get(t.strftime("%Y_%m_%d/%H_%M"), []) if t is not None else []
    return out


def _read_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix in (".pkl", ".pickle"):
            return pd.read_pickle(path)
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def rass_timeline(pat_id: str) -> list[dict]:
    """RASS rows of the flowsheet table inside the video window."""
    df = _read_table(data.flowsheet_file())
    if df is None or df.empty:
        return []
    start, end = _window_utc(pat_id)
    rows = df[df["pat_id"].astype(str) == str(pat_id)]
    rows = rows[rows["row_disp_name"].astype(str).str.contains("RASS", case=False, na=False)]
    out = []
    for _, r in rows.iterrows():
        try:
            t_utc = _local_naive_to_utc(r["recorded_time"])
        except Exception:
            continue
        if (start is not None and t_utc < start) or (end is not None and t_utc > end):
            continue
        try:
            score = float(str(r["meas_value"]).strip())
        except Exception:
            score = None
        out.append({"utc_time": t_utc.isoformat(), "pacific_time": _local_str(t_utc),
                    "rass_score": score, "wall_time": pd.Timestamp(r["recorded_time"]).isoformat()})
    out.sort(key=lambda x: x["utc_time"])
    return out


def medication_data(pat_id: str) -> list[dict]:
    """MAR rows inside the video window, in the shape the template's
    medication lanes expect."""
    df = _read_table(data.medications_file())
    if df is None or df.empty:
        return []
    start, end = _window_utc(pat_id)
    rows = df[df["pat_id"].astype(str) == str(pat_id)]
    out = []
    for _, r in rows.iterrows():
        if pd.isna(r.get("taken_time")):
            continue
        try:
            t_utc = _local_naive_to_utc(r["taken_time"])
        except Exception:
            continue
        if (start is not None and t_utc < start) or (end is not None and t_utc > end):
            continue

        def s(k):
            v = r.get(k)
            return None if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)

        rate = r.get("infusion_rate")
        out.append({
            "order_inst": None, "order_inst_pacific": None, "order_inst_utc": None,
            "taken_time": pd.Timestamp(r["taken_time"]).isoformat(),
            "taken_time_pacific": _local_str(t_utc), "taken_time_utc": t_utc.isoformat(),
            "discon_time": None, "discon_time_pacific": None, "discon_time_utc": None,
            "med_description": s("display_name"), "display_name": s("display_name"),
            "mar_action": s("mar_action"), "sig": s("sig"), "dose_unit": s("dose_unit"), "route": s("route"),
            "infusion_rate": None if rate is None or pd.isna(rate) else float(rate),
            "mar_inf_rate_unit": s("mar_inf_rate_unit"), "mar_duration": None, "mar_duration_unit": None,
            "comments": None, "thera_class_name": s("thera_class_name"),
            "pharm_class_name": s("pharm_class_name"), "mar_department_name": None,
        })
    out.sort(key=lambda x: x["taken_time_utc"])
    return out


def _activity_level(md_text: str) -> str | None:
    m = re.search(r"Overall movement level:\s*\**\s*([A-Za-z]+)", md_text)
    if not m:
        return None
    v = m.group(1).strip().lower()
    return v if v in ("none", "low", "moderate", "high") else None


def reports(pat_id: str) -> list[dict]:
    rdir = common.reports_path(pat_id)
    if not rdir.exists():
        return []
    out = []
    for f in sorted(rdir.glob("hour_*.md")):
        m = _REPORT_RE.match(f.name)
        if not m:
            continue
        hour = int(m["hour"])
        start_utc = _local_naive_to_utc(f"{m['date']} {hour:02d}:00:00")
        try:
            level = _activity_level(f.read_text(encoding="utf-8"))
        except Exception:
            level = None
        out.append({"filename": f.name, "date": m["date"], "hour": hour,
                    "hour_start_utc": start_utc.isoformat(),
                    "hour_end_utc": (start_utc + timedelta(hours=1)).isoformat(),
                    "activity_level": level})
    return out


_CAPTION_CACHE: dict[str, dict[str, str]] = {}


def minute_captions(pat_id: str, videos: list[str], available: set[str]) -> dict[str, str]:
    """{clip path: perception caption} from the per-clip perception outputs."""
    key = f"{data.perception_dir()}::{pat_id}"
    if key in _CAPTION_CACHE:
        caps = _CAPTION_CACHE[key]
    else:
        caps = {}
        pdir = data.perception_dir()
        if pdir.exists():
            for jf in pdir.rglob("*.json"):
                try:
                    rec = json.loads(jf.read_text(encoding="utf-8"))
                    vid = rec.get("video_id")
                    summary = (rec.get("parsed_response") or {}).get("summary")
                    if vid and isinstance(summary, str) and vid not in caps:
                        caps[vid] = summary
                except Exception:
                    continue
        _CAPTION_CACHE[key] = caps
    out = {}
    for v in videos:
        stem = v[:-7]
        for cand in (v, f"{stem}_lt.mp4", f"{stem}_rt.mp4"):
            if cand in caps:
                out[v] = caps[cand]
                break
    return out


def timeline_bounds(pat_id: str) -> dict | None:
    start, end = _window_utc(pat_id)
    if start is None:
        return None
    return {"start_utc": start.isoformat(), "end_utc": end.isoformat(),
            "start_pacific": _local_str(start), "end_pacific": _local_str(end)}


def encounter_info(pat_id: str) -> dict:
    try:
        tl = common.load_timeline(pat_id)
    except FileNotFoundError:
        return {"hosp_service_name": None, "patient_age": None}
    enc = (tl.get("encounters") or [{}])[0] if tl.get("encounters") else {}
    demo = tl.get("demographics") or {}
    return {"hosp_service_name": enc.get("hosp_service_name"),
            "patient_age": demo.get("age") or demo.get("age_years")}


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------
def create_app(pat_id: str | None = None, chat_model: str = DEFAULT_CHAT_MODEL,
               chat_models: dict[str, str] | None = None) -> Flask:
    app = Flask(__name__)
    models = dict(chat_models or CHAT_MODELS)
    if chat_model not in models:
        models[chat_model] = chat_model
    stays = _stays()
    if not stays:
        raise RuntimeError(f"No stays in {data.stays_file()} (TRACER_DATA_ROOT={config.data_root()})")
    state = {"index": stays.index(pat_id) if pat_id in stays else 0}
    registry = build_registry()
    histories: dict[tuple[str, str], list] = {}
    lock = threading.Lock()

    def current() -> str:
        return stays[state["index"]]

    def bundle() -> dict:
        pid = current()
        videos, available = list_videos(pid)
        heat, minute_to_labels = activity_heatmap(pid)
        b = {
            "index": state["index"], "max_index": len(stays) - 1,
            "patient_id": pid, "room": data.room_for(pid),
            "start_video": videos[0] if videos else "", "end_video": videos[-1] if videos else "",
            "videos": videos, "available_video_paths": sorted(available),
            "timeline_bounds": timeline_bounds(pid),
            "rass_timeline": rass_timeline(pid),
            "video_activities": video_activities(videos, minute_to_labels),
            "activity_labels": ACTIVITY_LABELS,
            "activity_heatmap": heat,
            "reports": reports(pid),
            "minute_captions": minute_captions(pid, videos, available),
            "medication_data": medication_data(pid),
            "medication_table": None, "show_medication_table": False,
            "manual_events": [],
            "encounter_info": encounter_info(pid),
            "ok": True,
        }
        return clean_for_json(b)

    def patient_state() -> dict:
        pid = current()
        videos, _ = list_videos(pid)
        return {"index": state["index"], "max_index": len(stays) - 1, "patient_id": pid,
                "room": data.room_for(pid), "start_video": videos[0] if videos else "",
                "end_video": videos[-1] if videos else ""}

    # ---- pages --------------------------------------------------------------
    @app.route("/")
    def index():
        return render_template("index.html", patient_id=current(), local_tz=_local_name())

    # ---- videos -------------------------------------------------------------
    @app.route("/video/<path:filename>")
    def serve_video(filename):
        base = data.video_base(current())
        return send_from_directory(str(base), filename, conditional=True)

    @app.route("/api/videos")
    def api_videos():
        return jsonify(list_videos(current())[0])

    @app.route("/api/video-exists/<path:filename>")
    def api_video_exists(filename):
        return jsonify({"exists": (data.video_base(current()) / filename).exists(), "filename": filename})

    @app.route("/api/video-info/<path:filename>")
    def api_video_info(filename):
        parts = filename.split("/")
        name = parts[-1].replace(".mp4", "").split("_")
        return jsonify({"filename": filename, "date_folder": parts[0] if len(parts) > 1 else "",
                        "video_name": parts[-1],
                        "parsed_info": {"month": name[0] if name else "", "day": name[1] if len(name) > 1 else "",
                                        "room": name[2] if len(name) > 2 else "", "side": name[3] if len(name) > 3 else ""}})

    # ---- bundle / navigation ------------------------------------------------
    @app.route("/api/patient/bundle")
    def api_bundle():
        return jsonify(bundle())

    @app.route("/api/patient", methods=["GET"])
    def api_patient():
        return jsonify(patient_state())

    @app.route("/api/patient/next", methods=["POST"])
    def api_next():
        if state["index"] >= len(stays) - 1:
            return jsonify({"ok": False, "error": "Already at the last stay."}), 400
        state["index"] += 1
        return jsonify(bundle())

    @app.route("/api/patient/prev", methods=["POST"])
    def api_prev():
        if state["index"] <= 0:
            return jsonify({"ok": False, "error": "Already at the first stay."}), 400
        state["index"] -= 1
        return jsonify(bundle())

    @app.route("/api/patient/jump", methods=["POST"])
    def api_jump():
        payload = request.get_json(silent=True) or {}
        idx = payload.get("index")
        pid = payload.get("patient_id")
        if pid in stays:
            state["index"] = stays.index(pid)
        elif isinstance(idx, int) and 0 <= idx < len(stays):
            state["index"] = idx
        return jsonify(bundle())

    @app.route("/api/timeline-bounds")
    def api_bounds():
        b = timeline_bounds(current())
        return (jsonify(b), 200) if b else (jsonify({"error": "no video window for this stay"}), 400)

    @app.route("/api/patient-rass-timeline")
    def api_rass():
        return jsonify(rass_timeline(current()))

    @app.route("/api/patient-video-activities")
    def api_activities():
        pid = current()
        videos, _ = list_videos(pid)
        _, m2l = activity_heatmap(pid)
        return jsonify({"activity_labels": ACTIVITY_LABELS, "video_activities": video_activities(videos, m2l)})

    @app.route("/api/patient-medication-data")
    def api_meds():
        return jsonify(clean_for_json(medication_data(current())))

    @app.route("/api/manual-events")
    def api_manual_events():
        return jsonify([])

    @app.route("/api/patient/reports")
    def api_reports():
        return jsonify({"patient_id": current(), "reports": reports(current())})

    @app.route("/api/patient/report/<filename>")
    def api_report(filename):
        if "/" in filename or "\\" in filename or ".." in filename or not _REPORT_RE.match(filename):
            return jsonify({"error": "Invalid filename"}), 400
        path = common.reports_path(current()) / filename
        if not path.exists():
            return jsonify({"error": "Report not found"}), 404
        raw = path.read_text(encoding="utf-8")
        try:
            import markdown
            html = markdown.markdown(raw)
        except ImportError:
            from html import escape
            html = "<pre>" + escape(raw) + "</pre>"
        return jsonify({"filename": filename, "html": html, "markdown": raw})

    # ---- chat ---------------------------------------------------------------
    @app.route("/chat/info")
    def chat_info():
        return jsonify({"models": models, "default_model": chat_model, "current_patient": current()})

    @app.route("/chat/tools")
    def chat_tools():
        return jsonify({"tools": [{"name": n, "description": registry[n].declaration.description}
                                  for n in ALL_ELEVEN if n in registry]})

    @app.route("/chat/message", methods=["POST"])
    def chat_message():
        payload = request.get_json(silent=True) or {}
        session_id = (payload.get("session_id") or "").strip()
        message = (payload.get("message") or "").strip()
        model = payload.get("model") or chat_model
        if not session_id:
            return jsonify({"error": "Missing session_id."}), 400
        if not message:
            return jsonify({"error": "Empty message."}), 400
        if model not in models:
            return jsonify({"error": f"Unknown model: {model}"}), 400
        pid = current()
        key = (session_id, pid)
        with lock:
            history = list(histories.get(key, []))

        def sse(ev: dict) -> str:
            return f"data: {json.dumps(clean_for_json(ev), default=str)}\n\n"

        def generate():
            try:
                for ev in stream_chat(message, history, pid, model=model, registry=registry):
                    if ev["type"] == "reply":
                        with lock:
                            histories[key] = ev["history"]
                        yield sse({"type": "reply", "text": ev["text"], "tool_calls": ev["tool_calls"]})
                    else:
                        yield sse(ev)
            except Exception as e:  # pragma: no cover - surfaced to the UI
                import traceback
                traceback.print_exc()
                yield sse({"type": "error", "message": f"{type(e).__name__}: {e}"})

        return Response(stream_with_context(generate()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/chat/reset", methods=["POST"])
    def chat_reset():
        payload = request.get_json(silent=True) or {}
        session_id = (payload.get("session_id") or "").strip()
        if not session_id:
            return jsonify({"error": "Missing session_id."}), 400
        with lock:
            histories.pop((session_id, current()), None)
        return jsonify({"ok": True})

    return app

"""Offline tests for the viewer (tracer.viewer): no network, no API key.

Builds the synthetic store the same way tests/test_offline.py does, then
drives the Flask app with its test client: page render, the bundle the
template reads, clips, reports, the tool list, and one streamed chat turn on
the scripted mock backend.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_TMP = Path(tempfile.mkdtemp(prefix="tracer_viewer_test_"))
os.environ["TRACER_DATA_ROOT"] = str(_TMP / "store")
os.environ["TRACER_RESULTS_ROOT"] = str(_TMP / "results")
os.environ["TRACER_CACHE_ROOT"] = str(_TMP / "cache")
os.environ["TRACER_LOCAL_TZ"] = "UTC"

from tracer.demo.__main__ import SAMPLE, PAT_ID, _expand_perception  # noqa: E402


def _build_store() -> Path:
    out = Path(os.environ["TRACER_DATA_ROOT"])
    if (out / "stays.csv").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(SAMPLE / "stays.csv", out / "stays.csv")
    for sub in ("ehr", "video", "reports"):
        shutil.copytree(SAMPLE / sub, out / sub, dirs_exist_ok=True)
    run_dir = out / "perception" / "synthetic" / "perception_prompt" / "run_0"
    _expand_perception(SAMPLE / "labels.csv", run_dir)
    from tracer.perception import build_timeline
    build_timeline.build(PAT_ID, run_dir, out / "timelines",
                         events=json.loads((SAMPLE / "events.json").read_text(encoding="utf-8")),
                         demographics=json.loads((SAMPLE / "demographics.json").read_text(encoding="utf-8")),
                         encounters=json.loads((SAMPLE / "encounters.json").read_text(encoding="utf-8")))
    from tracer.reports import generate_reports
    generate_reports.generate_for_stay(PAT_ID, run_dir, model="mock:report", skip_existing=True, log=lambda m: None)
    return out


def _client():
    _build_store()
    from tracer import data
    data.reset_caches()
    from tracer.tools import common
    common.reset_caches()
    from tracer.viewer.app import create_app
    app = create_app(chat_model="mock:agent", chat_models={"mock:agent": "mock"})
    app.config["TESTING"] = True
    return app.test_client()


def test_page_and_bundle():
    c = _client()
    r = c.get("/")
    assert r.status_code == 200
    assert b'const LOCAL_TZ = "UTC"' in r.data
    assert b"stay_A" in r.data

    b = c.get("/api/patient/bundle").get_json()
    assert b["ok"] and b["patient_id"] == PAT_ID and b["room"] == "r001"
    # the sample ships two views of one minute; the list has one entry per minute, rt preferred
    assert b["videos"] == ["2030_01_01/08_37_r001_rt.mp4"]
    assert set(b["available_video_paths"]) == {"2030_01_01/08_37_r001_lt.mp4", "2030_01_01/08_37_r001_rt.mp4"}
    assert len(b["activity_heatmap"]) == 180 and len(b["activity_heatmap"][0]["bits"]) == len(b["activity_labels"])
    assert b["video_activities"]["2030_01_01/08_37_r001_rt.mp4"] == ["patient_grabbing_medical_equipment",
                                                                    "patient_other_arm_movements"]
    assert b["timeline_bounds"]["start_utc"].startswith("2030-01-01T08:00")
    assert b["timeline_bounds"]["start_pacific"].startswith("2030-01-01 08:00")   # local zone is UTC here
    assert [r["filename"] for r in b["reports"]] == ["hour_2030-01-01_08.md", "hour_2030-01-01_09.md",
                                                     "hour_2030-01-01_10.md"]
    assert b["reports"][0]["hour_start_utc"].startswith("2030-01-01T08:00")
    assert "2030_01_01/08_37_r001_rt.mp4" in b["minute_captions"]
    assert len(b["medication_data"]) >= 2 and all(m["taken_time_utc"] for m in b["medication_data"])
    assert len(b["rass_timeline"]) >= 1 and b["rass_timeline"][0]["rass_score"] is not None
    assert b["index"] == 0 and b["max_index"] == 0


def test_clips_reports_and_tools():
    c = _client()
    assert c.get("/api/videos").get_json() == ["2030_01_01/08_37_r001_rt.mp4"]
    assert c.get("/api/video-exists/2030_01_01/08_37_r001_lt.mp4").get_json()["exists"] is True
    assert c.get("/api/video-exists/2030_01_01/08_38_r001_lt.mp4").get_json()["exists"] is False
    r = c.get("/video/2030_01_01/08_37_r001_rt.mp4")
    assert r.status_code == 200 and (r.content_length or len(r.data)) > 1000
    rep = c.get("/api/patient/report/hour_2030-01-01_08.md").get_json()
    assert rep["filename"] == "hour_2030-01-01_08.md" and "<" in rep["html"] and rep["markdown"]
    assert c.get("/api/patient/report/../stays.csv").status_code in (400, 404)
    tools = [t["name"] for t in c.get("/chat/tools").get_json()["tools"]]
    assert len(tools) == 11 and "query_video_minute" in tools and "get_patient_metadata" in tools
    assert c.get("/chat/info").get_json()["default_model"] == "mock:agent"
    assert c.post("/api/patient/next").status_code == 400     # single stay: navigation disabled


def test_chat_stream_with_mock_agent():
    c = _client()
    r = c.post("/chat/message", json={"session_id": "t", "model": "mock:agent",
                                      "message": "When did the patient first grab medical equipment?"})
    assert r.status_code == 200 and r.mimetype == "text/event-stream"
    events = [json.loads(l[6:]) for l in r.get_data(as_text=True).split("\n\n") if l.startswith("data: ")]
    types = [e["type"] for e in events]
    assert types[-1] == "reply" and "tool_call" in types and "tool_result" in types
    calls = [e["name"] for e in events if e["type"] == "tool_call"]
    assert "find_first_occurrence" in calls
    reply = events[-1]
    assert reply["text"] and len(reply["tool_calls"]) == len(calls)
    assert c.post("/chat/reset", json={"session_id": "t"}).get_json() == {"ok": True}
    assert c.post("/chat/message", json={"session_id": "", "message": "x"}).status_code == 400


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"{name}: ok")
    print("VIEWER TESTS OK")

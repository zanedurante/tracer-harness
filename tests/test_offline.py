"""Offline tests: no network, no API key. Run with `python -m pytest tests`
or `python tests/test_offline.py`.

The tests build a data store from the synthetic sample under a temporary
directory (the same steps as the demo) and exercise the tools, the agent
loop with the mock backend, the scorer, the perception evaluator and the
cohort scripts.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

_TMP = Path(tempfile.mkdtemp(prefix="tracer_test_"))
os.environ["TRACER_DATA_ROOT"] = str(_TMP / "store")
os.environ["TRACER_RESULTS_ROOT"] = str(_TMP / "results")
os.environ["TRACER_CACHE_ROOT"] = str(_TMP / "cache")
os.environ["TRACER_LOCAL_TZ"] = "UTC"  # the synthetic sample is authored in UTC

from tracer.demo.__main__ import SAMPLE, PAT_ID, _expand_perception  # noqa: E402


def _build_store():
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


def test_mcq_rotation_and_extraction():
    from tracer.eval.ingest import load_records
    from tracer.eval.mcq import item_with_gold_at, n_positions, build_item
    from tracer.eval.extract import extract_choice
    recs, skipped = load_records(SAMPLE / "questions.csv")
    assert len(recs) == 3 and not skipped
    assert n_positions(recs[0]) == 5
    for pos in range(5):
        it = item_with_gold_at(recs[0], pos)
        assert it.options[it.correct_letter] == recs[0].gold_answer
        assert it.correct_letter == "ABCDE"[pos]
    it = build_item(recs[1])
    assert extract_choice("blah blah\nAnswer: " + it.correct_letter, it).letter == it.correct_letter
    assert extract_choice("The final answer is $\\boxed{B}$", it).letter == "B"
    assert extract_choice("I think it is " + it.options[it.correct_letter], it).letter == it.correct_letter
    assert extract_choice("no idea", it).letter is None


def test_tools_on_synthetic_store():
    _build_store()
    from tracer import data
    from tracer.tools import common
    data.reset_caches()
    common.reset_caches()
    from tracer.tools import all_tools, THE_ELEVEN
    reg = all_tools()
    assert all(n in reg for n in THE_ELEVEN) and len(THE_ELEVEN) == 11
    md = reg["get_patient_metadata"].handler({}, PAT_ID, "mock:agent")
    assert md["video_start_pt"] == "2030-01-01 08:00" and md["n_hours"] == 3
    ffo = reg["find_first_occurrence"].handler({"labels": ["patient_grabbing_medical_equipment"]}, PAT_ID, "x")
    assert ffo["found"] and (ffo["hour_index"], ffo["minute"]) == (0, 37)
    qa = reg["query_actions"].handler({"labels": ["clinician_patient_contact"]}, PAT_ID, "x")
    per_hour = qa["per_label_per_hour_minutes_active"]["clinician_patient_contact"]
    assert per_hour.index(max(per_hour)) == 1
    aa = reg["get_atomic_actions"].handler({"hour_index": 0}, PAT_ID, "x")
    assert "patient_grabbing_medical_equipment" in aa["minutes"][37]["active_labels"]
    moments = reg["find_moments_of_interest"].handler({}, PAT_ID, "x")
    assert moments["total_moments"] >= 1 and moments["moments"][0]["signal_labels"]
    rep = reg["list_hourly_reports"].handler({}, PAT_ID, "x")
    assert [r["hour"] for r in rep["reports"]] == [8, 9, 10]
    md8 = reg["get_hourly_report"].handler({"date": "2030-01-01", "hour": 8}, PAT_ID, "x")
    assert "Patient Movement Summary" in md8["markdown"]
    meds = reg["query_medications"].handler({"start_pt": "2030-01-01 06:00", "end_pt": "2030-01-03 12:00",
                                             "name_filter": "acetamin"}, PAT_ID, "x")
    assert meds["n_events_total"] == 2
    disc = reg["query_flowsheet"].handler({}, PAT_ID, "x")
    assert any(d["disp_name"] == "Patient RASS Score" for d in disc["available_disp_names"])
    fs = reg["query_flowsheet"].handler({"disp_name_filter": "RASS", "start_pt": "2030-01-01 09:00",
                                         "end_pt": "2030-01-01 09:59"}, PAT_ID, "x")
    assert fs["rows"][0]["value"] == "-2"
    vid = reg["query_video_minute"].handler({"minute_pt": "2030-01-01 08:37", "prompt": "any movement?"}, PAT_ID, "mock:agent")
    assert "answer" in vid["left_view"] and "answer" in vid["right_view"]
    missing = reg["query_video_minute"].handler({"minute_pt": "2030-01-01 08:00", "prompt": "?"}, PAT_ID, "mock:agent")
    assert "error" in missing
    assert str(reg["convert_utc_local"].handler({"dt_str": "2030-01-01 08:37:00", "to_timezone": "Local"}, PAT_ID, "x")).startswith("2030-01-01 08:37")
    assert str(reg["convert_utc_local"].handler({"dt_str": "2030-01-01 08:37:00", "to_timezone": "UTC"}, PAT_ID, "x")).startswith("2030-01-01 08:37")


def test_prompt_assembly():
    _build_store()
    from tracer.harness.prompt import build_system_prompt, build_system_prompt_guided, BLIND_PROMPT
    from tracer.harness.registry import build_registry
    from tracer.harness.configs import resolve, build_named_configs, config_sig
    reg = build_registry()
    assert build_system_prompt([], reg) == BLIND_PROMPT
    p = build_system_prompt_guided(resolve("plus_ehr"), reg, start="2030-01-01 08:00", end="2030-01-01 10:59",
                                   n_hours=3, ehr_start="2030-01-01 06:00", ehr_end="2030-01-03 12:00",
                                   hour_examples="hour_index 0 = 2030-01-01 08:00")
    assert "Working rules" in p and "query_video_minute" not in p and "EHR data (query_medications" in p
    p2 = build_system_prompt_guided(resolve("timeline_only"), reg, start="s", end="e", n_hours=3,
                                    ehr_start="a", ehr_end="b", hour_examples="x")
    assert "EHR data" not in p2 and "start_hour 0 to end_hour 3" in p2
    named = build_named_configs()
    assert config_sig(named["blind"]) == "BLIND"
    assert config_sig(named["stack_hour_minute_ehr_v2_video"]) == config_sig(named["tracer"])
    assert len(named["tracer"]) == 11


def test_agent_sweep_and_scoring():
    out = _build_store()
    from tracer.harness import sweep
    from tracer.eval import position_stats, compare_sweeps
    run_dir = sweep.run_sweep(str(SAMPLE / "questions.csv"), ["text_only", "tracer"], "mock:agent",
                              positions=5, workers=2, run_name="t", quiet=True)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert abs(summary["configs"]["tracer"]["acc"] - 1.0) < 1e-9
    assert abs(summary["configs"]["text_only"]["acc"] - 0.2) < 1e-9
    stats = position_stats.position_stats(position_stats.load_cache(run_dir / "cache.jsonl"))
    assert stats["tracer"]["std"] == 0.0
    m = compare_sweeps.mcnemar_counts(compare_sweeps.load_config("t:text_only"), compare_sweeps.load_config("t:tracer"))
    assert m["n_pairs"] == 15 and m["a_wrong_b_correct"] == 12 and m["a_correct_b_wrong"] == 0


def test_perception_evaluate_on_synthetic():
    out = _build_store()
    from tracer.perception import evaluate
    # annotations = the left view's own labels -> a perfect score on the lt outputs
    ann = _TMP / "ann.csv"
    with open(SAMPLE / "labels.csv", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["view"] == "lt"]
    with open(ann, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id"] + evaluate.LABEL_KEYS)
        for r in rows:
            w.writerow([f"2030_01_01/{r['utc_hhmm']}_r001_lt.mp4"] + [r[k] for k in evaluate.LABEL_KEYS])
    gt = evaluate.load_annotations(ann)
    preds = evaluate.load_predictions("synthetic", "perception_prompt", run_id=0, root=out / "perception")
    results, active = evaluate.compute_metrics(gt, preds, exclude=evaluate.DEFAULT_EXCLUDE)
    assert len(active) == 15 and results["OVERALL (micro-avg)"]["accuracy"] == 1.0
    results, active = evaluate.compute_metrics(gt, preds, exclude=evaluate.RARE_CATEGORIES)
    assert len(active) == 13 and results["OVERALL (micro-avg)"]["accuracy"] == 1.0


def test_cohort_scripts_on_two_synthetic_stays():
    out = _build_store()
    from tracer.cohort import delirium_fr_sensitivity as dfs
    from tracer.cohort import delirium_exact_tests as det
    d = _TMP / "cohort"
    d.mkdir(exist_ok=True)
    for f in ("stay_A.npz", "stay_A_timeline.json", "activity_key.csv"):
        shutil.copy(out / "timelines" / f, d / f)
    # a second stay with no positive CAM-ICU, in which every label fires at least once
    import numpy as np
    z = np.load(d / "stay_A.npz")
    packed = {}
    for k in z.files:
        m = np.unpackbits(z[k], axis=1)[:, :15].copy()
        m[0, :] = 1
        packed[k] = np.packbits(m, axis=1)
    np.savez(d / "stay_B.npz", **packed)
    tl = json.loads((d / "stay_A_timeline.json").read_text(encoding="utf-8"))
    tl["pat_id"] = "stay_B"
    tl["events"] = [e for e in tl["events"] if e["type"] != "cam_icu_delirium_test"] + \
        [{"type": "cam_icu_delirium_test", "time": "2030-01-01T09:00:00+00:00", "data": {"cam_result": "Negative"}}]
    (d / "stay_B_timeline.json").write_text(json.dumps(tl), encoding="utf-8")
    patients, excl, missing = dfs.build_cohort(str(d))
    assert len(patients) == 2 and sum(p["delirium"] for p in patients) == 1
    dfs.main(["--data-dir", str(d), "--bootstrap", "50", "--out", str(_TMP / "fr.json")])
    det.main(["--data-dir", str(d), "--perms", "200", "--out", str(_TMP / "tests.json")])
    assert (_TMP / "fr.json").exists() and (_TMP / "tests.json").exists()


if __name__ == "__main__":
    import traceback
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception:
                failed += 1
                print(f"FAIL {name}")
                traceback.print_exc()
    shutil.rmtree(_TMP, ignore_errors=True)
    sys.exit(1 if failed else 0)

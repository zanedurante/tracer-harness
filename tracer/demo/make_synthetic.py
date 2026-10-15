"""Generate the synthetic sample stay under demo/data.

Everything here is fabricated: a 3-hour "stay" (stay_A, camera r001) on
2030-01-01, 08:00-10:59 site-local time (the sample is authored in UTC, so
local time and the UTC clip names coincide; the demo sets TRACER_LOCAL_TZ=UTC), with
  * demo/data/labels.csv        minute-level label vectors for both camera
                                views (15 actions) plus a fabricated caption
                                per view (expanded by the demo into the
                                per-clip perception output format)
  * demo/data/stays.csv         the stay row (pat_id, UTC window, room)
  * demo/data/ehr/medications.csv, flowsheet.csv   fabricated EHR tables
  * demo/data/events.json       the same EHR events in the timeline-JSON form
  * demo/data/demographics.json, encounters.json
  * demo/data/reports/stay_A/hour_2030-01-01_08.md   one hand-written report
  * demo/data/questions.csv     three multiple-choice questions about the data
  * demo/data/video/2030_01_01/08_37_r001_{lt,rt}.mp4  placeholder clips
                                (generated once with ffmpeg; see make_clips)

Planted facts the questions ask about:
  * the first patient_grabbing_medical_equipment minute is minute 37 (08:37);
  * acetaminophen is given twice (08:20 and 10:05);
  * hour index 1 (09:00-09:59) has the most clinician-contact minutes, and the
    RASS recorded in that hour is -2.

    python -m tracer.demo.make_synthetic [--out demo/data]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

LABELS = [
    "patient_grabbing_medical_equipment", "patient_using_non_medical_objects", "patient_gesticulating",
    "patient_adjusting_bedding", "patient_other_arm_movements", "patient_position_change",
    "patient_other_torso_movement", "patient_looking_at_others", "patient_head_turning",
    "patient_leg_ADL_movement", "patient_leg_in_bed_movement", "patient_communication",
    "clinician_patient_contact", "clinician_patient_object_exchange", "patient_restrained",
]
PAT_ID = "stay_A"
ROOM = "r001"
DATE_UTC = "2030_01_01"
START_UTC_HOUR = 8           # 08:00 site-local (TRACER_LOCAL_TZ=UTC for the sample)
END_UTC = "2030-01-01T10:59:00Z"
TZ_SUFFIX = "+00:00"         # ISO offset of the sample's local zone
N_MINUTES = 180

CAPTION_BITS = {
    "patient_grabbing_medical_equipment": "the patient's hand reaches for the tubing near the face",
    "patient_using_non_medical_objects": "the patient holds a cup",
    "patient_gesticulating": "the patient gestures with one hand",
    "patient_adjusting_bedding": "the patient pulls at the blanket",
    "patient_other_arm_movements": "the patient's arm shifts on the bed",
    "patient_position_change": "the patient rolls partly onto one side",
    "patient_looking_at_others": "the patient looks toward a person at the bedside",
    "patient_head_turning": "the patient turns the head",
    "patient_leg_ADL_movement": "the patient sits up and swings the legs over the bed edge",
    "patient_leg_in_bed_movement": "the patient's legs move under the covers",
    "patient_communication": "the patient appears to speak to a clinician",
    "clinician_patient_contact": "a clinician touches the patient",
    "clinician_patient_object_exchange": "a clinician hands the patient an object",
}


def plan_labels(rng: random.Random) -> list[dict[str, int]]:
    """One binary dict per minute (the merged truth)."""
    mins = [dict.fromkeys(LABELS, 0) for _ in range(N_MINUTES)]

    def on(label, a, b):
        for i in range(a, b + 1):
            mins[i][label] = 1

    # background: sparse arm / head / leg-in-bed movement
    for i in range(N_MINUTES):
        if rng.random() < 0.18:
            mins[i]["patient_other_arm_movements"] = 1
        if rng.random() < 0.12:
            mins[i]["patient_head_turning"] = 1
        if rng.random() < 0.08:
            mins[i]["patient_leg_in_bed_movement"] = 1
    # planted: first grab of medical equipment at minute 37 (08:37), again at 38 and 121
    on("patient_grabbing_medical_equipment", 37, 38)
    on("patient_grabbing_medical_equipment", 121, 121)
    on("patient_other_arm_movements", 36, 38)
    # clinician contact: hour 0 = 3 min, hour 1 = 16 min (the most), hour 2 = 6 min
    on("clinician_patient_contact", 20, 22)
    on("clinician_patient_contact", 70, 85)
    on("clinician_patient_contact", 150, 155)
    on("patient_looking_at_others", 70, 74)
    on("patient_communication", 72, 76)
    on("clinician_patient_object_exchange", 84, 84)
    # a moment of purposeful activity: cup + sitting up, minutes 130-141
    on("patient_using_non_medical_objects", 132, 139)
    on("patient_leg_ADL_movement", 130, 141)
    on("patient_position_change", 130, 131)
    on("patient_adjusting_bedding", 95, 96)
    on("patient_gesticulating", 75, 75)
    return mins


def caption_for(m: dict[str, int], rng: random.Random) -> str:
    bits = [CAPTION_BITS[l] for l in LABELS if m.get(l) and l in CAPTION_BITS]
    if not bits:
        return rng.choice(["The patient lies still in bed; no one else is in the room.",
                           "The patient rests quietly with no visible movement.",
                           "The room is quiet and the patient does not move."])
    return ("During this minute " + "; ".join(bits[:3]) + ".").capitalize()


def write_labels(out: Path, mins, rng):
    with open(out / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["minute", "view", "utc_hhmm"] + LABELS + ["caption"])
        for i, m in enumerate(mins):
            hh = START_UTC_HOUR + i // 60
            hhmm = f"{hh:02d}_{i % 60:02d}"
            for view in ("lt", "rt"):
                vals = []
                for l in LABELS:
                    v = "YES" if m[l] else "NO"
                    # the right camera misses ~15% of the positives (UNABLE TO IDENTIFY);
                    # the OR-merge across views still recovers the planted facts
                    if view == "rt" and m[l] and rng.random() < 0.15 and l != "clinician_patient_contact":
                        v = "UNABLE TO IDENTIFY"
                    vals.append(v)
                w.writerow([i, view, hhmm] + vals + [caption_for(m, rng)])


def write_ehr(out: Path):
    ehr = out / "ehr"
    ehr.mkdir(parents=True, exist_ok=True)
    meds = [
        # taken_time (local), display_name, action, sig, unit, route, rate, rate_unit, pharm class, thera class
        ("2030-01-01 07:30:00", "propofol 10 mg/mL infusion", "New Bag", 0, "", "Intravenous", 20, "mcg/kg/min", "GENERAL ANESTHETICS", "ANESTHETICS"),
        ("2030-01-01 08:20:00", "acetaminophen 325 mg tablet", "Given", 650, "mg", "Oral", "", "", "ANALGESICS, NON-OPIOID", "ANALGESICS"),
        ("2030-01-01 08:40:00", "fentanyl 50 mcg/mL injection", "Bolus", 25, "mcg", "Intravenous", "", "", "OPIOID AGONISTS", "ANALGESICS"),
        ("2030-01-01 09:15:00", "propofol 10 mg/mL infusion", "Rate Change", 0, "", "Intravenous", 30, "mcg/kg/min", "GENERAL ANESTHETICS", "ANESTHETICS"),
        ("2030-01-01 10:05:00", "acetaminophen 325 mg tablet", "Given", 650, "mg", "Oral", "", "", "ANALGESICS, NON-OPIOID", "ANALGESICS"),
        ("2030-01-01 10:30:00", "propofol 10 mg/mL infusion", "Stopped", 0, "", "Intravenous", 0, "mcg/kg/min", "GENERAL ANESTHETICS", "ANESTHETICS"),
        ("2030-01-01 11:40:00", "pantoprazole 40 mg tablet", "Given", 40, "mg", "Oral", "", "", "PROTON PUMP INHIBITORS", "GASTROINTESTINAL"),
    ]
    with open(ehr / "medications.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pat_id", "display_name", "taken_time", "mar_action", "sig", "dose_unit", "route",
                    "infusion_rate", "mar_inf_rate_unit", "pharm_class_name", "thera_class_name"])
        for t, name, act, sig, unit, route, rate, runit, pc, tc in meds:
            w.writerow([PAT_ID, name, t, act, sig, unit, route, rate, runit, pc, tc])
    rows = []
    for h, mm in ((8, 5), (9, 10), (10, 15)):
        rows.append((f"2030-01-01 {h:02d}:{mm:02d}:00", "Patient RASS Score", {8: "0", 9: "-2", 10: "-2"}[h], ""))
    for i in range(0, 180, 15):
        t = f"2030-01-01 {8 + i // 60:02d}:{i % 60:02d}:00"
        rows.append((t, "Heart Rate", str(78 + (i * 7) % 11), "bpm"))
        rows.append((t, "SpO2", str(95 + (i // 15) % 4), "%"))
    rows.append(("2030-01-01 08:30:00", "CAM-ICU", "Negative", ""))
    rows.append(("2030-01-01 10:45:00", "CAM-ICU", "Positive", ""))
    rows.sort()
    with open(ehr / "flowsheet.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pat_id", "row_disp_name", "recorded_time", "meas_value", "units"])
        for t, name, val, unit in rows:
            w.writerow([PAT_ID, name, t, val, unit])
    # timeline-JSON events (tz-aware ISO local time)
    events = []
    for t, name, act, sig, unit, route, rate, runit, pc, tc in meds:
        if pc in ("GENERAL ANESTHETICS", "OPIOID AGONISTS"):
            events.append({"type": "med_sedation", "time": t.replace(" ", "T") + TZ_SUFFIX,
                           "data": {"display_name": name, "mar_action": act, "route": route,
                                    "sig": sig, "dose_unit": unit, "infusion_rate": rate or None,
                                    "mar_inf_rate_unit": runit, "medication_id": name.split()[0],
                                    "pharm_class_name": pc, "thera_class_name": tc}})
    for t, name, val, unit in rows:
        if name == "Patient RASS Score":
            events.append({"type": "rass", "time": t.replace(" ", "T") + TZ_SUFFIX, "data": {"rass_score_int": int(val)}})
        elif name == "CAM-ICU":
            events.append({"type": "cam_icu_delirium_test", "time": t.replace(" ", "T") + TZ_SUFFIX, "data": {"cam_result": val}})
    events.append({"type": "ventilation", "time": "2030-01-01T06:30:00" + TZ_SUFFIX, "data": {"ventilation_type": "Invasive"}})
    events.sort(key=lambda e: e["time"])
    (out / "events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
    (out / "demographics.json").write_text(json.dumps({"age_group": "60-69", "sex": "unspecified"}, indent=2), encoding="utf-8")
    (out / "encounters.json").write_text(json.dumps([{"hosp_admsn_time": "2030-01-01T06:00:00", "hosp_disch_time": "2030-01-03T12:00:00",
                                                        "icu_days": 1.5}], indent=2), encoding="utf-8")


def write_stays(out: Path):
    with open(out / "stays.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pat_id", "start_time_utc", "end_time_utc", "room", "video_dir", "timeline_dir", "max_display_hours"])
        w.writerow([PAT_ID, f"2030-01-01T{START_UTC_HOUR:02d}:00:00Z", END_UTC, ROOM, "", "", ""])


def write_questions(out: Path):
    qs = [
        (PAT_ID, "Q: At what time (site-local, HH:MM) did the patient first grab medical equipment during the stay?\n\nA: 08:37",
         "08:12\n09:04\n09:41\n10:23"),
        (PAT_ID, "Q: How many doses of acetaminophen were given during the stay?\n\nA: 2", "0\n1\n3\n4"),
        (PAT_ID, "Q: During which hour index (hours from admission) did clinician contact occupy the most minutes, and what RASS score was recorded during that hour?\n\nA: Hour 1; RASS -2",
         "Hour 0; RASS 0\nHour 2; RASS -2\nHour 1; RASS 0\nHour 2; RASS -1"),
    ]
    with open(out / "questions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["PAT_ID", "Original Question", "Distractors (4)"])
        for row in qs:
            w.writerow(row)


REPORT_08 = """Observation window: 2030-01-01 08:00 to 2030-01-01 08:59 (60 minutes)

1. Patient Movement Summary
    - Overall movement level: low
    - Most common movement types: other arm movements, head turning
    - Arm movement observed at min 36-38 (both views), including a reach toward the tubing at min 37-38; head turning at scattered minutes (min 4, 11, 29, 47)
    - No discrepancies between left and right camera views beyond the right view marking min 38 as unable to identify

2. Patient-Clinician Interaction Summary
    - Interaction occurred: clinician contact min 20-22
    - No communication, restraint, or object exchange observed

3. Significant Moments of Interest
    - First reach toward medical equipment at min 37, after stillness min 23-35
    - Prolonged stillness min 39-58

4. Medication Context
    - 08:20 acetaminophen 325 mg tablet, 650 mg, Oral (Given)
    - 08:40 fentanyl 50 mcg/mL injection, 25 mcg, Intravenous (Bolus)
    - propofol 10 mg/mL infusion at 20 mcg/kg/min, Intravenous: continuous infusion started at 07:30 (still running)

5. Activity-Medication Alignment (observational only)
    - Arm movement and the reach toward tubing at min 36-38 preceded the fentanyl bolus at 08:40 (min 40)
    - Movement at min 36-38 while propofol infusing

6. Brief Summary
    The patient was mostly still during this hour with low overall movement. A clinician was at the bedside for three minutes early in the hour. The only notable event was a brief reach toward the tubing near the face at min 37-38, followed by a fentanyl bolus. Both camera views were available for every minute.
"""


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[2] / "demo" / "data"))
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    mins = plan_labels(rng)
    write_labels(out, mins, rng)
    write_stays(out)
    write_ehr(out)
    write_questions(out)
    rep = out / "reports" / PAT_ID
    rep.mkdir(parents=True, exist_ok=True)
    (rep / "hour_2030-01-01_08.md").write_text(REPORT_08, encoding="utf-8")
    print(f"wrote synthetic sample to {out}")


if __name__ == "__main__":
    main()

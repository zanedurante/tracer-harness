"""The atomic-action taxonomy as the tools see it: label order (from
activity_key.csv) and the rubric description of each label, inlined into the
label parameters of the timeline tools.
"""
from __future__ import annotations

import pandas as pd

from .. import config, data


def _load_activity_labels() -> list[str]:
    path = data.activity_key_file()
    if not path.exists():
        path = config.TAXONOMY_FILE
    df = pd.read_csv(path).sort_values("idx")
    return [str(row["label"]) for _, row in df.iterrows()]


ACTIVITY_LABELS: list[str] = _load_activity_labels()

# Source of truth for atomic-action descriptions (matches the annotation
# rubric used to label the perception benchmark).
LABEL_DESCRIPTIONS: dict[str, str] = {
    "patient_grabbing_medical_equipment": "Grabbing medical equipment.",
    "patient_using_non_medical_objects": "Grabbing or using other objects (e.g. drinking with a cup, using a towel, eating food, using a phone).",
    "patient_gesticulating": "Gesticulating.",
    "patient_adjusting_bedding": "Grabbing or adjusting pillows or bedsheets.",
    "patient_other_arm_movements": "Any arm motions (e.g. hand / wrist / arm fidgeting, arm movements, or lifting/dropping of arms).",
    "patient_position_change": "Patient changing their body position (e.g. lying flat to leaning on one side, sitting up to lying down).",
    "patient_other_torso_movement": "Other torso movements (e.g. quick or repeated body movements or seizures).",
    "patient_looking_at_others": "The patient is looking at others in the room.",
    "patient_head_turning": "The patient turns their head (e.g. head turning or movements).",
    "patient_leg_ADL_movement": "Leg movements associated with activities of daily living (e.g. standing up, taking a seat, walking, transferring from bed to chair).",
    "patient_leg_in_bed_movement": "Foot or leg movements in bed (e.g. leg fidgeting, leg repositioning, or repeated leg movements).",
    "patient_communication": "The patient is speaking or nodding to someone.",
    "clinician_patient_contact": "A person is touching the patient or is holding/using a tool or device that has direct contact with the patient.",
    "clinician_patient_object_exchange": "A person is handing the patient something, or the patient is handing a person something.",
    "patient_restrained": "A person is restraining the patient.",
}


def labels_inline() -> str:
    """Bullet list of `label: description` for inlining into FunctionDeclarations."""
    return "\n".join(
        f"  - {lbl}: {LABEL_DESCRIPTIONS.get(lbl, '(no description)')}"
        for lbl in ACTIVITY_LABELS
    )

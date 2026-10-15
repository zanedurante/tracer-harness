"""Cohort statistics (paper appendix, "Delirium and Atomic Action Frequency
Analysis"): minute-weighted and equal-patient-weight frequency ratios with
patient-level bootstrap intervals (delirium_fr_sensitivity), and Mann-Whitney
U, permutation and composite tests (delirium_exact_tests).

Both read only the per-stay timeline bundles (<pat_id>.npz +
<pat_id>_timeline.json + activity_key.csv) from <TRACER_DATA_ROOT>/timelines
or --data-dir; they run no model inference and emit aggregate statistics only.
"""

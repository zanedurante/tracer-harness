# Data layout

Everything is resolved from `TRACER_DATA_ROOT` (default `./data`) by
`tracer/data.py`. The demo builds a complete store of this shape under
`demo/out/` from `demo/data/`, so `python -m tracer.demo` followed by a look
at `demo/out/` is the quickest way to see every file.

```
$TRACER_DATA_ROOT/
  stays.csv                                   one row per stay (see below)
  video/<YYYY_MM_DD>/<HH_MM>_<room>_<lt|rt>.mp4   one-minute clips, two cameras;
                                              folders and file names are UTC
  perception/<model>/<prompt>/run_<N>/<YYYY_MM_DD>__<HH_MM>_<room>_<view>.json
                                              per-clip perception outputs (both views)
  timelines/activity_key.csv                  label order (idx,label); must equal the
                                              packaged tracer/perception/taxonomy/activity_key.csv
  timelines/<pat_id>.npz                      bit-packed minute-level action matrix
  timelines/<pat_id>_timeline.json            stay metadata + EHR event list
  reports/<pat_id>/hour_<YYYY-MM-DD>_<HH>.md  hourly reports (local-time date and hour)
  ehr/medications.csv                         MAR export (all stays)
  ehr/flowsheet.csv                           flowsheet export (all stays)
```

Times: clip folders/names are UTC; everything the tools and prompts expose is
site-local time (`TRACER_LOCAL_TZ`, an IANA zone name, default `UTC`; set it
to the site's zone). `icu_start`, `taken_time`, `recorded_time`
and report names are local-time-naive.

## stays.csv

| column | required | meaning |
|---|---|---|
| `pat_id` | yes | stay identifier (any string; the questions' `PAT_ID` must match) |
| `start_time_utc`, `end_time_utc` | yes | recorded video window, ISO 8601 UTC |
| `room` | yes | camera/room token in the clip names (`<HH_MM>_<room>_<view>.mp4`) |
| `video_dir` | no | per-stay clip root overriding `video/` |
| `timeline_dir` | no | per-stay directory overriding `timelines/` |
| `max_display_hours` | no | clamp the exposed window to this many hours after the start |

## Perception outputs

Written by `tracer.perception.run_perception`; `parsed_response` is the
model's JSON: `{"video_id": ..., "labels": {<15 labels>: YES|NO|UNABLE TO IDENTIFY}, "summary": "..."}`.
`tracer.perception.build_timeline` OR-merges the two views per minute
(UNABLE TO IDENTIFY = negative) into the matrix.

## Timeline files

`<pat_id>.npz`: one array per hour, key `hour_%04d`, shape `(60, 2)` uint8 =
`np.packbits` of the `(60, 15)` binary matrix (big-endian bits, padded to 16),
hour 0 starting at `icu_start`.

`<pat_id>_timeline.json`:

```json
{"pat_id": "stay_A",
 "icu_start": "2030-01-01 08:00:00",        local-time-naive, minute 0 of the matrix
 "n_hours": 3, "n_minutes": 180,
 "demographics": {...}, "encounters": [{"hosp_admsn_time": "...", "hosp_disch_time": "...", "icu_days": 1.5}],
 "events": [{"type": "med_sedation", "time": "2030-01-01T07:30:00+00:00", "data": {...}}, ...]}
```

Event types the code reads: `med_sedation` / `med_cns_adjunct` (report
context and `find_moments_of_interest`; `data` carries `display_name`,
`mar_action`, `route`, `sig`, `dose_unit`, `infusion_rate`,
`mar_inf_rate_unit`, `medication_id`), `rass` (`data.rass_score_int`),
`cam_icu_delirium_test` (`data.cam_result` Positive/Negative) and
`ventilation` (`data.ventilation_type` Invasive/NIV) for the cohort scripts.

## EHR tables

`ehr/medications.csv`: `pat_id, display_name, taken_time, mar_action, sig,
dose_unit, route, infusion_rate, mar_inf_rate_unit, pharm_class_name,
thera_class_name`.

`ehr/flowsheet.csv`: `pat_id, row_disp_name, recorded_time, meas_value,
units` (a `.parquet` or `.pkl` with the same columns is also accepted).

## Question bank

CSV with columns `PAT_ID`, `Original Question` (`"Q: <question>\n\nA: <gold>"`)
and `Distractors (4)` (one per line). Row index = question id (`qid`).
Optional extra columns are kept in `MCQRecord.extra`. For per-category
breakdowns (`tracer.eval.breakdown`) supply a separate CSV with `idx`,
`paper_category`, `patient`, `n_options`.

## Clip-benchmark annotations

CSV with `video_id` and one YES/NO column per label (`tracer.perception.evaluate`).

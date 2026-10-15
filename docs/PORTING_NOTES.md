# Porting notes

This repository is a re-packaged copy of the research code. The
agent loop, tools, prompt assembly, sweep drivers, evaluation and cohort
statistics preserve the behaviour of the code that produced the paper's
numbers. Where an internal dependency was cut, it was replaced by a minimal
shim; every such change is listed here.

## Agent-visible field names kept as-is

* `get_patient_metadata` returns the recorded admission time under the key
  `icu_start_admit_DO_NOT_CITE` (with a `note` field explaining it). The key
  name is part of the tool output the agent reads: the suffix is the
  instruction that keeps the agent from citing the admission time, which may
  pre-date the video window, as a timestamp. Every reported run used this
  exact key, so it is kept unchanged for fidelity to the paper's tool
  interface; it carries no data beyond the admission time itself.

## Verification

* `python -c "import tracer"`, the offline tests (`python -m pytest tests`)
  and `python -m tracer.demo` were run with the system Python 3.12 and in a
  fresh virtual environment with only `requirements.txt` installed
  (`import ok`; demo `DEMO OK`, about 4 s with system Python and 11 s in the
  fresh venv on the machine used to build the release).

## Data access shims

* **`tracer/data.py`** replaces the internal path resolver. The stay table
  (`stays.csv`) replaces an internal CSV with the same columns
  (`pat_id, start_time_utc, end_time_utc, room, video_dir, timeline_dir`);
  the default locations are now all under `TRACER_DATA_ROOT` instead of a
  home-directory mount and an in-repo `data/` folder.
* **Display clamp.** The original code clamped the exposed video window of
  one specific stay to 80 hours after its start (a hard-coded stay id). This
  is now the optional `max_display_hours` column of `stays.csv`, so the same
  clamp can be reproduced per stay without a hard-coded identifier.
* **EHR tables.** `query_medications` read one CSV from the raw EHR extract
  (only the columns it uses are required); `query_flowsheet` read a pickle
  pre-filtered to the two benchmark stays with a `PAT_ID` column. Both now
  read `<data root>/ehr/medications.csv` and `<data root>/ehr/flowsheet.csv`
  (parquet/pickle also accepted; `PAT_ID` is renamed to `pat_id`). Query
  logic, caps, filters and output fields are unchanged. `str.contains` is
  called with `regex=False`, so a filter containing regex metacharacters
  (e.g. a parenthesis) is matched literally rather than raising.
* **Room fallback.** `find_moments_of_interest` fell back to a hard-coded room
  token when a stay had no room; it now uses the stay's `room` (empty string
  if absent). The default stay id argument was removed (it is always passed).
* **Timezone.** All conversions use `TRACER_LOCAL_TZ` (an IANA zone name,
  default `UTC`; set it to the site's zone). The site's zone name was removed
  from tool names, prompt text and defaults; the `convert_utc_local` tool and
  the `_pt` parameter suffix (`start_pt`, `minute_pt`, ...) are as quoted in
  the paper.
* **Credentials.** The Gemini key was read from a file in the repository;
  it is now `GEMINI_API_KEY` (or `GEMINI_API_KEY_FILE`). The Vertex AI
  project id, the GCS staging bucket and the vLLM endpoint were hard-coded or
  defaulted to internal values; they are `GOOGLE_CLOUD_PROJECT`,
  `GEMINI_VIDEO_BUCKET` (no default) and `OPENAI_BASE_URL` (default
  `http://localhost:8000/v1`).
* **Caches.** Re-encoded clips, extracted frames and Socratic captions go to
  `TRACER_CACHE_ROOT` instead of directories inside the package.

## Tool set

* The application also exposed `list_patients` and `plot_activity_timeline`.
  `plot_activity_timeline` only emitted a directive for a front-end chart and
  had no effect on answers; it is not among the paper's eleven tools and was
  dropped. `list_patients` is registered for convenience but is in no
  configuration.
* Configuration names are the paper's row labels (`configs/tool_configs.yaml`).
  The historical names used in the original sweep caches (`blind`,
  `structured`, `narrative`, `stack_hour_minute`, `stack_hour_minute_ehr_v2`,
  `stack_hour_minute_ehr_v2_video`, ...) are kept as aliases that resolve to
  the identical tool set, so cache signatures (`sorted tool names joined by
  '+'`) are unchanged.
* The video tool's description defaults to variant D2 (hedged), as in every
  reported run; D1/D3 remain selectable with `VIDEO_DESC_VARIANT` for the
  description A/B test described in the code.

## Agent loop

* **Prompt variant.** The original loop defaulted the Gemini backend to the
  plain prompt and the OpenAI backend to the guided prompt unless
  `AGENT_PROMPT_VARIANT` was set; the reported runs all used the guided
  variant. The release defaults every backend to `guided`
  (`AGENT_PROMPT_VARIANT=base` restores the plain prompt). The former alias
  `open` is accepted for `guided`.
* **Context caps (OpenAI-compatible path).** The paper reports tool results
  truncated to 16,000 characters and the transcript to 100,000 characters.
  The source file's defaults were 32,000 / 180,000 with the same environment
  overrides (`TOOL_RESULT_CHARS`, `TRANSCRIPT_CHARS`); the release defaults
  to the paper's values. If your runs used the larger caps, set the
  variables. Gemini receives every payload in full on either setting.
* **Mock backend.** `mock:*` model names route to `tracer/backends/mock.py`,
  an offline scripted policy used by the demo and tests. It reuses the
  OpenAI-compatible loop unchanged; the video tool short-circuits to a canned
  description when the perceiver is a mock model (after checking the clip
  exists). No other code path is affected.
* Token usage on the OpenAI path is now accumulated into the same `tokens`
  record the Gemini path fills (the original recorded per-turn usage only in
  `turn_meta`).
* Errors returned by the agent (`error` set) are now surfaced by
  `tracer.eval.runners.run_tool_agent` as errored results instead of being
  scored as an empty reply.

## Evaluation

* The original `run_eval` had an additional `ehr` mode (the whole EHR export
  as text in context) that is not reported in the paper; it was dropped.
* `tracer.eval.score` adds `mcnemar_counts` (continuity-corrected chi-square
  as reported in the appendix, plus the exact binomial p-value the original
  `mcnemar` computed) and `position_mean_std`; `tracer.eval.compare_sweeps`
  generalizes the paper's paired-test scripts, which hard-coded run
  directories and configurations.
* `tracer.eval.breakdown` generalizes the per-category breakdown script,
  which hard-coded the benchmark's category file and result-file names; it
  takes both on the command line. The category assignments of the benchmark
  are not released with the questions.
* `tracer.eval.greedy` sampled the search subset with hard-coded per-stay
  counts for the two benchmark stays; it now takes `--per-patient`.
* The standalone drivers (`tracer/eval/standalone/`) hard-coded the two
  stays' rooms and windows and a home-directory caption path; they read a
  `stays.csv` and take the caption directory and prompt paths as arguments.
  Their logic is otherwise unchanged.
* The frames bundle no longer records a git SHA (the release has one commit).

## Perception and reports

* `tracer/perception` re-packages the perception runner with its four
  helpers (prompt loader, clip loader, result writer, evaluator). The
  runner's video-cropping options (`--crop`, `--center-crop`), which were
  exploratory and not used for reported numbers, were dropped, as was the
  machine-local `config.yaml`; the clip root and annotations file are
  arguments / `TRACER_DATA_ROOT`.
* `tracer.perception.build_timeline` is new: the production timeline builder
  was tied to the site's EHR extract format and patient index. It implements
  the merge rule stated in the paper (OR over views, UNABLE TO IDENTIFY as
  negative) and writes exactly the file formats the tools read
  (`docs/DATA_LAYOUT.md`). EHR events must be supplied as a JSON list in the
  timeline-JSON event format; no EHR ingestor is released.
* `tracer.reports.generate_reports` is the production generator with the
  patient-index / batch bookkeeping removed; it takes a perception run
  directory and reads events from the stay's timeline JSON (or `--events`).
  Both the `type`/`time` and the older `event_type`/`timestamp` spellings of
  events are accepted. Prompt construction is unchanged.
* The local-inference driver used for the open-weight perception models on
  the clip benchmark (Qwen3-VL, Qwen3.5, Gemma-4 with the perception prompt)
  lived in a separate cluster workspace and is not part of this release; the
  Socratic captioning driver (`standalone/socratic_caption_vlm.py`) shows the
  same loading and generation pattern.

## Cohort statistics

* `delirium_fr_sensitivity` and `delirium_exact_tests` are unchanged except
  that the bundle directory is `--data-dir` (default
  `<TRACER_DATA_ROOT>/timelines`) and the output path is `--out`.

## Not released

* Video, EHR data, the minute-level records of the cohort, the hourly
  reports of the benchmark stays, the clip-benchmark annotations, the
  question bank and its category assignments, result caches, transcripts.
* The internal browsing application from which the tools originated.

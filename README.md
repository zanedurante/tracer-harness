# TRACER

Code for the paper *TRACER: A Multimodal Harness for Patient Care in
Hospitals* (Durante, Tang, Chandrasegaran, et al., Stanford University).

TRACER answers clinical questions about a hospital stay from continuous bedside video and the electronic health record. A perception VLM
labels every minute of video with 15 atomic actions, a report LM summarizes
every hour, and an agent LM answers questions by calling eleven tools over the
minute-level timeline, the hourly reports, the EHR, and the raw video. None of
the models are trained on patient data, so the harness itself can be released
in full.

## Contents

- `tracer/tools/` the eleven tools and their declarations
- `tracer/harness/` the agent loop, prompt assembly, tool registry, and the
  named tool configurations from the paper (`configs/tool_configs.yaml`)
- `prompts/` every prompt, verbatim: perception, hourly report, agent system
  prompt with the guided addendum, text-only prompt, user and forced-choice
  messages, video tool description and preamble, HourVideo caption and
  question prompts, baseline prompts, SimpleQA grader
- `tracer/perception/` the taxonomy (`taxonomy/activity_key.csv`, rubric text
  in `tracer/tools/labels.py`), the perception runner, the 13-category
  evaluation, and the timeline builder
- `tracer/reports/` the hourly report generator
- `tracer/eval/` question ingestion, gold-position rotation, answer
  extraction, scoring with paired McNemar tests, and the text-only,
  long-context-frames, and Socratic baselines with their standalone drivers
- `tracer/cohort/` delirium frequency ratios, bootstrap intervals,
  Mann-Whitney and permutation tests
- `tracer/backends/` the model backends: Gemini (API key or Vertex AI), any
  OpenAI-compatible server such as vLLM, and the offline mock used by the
  demo and tests
- `tracer/quickstart/` the one-command run on the sample video (clips,
  perception, timeline, report, viewer)
- `tracer/viewer/` the browser viewer: video player, atomic-action timeline,
  minute captions, hourly reports, and the agent's chat panel
- `tracer/demo/` the synthetic-stay generator and the offline demo driver
- `tracer/config.py` and `tracer/data.py` the environment variables
  (`TRACER_DATA_ROOT`, `TRACER_LOCAL_TZ`, `GEMINI_API_KEY`, ...) and the
  store readers
- `configs/serving_settings.yaml` sampling and serving settings for every
  model in the paper
- `demo/` the synthetic stay (`demo/data/`) and the pre-computed store for
  the one-hour sample video (`demo/quickstart/`: atomic actions, minute
  captions, hourly report)
- `tests/` offline tests of the pipeline, the agent loop and the viewer

`docs/DATA_LAYOUT.md` gives the file formats the harness expects.
`docs/PORTING_NOTES.md` lists where an internal dependency was replaced by a
shim. `docs/RESULTS.md` maps each table and figure to the script behind it.

## What is not here

The video, the EHR data, the cohort's minute-level records and hourly
reports, the 397 annotated benchmark clips, and the benchmark questions. All
of it is protected health information or identifies a real patient's stay,
and none of it can be shared under the study's approval. Result caches and
agent transcripts are left out for the same reason.

## Install

Python 3.10 or newer. The code reads `prompts/` and `configs/` from the
repository root, so install from a clone rather than as a standalone
package.

```bash
git clone https://github.com/zanedurante/tracer-harness.git && cd tracer-harness
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests -q                              # offline; no key or network needed
```

`requirements.txt` covers everything in the README: the harness, tools,
evaluation and cohort statistics, both model backends, the viewer and the
quickstart. The same set is available as `pip install -e ".[viewer]"`.

To call Gemini, set one of:

```bash
export GEMINI_API_KEY=...                              # Google AI Studio key
export GEMINI_API_KEY_FILE=~/.keys/gemini.key          # or a file holding the key
export GEMINI_AUTH=adc GOOGLE_CLOUD_PROJECT=<project>  # or Vertex AI with application-default credentials
```

Open-weight models are reached through any OpenAI-compatible server
(`OPENAI_BASE_URL=http://localhost:8000/v1` for vLLM). `tracer/config.py`
lists every environment variable.

`ffmpeg` is used to cut the quickstart video into clips and to extract
frames for the perception and baseline runs. The `imageio-ffmpeg` wheel in
`requirements.txt` bundles a copy, so nothing has to be installed
separately; a working `ffmpeg` on the `PATH` is used first when present.
`google-cloud-storage` is only needed for GCS staging on the Vertex AI path,
and `torch transformers pillow accelerate av qwen-vl-utils` only for the
local-model drivers in `tracer/eval/standalone/`.

## Demo

```bash
python -m tracer.demo
```

No API key or network access needed. `demo/data/` holds a made-up three-hour
stay: per-minute labels and captions for two cameras, a medication list,
flowsheet rows, two placeholder clips, one hourly report, and three
multiple-choice questions. The demo builds the timeline from the labels,
fills in the missing hourly reports with a template model, runs the real
agent loop and all eleven tools with a scripted mock backend over the three
questions in three tool configurations with five-position gold rotation, and
scores the result with the same code the paper used. It takes a few seconds
and ends with `DEMO OK`. Details in `demo/README.md`.

To run it against a real model instead:

```bash
export GEMINI_API_KEY=...
python -m tracer.demo --model gemini-3.1-flash-lite

# Vertex AI
export GEMINI_AUTH=adc GOOGLE_CLOUD_PROJECT=<your project>

# any vLLM / OpenAI-compatible server
export OPENAI_BASE_URL=http://localhost:8000/v1
python -m tracer.demo --model qwen3.5-9b
```

## Run TRACER on the sample video (one command)

The quickstart runs the whole pipeline on a one-hour synthetic ICU video and
opens the viewer: the video player, the minute-level atomic-action timeline
under it (rows grouped and coloured as in the paper's taxonomy figure), the
minute captions, the hourly report, and a chat panel in which the paper's
agent answers questions with its eleven tools (you can watch each tool call
and click any cited timestamp to jump to that minute of video).

```bash
pip install -r requirements.txt            # adds flask, markdown, gdown, imageio-ffmpeg
export GEMINI_API_KEY=...                  # Google AI Studio key; needed for the chat panel
python -m tracer.quickstart                # downloads the sample video (1.7 GB) if it is not present
# then open http://127.0.0.1:5000/
```

The sample video (`demo/video/TRACER_SAMPLE_VIDEO.mp4`, one hour, 1920x1080,
15 fps, from [Google Drive](https://drive.google.com/file/d/15muTWlrIKeBEQpxo4loyLeIuFsifCZGU/view))
was generated with Veo, Google's video generation model, so no real
patient, clinician, or record appears in it. It needs `ffmpeg` on the `PATH` (or the `imageio-ffmpeg`
wheel, which the quickstart falls back to).

**What is pre-computed.** `demo/quickstart/` is committed with everything the
pipeline produced for the sample video, so the first run costs no model
calls: the per-minute perception outputs (one JSON per clip under
`perception/gemini-3.1-flash-lite/perception_prompt/run_0/`, each holding
the atomic-action labels for the minute and its one-to-two-sentence
caption), the minute-level timeline built from them
(`timelines/stay_A.npz` and `stay_A_timeline.json`), the hourly report
(`reports/stay_A/hour_2026-10-16_08.md`), and the synthetic EHR tables and
`stays.csv`, trimmed to the hour of video. Only the clips are missing, and
the quickstart cuts them from the video on the first run. With the cache in
place, the Gemini key is used only by the chat panel (and the agent's video
tool); without a key the viewer still shows the timeline, captions, and
report, and the chat panel reports the missing key when asked a question.

Every model role (perception VLM, report LM, agent, and the agent's video
tool) defaults to `gemini-3.1-flash-lite`; change all of them with `--model`
or the report model alone with `--report-model`. What the command does, in
order (each step is skipped when its outputs exist, so it can be re-run or
resumed):

1. splits the video into one-minute clips in the store layout
   (`demo/quickstart/video/<date>/<HH_MM>_r001_lt.mp4`, re-encoded at 540p so
   each clip stays under the inline-video size limit; about 90 s);
2. writes `stays.csv` and copies the synthetic, non-PHI EHR tables and events
   from `demo/data`, shifted onto the stay's clock (2026-10-16 08:00 UTC by
   default; `--start-utc` changes it) and trimmed to the span of the video,
   so the EHR tools and the reports' medication context line up with it;
3. runs minute-level perception with the paper's prompt for any clip whose
   output is not already in the store (`tracer.perception.run_perception`;
   none for the sample, about 60 calls and 20 s at 8 workers if you
   `--model` something else or delete the cache);
4. builds the minute-level timeline (`tracer.perception.build_timeline`);
5. generates the hourly report(s) (`tracer.reports.generate_reports`) when
   none exists or the perception outputs have changed;
6. starts the viewer (`tracer.viewer`) at <http://127.0.0.1:5000/>.

Useful flags: `--video my_hour.mp4` for your own continuous recording (one
camera; the second-camera slot simply reports no clip), `--max-minutes 5`
for a quick first run on a new video, `--local-tz America/Los_Angeles` to
display the site's clock, `--no-serve` to build the store without starting
the viewer, `--port`, and `--fresh` to delete the store (including the
committed cache) and recompute everything with the model. The store is a
complete `TRACER_DATA_ROOT`, so afterwards
`TRACER_DATA_ROOT=demo/quickstart python -m tracer.viewer` reopens it and
`python -m tracer.harness.sweep` can run the agent over your own questions.

The viewer (`tracer/viewer/`) is a port of the internal bedside video viewer
used in the study, with every data source re-pointed at the store layout and
the chat panel driven by `tracer/viewer/chat.py`: the same tools, handlers,
guided working rules and 25-call budget as the paper's agent, with free-form
answers and a conversation that persists across turns.

## Running on your own data

Point `TRACER_DATA_ROOT` at a directory laid out as in `docs/DATA_LAYOUT.md`:

```
$TRACER_DATA_ROOT/
  stays.csv                                     pat_id, start_time_utc, end_time_utc, room [, video_dir, timeline_dir, max_display_hours]
  video/<YYYY_MM_DD>/<HH_MM>_<room>_<lt|rt>.mp4  one-minute clips per camera (UTC folders and names)
  perception/<model>/<prompt>/run_<N>/*.json    per-clip perception outputs (written by the runner)
  timelines/<pat_id>.npz, <pat_id>_timeline.json, activity_key.csv   the action-label store
  reports/<pat_id>/hour_<YYYY-MM-DD>_<HH>.md    hourly reports
  ehr/medications.csv, ehr/flowsheet.csv        medication administrations and flowsheet rows
```

The agent, the baselines, and the cohort scripts read only this store. Set
`TRACER_LOCAL_TZ` to the site's IANA clock zone (default `UTC`); every
local-time conversion in the tools and prompts uses it.

### Minute-level perception

```bash
# every clip of a stay, one video id per line (e.g. 2030_01_01/08_37_r001_lt.mp4)
python -m tracer.perception.run_perception gemini-3.1-flash-lite --videos clips.txt --workers 32 --run 0

# the clip benchmark, five repeated runs at API defaults
for r in 0 1 2 3 4; do python -m tracer.perception.run_perception gemini-3.1-flash-lite --annotations labels.csv --run $r; done
python -m tracer.perception.evaluate gemini-3.1-flash-lite --annotations labels.csv --aggregate --csv
```

`evaluate` reports per-category accuracy, sensitivity, specificity, precision,
and F1, plus micro and macro averages. By default it scores all 15
categories, as in the paper's perception table; `--exclude-rare` drops
`patient_other_torso_movement` and `patient_restrained` for the 13-category
evaluation used in the per-category figure and the cohort analysis. Outputs go to
`<data root>/perception/<model>/<prompt>/run_<N>/`.

Then build the timeline the tools read (OR-merge of the two views, UNABLE TO
IDENTIFY treated as negative, bit-packed per hour):

```bash
python -m tracer.perception.build_timeline --pat-id stay_A \
    --perception-dir $TRACER_DATA_ROOT/perception/gemini-3.1-flash-lite/perception_prompt/run_0 \
    --events stay_A_events.json --encounters stay_A_encounters.json
```

### Hourly reports

```bash
python -m tracer.reports.generate_reports --pat-id stay_A \
    --perception-dir $TRACER_DATA_ROOT/perception/gemini-3.1-flash-lite/perception_prompt/run_0 \
    --model gemini-3.1-flash-lite            # --dry-run to plan; --save-prompt FILE to inspect one prompt
```

The generator groups both cameras' per-minute outputs into local-time hours,
attaches that hour's medication events and running infusions from the
timeline JSON, and sends `prompts/report_hourly_v1.yaml`.

### The agent

Configuration names follow the rows of the paper's harness-ablation table
(`configs/tool_configs.yaml`). Every configuration includes the two scaffold
tools, `get_patient_metadata` and `convert_utc_local`.

| Table row | `--configs` name | tool families |
|---|---|---|
| Text-only | `text_only` | none |
| Timeline only | `timeline_only` | timeline |
| Reports only | `reports_only` | report |
| EHR only | `ehr_only` | EHR |
| Video only | `video_only` | video |
| w/ timeline | `w_timeline` | timeline |
| + reports | `plus_reports` | timeline + report |
| + EHR | `plus_ehr` | timeline + report + EHR |
| + video (TRACER) | `tracer` | all eleven tools |

```bash
python -m tracer.harness.sweep --csv questions.csv \
    --configs text_only,timeline_only,reports_only,ehr_only,video_only,plus_reports,plus_ehr,tracer \
    --model gemini-3.1-flash-lite --positions 5 --workers 6 --run-name lite_table3
python -m tracer.eval.position_stats --run lite_table3                     # mean +/- std over gold positions
python -m tracer.eval.compare_sweeps --a lite_table3:plus_ehr --b lite_table3:tracer   # paired McNemar
python -m tracer.eval.breakdown --categories cats.csv --results "TRACER=results/lite_table3/cache.jsonl:tracer"
```

Each (config, question, gold position) trial is cached in
`<TRACER_RESULTS_ROOT>/<run>/cache.jsonl`, so interrupted runs resume. The
agent gets 25 turns and 25 tool calls, then is forced to answer with no tools.
The guided prompt variant is the default (`AGENT_PROMPT_VARIANT=base` for the
plain one), and the video tool's description is variant D2
(`VIDEO_DESC_VARIANT`). To serve the video tool with a different perceiver
than the agent, as in the paper's open-weight agent rows, set
`VIDEO_TOOL_MODEL=gemini-3.1-flash-lite`. Open-weight agents are addressed by
their served name on `OPENAI_BASE_URL`:

```bash
OPENAI_BASE_URL=http://localhost:8000/v1 VIDEO_TOOL_MODEL=gemini-3.1-flash-lite \
    python -m tracer.harness.sweep --csv questions.csv --configs tracer --model qwen3.5-9b --run-name qwen35_9b
```

The open-ended variant (no options, SimpleQA grader) is
`python -m tracer.harness.sweep_open --csv questions.csv --model gemini-2.5-flash --judge gemini-2.5-pro`.
Greedy forward selection over tools is `python -m tracer.harness.greedy --csv questions.csv`.

### Baselines

**Text-only** is the agent loop with zero tools (`--configs text_only`). A
single-turn version is `python -m tracer.eval.run_eval --mode mcq --systems blind`.

**Long-context VLM (475 / 950 frames)**:

```bash
python -m tracer.eval.run_eval --csv questions.csv --mode frames --model gemini-3.1-flash-lite --n-frames 950 --positions 5
```

samples frames uniformly over the stay's clips (left camera, 512 px) and sends
them inline. To run elsewhere, package once with
`python -m tracer.eval.bundle --csv questions.csv --name n950 --n-frames 950`
and run `tracer/eval/standalone/frames_qa_gemini.py` (Gemini) or
`run_local_vlm.py` (a local HuggingFace VLM, greedy decoding, bfloat16;
`--frame-stride 2` for 475 frames).

**Socratic Models (HourVideo protocol)**: caption every minute with
`python -m tracer.eval.socratic_caption --patients stay_A,stay_B --model gemini-3.1-flash-lite`
(or `standalone/socratic_caption_vlm.py` for a local VLM, 30 frames per
clip), then answer with
`python -m tracer.eval.run_eval --csv questions.csv --mode socratic --model gemini-3.1-flash-lite --positions 5`
(or `standalone/socratic_qa_gemini.py`, `socratic_qa_vlm.py`). Captions are
subsampled at a fixed linear stride to fit the answering model's context (286
tokens per caption in a 1M window for Gemini). To answer over the Gemini
captions with an open-weight model:
`standalone/socratic_qa_openai.py --model <served> --ctx <max_model_len>`.

### Cohort statistics

With every stay's timeline bundle under `<data root>/timelines/`:

```bash
python -m tracer.cohort.delirium_fr_sensitivity --bootstrap 10000 --seed 20260723
python -m tracer.cohort.delirium_exact_tests --perms 1_000_000 --seed 20260723
```

The first computes minute-weighted and equal-patient-weight frequency ratios
for every action between CAM-ICU-positive and negative patients, with
patient-level bootstrap intervals and the group characteristics table. The
second adds Mann-Whitney U tests (Holm and Benjamini-Hochberg adjusted),
permutation tests, and the composite comparisons.

## Model settings

`configs/serving_settings.yaml` is the machine-readable version of the
paper's appendix. In short:

- Gemini models in every role (perception benchmark, cohort run, report
  model, agent, baselines) are called with no generation config, so
  temperature, top-p, top-k, output length, and thinking follow the API
  defaults. The five repeated perception runs differ only in output
  directory. Gemini agents have no output cap.
- Open-weight perception models on the clip benchmark: bfloat16, temperature
  0.6, top-p 0.95, top-k 20, at most 2,048 new tokens, thinking off.
- Open-weight baselines (long-context frames and Socratic, Qwen3-VL 8B):
  greedy decoding in bfloat16; the Socratic captioner sees 30 frames per
  minute.
- Agent loop, every model: 25 turns and 25 tool calls, then a forced answer;
  up to three retries of a malformed call; tool results truncated to 16,000
  characters and the transcript to 100,000; guided prompt throughout.
- Open-weight agents under vLLM: Qwen3.5 9B in fp16 (65,536-token context,
  thinking off via the chat template); Qwen3-VL 8B in bfloat16 (131,072-token
  context); gpt-oss-20b in bfloat16 (131,072-token context, reasoning effort
  low, harmony tool parser); Gemma-4 31B in bfloat16 (131,072-token context,
  thinking off, Gemma-4 tool parser). Output capped at 8,192 tokens per turn. Gemma-4's
  vision tower is present but unused because the video tool is served by
  Gemini 3.1 Flash Lite.
- Video tool: Gemini agents get the original one-minute clip with a 5 fps
  sampling hint (the 1 fps default misses brief motion, at 4.5x the token
  cost); open-weight agents get the clip re-encoded at 2 fps with the longer
  side at 512 px.
- Cohort perception run: Gemini 3.1 Flash Lite at API defaults with the same
  perception prompt as the clip benchmark, over every minute of every stay.
  Report model: Gemini 3.1 Flash Lite at API defaults, over the two benchmark
  stays only.

## Layout

```
tracer/harness/     agent loop, prompt assembly, registry, named configs, sweep drivers
tracer/tools/       the eleven tools (scaffold, timeline, report, EHR, video) and the taxonomy descriptions
tracer/perception/  perception runner, evaluation, timeline builder, taxonomy
tracer/reports/     hourly report generator
tracer/eval/        question ingestion, rotation, extraction, scoring, baselines, standalone drivers
tracer/cohort/      delirium frequency ratios, bootstrap, exact tests
tracer/backends/    Gemini, OpenAI-compatible (vLLM), and the offline mock backend
tracer/demo/        synthetic sample generator and the demo driver
tracer/quickstart/  one-command run on the sample video (clips, perception, timeline, report, viewer)
tracer/viewer/      browser viewer: player, atomic-action timeline, reports, agent chat panel
prompts/            every prompt verbatim
configs/            tool configurations (paper rows) and serving settings
demo/               synthetic sample data, the demo entry point, and the pre-computed store for the sample video
docs/               DATA_LAYOUT.md, RESULTS.md, PORTING_NOTES.md
tests/              offline tests (python -m pytest)
```

## License

Apache License 2.0 (`LICENSE`), copyright the TRACER authors. The HourVideo
caption and question prompts (`prompts/hour_video_caption.yaml`,
`prompts/hour_video_qa.yaml`) are reproduced from Chandrasegaran et al.,
*HourVideo* (NeurIPS 2024). The SimpleQA grader in `tracer/eval/judge.py` is
reproduced from openai/simple-evals (MIT). The synthetic data in `demo/data/`
is fabricated. Model weights and APIs (Gemini, Qwen, Gemma, gpt-oss) are
subject to their providers' terms.

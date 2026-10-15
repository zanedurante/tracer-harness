# Demo

`python -m tracer.demo` (or `python demo/run_demo.py`) runs the whole
pipeline on the synthetic, non-PHI sample in `demo/data/` with no API key and
no network, in a few seconds:

1. expands the sample into a data store (`demo/out/`) in the layout of
   `docs/DATA_LAYOUT.md`, including per-clip perception outputs for both
   cameras generated from `labels.csv`;
2. builds the minute-level timeline (`tracer.perception.build_timeline`);
3. generates hourly reports with the offline template model (`mock:report`),
   keeping the hand-written `reports/stay_A/hour_2030-01-01_08.md`;
4. runs the real agent loop, prompt assembly and all eleven tools with the
   scripted mock backend (`mock:agent`) over the three questions in
   `questions.csv`, five gold positions each, in three tool configurations
   (`text_only`, `timeline_only`, `tracer`);
5. scores the run (mean +/- std over positions, paired McNemar) and prints one
   agent transcript.

Expected result: `text_only` 20% (it always answers A, so it is right only
when the gold sits in slot A; this is the positional prior the rotation is
designed to expose), `timeline_only` 47%, `tracer` 100%.

The sample: a fabricated 3-hour stay (`stay_A`, camera `r001`, 2030-01-01
08:00-10:59 site-local time) with minute-level label vectors and captions for
two camera views (`labels.csv`), a medication list and flowsheet rows
(`ehr/`), the same EHR events in timeline-JSON form (`events.json`), two
placeholder one-minute clips for the planted minute (`video/2030_01_01/08_37_r001_{lt,rt}.mp4`,
solid colour, generated with ffmpeg), one hand-written hourly report, and
three questions. The sample is authored in UTC (the demo sets
`TRACER_LOCAL_TZ=UTC`, so clip names and local times coincide).
`python -m tracer.demo.make_synthetic` regenerates the
sample; `python -m tracer.demo.make_clips` regenerates the clips (needs
ffmpeg or `pip install imageio-ffmpeg`).

To run the same steps with a real model: `python -m tracer.demo --model
gemini-3.1-flash-lite` with `GEMINI_API_KEY` set (the agent and the video
tool then call Gemini on the synthetic store; the placeholder clips contain no
patient, so the video tool will report an empty room). With a vLLM server:
`OPENAI_BASE_URL=http://localhost:8000/v1 python -m tracer.demo --model qwen3.5-9b`.

## Quickstart on the sample video

`python -m tracer.quickstart` (see the main README) runs the real pipeline on
the one-hour synthetic ICU video `demo/video/TRACER_SAMPLE_VIDEO.mp4`
(generated with Veo; downloaded on first use; not committed), writes a complete store to
`demo/quickstart/`, and opens the viewer. The synthetic EHR tables and events
in `demo/data/` are reused as the stay's record, shifted onto the video's
clock (2026-10-16 08:00 UTC) and trimmed to its hour, so the agent can answer
questions that cross video and EHR.

`demo/quickstart/` is committed with the outputs of one full run with
`gemini-3.1-flash-lite`, so the viewer opens without any model calls:

- `perception/gemini-3.1-flash-lite/perception_prompt/run_0/*.json`: one file
  per minute with the atomic-action labels (the paper's taxonomy) and the
  minute caption written by the perception model;
- `timelines/stay_A.npz`, `stay_A_timeline.json`: the minute-level timeline
  built from those outputs, with the demographics, encounter and events;
- `reports/stay_A/hour_2026-10-16_08.md`: the hourly report
  (`.built_from` records the inputs it was generated from; the quickstart
  regenerates the report when they change);
- `ehr/medications.csv`, `ehr/flowsheet.csv`, `stays.csv`, `clips.txt`.

Only `video/` (the one-minute clips) is left out and cut from the video on
the first run. A Gemini API key is needed for the chat panel and for any
step whose outputs are missing (another `--model`, `--fresh`, or your own
`--video`). Unlike the offline demo above, the atomic-action labels come from
the perception model, not from `labels.csv`, so the three planted questions
in `questions.csv` do not apply to it.

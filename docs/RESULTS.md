# Paper table / figure -> script

All commands assume `TRACER_DATA_ROOT` points at a data store laid out as in
`docs/DATA_LAYOUT.md`, a question bank CSV, and credentials for the model
(`GEMINI_API_KEY`, or `GEMINI_AUTH=adc` + `GOOGLE_CLOUD_PROJECT`, or
`OPENAI_BASE_URL` for a vLLM server). Every accuracy in the paper is the mean
over gold placements (`--positions 5`).

| Paper | Script | Notes |
|---|---|---|
| Table 1 (main benchmark, TRACER vs baselines) | `tracer.harness.sweep --configs text_only,tracer`; `tracer.eval.run_eval --mode frames`; `tracer.eval.socratic_caption` + `run_eval --mode socratic`; `tracer.eval.position_stats` | text-only = agent loop with zero tools |
| Table 2 (perception, 397-clip benchmark) | `tracer.perception.run_perception <model> --annotations labels.csv --run N` x5; `tracer.perception.evaluate <model> --annotations labels.csv --aggregate` | all 15 categories (default; `--exclude-rare` gives the 13-category figure set); open-weight perception models were run with a separate cluster driver (not released, see PORTING_NOTES) |
| Table 3 (harness ablation, single and cumulative toolsets) | `tracer.harness.sweep --configs text_only,timeline_only,reports_only,ehr_only,video_only,plus_reports,plus_ehr,tracer --positions 5`, per agent model; `tracer.eval.breakdown --categories cats.csv --results ...` for the category columns | config names = `configs/tool_configs.yaml`; average the three agent models |
| Table 4 (fixed perceiver, varying agent) | same sweep with `VIDEO_TOOL_MODEL=gemini-3.1-flash-lite` and `--model <served name>` on `OPENAI_BASE_URL` | perception records are the Gemini run in any case |
| Appendix, Tool Interface (Table: tools) | `tracer/tools/*.py`, `prompts/system_prompt.md`, `prompts/video_tool.md` | |
| Appendix, Baseline Implementations | `tracer.eval.bundle` + `tracer/eval/standalone/frames_qa_gemini.py` / `run_local_vlm.py` (frames); `tracer.eval.socratic_caption`, `standalone/socratic_qa_gemini.py`, `socratic_caption_vlm.py`, `socratic_qa_vlm.py` (Socratic) | |
| Appendix, Table: Socratic with Gemini captions, open-weight answerers | `tracer/eval/standalone/socratic_qa_openai.py --model <served> --ctx <n>` | tokens per caption measured on `/tokenize` |
| Appendix, Model Instantiation / Sampling and serving settings | `configs/serving_settings.yaml` | |
| Appendix, Perception Benchmark Details (cost figure) | `tracer.perception.evaluate` profiling (`MODEL_PRICING`, elapsed) | plotting script not released |
| Appendix, Hourly Report Faithfulness | `tracer.reports.generate_reports --save-prompt` to inspect a prompt; audit was manual | |
| Appendix, Harness Across More Agent Models; Per-Model Cumulative Ablation; Single Evidence Sources | `tracer.harness.sweep` per model; `tracer.eval.compare_sweeps --a run:cfgA --b run:cfgB` for the paired McNemar tests | |
| Appendix, Answer-Position Rotation | `tracer.eval.position_stats --run <run>` (per-position accuracies) | |
| Appendix, Delirium and Atomic Action Frequency Analysis (Tables: tests, FR, cohort) | `tracer.cohort.delirium_fr_sensitivity`; `tracer.cohort.delirium_exact_tests` | reads the per-stay timeline bundles of the cohort |
| Appendix, Token Cost of Raw Video | derived from `prompt_tokens` recorded by the perception runner and the frames baseline | |
| Open-ended (SimpleQA-graded) variant | `tracer.harness.sweep_open --judge gemini-2.5-pro` | |
| Greedy tool selection (diagnostic) | `tracer.harness.greedy` | |

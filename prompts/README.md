# Prompts

Every prompt used in the paper, verbatim.

| File | Used by | Paper section |
|---|---|---|
| `perception_prompt.yaml` | `tracer.perception.run_perception` (minute-level perception, clip benchmark and cohort run) | Appendix, Minute-Level Inference Details |
| `report_hourly_v1.yaml` | `tracer.reports.generate_reports` (system prompt + user turn) | Appendix, Minute-Level Inference Details |
| `system_prompt.md` | `tracer.harness.prompt` (agent system prompt, guided addendum, text-only prompt, user message, forced-choice message) | Appendix, Tool Interface |
| `video_tool.md` | `tracer.tools.video` (tool description D1/D2/D3 and the perception preamble) | Appendix, Tool Interface |
| `hour_video_caption.yaml` | Socratic baseline step 1 (HourVideo caption prompt) | Appendix, Baseline Implementations |
| `hour_video_qa.yaml` | Socratic baseline step 2 (HourVideo question prompt) | Appendix, Baseline Implementations |
| `baselines.md` | frames-only prompt, single-turn text-only prompt, open-ended preambles | Appendix, Baseline Implementations |

The tool descriptions and parameter schemas (which the agent also reads) are
in `tracer/tools/*.py`; the label descriptions inlined into the timeline
tools' label parameters are in `tracer/tools/labels.py`.

The HourVideo prompts are reproduced from Chandrasegaran et al., HourVideo
(NeurIPS 2024). The SimpleQA grader in `tracer/eval/judge.py` is reproduced
from openai/simple-evals (MIT).

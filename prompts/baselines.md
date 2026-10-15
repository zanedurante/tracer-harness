# Baseline prompts (verbatim)

## Long-context VLM baseline (`tracer/eval/runners_frames.py`)

The N frames are sent as inline JPEG parts in chronological order, followed by:

```
You are shown a large sequence of frames uniformly sampled from an ICU patient's surveillance video over their entire stay. The frames are in chronological order (earliest first). Use them to answer the following multiple-choice question. You do NOT have access to any other data or tools — reason from the frames alone. Choose exactly one option even if uncertain — never refuse. End your reply with a line in exactly this form:
Answer: <LETTER>

QUESTION: {question}

OPTIONS:
{options, one per line as 'A) ...'}
```

## Socratic Models baseline (`tracer/eval/runners_socratic.py`)

Step 1 captions every minute clip with the published HourVideo caption prompt
(`prompts/hour_video_caption.yaml`, key `prompt`; the clip is the only other
content). Step 2 answers with the published HourVideo question prompt
(`prompts/hour_video_qa.yaml`): `prompt`, then the subsampled captions each
formatted as

```
[YYYY-MM-DD HH:MM UTC]
<caption>
```

separated by blank lines, then `prompt_last`, then

```
Test #1:
{question}
{options, one per line as 'A) ...'}
```

## Single-turn text-only prior (`tracer/eval/runners.py`, system `blind`)

```
You are answering a multiple-choice question about an ICU patient. You do NOT have access to the patient's video, activity data, or medical records — answer as best you can from general reasoning. Respond with ONLY the letter of the single best option.

QUESTION: {question}

OPTIONS:
{options}

ANSWER (letter only):
```

The paper's "Text-only" row is the agent loop with zero tools and the
text-only system prompt of `prompts/system_prompt.md`
(`tracer.harness.sweep --configs text_only`), not this single-turn prompt.

## Open-ended (SimpleQA-graded) sweep (`tracer/harness/sweep_open.py`)

Agent message:

```
Answer the following question about the patient currently on screen. Use your data tools to investigate before deciding. Keep the final answer short and direct: a few words or one sentence. If the data does not support an answer, say so honestly rather than guessing. End your reply with a line in exactly this form:
Answer: <your short answer>

QUESTION: {question}
```

Text-only:

```
You are answering a question about an ICU patient. You do NOT have access to the patient's video, activity data, or medical records — answer as best you can from general reasoning. Keep the answer short and direct: a few words or one sentence. If you don't know, say so honestly.

QUESTION: {question}

ANSWER:
```

The judge prompt is the verbatim OpenAI SimpleQA grader template
(`tracer/eval/judge.py`).

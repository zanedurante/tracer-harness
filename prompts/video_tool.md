# Video tool prompts (verbatim)

`query_video_minute` sends the original one-minute clip of each camera view,
in parallel with the same agent-written prompt, to the perception VLM (the
agent's own model unless `VIDEO_TOOL_MODEL` is set), with a 5 fps sampling
hint on the Gemini path and the re-encoded 2 fps, 512 px clip on the
open-weight path. A fixed preamble precedes the agent's prompt on every call,
and both views' answers are returned for the agent to reconcile. Source:
`tracer/tools/video.py`.

## Tool description, as the agent sees it (variant D2, the default and the one every reported run used)

```
Look at the ICU surveillance video for ONE specific minute and answer a free-form question about what is visible -- patient movements/reactions, what a clinician is doing, devices/objects, body position, who is in the room.
Treat its answer as EVIDENCE TO WEIGH, not ground truth: the video reader can miss brief events or misread motion, and the two camera views may disagree. When the answer matters, re-query (a different prompt or an adjacent minute) and cross-check against your other tools before committing.
This tool is useful for zooming in on a single minute of interest. You can call it on the SAME minute several times with DIFFERENT prompts to probe different details (e.g. first ask what the clinician is doing, then whether the patient moved). It returns BOTH camera angles (left and right view) answered separately.
TIME: everything here is local time. Pass the minute exactly as it appears in the question, 'YYYY-MM-DD HH:MM' (24-hour, local time); the tool finds the right clip for you and you never convert timezones. The clip is one minute long, so if an event might straddle a boundary, also query the adjacent minute.
```

Parameters: `minute_pt` (string, "The minute to view, local time, format
'YYYY-MM-DD HH:MM' (24-hour)."), `prompt` (string, "The specific
question to ask about what is visible in that minute of video."); both required.

Alternative descriptions selectable with `VIDEO_DESC_VARIANT` (used only in
a description A/B test, not in any reported number):

* D1 (oversells): "Look at the ACTUAL ICU surveillance video for ONE specific
  minute and answer a free-form question about what is visible -- patient
  movements/reactions, what a clinician is doing, devices/objects, body
  position, who is in the room. This is ground truth: prefer it over
  precomputed action labels, which can be unreliable for this patient." + the
  common usage paragraph.
* D3 (neutral): "Look at the ICU surveillance video for one minute and answer
  a question about what is visible. Returns both camera views (left and
  right) answered separately." + the common usage paragraph.

## Preamble prepended to the agent's prompt on every perception call

```
You are an expert ICU video analyst. Patients here are often sedated, so subtle patient movement (a brief head turn, eye flutter, blink, cough, swallow, finger/limb twitch) is clinically relevant. Review the one-minute clip carefully and list ONLY movements you can actually see, each with an approximate timestamp (mm:ss). Do NOT infer or assume movements that are not clearly visible -- if the patient appears still, say so explicitly. Then answer the custom prompt below.
```

The call contents are `[<clip>, PREAMBLE + "\n\n" + <agent prompt>]`.

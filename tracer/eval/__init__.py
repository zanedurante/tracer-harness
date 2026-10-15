"""Question-answering evaluation.

  ingest          parse a question bank CSV into records (question, gold, distractors)
  mcq             deterministic option shuffling and gold-position rotation
  extract         parse the chosen letter out of a free-text reply
  score           accuracy, position diagnostics, paired McNemar tests, SimpleQA metrics
  judge           the SimpleQA grader prompt (open-ended answers)
  retry           transient-error classification and backoff
  runners         RunResult + the text-only and tool-agent systems
  runners_frames  long-context VLM baseline (N uniformly sampled frames in context)
  runners_socratic  Socratic Models baseline (per-minute captions, subsampled to context)
  socratic_caption  batch captioning of every minute clip with the HourVideo prompt
  frame_sampler / frame_extract   uniform frame sampling over a stay's clips
  run_eval        orchestrator for the baselines with position rotation and caching
  bundle          package frames + questions for offline (cluster) VLM inference
  standalone/     self-contained drivers for remote machines (Gemini/Vertex, vLLM, HF)
  position_stats  mean +/- std over gold positions from a sweep cache
  compare_sweeps  paired McNemar between two configs/runs
  breakdown       per-category / per-stay breakdown of any result JSONL
"""

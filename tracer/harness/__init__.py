"""The agent harness.

  registry   name -> Tool for the current run (build_registry)
  configs    named tool configurations; the paper's Table 3 rows
  prompt     per-configuration system prompt assembly (header, one line per
             exposed tool, footer, and the guided addendum)
  agent      the gated agent loop (Gemini and OpenAI-compatible/mock paths)
  sweep      MCQ sweep: configs x questions x gold positions, cached
  sweep_open open-ended sweep graded with the SimpleQA judge
  greedy     greedy forward selection over tools
  prompt_cache / tool_cache   cost-saving caches that do not change outputs
"""

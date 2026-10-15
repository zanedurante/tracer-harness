"""End-to-end demo on synthetic, non-PHI data (``python -m tracer.demo``).

  make_synthetic   writes the synthetic sample under demo/data (already
                   committed; rerun to regenerate)
  __main__         the runnable demo: expand the sample into a data store,
                   build the timeline from per-minute labels, generate hourly
                   reports, run the agent loop with the scripted mock backend
                   over three questions x five gold positions x three tool
                   configurations, and score the result
"""

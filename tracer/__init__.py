"""TRACER: a multimodal harness for longitudinal question answering over
bedside video and electronic health records.

Subpackages
-----------
tracer.tools       the eleven agent tools (scaffold, timeline, report, EHR, video)
tracer.harness     agent loop, per-configuration prompt assembly, tool registry,
                   named tool configurations, sweep drivers
tracer.perception  minute-level perception runner, taxonomy, 13-category
                   evaluation, timeline construction from per-minute labels
tracer.reports     hourly report generator
tracer.eval        MCQ ingestion, gold-position rotation, answer extraction,
                   scoring (accuracy, McNemar), baselines (text-only,
                   long-context frames, Socratic)
tracer.cohort      delirium frequency ratios, bootstrap intervals, exact tests
tracer.backends    Gemini, OpenAI-compatible (vLLM) and mock model backends
tracer.demo        end-to-end demo on synthetic data (python -m tracer.demo)

All data locations are resolved through tracer.config / tracer.data from the
TRACER_DATA_ROOT environment variable; nothing in the package is tied to a
particular site, cluster, or cloud project.
"""
__version__ = "1.0.0"

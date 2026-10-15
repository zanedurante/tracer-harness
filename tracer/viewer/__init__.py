"""Browser viewer for a TRACER data store: video player, atomic-action
timeline, hourly reports, and a chat panel driven by the harness's own agent
loop and tools.

    python -m tracer.viewer [--port 5000] [--pat-id stay_A] [--model gemini-3.1-flash-lite]

Reads the store at TRACER_DATA_ROOT (docs/DATA_LAYOUT.md). The chat endpoint
runs the same eleven tools, tool handlers and guided working rules as the
paper's agent; only the answer contract differs (free-form answers instead of
a multiple-choice letter).
"""

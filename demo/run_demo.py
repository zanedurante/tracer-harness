#!/usr/bin/env python3
"""Thin wrapper: `python demo/run_demo.py` == `python -m tracer.demo`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tracer.demo.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

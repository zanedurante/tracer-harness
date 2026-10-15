"""Environment-driven configuration.

Every path and credential is read from the environment so the same code runs
on any machine. Nothing here is evaluated eagerly except the repository root.

Environment variables
---------------------
TRACER_DATA_ROOT      root of the data store (default: ./data). See
                      docs/DATA_LAYOUT.md for the expected directory layout.
TRACER_LOCAL_TZ       IANA zone of the site's local clock (default: UTC).
                      Set it to the deployment site's zone; every local-time
                      conversion in the tools and prompts uses it.
TRACER_RESULTS_ROOT   where sweep caches and summaries are written
                      (default: ./results).
TRACER_CACHE_ROOT     scratch space for re-encoded clips and extracted frames
                      (default: ./cache).

GEMINI_API_KEY        Gemini API key (or GOOGLE_API_KEY, or a file path in
                      GEMINI_API_KEY_FILE).
GEMINI_AUTH           "api_key" (default) or "adc" (Vertex AI with
                      Application Default Credentials; needs
                      GOOGLE_CLOUD_PROJECT, optional GOOGLE_CLOUD_LOCATION).
GEMINI_VIDEO_BUCKET   GCS bucket used to stage clips on the Vertex path when a
                      clip is too large to inline (optional).
OPENAI_BASE_URL       OpenAI-compatible endpoint (vLLM) for open-weight
                      models; OPENAI_API_KEY may be any non-empty string.
"""
from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = REPO_ROOT / "prompts"
CONFIGS_DIR = REPO_ROOT / "configs"
TAXONOMY_FILE = Path(__file__).resolve().parent / "perception" / "taxonomy" / "activity_key.csv"


def data_root() -> Path:
    return Path(os.environ.get("TRACER_DATA_ROOT", "data")).expanduser().resolve()


def results_root() -> Path:
    return Path(os.environ.get("TRACER_RESULTS_ROOT", "results")).expanduser().resolve()


def cache_root() -> Path:
    return Path(os.environ.get("TRACER_CACHE_ROOT", "cache")).expanduser().resolve()


def local_tz_name() -> str:
    return os.environ.get("TRACER_LOCAL_TZ", "UTC")


def local_tz() -> ZoneInfo:
    return ZoneInfo(local_tz_name())


UTC = ZoneInfo("UTC")

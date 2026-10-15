"""Gemini client: key loading, client setup, and model registry.

Supports two authentication modes (GEMINI_AUTH env var):

  GEMINI_AUTH=api_key  (default)
      Reads the key from GEMINI_API_KEY (or GOOGLE_API_KEY, or the file named
      by GEMINI_API_KEY_FILE) and uses the Google AI Studio endpoint.

  GEMINI_AUTH=adc
      Uses Application Default Credentials via the Vertex AI endpoint.
      Required: GOOGLE_CLOUD_PROJECT. Optional: GOOGLE_CLOUD_LOCATION
      (default "global"; Gemini 3.x models require the global endpoint) and
      GOOGLE_APPLICATION_CREDENTIALS.

Every Gemini call in the harness, the baselines and the perception runner is
made with no generation configuration, so temperature, top-p, top-k, output
length and thinking follow the API defaults of each model (paper appendix,
"Sampling and serving settings").
"""
from __future__ import annotations

import os
from pathlib import Path

from google import genai
from google.genai.types import HttpOptions


def get_gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if key:
        return key.strip()
    f = os.environ.get("GEMINI_API_KEY_FILE")
    if f and Path(f).expanduser().exists():
        return Path(f).expanduser().read_text().strip()
    raise RuntimeError(
        "No Gemini API key: set GEMINI_API_KEY (or GOOGLE_API_KEY, or "
        "GEMINI_API_KEY_FILE=<path>), or set GEMINI_AUTH=adc with GOOGLE_CLOUD_PROJECT."
    )


_client: genai.Client | None = None


def auth_mode() -> str:
    m = os.environ.get("GEMINI_AUTH", "api_key")
    return m if m in ("api_key", "adc") else "api_key"


def is_adc() -> bool:
    return auth_mode() == "adc"


def get_gemini_client() -> genai.Client:
    """Return a reusable, authenticated genai.Client."""
    global _client
    if _client is None:
        if is_adc():
            project = os.environ.get("GOOGLE_CLOUD_PROJECT")
            location = os.environ.get("GOOGLE_CLOUD_LOCATION", "global")
            if not project:
                raise RuntimeError("GEMINI_AUTH=adc requires GOOGLE_CLOUD_PROJECT.")
            _client = genai.Client(
                vertexai=True, project=project, location=location,
                http_options=HttpOptions(timeout=180_000),
            )
        else:
            _client = genai.Client(
                api_key=get_gemini_api_key(),
                http_options=HttpOptions(
                    api_version="v1alpha",
                    base_url="https://generativelanguage.googleapis.com/",
                    timeout=180_000,
                ),
            )
    return _client


def get_files_client() -> genai.Client:
    """Client for the Files API (video upload). The v1alpha chat client cannot
    fetch an uploaded file's v1beta URI, so uploads use a default client; on
    the Vertex path the main client handles files natively."""
    if is_adc():
        return get_gemini_client()
    return genai.Client(api_key=get_gemini_api_key())


GEMINI_MODELS: dict[str, str] = {
    "gemini-2.5-flash-lite":  "Gemini 2.5 Flash Lite",
    "gemini-2.5-flash":       "Gemini 2.5 Flash",
    "gemini-2.5-pro":         "Gemini 2.5 Pro",
    "gemini-3-flash-preview": "Gemini 3 Flash Preview",
    "gemini-3-pro-preview":   "Gemini 3 Pro Preview",
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite Preview",
    "gemini-3.1-flash-lite":  "Gemini 3.1 Flash Lite",
    "gemini-3.5-flash-lite":  "Gemini 3.5 Flash Lite",
    "gemini-3.5-flash":       "Gemini 3.5 Flash",
}

DEFAULT_MODEL = "gemini-3.1-flash-lite"


def call_gemini(prompt: str, *, model: str = DEFAULT_MODEL) -> str:
    """Send a text prompt to Gemini (API defaults) and return the response text."""
    client = get_gemini_client()
    response = client.models.generate_content(model=model, contents=prompt)
    return response.text or ""

"""Model backends.

  gemini          google-genai client (API key or Vertex/ADC), model registry
  openai_backend  any OpenAI-compatible /v1/chat/completions server (vLLM)
  mock            offline scripted backend used by the demo and tests

Backend selection is by model name (see openai_backend.is_openai_model):
``gemini-*`` -> Gemini; ``mock:*`` -> mock; anything else -> OpenAI-compatible.
"""

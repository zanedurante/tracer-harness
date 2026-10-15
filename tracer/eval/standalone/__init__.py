"""Self-contained drivers for machines where the package is not installed
(a machine next to the data). They import nothing from
`tracer`; each reads stays from a stays.csv passed on the command line and
the prompts from the YAML files in prompts/.

  frames_qa_gemini.py     long-context frames baseline, Gemini via Vertex AI/ADC, from a bundle
  socratic_qa_gemini.py   Socratic QA, Gemini, over a caption store
  socratic_qa_openai.py   Socratic QA with an OpenAI-compatible (vLLM) answerer over Gemini captions
  socratic_caption_vlm.py Socratic captioning with a local HF VLM (30 frames per clip)
  socratic_qa_vlm.py      Socratic QA with a local HF model over JSONL captions
  run_local_vlm.py        frames baseline with a local HF VLM from a bundle
"""

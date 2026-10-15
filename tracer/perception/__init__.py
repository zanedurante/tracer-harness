"""Minute-level perception layer.

  taxonomy/activity_key.csv   the 15 atomic actions in matrix-column order
  prompts                     loader for the YAML prompt templates in prompts/
  video                       one-minute clip -> Gemini inline video part
  save_results                per-clip output JSON (raw + parsed labels + usage)
  run_perception              concurrent perception runner over clips (Gemini)
  evaluate                    per-category / micro / macro metrics against
                              annotations, all 15 categories by default
                              (--exclude-rare for the 13-category set)
  build_timeline              per-minute outputs -> bit-packed action matrix
                              (npz) + stay timeline JSON for the tools
"""

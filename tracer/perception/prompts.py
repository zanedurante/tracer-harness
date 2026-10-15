"""Loader for prompt templates (YAML files in prompts/)."""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import yaml

from .. import config


class Prompt(NamedTuple):
    name: str          # stem of the YAML file
    prompt: str        # main system / instruction text
    user_prompt: str   # short user-turn text (accompanies the frames)
    label_aliases: dict | None = None   # optional prompt-specific key -> canonical label


def list_prompts(prompts_dir: Path | None = None) -> list[str]:
    d = prompts_dir or config.PROMPTS_DIR
    return sorted(p.stem for p in d.glob("*.yaml"))


def load_prompt(name: str, prompts_dir: Path | None = None) -> Prompt:
    d = prompts_dir or config.PROMPTS_DIR
    path = Path(name) if name.endswith(".yaml") and Path(name).exists() else d / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt {name!r} not found in {d}. Available: {list_prompts(d)}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    missing = [k for k in ("prompt", "user_prompt") if k not in data]
    if missing:
        raise KeyError(f"Prompt YAML {path.name} is missing required key(s): {missing}")
    aliases = data.get("label_aliases")
    if aliases is not None and not isinstance(aliases, dict):
        raise KeyError(f"Prompt YAML {path.name} has a 'label_aliases' entry that is not a mapping.")
    return Prompt(name=path.stem, prompt=data["prompt"].strip(), user_prompt=data["user_prompt"].strip(),
                  label_aliases=aliases)

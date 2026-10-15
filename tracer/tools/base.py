"""The Tool record shared by every tool module and the harness registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Tool:
    name: str
    declaration: object          # google.genai.types.FunctionDeclaration
    handler: Callable            # (args: dict, pat_id: str | None, model: str) -> dict
    group: str = "other"         # scaffold | timeline | report | ehr | video

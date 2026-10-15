"""Tool registry for a run: every tool bundles its FunctionDeclaration with a
handler `(args, pat_id, model) -> dict`."""
from __future__ import annotations

from .. import tools as _tools
from ..tools.base import Tool

__all__ = ["Tool", "build_registry", "all_tool_names"]


def build_registry(include_video: bool = True, include_med: bool = True,
                   include_flowsheet: bool = True) -> dict[str, Tool]:
    reg = _tools.all_tools()
    if not include_video:
        reg.pop("query_video_minute", None)
    if not include_med:
        reg.pop("query_medications", None)
    if not include_flowsheet:
        reg.pop("query_flowsheet", None)
    return reg


def all_tool_names(include_video: bool = True, include_med: bool = True,
                   include_flowsheet: bool = True) -> list[str]:
    return list(build_registry(include_video, include_med, include_flowsheet).keys())

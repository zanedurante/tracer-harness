"""The systems under test with a common result shape.

  - run_blind_baseline  a plain single-turn LLM call with the question and
    options and NO patient data (a text-only prior). The paper's "Text-only"
    row instead runs the agent loop with zero tools (tracer.harness.sweep with
    --configs text_only); both are provided.
  - run_tool_agent      the harness (all eleven tools by default; pass
    `tools=` for another configuration).

Both return a RunResult; the chosen letter is parsed by tracer.eval.extract.
External calls are injectable so unit tests run offline.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .extract import extract_choice
from .mcq import MCQItem, render_options

BLIND_PREAMBLE = (
    "You are answering a multiple-choice question about an ICU patient. You do "
    "NOT have access to the patient's video, activity data, or medical records "
    "— answer as best you can from general reasoning. Respond with ONLY the "
    "letter of the single best option."
)


@dataclass
class RunResult:
    system: str
    item_id: int
    patient_id: str
    model: str
    chosen_letter: str | None
    correct_letter: str
    is_correct: bool
    raw_text: str
    extraction_method: str
    n_tool_calls: int = 0
    tool_calls: list = field(default_factory=list)
    error: str | None = None
    pos: int | None = None
    grader_model: str | None = None
    grade_label: str | None = None
    grader_raw: str | None = None


def build_blind_prompt(item: MCQItem) -> str:
    return (f"{BLIND_PREAMBLE}\n\nQUESTION: {item.question}\n\n"
            f"OPTIONS:\n{render_options(item)}\n\nANSWER (letter only):")


def _result(system, item, model, raw, extraction, *, n_tool_calls=0, tool_calls=None, error=None) -> RunResult:
    letter = extraction.letter if extraction else None
    return RunResult(system=system, item_id=item.id, patient_id=item.patient_id, model=model,
                     chosen_letter=letter, correct_letter=item.correct_letter,
                     is_correct=(letter == item.correct_letter), raw_text=raw,
                     extraction_method=extraction.method if extraction else "none",
                     n_tool_calls=n_tool_calls, tool_calls=tool_calls or [], error=error)


def run_blind_baseline(item: MCQItem, model: str, *, gemini_fn=None, allow_llm_extract=False,
                       extract_fn=None) -> RunResult:
    if gemini_fn is None:
        from ..backends.gemini import call_gemini as gemini_fn
    ex = extract_fn or (lambda r: extract_choice(r, item, allow_llm=allow_llm_extract, llm_model=model))
    try:
        raw = gemini_fn(build_blind_prompt(item), model=model)
    except Exception as e:
        return _result("blind", item, model, "", None, error=f"{type(e).__name__}: {e}")
    return _result("blind", item, model, raw, ex(raw))


def run_tool_agent(item: MCQItem, model: str, *, tools: list[str] | None = None, agent_fn=None,
                   allow_llm_extract=False, extract_fn=None) -> RunResult:
    if agent_fn is None:
        from ..harness import agent as aa
        from ..harness.configs import resolve
        from ..harness.registry import build_registry
        registry = build_registry()
        allowed = tools if tools is not None else resolve("tracer")

        def agent_fn(message, pat_id, model):
            return aa.run_ablated(message, pat_id, allowed, model, registry)
    from ..harness.agent import build_message
    ex = extract_fn or (lambda r: extract_choice(r, item, allow_llm=allow_llm_extract, llm_model=model))
    try:
        out = agent_fn(build_message(item), item.patient_id, model)
    except Exception as e:
        return _result("tool_agent", item, model, "", None, error=f"{type(e).__name__}: {e}")
    if isinstance(out, dict) and out.get("error"):
        return _result("tool_agent", item, model, "", None, error=out["error"],
                       n_tool_calls=len(out.get("tool_calls", [])), tool_calls=out.get("tool_calls", []))
    raw = out.get("reply", "") if isinstance(out, dict) else str(out)
    tcs = out.get("tool_calls", []) if isinstance(out, dict) else []
    return _result("tool_agent", item, model, raw, ex(raw), n_tool_calls=len(tcs), tool_calls=tcs)


SYSTEMS = {"blind": run_blind_baseline, "tool_agent": run_tool_agent}

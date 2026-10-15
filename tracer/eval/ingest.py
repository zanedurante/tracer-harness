"""Ingest a question bank (CSV) into normalized records.

Expected columns (docs/DATA_LAYOUT.md):
  PAT_ID              stay identifier the question is about
  Original Question   "Q: <question>\\n\\nA: <gold answer>" (or "<question>\\n<gold>")
  Distractors (4)     one distractor per line (list cruft such as "- ", "1)", "B." is stripped)

Rows are separated into buildable records (a clean question + single gold
answer + >=1 distractor) and skipped rows, each tagged with a reason. Cells
containing "TODO" / "<...correct answer>" or embedded "A) ..." option lists are
skipped. Pure parsing; no network.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

COL_PATIENT = "PAT_ID"
COL_QA = "Original Question"
COL_DISTRACTORS = "Distractors (4)"

_Q_PREFIX = re.compile(r"^\s*Q\s*:\s*", re.IGNORECASE)
_A_MARKER = re.compile(r"(?m)^\s*A\s*:\s*")
_TODO = re.compile(r"todo|<[^>]*correct answer[^>]*>", re.IGNORECASE)
_EMBEDDED_OPTION = re.compile(r"(?m)^\s*-?\s*[A-E][\).]\s+\S")
_DISTRACTOR_CRUFT = re.compile(r"^\s*(?:-\s*|\d+[\).]\s*|distractor\s*\d*\s*[:\).-]?\s*|[A-E][\).]\s*)", re.IGNORECASE)


@dataclass
class MCQRecord:
    """A buildable multiple-choice question (pre-shuffle)."""
    id: int                      # source row index (0-based, matches the CSV body)
    patient_id: str
    question: str
    gold_answer: str
    distractors: list[str]
    parsed_fallback: bool = False
    extra: dict = field(default_factory=dict)


@dataclass
class SkippedRow:
    id: int
    patient_id: str
    reason: str
    detail: str = ""


def _norm_line(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _norm_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    lines = [_norm_line(ln) for ln in s.splitlines()]
    return " ".join(ln for ln in lines if ln).strip()


def _parse_distractors(cell) -> list[str]:
    if not isinstance(cell, str):
        return []
    out: list[str] = []
    for ln in cell.splitlines():
        ln = _DISTRACTOR_CRUFT.sub("", ln)
        ln = _norm_line(ln)
        if ln:
            out.append(ln)
    return out


def _split_qa(cell: str) -> tuple[str, str, bool]:
    m = _A_MARKER.search(cell)
    if m:
        q = _norm_text(_Q_PREFIX.sub("", cell[: m.start()]))
        a = _norm_text(cell[m.end():])
        return q, a, False
    lines = [_norm_line(ln) for ln in cell.splitlines()]
    lines = [ln for ln in lines if ln]
    if len(lines) >= 2:
        return lines[0], " ".join(lines[1:]), True
    if len(lines) == 1:
        return lines[0], "", True
    return "", "", True


def load_records(csv_path: str | Path) -> tuple[list[MCQRecord], list[SkippedRow]]:
    """Parse the bank. Returns (buildable_records, skipped_rows)."""
    df = pd.read_csv(csv_path, encoding="utf-8")
    for col in (COL_PATIENT, COL_QA, COL_DISTRACTORS):
        if col not in df.columns:
            raise KeyError(f"Expected column {col!r} not found. Have: {list(df.columns)}")

    records: list[MCQRecord] = []
    skipped: list[SkippedRow] = []
    for idx, row in df.iterrows():
        pat = str(row[COL_PATIENT]).strip() if isinstance(row[COL_PATIENT], str) else ""
        qa_cell = row[COL_QA]
        distractors = _parse_distractors(row[COL_DISTRACTORS])
        if not isinstance(qa_cell, str) or not qa_cell.strip():
            skipped.append(SkippedRow(idx, pat, "empty_qa", "question cell is blank"))
            continue
        if _TODO.search(qa_cell):
            skipped.append(SkippedRow(idx, pat, "todo_unresolved", "answer not yet filled in (TODO marker)"))
            continue
        question, gold, fallback = _split_qa(qa_cell)
        if _EMBEDDED_OPTION.search(gold):
            skipped.append(SkippedRow(idx, pat, "embedded_options",
                                      "answer cell lists A)/B)... options with no single resolved answer"))
            continue
        if not question:
            skipped.append(SkippedRow(idx, pat, "no_question", "could not extract a question"))
            continue
        if not gold:
            skipped.append(SkippedRow(idx, pat, "no_gold_answer", "no answer text after Q:/A: split"))
            continue
        if not distractors:
            skipped.append(SkippedRow(idx, pat, "no_distractors", "distractor cell empty"))
            continue
        if not pat:
            skipped.append(SkippedRow(idx, pat, "no_patient", "PAT_ID missing"))
            continue
        extra = {k: row[k] for k in df.columns if k not in (COL_PATIENT, COL_QA, COL_DISTRACTORS)}
        records.append(MCQRecord(id=int(idx), patient_id=pat, question=question, gold_answer=gold,
                                 distractors=distractors, parsed_fallback=fallback, extra=extra))
    return records, skipped


def summarize(records: list[MCQRecord], skipped: list[SkippedRow]) -> str:
    from collections import Counter
    lines = [f"Buildable MCQs: {len(records)}   Skipped: {len(skipped)}"]
    by_pat = Counter(r.patient_id for r in records)
    lines.append("  buildable by stay: " + ", ".join(f"{k}={v}" for k, v in sorted(by_pat.items())))
    if skipped:
        by_reason = Counter(s.reason for s in skipped)
        lines.append("  skipped reasons: " + ", ".join(f"{k}={v}" for k, v in by_reason.items()))
    return "\n".join(lines)

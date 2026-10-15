"""Turn ingested records into labeled multiple-choice items.

Each MCQRecord (one gold answer + N distractors) becomes an MCQItem with
options labeled A, B, C, ... and a recorded correct letter.

Two arrangements are provided:
  * build_item     deterministic, content-addressed shuffle (hash of seed +
                   question text), stable across runs and machines;
  * item_with_gold_at   the gold answer forced into slot `pos`. The paper
                   evaluates every question once per gold position
                   (n_positions = number of distinct options) and reports the
                   mean over positions (appendix, "Answer-Position Rotation").

Pure / offline.
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter
from dataclasses import dataclass

from .ingest import MCQRecord

LETTERS = "ABCDEFGH"
DEFAULT_SEED = 1234


@dataclass
class MCQItem:
    id: int
    patient_id: str
    question: str
    options: dict[str, str]   # letter -> option text
    correct_letter: str
    gold_answer: str
    n_options: int


def _dedup_distractors(gold: str, distractors: list[str]) -> list[str]:
    """Drop distractors that duplicate the gold answer or each other
    (case-insensitive, whitespace-insensitive)."""
    seen = {gold.strip().lower()}
    out = []
    for d in distractors:
        k = d.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(d)
    return out


def _seeded_rng(seed: int, question: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}\x00{question}".encode("utf-8")).hexdigest()
    return random.Random(digest)


def build_item(record: MCQRecord, seed: int = DEFAULT_SEED) -> MCQItem:
    distractors = _dedup_distractors(record.gold_answer, record.distractors)
    option_texts = [record.gold_answer] + distractors
    if len(option_texts) > len(LETTERS):
        raise ValueError(f"row {record.id}: {len(option_texts)} options exceeds {len(LETTERS)} letters")
    order = list(range(len(option_texts)))
    _seeded_rng(seed, record.question).shuffle(order)
    shuffled = [option_texts[i] for i in order]
    letters = LETTERS[: len(shuffled)]
    options = dict(zip(letters, shuffled))
    correct_letter = letters[shuffled.index(record.gold_answer)]
    return MCQItem(id=record.id, patient_id=record.patient_id, question=record.question,
                   options=options, correct_letter=correct_letter, gold_answer=record.gold_answer,
                   n_options=len(shuffled))


def build_items(records: list[MCQRecord], seed: int = DEFAULT_SEED) -> tuple[list[MCQItem], list[tuple[int, str]]]:
    items: list[MCQItem] = []
    degenerate: list[tuple[int, str]] = []
    for rec in records:
        item = build_item(rec, seed=seed)
        if item.n_options < 2:
            degenerate.append((rec.id, "fewer than 2 distinct options after dedup"))
            continue
        items.append(item)
    return items, degenerate


def item_with_gold_at(record: MCQRecord, pos: int) -> MCQItem:
    """Build an MCQItem with the gold answer forced into option slot `pos`."""
    opts = list(_dedup_distractors(record.gold_answer, record.distractors))
    opts.insert(pos, record.gold_answer)
    letters = LETTERS[: len(opts)]
    return MCQItem(record.id, record.patient_id, record.question,
                   dict(zip(letters, opts)), letters[pos], record.gold_answer, len(opts))


def n_positions(record: MCQRecord) -> int:
    """Number of valid option slots for position rotation."""
    return len(_dedup_distractors(record.gold_answer, record.distractors)) + 1


def render_options(item: MCQItem) -> str:
    """LLM-readable option block, e.g. 'A) ...\\nB) ...'."""
    return "\n".join(f"{L}) {item.options[L]}" for L in sorted(item.options))


def position_distribution(items: list[MCQItem]) -> dict[str, int]:
    return dict(sorted(Counter(it.correct_letter for it in items).items()))

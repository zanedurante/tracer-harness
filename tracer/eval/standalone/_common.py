"""Shared helpers for the standalone drivers (kept dependency-free: csv, json,
re, datetime only). Each driver copies what it needs from here so it can also
be shipped as a single file; importing works when the directory is on the path.
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
import re
from collections import defaultdict
from pathlib import Path

LETTERS = "ABCDEFGH"


def load_stays(csv_path) -> dict:
    """{pat_id: {room, start_utc, end_utc}} from a stays.csv (naive UTC datetimes)."""
    out = {}
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            def _p(s):
                s = s.strip().replace("Z", "+00:00")
                d = dt.datetime.fromisoformat(s)
                return (d.astimezone(dt.timezone.utc) if d.tzinfo else d).replace(tzinfo=None)
            out[row["pat_id"].strip()] = {"room": row["room"].strip(),
                                          "start_utc": _p(row["start_time_utc"]),
                                          "end_utc": _p(row["end_time_utc"])}
    return out


# ---------------------------------------------------------------------------
# MCQ record parsing (mirrors tracer.eval.ingest / mcq)
# ---------------------------------------------------------------------------
def _norm(s):
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s).strip()


def _split_qa(cell):
    m = re.search(r"(?m)^\s*A\s*:\s*", cell)
    if m:
        q = _norm(re.sub(r"^\s*Q\s*:\s*", "", cell[:m.start()], flags=re.I))
        return q, _norm(cell[m.end():])
    lines = [_norm(ln) for ln in cell.splitlines() if _norm(ln)]
    if len(lines) >= 2:
        return lines[0], " ".join(lines[1:])
    return (lines[0] if lines else ""), ""


def _parse_distractors(cell):
    if not isinstance(cell, str):
        return []
    out = []
    for ln in cell.splitlines():
        ln = re.sub(r"^\s*(?:-\s*|\d+[).]\s*|distractor\s*\d*\s*[:\).-]?\s*|[A-E][).]\s*)", "", ln, flags=re.I)
        ln = _norm(ln)
        if ln:
            out.append(ln)
    return out


def _dedup_distractors(gold, distractors):
    seen = {gold.strip().lower()}
    out = []
    for d in distractors:
        k = d.strip().lower()
        if k and k not in seen:
            seen.add(k)
            out.append(d)
    return out


def load_mcqs(csv_path):
    records = []
    with open(csv_path, encoding="utf-8") as f:
        for idx, row in enumerate(csv.DictReader(f)):
            pat = (row.get("PAT_ID") or "").strip()
            qa_cell = row.get("Original Question", "")
            if not qa_cell or not pat or re.search(r"todo|<[^>]*correct answer[^>]*>", qa_cell, re.I):
                continue
            q, gold = _split_qa(qa_cell)
            if not q or not gold:
                continue
            distractors = _dedup_distractors(gold, _parse_distractors(row.get("Distractors (4)", "")))
            if not distractors:
                continue
            records.append({"id": idx, "patient_id": pat, "question": q, "gold": gold, "distractors": distractors})
    return records


def item_with_gold_at(rec, pos):
    opts = list(rec["distractors"])
    opts.insert(pos, rec["gold"])
    letters = LETTERS[:len(opts)]
    return dict(zip(letters, opts)), letters[pos]


def n_positions(rec):
    return len(rec["distractors"]) + 1


def render_options(options):
    return "\n".join(f"{L}) {options[L]}" for L in sorted(options))


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------
LETTER_RE = re.compile(r"answer\s*[:\-]?\s*\*?\*?\s*([A-H])\b", re.IGNORECASE)
FALLBACK_RE = re.compile(r"\b([A-H])\b")


def extract_letter(reply, valid_letters):
    if not reply or not reply.strip():
        return None, "empty"
    m = LETTER_RE.search(reply)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper(), "regex:answer"
    for c in reversed(FALLBACK_RE.findall(reply)):
        if c.upper() in valid_letters:
            return c.upper(), "regex:last_letter"
    return None, "unmatched"


def extract_socratic_answer(raw, n_opts):
    """JSON YOUR_ANSWER first, then a standalone letter."""
    try:
        start, end = raw.find("["), raw.rfind("]") + 1
        if start >= 0 and end > start:
            answers = json.loads(raw[start:end])
            if answers and isinstance(answers, list):
                m = re.match(r"[A-H]", answers[0].get("YOUR_ANSWER", "").strip())
                if m:
                    return m.group(0)
    except (json.JSONDecodeError, KeyError, IndexError, AttributeError):
        pass
    m = re.search(r"\b([" + LETTERS[:n_opts] + r"])\b", raw)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Cache + summary
# ---------------------------------------------------------------------------
class Cache:
    def __init__(self, path):
        self.path = Path(path)
        self.done = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("error") is None and rec.get("chosen_letter") is not None:
                        self.done[(rec["system"], rec["model"], rec["item_id"], rec.get("pos"))] = rec

    def get(self, system, model, item_id, pos):
        return self.done.get((system, model, item_id, pos))

    def put(self, rec):
        self.done[(rec["system"], rec["model"], rec["item_id"], rec.get("pos"))] = rec
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


def fmt_duration(seconds):
    s = int(seconds)
    if s < 60:
        return "%ds" % s
    if s < 3600:
        return "%dm %ds" % (s // 60, s % 60)
    h, r = divmod(s, 3600)
    return "%dh %dm" % (h, r // 60)


def print_position_summary(cache: Cache, model_label: str, system: str):
    pos_correct, pos_total = defaultdict(int), defaultdict(int)
    for rec in cache.done.values():
        if rec["model"] == model_label and rec["system"] == system:
            pos_total[rec.get("pos")] += 1
            pos_correct[rec.get("pos")] += bool(rec["is_correct"])
    accs = []
    for p in sorted(pos_total, key=lambda x: (x is None, x)):
        acc = pos_correct[p] / pos_total[p]
        accs.append(acc)
        print("  pos=%s: %d/%d = %.1f%%" % (p, pos_correct[p], pos_total[p], 100 * acc), flush=True)
    if len(accs) > 1:
        mean = sum(accs) / len(accs)
        std = math.sqrt(sum((a - mean) ** 2 for a in accs) / (len(accs) - 1))
        print("\nMean: %.1f%% +/- %.1f%%" % (100 * mean, 100 * std), flush=True)


def is_retryable(exc):
    err = str(exc)
    status = getattr(exc, "status_code", None)
    return (status in (429, 500, 502, 503) or "429" in err or "RESOURCE_EXHAUSTED" in err
            or "500" in err or "502" in err or "503" in err or "timeout" in err.lower()
            or "CANCELLED" in err or isinstance(exc, (TimeoutError, ConnectionError, OSError)))

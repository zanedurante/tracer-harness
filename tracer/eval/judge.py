"""LLM judge for the open-ended eval (SimpleQA grader).

This is a verbatim port of OpenAI SimpleQA's `GRADER_TEMPLATE` from
`openai/simple-evals/simpleqa_eval.py`. It grades a free-form prediction
against a single gold answer into one of three buckets:

    A = CORRECT         prediction fully contains the gold, no contradictions
    B = INCORRECT       prediction contradicts the gold (even hedged)
    C = NOT_ATTEMPTED   prediction neither includes nor contradicts the gold

Source: https://github.com/openai/simple-evals/blob/main/simpleqa_eval.py
License: MIT (OpenAI).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SIMPLEQA_GRADER_TEMPLATE = """
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade of either ["CORRECT", "INCORRECT", "NOT_ATTEMPTED"].
First, I will give examples of each grade, and then you will grade a new example.


The following are examples of CORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: most people would say Malia and Sasha, but I'm not sure and would have to double check
Predicted answer 3: Barack Obama has two daughters. Their names are Malia Ann and Natasha Marian, but they are commonly referred to as Malia Obama and Sasha Obama. Malia was born on July 4, 1998, and Sasha was born on June 10, 2001.
```
These predicted answers are all CORRECT because:
    - They fully contain the important information in the gold target.
    - They do not contain any information that contradicts the gold target.
    - Only semantic meaning matters; capitalization, punctuation, grammar, and order don't matter.
    - Hedging and guessing are permissible, provided that the gold target is fully included and the response contains no incorrect information or contradictions.


The following are examples of INCORRECT predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Barack Obama does not have any children.
Predicted answer 4: I think it's either Malia and Sasha. Or it could be Malia and Jackie. Or it could be Joey and Malia.
Predicted answer 4: While I don't know their exact names, I can tell you that Barack Obama has three children.
Predicted answer 5: It's possible you may mean Betsy and Olivia. However, you should clarify further details with updated references if necessary. Is that the correct answer?
Predicted answer 6: It may be the case that Obama's child is named James. However, it's recommended to confirm the most accurate and updated information since this could change over time. This model may not always reflect the most current information.
```
These predicted answers are all INCORRECT because:
    - A factual statement in the answer contradicts the gold target. Incorrect statements that have some hedging (e.g., "it is possible that", "although i'm not sure, i think") are also considered incorrect.


The following are examples of NOT_ATTEMPTED predicted answers.
```
Question: What are the names of Barack Obama's children?
Gold target: Malia and Sasha
Predicted answer 1: I don't know.
Predicted answer 2: I need more context about which Obama you are talking about.
Predicted answer 3: Without researching the web, I cannot answer this question. However, I can tell you that Barack Obama has two children.
Predicted answer 4: Barack Obama has two children. I know that one of them is Malia, but I'm not sure about the other one.
```
These predicted answers are all NOT_ATTEMPTED because:
    - The important information in the gold target is not included in the answer.
    - No statements in the answer contradict the gold target.


Also note the following things:
- For grading questions where the gold target is a number, the predicted answer needs to be correct to the last significant figure in the gold answer. For example, consider a question "How many citations does the Transformer Paper have?" with gold target "120k".
    - Predicted answers "120k", "124k", and 115k" are all CORRECT.
    - Predicted answers "100k" and "113k" are INCORRECT.
    - Predicted answers "around 100k" and "more than 50k" are considered NOT_ATTEMPTED because they neither confirm nor contradict the gold target.
- The gold target may contain more information than the question. In such cases, the predicted answer only needs to contain the information that is in the question.
    - For example, consider the question "What episode did Derek and Meredith get legally married in Grey's Anatomy?" with gold target "Season 7, Episode 20: White Wedding". Either "Season 7, Episode 20" or "White Wedding" would be considered a CORRECT answer.
- Do not punish predicted answers if they omit information that would be clearly inferred from the question.
    - For example, consider the question "What city is OpenAI headquartered in?" and the gold target "San Francisco, California". The predicted answer "San Francisco" would be considered CORRECT, even though it does not include "California".
    - Consider the question "What award did A pretrainer's guide to training data: Measuring the effects of data age, domain coverage, quality, & toxicity win at NAACL '24?", the gold target is "Outstanding Paper Award". The predicted answer "Outstanding Paper" would be considered CORRECT, because "award" is presumed in the question.
    - For the question "What is the height of Jason Wei in meters?", the gold target is "1.73 m". The predicted answer "1.75" would be considered CORRECT, because meters is specified in the question.
    - For the question "What is the name of Barack Obama's wife?", the gold target is "Michelle Obama". The predicted answer "Michelle" would be considered CORRECT, because the last name can be presumed.
- Do not punish for typos in people's name if it's clearly the same name.
    - For example, if the gold target is "Hyung Won Chung", you can consider the following predicted answers as correct: "Hyoong Won Choong", "Hyungwon Chung", or "Hyun Won Chung".


Here is a new example. Simply reply with either CORRECT, INCORRECT, NOT ATTEMPTED. Don't apologize or correct yourself if there was a mistake; we are just trying to grade the answer.
```
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
```

Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: NOT_ATTEMPTED

Just return the letters "A", "B", or "C", with no text around it.
""".strip()

DEFAULT_JUDGE_MODEL = "gemini-2.5-pro"

_LETTER_TOKEN = re.compile(r"(?<![A-Za-z])([ABCabc])(?![A-Za-z])")
_WORD_TO_LETTER = [
    (re.compile(r"\bnot[\s_-]*attempted\b", re.IGNORECASE), "C"),
    (re.compile(r"\bincorrect\b", re.IGNORECASE), "B"),
    (re.compile(r"\bcorrect\b", re.IGNORECASE), "A"),
]


@dataclass
class Grade:
    letter: str | None
    label: str
    raw: str
    method: str


LETTER_TO_LABEL = {"A": "CORRECT", "B": "INCORRECT", "C": "NOT_ATTEMPTED"}


def build_judge_prompt(question: str, gold: str, prediction: str) -> str:
    return SIMPLEQA_GRADER_TEMPLATE.format(question=question, target=gold, predicted_answer=prediction)


def parse_grade(reply: str) -> Grade:
    """Extract a grade letter; unmatched defaults to C (NOT_ATTEMPTED), matching
    openai/simple-evals behaviour."""
    if not reply:
        return Grade(None, "UNPARSEABLE", "", "none")
    letters = [m.group(1).upper() for m in _LETTER_TOKEN.finditer(reply)]
    letters = [c for c in letters if c in "ABC"]
    if letters:
        L = letters[-1]
        return Grade(L, LETTER_TO_LABEL[L], reply, "letter")
    hits: list[tuple[int, str]] = []
    for pat, letter in _WORD_TO_LETTER:
        for m in pat.finditer(reply):
            hits.append((m.start(), letter))
    if hits:
        hits.sort()
        L = hits[-1][1]
        return Grade(L, LETTER_TO_LABEL[L], reply, "word")
    return Grade("C", "NOT_ATTEMPTED", reply, "default_C")


def grade_open(question: str, gold: str, prediction: str, *, judge_model: str, gemini_fn=None) -> Grade:
    """Grade one (gold, prediction) pair with the SimpleQA rubric. `gemini_fn`
    is injectable so tests stay offline."""
    if gemini_fn is None:
        from ..backends.gemini import call_gemini as gemini_fn
    reply = gemini_fn(build_judge_prompt(question, gold, prediction), model=judge_model)
    return parse_grade(reply)

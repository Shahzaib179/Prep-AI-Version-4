"""Pure mastery maths for Prep AI (no database, no Streamlit - easy to test).

How a mastery score (0-100) is calculated
-----------------------------------------
1. Skipped questions count as a soft miss (half of a wrong answer), so skipping every
   hard question cannot make a topic look strong. They are NOT counted in accuracy.
2. Recent answers count more than old ones (each older answer is worth 10% less).
3. Difficulty matters fairly: a correct Hard answer earns more credit, and a wrong
   Easy answer costs more than a wrong Hard answer.
4. Small samples are pulled toward 50%. One lucky answer can no longer make a topic
   "Mastered" and one slip can no longer make it "Very Weak".

    mastery = (weighted_correct + 2 * 0.5) / (weighted_total + 2)
"""
from __future__ import annotations

from typing import Any, Iterable

PRIOR_MEAN = 0.5          # what we assume before we have any evidence
PRIOR_WEIGHT = 2.0        # worth about 2 answers
RECENCY_DECAY = 0.90      # each older answer counts 10% less than the next one
SKIP_WEIGHT = 0.5         # a skipped question counts as half of a wrong answer
RECENT_WINDOW = 5         # "recent accuracy" looks at the last 5 answers

MIN_ATTEMPTS_FOR_LABEL = 3   # below this we say "not enough data yet"
WEAK_BELOW = 60              # < 60  -> Weak Area
STRONG_FROM = 75             # >= 75 -> Strong Area

MASTERY_LABELS = [
    (0, 39, "Very Weak"),
    (40, 59, "Weak"),
    (60, 74, "Developing"),
    (75, 89, "Strong"),
    (90, 100, "Mastered"),
]

_DIFF_CREDIT = {"easy": 0.8, "medium": 1.0, "hard": 1.25}   # weight of a CORRECT answer
_DIFF_SCORE = {"easy": 25.0, "medium": 50.0, "hard": 100.0}


def norm(text: Any) -> str:
    """Case/space-insensitive key so 'Biology', 'biology ' and 'BIOLOGY' are one topic."""
    return " ".join(str(text or "").split()).lower()


def mastery_label(score: float) -> str:
    if score < 40:
        return "Very Weak"
    if score < 60:
        return "Weak"
    if score < 75:
        return "Developing"
    if score < 90:
        return "Strong"
    return "Mastered"


def _diff(value: Any) -> str:
    v = norm(value)
    return v if v in _DIFF_CREDIT else "medium"


def compute_mastery(attempts: Iterable[dict[str, Any]]) -> dict[str, float | int]:
    """attempts: questions shown, OLDEST first. Each needs is_correct, difficulty, optional skipped."""
    items = list(attempts)
    answered = [x for x in items if not x.get("skipped")]
    n_ans = len(answered)
    if not items:
        return {"score": 0.0, "attempts": 0, "skipped": 0, "correct": 0, "wrong": 0,
                "accuracy": 0.0, "recent_accuracy": 0.0, "difficulty_score": 0.0}

    weighted_correct = 0.0
    weighted_total = 0.0
    for age, item in enumerate(reversed(items)):          # age 0 = newest question
        recency = RECENCY_DECAY ** age
        d = _diff(item.get("difficulty"))
        skipped = bool(item.get("skipped"))
        correct = bool(item.get("is_correct")) and not skipped
        # correct: Hard counts more.  wrong: Easy counts more (inverse weight).
        weight = recency * (_DIFF_CREDIT[d] if correct else 1.0 / _DIFF_CREDIT[d])
        if skipped:
            weight *= SKIP_WEIGHT
        weighted_total += weight
        weighted_correct += weight if correct else 0.0

    score = 100.0 * (weighted_correct + PRIOR_WEIGHT * PRIOR_MEAN) / (weighted_total + PRIOR_WEIGHT)
    correct_n = sum(1 for x in answered if x.get("is_correct"))
    recent = answered[-RECENT_WINDOW:]
    return {
        "score": round(max(0.0, min(100.0, score)), 1),
        "attempts": n_ans,                               # ANSWERED questions
        "skipped": len(items) - n_ans,
        "correct": correct_n,
        "wrong": n_ans - correct_n,
        "accuracy": round(100.0 * correct_n / n_ans, 1) if n_ans else 0.0,
        "recent_accuracy": round(100.0 * sum(1 for x in recent if x.get("is_correct")) / len(recent), 1) if recent else 0.0,
        "difficulty_score": round(sum(_DIFF_SCORE[_diff(x.get("difficulty"))] for x in items) / len(items), 1),
    }

"""Timer maths for timed quizzes. Pure Python (no Streamlit) so it is easy to test."""
from __future__ import annotations

import time
from datetime import datetime, timezone

from config import QUIZ_MAX_MINUTES


def minutes_to_seconds(minutes: float | int | None) -> int:
    """0 / None / negative -> 0 (= no time limit). Capped at QUIZ_MAX_MINUTES."""
    try:
        m = float(minutes or 0)
    except (TypeError, ValueError):
        return 0
    if m <= 0:
        return 0
    return int(min(m, QUIZ_MAX_MINUTES) * 60)


def suggested_minutes(question_count: int) -> int:
    """A relaxed default: about 1 minute per question."""
    return max(1, int(question_count))


def iso_to_epoch(iso: str) -> float:
    """Parse the UTC ISO strings stored in SQLite into epoch seconds."""
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def remaining_seconds(started_epoch: float | None, limit_sec: int, now: float | None = None) -> int | None:
    """Seconds left, never below 0. None means the quiz is untimed (or not started)."""
    if not limit_sec or started_epoch is None:
        return None
    now = time.time() if now is None else now
    return max(0, int(round(started_epoch + limit_sec - now)))


def is_expired(started_epoch: float | None, limit_sec: int, now: float | None = None) -> bool:
    left = remaining_seconds(started_epoch, limit_sec, now)
    return left is not None and left <= 0


def elapsed_seconds(started_epoch: float | None, limit_sec: int = 0, now: float | None = None) -> int:
    """Time spent so far, clipped to the limit when there is one."""
    if started_epoch is None:
        return 0
    now = time.time() if now is None else now
    spent = max(0, int(round(now - started_epoch)))
    return min(spent, limit_sec) if limit_sec else spent


def clock(seconds: int | None) -> str:
    if seconds is None:
        return "No limit"
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

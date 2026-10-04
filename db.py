from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from config import DB_PATH
from mastery_model import (
    MIN_ATTEMPTS_FOR_LABEL,
    STRONG_FROM,
    WEAK_BELOW,
    compute_mastery,
    mastery_label,
    norm,
)


class ClosingConnection(sqlite3.Connection):
    """SQLite connection that closes after a ``with connect()`` block.

    sqlite3.Connection commits/rolls back on context exit but does not close
    the connection. Closing here prevents stale connections from locking the
    SQLite database during quiz -> mastery -> revision updates.
    """

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


MASTERY_MODEL_VERSION = 2  # bump to force every student's mastery to be recomputed


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH, check_same_thread=False, factory=ClosingConnection)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.execute("PRAGMA busy_timeout = 5000")
    return con


def _ensure_column(con: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row[1] for row in con.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def init_db() -> None:
    with connect() as con:
        con.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, exam_name TEXT DEFAULT '',
            exam_date TEXT, level TEXT DEFAULT 'MDCAT', daily_minutes INTEGER DEFAULT 60,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS student_preferences (
            student_id TEXT PRIMARY KEY REFERENCES students(id) ON DELETE CASCADE,
            preferred_difficulty TEXT DEFAULT 'Medium', preferred_language TEXT DEFAULT 'English',
            explanation_style TEXT DEFAULT 'Detailed', learning_style TEXT DEFAULT 'Examples + Practice',
            llm_model TEXT DEFAULT 'openai/gpt-oss-120b', ui_color TEXT DEFAULT 'Blue'
        );
        CREATE TABLE IF NOT EXISTS study_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT REFERENCES students(id),
            subject TEXT, chapter TEXT, topic TEXT, mode TEXT, started_at TEXT, minutes INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, question_text TEXT, subject TEXT,
            topic TEXT, concept TEXT, difficulty TEXT, correct_answer TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS quiz_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, subject TEXT, topic TEXT,
            total INTEGER, correct INTEGER, incorrect INTEGER, skipped INTEGER, score REAL,
            difficulty TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS question_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT, quiz_id INTEGER, student_id TEXT,
            question_text TEXT, selected_answer TEXT, correct_answer TEXT, is_correct INTEGER,
            subject TEXT, topic TEXT, concept TEXT, difficulty TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS mistakes (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, subject TEXT, topic TEXT,
            concept TEXT, question_text TEXT, wrong_answer TEXT, correct_answer TEXT,
            explanation TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS mastery (
            student_id TEXT, subject TEXT, chapter TEXT DEFAULT '', topic TEXT DEFAULT '', concept TEXT DEFAULT '',
            attempts INTEGER DEFAULT 0, correct INTEGER DEFAULT 0, accuracy REAL DEFAULT 0,
            difficulty_score REAL DEFAULT 0, recent_accuracy REAL DEFAULT 0, repeated_mistakes INTEGER DEFAULT 0,
            mastery_score REAL DEFAULT 0, last_studied TEXT, PRIMARY KEY(student_id, subject, chapter, topic, concept)
        );
        CREATE TABLE IF NOT EXISTS revision_schedule (
            student_id TEXT, subject TEXT, topic TEXT, next_review TEXT, interval_days INTEGER DEFAULT 1,
            mastery REAL DEFAULT 0, last_studied TEXT, PRIMARY KEY(student_id, subject, topic)
        );
        CREATE TABLE IF NOT EXISTS study_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, exam_name TEXT, exam_date TEXT,
            plan_json TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS achievements (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, code TEXT, title TEXT,
            unlocked_at TEXT, UNIQUE(student_id, code)
        );
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, memory_type TEXT, content TEXT,
            subject TEXT DEFAULT '', topic TEXT DEFAULT '', importance REAL DEFAULT 0.5,
            confidence REAL DEFAULT 0.7, created_at TEXT, updated_at TEXT, last_used_at TEXT, is_active INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS agent_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, agent_name TEXT, user_input TEXT,
            output TEXT, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS merit_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, exam TEXT, formula_id TEXT,
            formula_name TEXT, program TEXT, marks_json TEXT, aggregate REAL, created_at TEXT
        );
        CREATE TABLE IF NOT EXISTS shared_quizzes (
            code TEXT PRIMARY KEY, title TEXT, subject TEXT, topic TEXT, difficulty TEXT,
            questions_json TEXT NOT NULL, time_limit_sec INTEGER DEFAULT 0,
            created_by TEXT, created_at TEXT, is_open INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS shared_quiz_starts (
            code TEXT, student_id TEXT, started_at TEXT, PRIMARY KEY(code, student_id)
        );
        CREATE TABLE IF NOT EXISTS goals (
            id INTEGER PRIMARY KEY AUTOINCREMENT, student_id TEXT, goal TEXT, target_date TEXT, status TEXT DEFAULT 'active'
        );
        """)
        # Lightweight schema migration for V4 settings added after the initial release.
        _ensure_column(con, "student_preferences", "llm_model", "TEXT DEFAULT 'openai/gpt-oss-120b'")
        _ensure_column(con, "student_preferences", "ui_color", "TEXT DEFAULT 'Blue'")
        _ensure_column(con, "students", "role", "TEXT DEFAULT 'student'")
        _ensure_column(con, "students", "last_active", "TEXT")
        _ensure_column(con, "student_preferences", "theme_preset", "TEXT DEFAULT 'Default'")
        _ensure_column(con, "student_preferences", "theme_bg", "TEXT DEFAULT ''")
        _ensure_column(con, "student_preferences", "theme_text", "TEXT DEFAULT ''")
        _ensure_column(con, "quiz_attempts", "shared_code", "TEXT")
        _ensure_column(con, "quiz_attempts", "time_taken_sec", "INTEGER")
        _ensure_column(con, "quiz_attempts", "time_limit_sec", "INTEGER DEFAULT 0")
        _ensure_column(con, "quiz_attempts", "timed_out", "INTEGER DEFAULT 0")
        con.execute("CREATE UNIQUE INDEX IF NOT EXISTS ux_attempt_shared ON quiz_attempts(shared_code, student_id) WHERE shared_code IS NOT NULL")
        version = con.execute("PRAGMA user_version").fetchone()[0]
        needs_rebuild = version < MASTERY_MODEL_VERSION
    if needs_rebuild:
        # Old scores were saved with the previous (inaccurate) formula: recompute them
        # from the raw answers, which were always stored correctly.
        with connect() as con:
            ids = [r[0] for r in con.execute("SELECT DISTINCT student_id FROM question_attempts").fetchall()]
        for sid in ids:
            rebuild_mastery(sid)
        with connect() as con:
            con.execute(f"PRAGMA user_version = {MASTERY_MODEL_VERSION}")


def ensure_student(student_id: str, name: str = "Student") -> None:
    now = datetime.utcnow().isoformat()
    with connect() as con:
        con.execute("INSERT OR IGNORE INTO students(id,name,created_at,updated_at) VALUES(?,?,?,?)", (student_id, name, now, now))
        con.execute("UPDATE students SET name=?, updated_at=? WHERE id=?", (name, now, student_id))
        con.execute("INSERT OR IGNORE INTO student_preferences(student_id) VALUES(?)", (student_id,))


def get_student(student_id: str) -> dict[str, Any]:
    with connect() as con:
        row = con.execute("SELECT * FROM students WHERE id=?", (student_id,)).fetchone()
    return dict(row) if row else {}


def update_student(student_id: str, **fields: Any) -> None:
    allowed = {"name", "exam_name", "exam_date", "level", "daily_minutes"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    fields["updated_at"] = datetime.utcnow().isoformat()
    sql = ", ".join(f"{k}=?" for k in fields)
    with connect() as con:
        con.execute(f"UPDATE students SET {sql} WHERE id=?", (*fields.values(), student_id))


def get_preferences(student_id: str) -> dict[str, Any]:
    with connect() as con:
        row = con.execute("SELECT * FROM student_preferences WHERE student_id=?", (student_id,)).fetchone()
    return dict(row) if row else {}


def update_preferences(student_id: str, **fields: str) -> None:
    allowed = {"preferred_difficulty", "preferred_language", "explanation_style", "learning_style", "llm_model", "ui_color", "theme_preset", "theme_bg", "theme_text"}
    fields = {k: v for k, v in fields.items() if k in allowed}
    if not fields:
        return
    with connect() as con:
        sets = ", ".join(f"{k}=?" for k in fields)
        con.execute(f"UPDATE student_preferences SET {sets} WHERE student_id=?", (*fields.values(), student_id))


def record_quiz(student_id: str, subject: str, topic: str, questions: list[dict[str, Any]], answers: dict[int, str], difficulty: str,
                *, shared_code: str | None = None, time_taken_sec: int | None = None, time_limit_sec: int = 0, timed_out: bool = False) -> int:
    correct = incorrect = skipped = 0
    now = datetime.utcnow().isoformat()
    with connect() as con:
        cur = con.execute("INSERT INTO quiz_attempts(student_id,subject,topic,total,correct,incorrect,skipped,score,difficulty,created_at,shared_code,time_taken_sec,time_limit_sec,timed_out) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (student_id, subject, topic, len(questions), 0, 0, 0, 0, difficulty, now, shared_code, time_taken_sec, int(time_limit_sec or 0), int(bool(timed_out))))
        quiz_id = cur.lastrowid
        for i, q in enumerate(questions):
            selected = answers.get(i, "")
            correct_answer = str(q.get("answer", "")).strip()
            is_correct = bool(selected and selected == correct_answer)
            correct += int(is_correct)
            if not selected:
                skipped += 1
            else:
                incorrect += int(not is_correct)
            con.execute("INSERT INTO question_attempts(quiz_id,student_id,question_text,selected_answer,correct_answer,is_correct,subject,topic,concept,difficulty,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)", (quiz_id, student_id, q.get("question", ""), selected, correct_answer, int(is_correct), subject, topic, q.get("concept", ""), q.get("difficulty", difficulty), now))
            if not is_correct and selected:
                con.execute("INSERT INTO mistakes(student_id,subject,topic,concept,question_text,wrong_answer,correct_answer,explanation,created_at) VALUES(?,?,?,?,?,?,?,?,?)", (student_id, subject, topic, q.get("concept", ""), q.get("question", ""), selected, correct_answer, q.get("explanation", ""), now))
        total = len(questions)
        score = (correct / total * 100) if total else 0
        con.execute("UPDATE quiz_attempts SET correct=?,incorrect=?,skipped=?,score=? WHERE id=?", (correct, incorrect, skipped, score, quiz_id))
    update_mastery_from_quiz(student_id, subject, topic, questions, answers)
    return int(quiz_id)


def _attempt_rows(student_id: str) -> list[sqlite3.Row]:
    """Every question the student was shown (answered or skipped), oldest first."""
    with connect() as con:
        return con.execute(
            "SELECT subject,topic,concept,difficulty,is_correct,created_at,"
            "(TRIM(COALESCE(selected_answer,''))='') AS skipped FROM question_attempts "
            "WHERE student_id=? ORDER BY id ASC",
            (student_id,),
        ).fetchall()


def _group(rows: list[sqlite3.Row], level: str) -> dict[tuple, dict[str, Any]]:
    """Group attempts by (subject, topic) or (subject, topic, concept), ignoring case/spaces."""
    groups: dict[tuple, dict[str, Any]] = {}
    for r in rows:
        topic = (r["topic"] or "").strip() or "General"
        concept = (r["concept"] or "").strip() or topic
        key = (norm(r["subject"]), norm(topic)) + ((norm(concept),) if level == "concept" else ())
        g = groups.setdefault(key, {"attempts": []})
        g["attempts"].append({"is_correct": bool(r["is_correct"]), "difficulty": r["difficulty"], "skipped": bool(r["skipped"])})
        # keep the FIRST spelling for display so 'Biology' and 'biology ' show as one stable name
        g.setdefault("subject", (r["subject"] or "").strip() or "General")
        g.setdefault("topic", topic)
        g.setdefault("concept", concept)
        g["last_studied"] = r["created_at"]
    return groups


def rebuild_mastery(student_id: str) -> None:
    """Recompute concept mastery + revision schedule from the raw answers (single source of truth)."""
    rows = _attempt_rows(student_id)
    concept_groups = _group(rows, "concept")
    topic_groups = _group(rows, "topic")
    with connect() as con:
        con.execute("DELETE FROM mastery WHERE student_id=?", (student_id,))
        for key, g in concept_groups.items():
            m = compute_mastery(g["attempts"])
            parent = topic_groups[key[:2]]          # use the topic's display spelling everywhere
            g["subject"], g["topic"] = parent["subject"], parent["topic"]
            con.execute(
                "INSERT OR REPLACE INTO mastery(student_id,subject,chapter,topic,concept,attempts,correct,accuracy,"
                "difficulty_score,recent_accuracy,repeated_mistakes,mastery_score,last_studied) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (student_id, g["subject"], "", g["topic"], g["concept"], m["attempts"], m["correct"],
                 m["accuracy"], m["difficulty_score"], m["recent_accuracy"], m["wrong"], m["score"],
                 g["last_studied"]),
            )
        con.execute("DELETE FROM revision_schedule WHERE student_id=?", (student_id,))
    for g in topic_groups.values():
        schedule_revision(student_id, g["subject"], g["topic"], compute_mastery(g["attempts"])["score"],
                          studied_on=str(g["last_studied"])[:10])


def update_mastery_from_quiz(student_id: str, subject: str, topic: str, questions: list[dict[str, Any]], answers: dict[int, str]) -> None:
    """Called after every quiz. Answers are already saved, so just recompute from them."""
    rebuild_mastery(student_id)


def schedule_revision(student_id: str, subject: str, topic: str, mastery_score: float, studied_on: str | None = None) -> None:
    if mastery_score < 40:
        interval = 1
    elif mastery_score < 60:
        interval = 2
    elif mastery_score < 75:
        interval = 4
    elif mastery_score < 90:
        interval = 7
    else:
        interval = 14
    try:
        base = date.fromisoformat(studied_on[:10]) if studied_on else datetime.utcnow().date()
    except ValueError:
        base = datetime.utcnow().date()
    next_review = (base + timedelta(days=interval)).isoformat()
    with connect() as con:
        con.execute("INSERT INTO revision_schedule(student_id,subject,topic,next_review,interval_days,mastery,last_studied) VALUES(?,?,?,?,?,?,?) ON CONFLICT(student_id,subject,topic) DO UPDATE SET next_review=excluded.next_review,interval_days=excluded.interval_days,mastery=excluded.mastery,last_studied=excluded.last_studied", (student_id, subject, topic, next_review, interval, mastery_score, studied_on or datetime.utcnow().isoformat()))


def topic_mastery(student_id: str) -> list[dict[str, Any]]:
    """One row per (subject, topic): mastery pooled over ALL answers in that topic."""
    out = []
    for g in _group(_attempt_rows(student_id), "topic").values():
        m = compute_mastery(g["attempts"])
        out.append({
            "subject": g["subject"], "topic": g["topic"], "mastery_score": m["score"],
            "attempts": m["attempts"], "accuracy": m["accuracy"], "recent_accuracy": m["recent_accuracy"],
            "wrong": m["wrong"], "skipped": m["skipped"], "label": mastery_label(m["score"]),
            "enough_data": m["attempts"] >= MIN_ATTEMPTS_FOR_LABEL,
        })
    return out


def weak_strong_areas(student_id: str, limit: int = 5) -> dict[str, list[dict[str, Any]]]:
    """Dashboard buckets. Topics with too few answers go to 'building' instead of being mislabeled."""
    rows = topic_mastery(student_id)
    solid = [r for r in rows if r["enough_data"]]
    return {
        "weak": sorted((r for r in solid if r["mastery_score"] < WEAK_BELOW), key=lambda r: (r["mastery_score"], -r["attempts"]))[:limit],
        "strong": sorted((r for r in solid if r["mastery_score"] >= STRONG_FROM), key=lambda r: (-r["mastery_score"], -r["attempts"]))[:limit],
        "developing": [r for r in solid if WEAK_BELOW <= r["mastery_score"] < STRONG_FROM],
        "building": [r for r in rows if not r["enough_data"]],
    }


def get_topic_mastery(student_id: str, subject: str, topic: str) -> dict[str, Any] | None:
    for r in topic_mastery(student_id):
        if norm(r["subject"]) == norm(subject) and norm(r["topic"]) == norm(topic):
            return r
    return None


def dashboard_stats(student_id: str) -> dict[str, Any]:
    rows = _attempt_rows(student_id)
    topics = topic_mastery(student_id)
    answered = [r for r in rows if not r["skipped"]]
    correct = sum(1 for r in answered if r["is_correct"])
    with connect() as con:
        due = con.execute("SELECT COUNT(*) FROM revision_schedule WHERE student_id=? AND next_review<=date('now')", (student_id,)).fetchone()[0]
    overall = sum(t["mastery_score"] for t in topics) / len(topics) if topics else 0.0
    ranked = sorted(topics, key=lambda t: t["mastery_score"])
    return {
        "overall": round(overall, 1), "attempted": len(answered), "skipped": len(rows) - len(answered),
        "correct": correct, "incorrect": len(answered) - correct,
        "accuracy": round(100 * correct / max(1, len(answered)), 1),
        "weak": ranked[0] if ranked else None, "strong": ranked[-1] if ranked else None,
        "topics_tracked": len(topics), "revision_due": due,
    }


def weak_topics(student_id: str, limit: int = 10, max_score: float | None = STRONG_FROM) -> list[dict[str, Any]]:
    """Concept rows that still need practice (below 75%), weakest first."""
    sql = ("SELECT subject,topic,concept,mastery_score,attempts,repeated_mistakes FROM mastery WHERE student_id=?")
    params: list[Any] = [student_id]
    if max_score is not None:
        sql += " AND mastery_score < ?"
        params.append(max_score)
    sql += " ORDER BY mastery_score ASC, attempts DESC LIMIT ?"
    params.append(limit)
    with connect() as con:
        rows = con.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def recent_mistakes(student_id: str, limit: int = 10) -> list[dict[str, Any]]:
    with connect() as con:
        rows = con.execute("SELECT * FROM mistakes WHERE student_id=? ORDER BY id DESC LIMIT ?", (student_id, limit)).fetchall()
    return [dict(r) for r in rows]


def due_revisions(student_id: str) -> list[dict[str, Any]]:
    with connect() as con:
        rows = con.execute("SELECT * FROM revision_schedule WHERE student_id=? AND next_review<=date('now') ORDER BY mastery ASC", (student_id,)).fetchall()
    return [dict(r) for r in rows]


def revision_recommendations(student_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return weak topics that are worth revising even before their scheduled date."""
    with connect() as con:
        rows = con.execute(
            """
            SELECT
                m.subject,
                m.topic,
                m.concept,
                m.mastery_score,
                m.attempts,
                m.repeated_mistakes,
                m.last_studied,
                r.next_review,
                r.interval_days
            FROM mastery AS m
            LEFT JOIN revision_schedule AS r
              ON r.student_id = m.student_id
             AND r.subject = m.subject
             AND r.topic = m.topic
            WHERE m.student_id = ?
              AND m.mastery_score < 75
            ORDER BY m.mastery_score ASC,
                     m.repeated_mistakes DESC,
                     m.attempts DESC
            LIMIT ?
            """,
            (student_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def upcoming_revisions(student_id: str, limit: int = 10) -> list[dict[str, Any]]:
    """Return scheduled revisions whose review date is still in the future."""
    with connect() as con:
        rows = con.execute(
            """
            SELECT *
            FROM revision_schedule
            WHERE student_id = ?
              AND next_review > date('now')
            ORDER BY next_review ASC, mastery ASC
            LIMIT ?
            """,
            (student_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def history(student_id: str, limit: int = 50) -> list[dict[str, Any]]:
    with connect() as con:
        rows = con.execute("SELECT * FROM quiz_attempts WHERE student_id=? ORDER BY id DESC LIMIT ?", (student_id, limit)).fetchall()
    return [dict(r) for r in rows]


def save_plan(student_id: str, exam_name: str, exam_date: str, plan: Any) -> None:
    with connect() as con:
        con.execute("INSERT INTO study_plans(student_id,exam_name,exam_date,plan_json,created_at) VALUES(?,?,?,?,?)", (student_id, exam_name, exam_date, json.dumps(plan), datetime.utcnow().isoformat()))


def add_achievement(student_id: str, code: str, title: str) -> None:
    with connect() as con:
        con.execute("INSERT OR IGNORE INTO achievements(student_id,code,title,unlocked_at) VALUES(?,?,?,?)", (student_id, code, title, datetime.utcnow().isoformat()))


def achievements(student_id: str) -> list[dict[str, Any]]:
    with connect() as con:
        return [dict(r) for r in con.execute("SELECT * FROM achievements WHERE student_id=? ORDER BY id DESC", (student_id,)).fetchall()]


def save_agent_session(student_id: str, agent_name: str, user_input: str, output: str) -> None:
    with connect() as con:
        con.execute("INSERT INTO agent_sessions(student_id,agent_name,user_input,output,created_at) VALUES(?,?,?,?,?)", (student_id, agent_name, user_input, output, datetime.utcnow().isoformat()))


def get_agent_sessions(student_id: str, agent_name: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    with connect() as con:
        if agent_name:
            rows = con.execute(
                "SELECT * FROM agent_sessions WHERE student_id=? AND agent_name=? ORDER BY id DESC LIMIT ?",
                (student_id, agent_name, limit),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM agent_sessions WHERE student_id=? ORDER BY id DESC LIMIT ?",
                (student_id, limit),
            ).fetchall()
    return [dict(r) for r in rows]


# -----------------------------------------------------------------------------
# Merit Aggregate Agent history (used to pre-fill Path Finder)
# -----------------------------------------------------------------------------
def save_merit_result(student_id: str, exam: str, formula_id: str, formula_name: str, program: str, marks: dict[str, Any], aggregate: float) -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO merit_results(student_id,exam,formula_id,formula_name,program,marks_json,aggregate,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (student_id, exam, formula_id, formula_name, program or "", json.dumps(marks), float(aggregate), datetime.utcnow().isoformat()),
        )


def merit_results(student_id: str, limit: int = 20) -> list[dict[str, Any]]:
    with connect() as con:
        rows = con.execute("SELECT * FROM merit_results WHERE student_id=? ORDER BY id DESC LIMIT ?", (student_id, limit)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        try:
            d["marks"] = json.loads(d.pop("marks_json") or "{}")
        except json.JSONDecodeError:
            d["marks"] = {}
        out.append(d)
    return out


# -----------------------------------------------------------------------------
# Roles (student / tutor) and activity
# -----------------------------------------------------------------------------
def set_role(student_id: str, role: str) -> None:
    if role not in ("student", "tutor"):
        raise ValueError("role must be 'student' or 'tutor'")
    with connect() as con:
        con.execute("UPDATE students SET role=? WHERE id=?", (role, student_id))


def get_role(student_id: str) -> str:
    with connect() as con:
        row = con.execute("SELECT role FROM students WHERE id=?", (student_id,)).fetchone()
    return (row["role"] if row and row["role"] else "student")


def touch_active(student_id: str) -> None:
    with connect() as con:
        con.execute("UPDATE students SET last_active=? WHERE id=?", (datetime.utcnow().isoformat(), student_id))


def delete_agent_sessions(student_id: str) -> int:
    """Remove saved tutor/research conversation history for one student."""
    with connect() as con:
        return con.execute("DELETE FROM agent_sessions WHERE student_id=?", (student_id,)).rowcount


# -----------------------------------------------------------------------------
# Shared quizzes: one question set, many students, each attempt stored separately
# -----------------------------------------------------------------------------
_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no 0/O/1/I/L lookalikes


def normalize_code(code: str) -> str:
    return "".join(ch for ch in (code or "").upper() if ch.isalnum())


def create_shared_quiz(created_by: str, title: str, subject: str, topic: str, difficulty: str,
                       questions: list[dict[str, Any]], time_limit_sec: int = 0, code_length: int = 6) -> str:
    import secrets
    now = datetime.utcnow().isoformat()
    for _ in range(20):
        code = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(code_length))
        try:
            with connect() as con:
                con.execute(
                    "INSERT INTO shared_quizzes(code,title,subject,topic,difficulty,questions_json,time_limit_sec,created_by,created_at,is_open) VALUES(?,?,?,?,?,?,?,?,?,1)",
                    (code, title, subject, topic, difficulty, json.dumps(questions), int(time_limit_sec or 0), created_by, now),
                )
            return code
        except sqlite3.IntegrityError:
            continue
    raise RuntimeError("Could not generate a unique quiz code.")


def get_shared_quiz(code: str) -> dict[str, Any] | None:
    with connect() as con:
        row = con.execute("SELECT * FROM shared_quizzes WHERE code=?", (normalize_code(code),)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["questions"] = json.loads(d.pop("questions_json") or "[]")
    return d


def set_shared_quiz_open(code: str, is_open: bool) -> None:
    with connect() as con:
        con.execute("UPDATE shared_quizzes SET is_open=? WHERE code=?", (int(is_open), normalize_code(code)))


def list_shared_quizzes(created_by: str | None = None) -> list[dict[str, Any]]:
    sql = ("SELECT s.code,s.title,s.subject,s.topic,s.difficulty,s.time_limit_sec,s.created_by,s.created_at,s.is_open,"
           "(SELECT COUNT(*) FROM quiz_attempts a WHERE a.shared_code=s.code) AS attempts,"
           "(SELECT ROUND(AVG(a.score),1) FROM quiz_attempts a WHERE a.shared_code=s.code) AS avg_score,"
           "(SELECT name FROM students WHERE id=s.created_by) AS creator_name FROM shared_quizzes s")
    params: tuple = ()
    if created_by:
        sql += " WHERE s.created_by=?"
        params = (created_by,)
    sql += " ORDER BY s.created_at DESC"
    with connect() as con:
        return [dict(r) for r in con.execute(sql, params).fetchall()]


def shared_quiz_start(code: str, student_id: str) -> str:
    """Record (once) when a student opened a shared quiz; return the ORIGINAL start time.

    Stored server-side so reloading the browser cannot restart a timed quiz.
    """
    code = normalize_code(code)
    with connect() as con:
        con.execute("INSERT OR IGNORE INTO shared_quiz_starts(code,student_id,started_at) VALUES(?,?,?)", (code, student_id, datetime.utcnow().isoformat()))
        row = con.execute("SELECT started_at FROM shared_quiz_starts WHERE code=? AND student_id=?", (code, student_id)).fetchone()
    return row["started_at"]


def has_attempted_shared(code: str, student_id: str) -> bool:
    with connect() as con:
        return con.execute("SELECT 1 FROM quiz_attempts WHERE shared_code=? AND student_id=?", (normalize_code(code), student_id)).fetchone() is not None


def record_shared_timeout(student_id: str, quiz: dict[str, Any], time_taken_sec: int) -> int:
    """The student's time ran out before they came back: store a 0-answered attempt.

    Deliberately NOT sent through mastery (no question_attempts rows) so a lost
    connection does not damage the student's topic scores.
    """
    now = datetime.utcnow().isoformat()
    total = len(quiz["questions"])
    with connect() as con:
        cur = con.execute(
            "INSERT INTO quiz_attempts(student_id,subject,topic,total,correct,incorrect,skipped,score,difficulty,created_at,shared_code,time_taken_sec,time_limit_sec,timed_out) VALUES(?,?,?,?,0,0,?,0,?,?,?,?,?,1)",
            (student_id, quiz["subject"], quiz["topic"], total, total, quiz["difficulty"], now, quiz["code"], int(time_taken_sec), int(quiz.get("time_limit_sec") or 0)),
        )
        return int(cur.lastrowid)


def shared_quiz_results(code: str) -> list[dict[str, Any]]:
    """Every student's attempt on one shared quiz, best first (ties: faster first)."""
    with connect() as con:
        rows = con.execute(
            "SELECT a.student_id, COALESCE(s.name,a.student_id) AS name, a.score, a.correct, a.incorrect, a.skipped, a.total, "
            "a.time_taken_sec, a.timed_out, a.created_at FROM quiz_attempts a LEFT JOIN students s ON s.id=a.student_id "
            "WHERE a.shared_code=? ORDER BY a.score DESC, COALESCE(a.time_taken_sec, 999999) ASC",
            (normalize_code(code),),
        ).fetchall()
    return [dict(r) for r in rows]


# -----------------------------------------------------------------------------
# Tutor view: performance of every individual student
# -----------------------------------------------------------------------------
def roster() -> list[dict[str, Any]]:
    """One summary row per student (tutors are excluded)."""
    with connect() as con:
        studs = [dict(r) for r in con.execute("SELECT id,name,level,created_at,last_active FROM students WHERE COALESCE(role,'student')='student' ORDER BY name COLLATE NOCASE").fetchall()]
        quiz = {r["student_id"]: dict(r) for r in con.execute(
            "SELECT student_id, COUNT(*) AS quizzes, ROUND(AVG(score),1) AS avg_score, MAX(created_at) AS last_quiz, SUM(timed_out) AS timeouts FROM quiz_attempts GROUP BY student_id").fetchall()}
    out = []
    for s in studs:
        q = quiz.get(s["id"], {})
        st = dashboard_stats(s["id"])
        out.append({
            "student_id": s["id"], "name": s["name"], "level": s["level"],
            "quizzes": q.get("quizzes", 0), "avg_score": q.get("avg_score") or 0.0,
            "questions_answered": st["attempted"], "accuracy": st["accuracy"], "mastery": st["overall"],
            "revision_due": st["revision_due"], "timeouts": int(q.get("timeouts") or 0),
            "last_active": q.get("last_quiz") or s.get("last_active") or s["created_at"],
        })
    return out


def student_quiz_trend(student_id: str, limit: int = 200) -> list[dict[str, Any]]:
    """Quiz attempts oldest -> newest (for line charts)."""
    with connect() as con:
        rows = con.execute(
            "SELECT id,subject,topic,total,correct,incorrect,skipped,score,difficulty,created_at,shared_code,time_taken_sec,time_limit_sec,timed_out "
            "FROM quiz_attempts WHERE student_id=? ORDER BY id DESC LIMIT ?", (student_id, limit)).fetchall()
    return [dict(r) for r in reversed(rows)]


def subject_breakdown(student_id: str) -> list[dict[str, Any]]:
    """Accuracy per subject across every answered question."""
    groups: dict[str, list[int]] = {}
    for r in _attempt_rows(student_id):
        if r["skipped"]:
            continue
        groups.setdefault((r["subject"] or "General").strip() or "General", []).append(int(r["is_correct"]))
    return [{"subject": k, "answered": len(v), "accuracy": round(100 * sum(v) / len(v), 1)} for k, v in sorted(groups.items())]

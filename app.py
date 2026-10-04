from __future__ import annotations

import hashlib
import hmac
import os
import random
import sqlite3
import time
import uuid
from datetime import date

import streamlit as st

from adaptive import difficulty_for_topic, next_best_action
from agent_pages import render_merit, render_pathfinder
from agent_system import AgentContext, Orchestrator
import charts
from config import (
    DEFAULT_GROQ_MODEL,
    GROQ_MODELS,
    MEMORY_MAX_ITEMS,
    QUIZ_MAX_MINUTES,
    SHARED_CODE_LENGTH,
    THEME_PRESETS,
    UI_COLORS,
    get_student_id,
    get_tutor_code,
)
from db import (
    add_achievement,
    achievements,
    create_shared_quiz,
    dashboard_stats,
    get_shared_quiz,
    list_shared_quizzes,
    has_attempted_shared,
    normalize_code,
    record_shared_timeout,
    set_role,
    set_shared_quiz_open,
    shared_quiz_results,
    shared_quiz_start,
    topic_mastery,
    touch_active,
    due_revisions,
    weak_strong_areas,
    revision_recommendations,
    student_quiz_trend,
    upcoming_revisions,
    ensure_student,
    get_agent_sessions,
    get_preferences,
    get_student,
    history,
    init_db,
    recent_mistakes,
    record_quiz,
    save_agent_session,
    save_plan,
    update_preferences,
    update_student,
    weak_topics,
)
from documents import chunk_pages, download_drive, extract_document
from groq_service import grounded_answer
from memory import LongTermMemory, memory_prompt, reset_memory
from quiz_runtime import clock, elapsed_seconds, is_expired, iso_to_epoch, minutes_to_seconds, remaining_seconds, suggested_minutes
from theme import resolve_theme, validate_theme
from tutor_pages import render_quiz_results, render_student_performance
from pdf_export import questions_to_pdf, text_to_pdf, research_to_pdf
from rag import build_index, hybrid_search, load_database_index
from ui import apply_theme, hero, source_cards
from voice_service import speak, transcribe


st.set_page_config(
    page_title="Prep AI V4",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()

# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "student_id" not in st.session_state:
    st.session_state.student_id = ""
if "profile_complete" not in st.session_state:
    st.session_state.profile_complete = False
if "student_name" not in st.session_state:
    st.session_state.student_name = ""
if "personal_chunks" not in st.session_state:
    st.session_state.personal_chunks = []
if "personal_index" not in st.session_state:
    st.session_state.personal_index = None
if "personal_info" not in st.session_state:
    st.session_state.personal_info = []
if "quiz" not in st.session_state:
    st.session_state.quiz = None
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "last_quiz_result" not in st.session_state:
    st.session_state.last_quiz_result = None
if "tutor_messages" not in st.session_state:
    st.session_state.tutor_messages = []
if "is_tutor" not in st.session_state:
    st.session_state.is_tutor = False
if "shared_quiz" not in st.session_state:
    st.session_state.shared_quiz = None
if "theme_pair" not in st.session_state:
    st.session_state.theme_pair = None


def _load_theme_prefs(prefs: dict) -> None:
    """Copy the saved theme of the active student into the session."""
    st.session_state.theme_preset = prefs.get("theme_preset") or "Default"
    st.session_state.theme_bg = prefs.get("theme_bg") or ""
    st.session_state.theme_text = prefs.get("theme_text") or ""


# -----------------------------------------------------------------------------
# First-run student profile
# -----------------------------------------------------------------------------
def render_profile_setup() -> None:
    st.markdown(
        """
        <div style="max-width:760px;margin:4rem auto 1rem auto;text-align:center;">
            <h1>🎓 Welcome to Prep AI V4</h1>
            <p style="font-size:1.1rem;color:#6b7280;">
                Students: create your profile. Your dashboard, learning history,
                tutor conversations, research memories, quiz performance and adaptive
                recommendations will be linked to your Student ID.
                Tutors: choose <b>Tutor</b> and enter the access code.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # The role picker sits OUTSIDE the form so the fields below change as soon as it is clicked.
    role = st.radio("I am a", ["Student", "Tutor"], horizontal=True, key="login_role")

    if role == "Tutor":
        with st.form("tutor_login"):
            tutor_code = st.text_input("Tutor access code", type="password", help="Ask the app administrator for the tutor access code.")
            tutor_submit = st.form_submit_button("Open Tutor Dashboard", type="primary")
        if tutor_submit:
            configured = get_tutor_code()
            if not configured:
                st.error("The Tutor role is not enabled. The administrator must set TUTOR_ACCESS_CODE in Streamlit Secrets.")
                return
            if not hmac.compare_digest(tutor_code.strip().encode(), configured.encode()):
                st.error("Incorrect tutor access code.")
                return
            # Tutors do not enter a name or ID: a stable private ID is derived from the code.
            tutor_id = "tutor_" + hashlib.sha256(configured.encode()).hexdigest()[:10]
            _start_session(tutor_id, "Tutor", "MDCAT", is_tutor=True)
        return

    with st.form("student_profile_setup"):
        name = st.text_input("Student name", placeholder="Enter your full name")
        sid = st.text_input(
            "Student ID",
            placeholder="Example: STU-001",
            help="Use a unique ID. This ID connects all future learning data to you.",
        )
        level = st.selectbox("Current level", ["MDCAT", "University", "Intermediate", "Beginner"])
        submitted = st.form_submit_button("Create / Open My Student Profile", type="primary")

    if submitted:
        name = name.strip()
        sid = "".join(ch for ch in sid.strip() if ch.isalnum() or ch in "_-. ").replace(" ", "_")[:64]
        if not name:
            st.error("Please enter your student name.")
            return
        if not sid:
            st.error("Please enter a valid Student ID.")
            return
        if sid.lower().startswith("tutor_"):
            st.error("Student IDs cannot start with 'tutor_'. Please choose another ID.")
            return
        _start_session(sid, name, level, is_tutor=False)


def _start_session(sid: str, name: str, level: str, is_tutor: bool) -> None:
    st.session_state.is_tutor = is_tutor
    st.session_state.shared_quiz = None
    st.session_state.exam = None
    st.session_state.student_id = sid
    st.session_state.student_name = name
    st.session_state.profile_complete = True
    st.session_state.llm_model = DEFAULT_GROQ_MODEL
    st.session_state.ui_color = "Blue"
    st.session_state.quiz = None
    st.session_state.quiz_answers = {}
    st.session_state.last_quiz_result = None
    st.session_state.personal_chunks = []
    st.session_state.personal_index = None
    st.session_state.personal_info = []
    st.session_state.tutor_messages = []
    ensure_student(sid, name)
    update_student(sid, name=name, level=level)
    if is_tutor:
        set_role(sid, "tutor")
    prefs = get_preferences(sid)
    _load_theme_prefs(prefs)
    st.session_state.llm_model = prefs.get("llm_model") or DEFAULT_GROQ_MODEL
    st.session_state.ui_color = prefs.get("ui_color") or "Blue"
    st.rerun()


if not st.session_state.profile_complete or not st.session_state.student_id:
    render_profile_setup()
    st.stop()


student_id = get_student_id()
student = get_student(student_id)
if not student:
    ensure_student(student_id, st.session_state.student_name or "Student")
    student = get_student(student_id)

# Load the saved name for this exact student. Do not overwrite it with a default
# value on every Streamlit rerun.
st.session_state.student_name = student.get("name") or st.session_state.student_name or "Student"

preferences = get_preferences(student_id)
if "llm_model" not in st.session_state:
    st.session_state.llm_model = preferences.get("llm_model") or DEFAULT_GROQ_MODEL
if st.session_state.llm_model not in GROQ_MODELS:
    st.session_state.llm_model = DEFAULT_GROQ_MODEL
if "ui_color" not in st.session_state:
    st.session_state.ui_color = preferences.get("ui_color") or "Blue"
if st.session_state.ui_color not in UI_COLORS:
    st.session_state.ui_color = "Blue"

if "theme_preset" not in st.session_state:
    _load_theme_prefs(preferences)
st.session_state.theme_pair = resolve_theme(st.session_state.theme_preset, st.session_state.theme_bg, st.session_state.theme_text)
apply_theme(UI_COLORS[st.session_state.ui_color], *(st.session_state.theme_pair or (None, None)))

if not st.session_state.get("_activity_logged"):
    touch_active(student_id)
    st.session_state._activity_logged = True


# -----------------------------------------------------------------------------
# Sidebar / identity
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 Prep AI V4")
    if st.session_state.is_tutor:
        st.markdown("**👩‍🏫 Tutor**")
        st.caption("Tutor access active")
        nav_pages = ["Tutor Dashboard"]
    else:
        st.markdown(f"**Student:** {st.session_state.student_name}")
        st.caption(f"Student ID: `{student_id}`")
        nav_pages = [
            "Dashboard",
            "Learn",
            "Practice",
            "Exam",
            "Published Quizzes",
            "AI Tutor",
            "Voice Tutor",
            "Research Agent",
            "Merit Calculator",
            "Path Finder",
            "Study Plan",
            "Memory",
            "History",
            "Settings",
        ]
    st.divider()
    page = st.radio("Navigation", nav_pages, label_visibility="collapsed")

    if st.button("Log out" if st.session_state.is_tutor else "Change Student Profile"):
        st.session_state.is_tutor = False
        st.session_state.shared_quiz = None
        st.session_state.exam = None
        st.session_state._activity_logged = False
        for _k in ("theme_preset", "theme_bg", "theme_text"):
            st.session_state.pop(_k, None)
        st.session_state.profile_complete = False
        st.session_state.student_id = ""
        st.session_state.student_name = ""
        st.session_state.tutor_messages = []
        st.session_state.quiz = None
        st.session_state.quiz_answers = {}
        st.session_state.last_quiz_result = None
        st.session_state.personal_chunks = []
        st.session_state.personal_index = None
        st.session_state.personal_info = []
        st.rerun()

    st.divider()
    st.caption("V4 = Adaptive learning + long-term memory + multi-agent AI")

orch = Orchestrator(student_id)


# -----------------------------------------------------------------------------
# Common helpers
# -----------------------------------------------------------------------------
def render_questions(questions: list[dict], title: str = "Generated MCQs", download_key: str = "mcqs") -> None:
    """Render MCQs as student-friendly cards and provide a PDF download."""
    st.subheader(title)
    for i, q in enumerate(questions, 1):
        st.markdown(f"### Q{i}. {q.get('question', 'Question unavailable')}")
        options = q.get("options") or {}
        for letter in ("A", "B", "C", "D"):
            if letter in options:
                st.markdown(f"**{letter}.** {options[letter]}")
        with st.expander("Answer & Explanation"):
            st.success(f"Correct answer: {q.get('answer', 'Not provided')}")
            st.write(q.get("explanation", "No explanation was provided."))
            if q.get("concept"):
                st.caption(f"Concept: {q['concept']}")
            if q.get("difficulty"):
                st.caption(f"Difficulty: {q['difficulty']}")

    st.download_button(
        "📥 Download MCQs as PDF",
        data=questions_to_pdf(title, questions),
        file_name="prep_ai_mcqs.pdf",
        mime="application/pdf",
        key=f"download_{download_key}",
    )


def create_context(
    request: str,
    subject: str,
    topic: str,
    difficulty: str,
    context: str,
) -> AgentContext:
    return AgentContext(
        student_id,
        request,
        subject,
        topic,
        student.get("level") or "MDCAT",
        difficulty,
        context,
    )


def build_rag_context(query: str, chunks: list[dict], index, top_k: int = 8) -> tuple[str, list[dict]]:
    results = hybrid_search(query, chunks, index, top_k)
    context = "\n\n".join(
        f"[{r.get('filename', 'Source')} | page {r.get('page') or 'N/A'}]\n{r.get('text', '')}"
        for r in results
    )
    return context, results


def save_tutor_turn(user_text: str, assistant_text: str, subject: str, topic: str) -> None:
    """Persist both sides of a tutor conversation in SQLite and semantic memory."""
    save_agent_session(student_id, "Tutor Agent", user_text, assistant_text)
    memory = LongTermMemory(student_id)
    memory.add(user_text, "tutor_student_message", subject, topic, importance=0.65, confidence=1.0)
    memory.add(assistant_text, "tutor_response", subject, topic, importance=0.75, confidence=0.85)


def save_research_memory(request: str, response: str) -> None:
    """Persist both research request and research response in long-term memory."""
    save_agent_session(student_id, "Research Agent", request, response)
    memory = LongTermMemory(student_id)
    memory.add(request, "research_request", "", request[:120], importance=0.65, confidence=1.0)
    memory.add(response, "research_response", "", request[:120], importance=0.85, confidence=0.85)


# -----------------------------------------------------------------------------
# Quiz engine: timer, one-attempt shared quizzes, results with charts
# -----------------------------------------------------------------------------
def new_quiz(questions: list[dict], subject: str, topic: str, difficulty: str, sources: list | None = None,
             limit_sec: int = 0, **extra) -> dict:
    """Every quiz (practice, exam, shared) is created here so they all behave the same way."""
    quiz = {
        "uid": uuid.uuid4().hex[:10],          # unique widget keys: a new quiz never inherits old selections
        "questions": questions, "subject": subject, "topic": topic, "difficulty": difficulty,
        "sources": sources or [], "time_limit_sec": int(limit_sec or 0),
        "started_at": None, "finished": False, "answers": {}, "answer_ts": {}, "result": None,
    }
    quiz.update(extra)
    return quiz


def time_limit_input(prefix: str, question_count: int) -> int:
    """Minutes input shown next to the question-count control. Returns SECONDS (0 = no limit)."""
    minutes = st.number_input(
        "Time limit (minutes) — 0 means no limit",
        min_value=0, max_value=QUIZ_MAX_MINUTES, value=0, step=1, key=f"{prefix}_minutes",
        help="The quiz is submitted automatically when the time is up.",
    )
    st.caption(f"Suggestion: about {suggested_minutes(question_count)} minutes for {question_count} questions.")
    return minutes_to_seconds(minutes)


@st.fragment(run_every="1s")
def quiz_timer(state_key: str) -> None:
    """Live countdown. Only this small fragment reruns every second, not the whole page."""
    quiz = st.session_state.get(state_key)
    if not quiz or quiz.get("result") or not quiz.get("time_limit_sec") or quiz.get("started_at") is None:
        return
    limit = quiz["time_limit_sec"]
    left = remaining_seconds(quiz["started_at"], limit) or 0
    low = left <= max(30, limit * 0.1)
    st.progress(max(0.0, min(1.0, left / limit)), text=f"{'⚠️' if low else '⏱'} Time left: {clock(left)}")
    if left <= 0 and not quiz.get("_expiry_rerun"):
        quiz["_expiry_rerun"] = True
        st.rerun()  # full rerun -> render_quiz_taker sees the expiry and auto-submits


def submit_quiz(state_key: str, timed_out: bool) -> None:
    quiz = st.session_state[state_key]
    questions, limit, started = quiz["questions"], quiz["time_limit_sec"], quiz["started_at"]
    deadline = started + limit if (limit and started is not None) else None
    # Answers picked after the deadline (+2s network grace) do not count.
    answers = {i: a for i, a in quiz["answers"].items() if deadline is None or quiz["answer_ts"].get(i, 0) <= deadline + 2}
    taken = elapsed_seconds(started, limit)
    try:
        quiz_id = record_quiz(
            student_id, quiz["subject"], quiz["topic"], questions, answers, quiz["difficulty"],
            shared_code=quiz.get("shared_code"), time_taken_sec=taken, time_limit_sec=limit, timed_out=timed_out,
        )
    except sqlite3.IntegrityError:
        st.error("This shared quiz was already submitted from your Student ID (maybe in another tab).")
        quiz["result"] = {"timeout_only": False, "already": True}
        return
    except Exception as exc:
        st.error("Prep AI could not save this quiz result. Please try again.")
        if st.session_state.get("debug_mode"):
            st.exception(exc)
        return
    n = len(questions)
    correct = sum(answers.get(i) == str(q.get("answer", "")).strip() for i, q in enumerate(questions))
    wrong = sum(bool(answers.get(i)) and answers.get(i) != str(q.get("answer", "")).strip() for i, q in enumerate(questions))
    quiz["finished"] = True
    quiz["result"] = {
        "answers": answers, "correct": correct, "incorrect": wrong, "skipped": n - correct - wrong, "total": n,
        "pct": (100 * correct / n) if n else 0.0, "final": correct - (wrong * 0.25 if quiz.get("negative") else 0),
        "taken": taken, "timed_out": timed_out, "quiz_id": quiz_id,
    }
    if n and correct == n and not quiz.get("shared_code"):
        add_achievement(student_id, "perfect_quiz", "Perfect Quiz")


def render_quiz_result(quiz: dict) -> None:
    r = quiz["result"]
    if r.get("already"):
        st.warning("You have already submitted this shared quiz from your Student ID, so this attempt was not saved again.")
        return
    if r.get("timeout_only"):
        st.error("⏰ Your time for this quiz ended before you came back. An unanswered attempt was recorded for your tutor.")
        return
    if r["timed_out"]:
        st.warning("⏰ Time is up — your quiz was submitted automatically with the answers you had selected.")
    st.markdown("## Quiz Result")
    c1, c2, c3 = st.columns(3)
    with c1:
        charts.show(charts.ring(r["pct"], "Score"))
    with c2:
        charts.show(charts.donut({"Correct": r["correct"], "Incorrect": r["incorrect"], "Skipped": r["skipped"]}, charts.OUTCOME_COLORS, "Answers", center_text=f"{r['correct']}/{r['total']}"))
    with c3:
        st.metric("Time used", clock(r["taken"]), help=f"Limit: {clock(quiz['time_limit_sec'])}" if quiz["time_limit_sec"] else "No time limit")
        if quiz.get("negative"):
            st.metric("Marks (with −0.25 negative marking)", f"{r['final']:.2f} / {r['total']}")
        else:
            st.metric("Correct", f"{r['correct']} of {r['total']}")
    if quiz.get("shared_code"):
        results = shared_quiz_results(quiz["shared_code"])
        mine = next((i for i, x in enumerate(results) if x["student_id"] == student_id), None)
        if mine is not None:
            avg = sum(x["score"] for x in results) / len(results)
            st.info(f"🏁 Rank **{mine + 1} of {len(results)}** so far · class average **{avg:.0f}%**. (Other students' names are visible to tutors only.)")
    for i, q in enumerate(quiz["questions"]):
        selected = r["answers"].get(i)
        correct_answer = str(q.get("answer", "")).strip()
        mark = "⏭ Skipped" if not selected else ("✅ Correct" if selected == correct_answer else "❌ Incorrect")
        st.markdown(f"### Q{i + 1}. {q.get('question', '')}")
        st.write(f"**Result:** {mark}")
        st.write(f"**Your answer:** {selected or 'Skipped'}")
        st.write(f"**Correct answer:** {correct_answer}")
        st.write(f"**Explanation:** {q.get('explanation', 'No explanation provided.')}")
    st.download_button("Download Quiz PDF", questions_to_pdf("Prep AI Quiz", quiz["questions"]), "prep_ai_quiz.pdf", "application/pdf", key=f"dl_result_{quiz['uid']}")


def render_quiz_taker(state_key: str) -> None:
    """Start screen -> live timer + questions -> auto/manual submit -> result. Used by Practice, Exam and Published Quizzes."""
    quiz = st.session_state.get(state_key)
    if not quiz:
        return
    questions, limit, uid, shared = quiz["questions"], quiz["time_limit_sec"], quiz["uid"], quiz.get("shared_code")
    n = len(questions)
    st.write(f"**{quiz['subject']} → {quiz['topic']}** · {quiz['difficulty']} · {n} questions · " + (f"⏱ {clock(limit)} limit" if limit else "no time limit"))

    if quiz.get("result"):
        render_quiz_result(quiz)
        return

    if quiz["started_at"] is None:
        if limit or shared:   # timed or one-attempt quizzes start on an explicit click
            rules = []
            if limit:
                rules.append(f"You have **{clock(limit)}**. The quiz is submitted automatically when time runs out.")
            if shared:
                rules.append("This shared quiz can be attempted **once**. The clock keeps running even if you reload the page.")
            st.info(" ".join(rules))
            if st.button("▶ Start Quiz", type="primary", key=f"start_{uid}"):
                if shared:
                    started = iso_to_epoch(shared_quiz_start(shared, student_id))
                    quiz["started_at"] = started
                    if limit and is_expired(started, limit):
                        try:
                            record_shared_timeout(student_id, {"questions": questions, "subject": quiz["subject"], "topic": quiz["topic"], "difficulty": quiz["difficulty"], "code": shared, "time_limit_sec": limit}, limit)
                        except sqlite3.IntegrityError:
                            pass
                        quiz["result"] = {"timeout_only": True}
                else:
                    quiz["started_at"] = time.time()
                st.rerun()
            return
        quiz["started_at"] = time.time()

    expired = is_expired(quiz["started_at"], limit)
    if limit and not expired:
        quiz_timer(state_key)

    answers, stamps = quiz["answers"], quiz["answer_ts"]
    for i, q in enumerate(questions):
        st.markdown(f"### Q{i + 1}. {q.get('question', '')}")
        options = q.get("options") or {}
        choice = st.radio(
            "Choose an answer", list(options.keys()),
            format_func=lambda k, q=q: f"{k}. {q['options'][k]}",
            key=f"qc_{uid}_{i}", index=None,
        )
        if choice and answers.get(i) != choice:
            answers[i] = choice
            stamps[i] = time.time()

    st.progress(len(answers) / n if n else 0.0, text=f"{len(answers)} of {n} answered")
    clicked = st.button("Submit Quiz", type="primary", key=f"submit_{uid}")
    expired = expired or is_expired(quiz["started_at"], limit)
    if clicked or expired:
        submit_quiz(state_key, timed_out=expired)
        if quiz.get("result"):
            st.rerun()


def _share_form(quiz: dict) -> None:
    """Title + time limit + 'publish' button. Shows the code once the quiz is published."""
    if quiz.get("share_code_created"):
        st.success("Quiz published. Give this code to your students:")
        st.code(quiz["share_code_created"])
        st.caption("They open **Published Quizzes → Join Quiz** and enter the code. Everyone gets the same questions in a different order; each result is saved under the student's own ID and is visible to the tutor only.")
        return
    st.caption("Anyone with the code can take this exact quiz. Each person gets one attempt, and only the tutor can see the results.")
    title = st.text_input("Quiz title", value=f"{quiz['subject']} – {quiz['topic']}", key=f"share_title_{quiz['uid']}")
    mins = st.number_input("Time limit for everyone (minutes, 0 = none)", 0, QUIZ_MAX_MINUTES, int(quiz["time_limit_sec"] // 60), key=f"share_min_{quiz['uid']}")
    if st.button("Publish quiz", type="primary", key=f"share_btn_{quiz['uid']}"):
        quiz["share_code_created"] = create_shared_quiz(
            student_id, title.strip() or "Shared quiz", quiz["subject"], quiz["topic"], quiz["difficulty"],
            quiz["questions"], minutes_to_seconds(mins), SHARED_CODE_LENGTH,
        )
        st.rerun()


def render_share_panel(quiz: dict) -> None:
    """Small shortcut inside Practice → Current Quiz."""
    if quiz.get("shared_code"):
        return
    with st.expander("👥 Publish this quiz for other students"):
        _share_form(quiz)


def render_publish_tab() -> None:
    st.subheader("Publish the current quiz")
    quiz = st.session_state.get("quiz")
    if not quiz:
        st.info("Create a quiz from Learn first, then return here to publish it." if not st.session_state.is_tutor
                else "Create a quiz in the **Create Quiz** tab first, then return here to publish it.")
        return
    st.write(f"**{quiz['subject']} → {quiz['topic']}** · {quiz['difficulty']} · {len(quiz['questions'])} questions")
    if quiz.get("shared_code"):
        st.info("This quiz was joined from a code, so it cannot be published again.")
        return
    _share_form(quiz)


def render_join_tab() -> None:
    st.subheader("Join a published quiz")
    active = st.session_state.get("shared_quiz")
    if active:
        render_quiz_taker("shared_quiz")
        if active.get("result") or active["started_at"] is None:
            if st.button("← Enter a different code", key="leave_shared"):
                st.session_state.shared_quiz = None
                st.rerun()
        return
    st.caption("Enter the code from your tutor. Everyone gets the same questions in a different order; your result is saved under your own Student ID.")
    with st.form("join_form"):
        code_in = st.text_input("Quiz code", placeholder="e.g. K7M2QX", max_chars=12)
        go = st.form_submit_button("Open quiz", type="primary")
    if go:
        code = normalize_code(code_in)
        found = get_shared_quiz(code) if code else None
        if not found:
            st.error("No quiz found with that code. Please check it and try again.")
            return
        if not found["is_open"]:
            st.warning("This quiz has been closed by its creator.")
            return
        if has_attempted_shared(code, student_id):
            mine = next((x for x in shared_quiz_results(code) if x["student_id"] == student_id), None)
            st.info("You have already completed this quiz." + (f" Your score: **{mine['score']:.0f}%**." if mine else ""))
            return
        # Each student gets the same questions in their own (stable) random order.
        questions = list(found["questions"])
        random.Random(f"{code}:{student_id}").shuffle(questions)
        st.session_state.shared_quiz = new_quiz(questions, found["subject"], found["topic"], found["difficulty"], [], found["time_limit_sec"], shared_code=code, title=found["title"])
        st.rerun()


def render_my_published_tab() -> None:
    st.subheader("My published quizzes")
    mine = list_shared_quizzes(student_id)
    if not mine:
        st.info("You have not published any quiz yet. Use the **Publish** tab.")
        return
    for q in mine:
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{q['title']}** · {q['subject']} → {q['topic']} · {q['difficulty']}")
                st.caption(f"Code: `{q['code']}` · {q['attempts']} attempt(s) · time limit {clock(q['time_limit_sec']) if q['time_limit_sec'] else 'none'} · {'OPEN' if q['is_open'] else 'CLOSED'}")
            with c2:
                if q["is_open"]:
                    if st.button("Close", key=f"mp_close_{q['code']}"):
                        set_shared_quiz_open(q["code"], False)
                        st.rerun()
                else:
                    if st.button("Re-open", key=f"mp_open_{q['code']}"):
                        set_shared_quiz_open(q["code"], True)
                        st.rerun()


def render_published_quizzes() -> None:
    hero("🌐 Published Quizzes", "Publish one quiz for multiple students, give each student a randomized sequence, and keep results visible only to the tutor.")
    tabs = st.tabs(["Publish", "Join Quiz", "My Published Quizzes", "Tutor Results"])
    with tabs[0]:
        render_publish_tab()
    with tabs[1]:
        render_join_tab()
    with tabs[2]:
        render_my_published_tab()
    with tabs[3]:
        if st.session_state.is_tutor:
            render_quiz_results()
        else:
            st.info("🔒 Results are visible only to the tutor. After you submit a quiz you can see your own score and rank on the result screen.")


def render_tutor_create_quiz() -> None:
    st.subheader("Create a quiz")
    subject = st.selectbox("Subject", ["Biology", "Chemistry", "Physics", "English"], key="tq_subject")
    topic = st.text_input("Chapter / Topic", key="tq_topic")
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1, key="tq_difficulty")
    count = st.slider("Number of MCQs", 5, 50, 20, key="tq_count")
    limit_sec = time_limit_input("tq", count)
    instructions = st.text_area("Optional instructions", key="tq_instructions")

    if st.button("Generate Quiz", type="primary", key="tq_generate"):
        if not topic.strip():
            st.warning("Please enter a chapter or topic.")
            return
        db_index, metadata = load_database_index()
        if db_index is None:
            st.error("Database FAISS files are missing. Place database.faiss and metadata.json inside faiss_index/.")
            return
        try:
            context, results = build_rag_context(f"{subject} {topic} {instructions}".strip(), metadata, db_index, min(10, count))
            ctx = create_context(f"Generate {count} MCQs about {topic}. {instructions}", subject, topic, difficulty, context)
            questions = orch.practice_request(ctx, count)
        except Exception as exc:
            st.error("Prep AI could not generate this quiz. Check your Groq configuration and try again.")
            if st.session_state.get("debug_mode"):
                st.exception(exc)
            return
        if not questions:
            st.warning("No valid questions were generated. Try a more specific topic or a smaller question count.")
            return
        st.session_state.quiz = new_quiz(questions, subject, topic, difficulty, results, limit_sec)
        st.rerun()

    quiz = st.session_state.get("quiz")
    if quiz and not quiz.get("shared_code"):
        st.success(f"Quiz ready: **{quiz['subject']} → {quiz['topic']}** · {len(quiz['questions'])} questions. Open the **Publish** tab to get the student code.")
        with st.expander("Preview questions and answers"):
            render_questions(quiz["questions"], download_key=f"tq_{quiz['uid']}")


def render_tutor_home() -> None:
    hero("👩‍🏫 Tutor Dashboard", "Create a quiz, publish it for your students, and check how every student performed.")
    tabs = st.tabs(["Create Quiz", "Publish", "My Published Quizzes", "Quiz Results", "Student Performance"])
    with tabs[0]:
        render_tutor_create_quiz()
    with tabs[1]:
        render_publish_tab()
    with tabs[2]:
        render_my_published_tab()
    with tabs[3]:
        render_quiz_results()
    with tabs[4]:
        render_student_performance()


def render_memory_meter(memory: LongTermMemory, detailed: bool = False) -> dict:
    """Ring chart of how much of the memory allowance is used."""
    u = memory.usage()
    colour = "#DC2626" if u["full"] else "#F59E0B" if u["near_limit"] else None
    c1, c2 = st.columns([1, 2])
    with c1:
        charts.show(charts.ring(u["percent"], "Memory used", color=colour, center_text=f"{u['used']}/{u['limit']}"))
    with c2:
        m1, m2 = st.columns(2)
        m1.metric("Memories stored", f"{u['used']} of {u['limit']}")
        m2.metric("Free slots", u["free"])
        kb = (u["text_bytes"] + u["disk_bytes"]) / 1024
        st.caption(f"Approx. storage: {kb:,.0f} KB (text + search index).")
        if u["full"]:
            st.error("Memory is full. New memories replace the least important and oldest ones. You can reset memory below.")
        elif u["near_limit"]:
            st.warning(f"Memory is {u['percent']:.0f}% full. When it reaches 100%, the least important and oldest memories are replaced automatically.")
        if detailed and u["by_type"]:
            charts.show(charts.donut(u["by_type"], title="What your memory contains", center_text=str(u["used"])))
    return u


# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------
def render_dashboard() -> None:
    hero(
        f"Welcome, {st.session_state.student_name} 👋",
        "Your adaptive learning dashboard is personalized to this Student ID.",
    )
    stats = dashboard_stats(student_id)
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        charts.show(charts.ring(stats["overall"], "Overall mastery"))
    with g2:
        charts.show(charts.ring(stats["accuracy"], "Accuracy", color="#16A34A"))
    with g3:
        charts.show(charts.donut(
            {"Correct": stats["correct"], "Incorrect": stats["incorrect"], "Skipped": stats["skipped"]},
            charts.OUTCOME_COLORS, "Your answers", center_text=str(stats["attempted"] + stats["skipped"]),
        ))
    with g4:
        st.metric("Questions Answered", stats["attempted"], help=f"{stats['skipped']} skipped question(s) are not counted here." if stats["skipped"] else None)
        st.metric("Revision Due", stats["revision_due"])

    action = next_best_action(student_id)
    st.markdown("### 🎯 Next Best Action")
    st.info(f"**{action['title']}** — {action['reason']}")
    if action["topic"]:
        st.write(f"Recommended: **{action['subject']} → {action['topic']}**")

    if 0 < stats["attempted"] < 10:
        st.caption(f"ℹ️ Based on only {stats['attempted']} answered question(s) - scores become more reliable as you practice.")

    areas = weak_strong_areas(student_id, 5)
    topics = topic_mastery(student_id)
    st.markdown("### 📊 Mastery by topic")
    if topics:
        charts.show(charts.mastery_bars(topics))
        st.caption("Dashed lines mark 60% and 75%. Faded bars have fewer than 3 answers, so they are not called weak or strong yet.")
        if areas["weak"]:
            st.markdown("⚠️ **Focus on:** " + " · ".join(f"{x['subject']} → {x['topic']}" for x in areas["weak"]))
        elif stats["topics_tracked"]:
            st.success("No weak areas right now. Keep practicing!")
        if areas["strong"]:
            st.markdown("🧠 **Strong in:** " + " · ".join(f"{x['subject']} → {x['topic']}" for x in areas["strong"]))
    else:
        st.info("Complete a quiz to build your learning profile.")

    trend = list(reversed(history(student_id, 100)))
    if trend:
        charts.show(charts.score_trend(trend))

    with st.expander("How is mastery calculated?"):
        st.markdown(
            "- **Skipped** questions count as a soft miss (half a wrong answer) but are not included in accuracy.\n"
            "- **Recent answers count more** than old ones.\n"
            "- Correct **Hard** answers earn more credit; wrong **Easy** answers cost more.\n"
            "- A topic needs **at least 3 answers** before it is called weak or strong, and small "
            "samples are pulled toward 50% so one lucky answer cannot make you a master.\n"
            "- **Weak** = below 60% · **Developing** = 60-74% · **Strong** = 75% and above · **Mastered** = 90%+"
        )

    st.markdown("### 🧠 Long-Term Memory")
    dash_memory = LongTermMemory(student_id)
    render_memory_meter(dash_memory)
    memories = dash_memory.recent(5)
    if memories:
        for m in memories:
            st.write(f"• **{m['memory_type']}** — {m['content'][:180]}")
    else:
        st.info("Tutor and research conversations will appear here after you use them.")

    st.markdown("### 🏆 Achievements")
    for a in achievements(student_id)[:8]:
        st.success(a["title"])


# -----------------------------------------------------------------------------
# Personalized learning
# -----------------------------------------------------------------------------
def load_personalized() -> None:
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, TXT or MD",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )
    drive_url = st.text_input("Or paste a public Google Drive file/folder link")

    if st.button("Process Learning Material", type="primary"):
        files = []
        if uploaded:
            for f in uploaded:
                files.append((os.path.basename(f.name), f.getvalue()))
        if drive_url.strip():
            try:
                drive_files, _ = download_drive(drive_url)
                files.extend(drive_files)
            except Exception as exc:
                st.error(f"Google Drive error: {exc}")
                return

        if not files:
            st.warning("Upload a document or provide a Google Drive link.")
            return

        pages, info = [], []
        try:
            for name, data in files:
                extracted = extract_document(data, name, name)
                pages.extend(extracted)
                info.append(
                    {
                        "filename": name,
                        "file_type": name.rsplit(".", 1)[-1].upper() if "." in name else "Unknown",
                        "pages": len(extracted),
                        "characters": sum(len(p.get("text", "")) for p in extracted),
                    }
                )
            chunks = chunk_pages(pages)
            index, _ = build_index(chunks)
            st.session_state.personal_chunks = chunks
            st.session_state.personal_index = index
            st.session_state.personal_info = info
            st.success(f"Processed {len(info)} document(s) and created {len(chunks)} overlapping chunks.")
        except Exception as exc:
            st.error(f"Processing error: {exc}")
            return

    if st.session_state.personal_info:
        st.markdown("### Document Information")
        st.dataframe(st.session_state.personal_info, use_container_width=True, hide_index=True)
        st.write(f"**Total chunks:** {len(st.session_state.personal_chunks)}")


def render_personalized_controls() -> None:
    st.markdown("### 🎯 Personalized Study")
    topic = st.text_input("Topic / Chapter", key="personal_topic", placeholder="e.g. Genetics, Thermodynamics, Cell Biology")
    mode_name = st.selectbox("Mode", ["MCQs", "Answer explanation", "Quiz"], key="personal_mode")
    difficulty = st.selectbox("Difficulty", ["Adaptive", "Easy", "Medium", "Hard"], key="personal_difficulty")
    count = st.slider("Number of MCQs", 5, 50, 20, key="personal_count")
    limit_sec = time_limit_input("personal", count) if mode_name == "Quiz" else 0
    instructions = st.text_area("Optional instructions", key="personal_instructions")

    if st.button("Start Personalized Learning", type="primary"):
        if not topic.strip():
            st.warning("Please enter a topic or chapter.")
            return
        if not st.session_state.personal_chunks or st.session_state.personal_index is None:
            st.warning("Process your learning material first.")
            return

        actual = difficulty
        if difficulty == "Adaptive":
            actual = difficulty_for_topic(student_id, "Personalized", topic)

        query = f"{topic} {instructions}".strip()
        context, results = build_rag_context(query, st.session_state.personal_chunks, st.session_state.personal_index, 8)
        ctx = create_context(
            f"Generate {count} {mode_name} for the topic {topic}. {instructions}",
            "Personalized Material",
            topic,
            actual,
            context,
        )

        try:
            if mode_name == "Answer explanation":
                answer = grounded_answer(
                    f"Explain the topic '{topic}' accurately at {student.get('level') or 'MDCAT'} level. {instructions}",
                    context,
                    memory_prompt(LongTermMemory(student_id).retrieve(topic)),
                )
                st.markdown("## Answer Explanation")
                st.markdown(answer)
                st.download_button(
                    "📥 Download Explanation as PDF",
                    data=text_to_pdf("Prep AI — Answer Explanation", answer, f"Topic: {topic}"),
                    file_name="prep_ai_answer_explanation.pdf",
                    mime="application/pdf",
                    key="download_personal_explanation",
                )
                source_cards(results)
            else:
                questions = orch.practice_request(ctx, count)
                if not questions:
                    st.warning("No reliable questions were generated from this material. Try a more specific topic.")
                    return
                if mode_name == "Quiz":
                    st.session_state.quiz = new_quiz(questions, "Personalized Material", topic, actual, results, limit_sec)
                    st.success("Quiz created. Open Practice → Current Quiz." + (f" Time limit: {clock(limit_sec)}." if limit_sec else ""))
                    source_cards(results)
                else:
                    render_questions(questions, "Generated MCQs", download_key="personal_mcqs")
                    source_cards(results)
        except Exception as exc:
            st.error("Prep AI could not generate this learning activity. Please check your Groq configuration and try again.")
            if st.session_state.get("debug_mode"):
                st.exception(exc)


def render_learn() -> None:
    hero("📚 Learn", "Study your own material or the prebuilt Biology, Chemistry, Physics and English database.")
    tabs = st.tabs(["Personalized Learning", "Database Learning"])

    with tabs[0]:
        load_personalized()
        if st.session_state.personal_chunks:
            st.divider()
            render_personalized_controls()

    with tabs[1]:
        subject = st.selectbox("Subject", ["Biology", "Chemistry", "Physics", "English"], key="db_subject")
        topic = st.text_input("Chapter / Topic", key="db_topic")
        mode_name = st.selectbox("Mode", ["MCQs", "Answer explanation", "Quiz"], key="db_mode")
        difficulty = st.selectbox("Difficulty", ["Adaptive", "Easy", "Medium", "Hard"], key="db_difficulty")
        count = st.slider("Number of MCQs", 5, 50, 20, key="db_count")
        db_limit_sec = time_limit_input("db", count) if mode_name == "Quiz" else 0
        instructions = st.text_area("Optional instructions", key="db_instructions")

        if st.button("Start Database Learning", type="primary"):
            if not topic.strip():
                st.warning("Please enter a chapter or topic.")
                return

            db_index, metadata = load_database_index()
            if db_index is None:
                st.error("Database FAISS files are missing. Place database.faiss and metadata.json inside faiss_index/.")
                return

            actual = difficulty
            if difficulty == "Adaptive":
                actual = difficulty_for_topic(student_id, subject, topic)

            query = f"{subject} {topic} {instructions}".strip()
            context, results = build_rag_context(query, metadata, db_index, min(10, count))
            ctx = create_context(
                f"Generate {count} {mode_name} about {topic}. {instructions}",
                subject,
                topic,
                actual,
                context,
            )

            try:
                if mode_name == "Answer explanation":
                    answer = grounded_answer(
                        f"Explain '{topic}' accurately using only the provided learning context. {instructions}",
                        context,
                        memory_prompt(LongTermMemory(student_id).retrieve(topic)),
                    )
                    st.markdown("## Answer Explanation")
                    st.markdown(answer)
                    st.download_button(
                        "📥 Download Explanation as PDF",
                        data=text_to_pdf("Prep AI — Answer Explanation", answer, f"{subject}: {topic}"),
                        file_name="prep_ai_answer_explanation.pdf",
                        mime="application/pdf",
                        key="download_database_explanation",
                    )
                    source_cards(results)
                else:
                    questions = orch.practice_request(ctx, count)
                    if not questions:
                        st.warning("No valid questions were generated. Try a more specific topic or smaller question count.")
                    elif mode_name == "Quiz":
                        st.session_state.quiz = new_quiz(questions, subject, topic, actual, results, db_limit_sec)
                        st.success("Quiz created. Open Practice → Current Quiz to take it." + (f" Time limit: {clock(db_limit_sec)}." if db_limit_sec else ""))
                        source_cards(results)
                    else:
                        render_questions(questions, download_key="database_mcqs")
                        source_cards(results)
            except Exception as exc:
                st.error("Prep AI could not complete this learning request. Check your Groq configuration and try again.")
                if st.session_state.get("debug_mode"):
                    st.exception(exc)


# -----------------------------------------------------------------------------
# Practice
# -----------------------------------------------------------------------------
def render_practice() -> None:
    hero("📝 Practice", "Practice current quizzes, weak topics, smart revision, mistakes and flashcards.")
    tabs = st.tabs(["Current Quiz", "Weak Topics", "Smart Revision", "Teach Me My Mistakes", "Flashcards"])

    with tabs[0]:
        quiz = st.session_state.get("quiz")
        if not quiz:
            st.info("No active quiz. Start one from Learn or Practice My Weak Topics, or open **Published Quizzes → Join Quiz** to enter a code.")
        else:
            if not quiz.get("result"):
                render_share_panel(quiz)
            render_quiz_taker("quiz")

    with tabs[1]:
        weak = weak_topics(student_id, 10)
        if not weak:
            st.info("No topic is below 75% right now. Complete more quizzes to keep your profile up to date.")
        else:
            charts.show(charts.mastery_bars(
                [{"subject": x["subject"], "topic": f"{x['topic']} → {x['concept']}", "mastery_score": x["mastery_score"], "attempts": x["attempts"]} for x in weak],
                "Weakest concepts",
            ))
            weak_count = st.slider("Number of questions", 5, 30, 10, key="weak_count")
            weak_limit_sec = time_limit_input("weak", weak_count)

            if st.button("Practice My Weak Topics", type="primary"):
                x = weak[0]
                try:
                    db_index, metadata = load_database_index()
                    context = ""
                    sources = []
                    if db_index is not None:
                        context, sources = build_rag_context(
                            f"{x['subject']} {x['topic']} {x['concept']}",
                            metadata,
                            db_index,
                            8,
                        )

                    difficulty = "Easy" if float(x["mastery_score"]) < 45 else "Medium"
                    ctx = create_context(
                        f"Create a targeted quiz for my weak topic {x['topic']} and concept {x['concept']}",
                        x["subject"],
                        x["topic"],
                        difficulty,
                        context,
                    )
                    questions = orch.practice_request(ctx, weak_count)
                    if not questions:
                        st.warning("Prep AI could not generate reliable questions for this weak topic. Try again or check your database material.")
                    else:
                        st.session_state.quiz = new_quiz(questions, x["subject"], x["topic"], difficulty, sources, weak_limit_sec)
                        st.success("Targeted quiz created. The page will open Current Quiz now.")
                        st.rerun()
                except Exception as exc:
                    st.error("Could not create the weak-topic quiz. Check the database material and Groq configuration.")
                    if st.session_state.get("debug_mode"):
                        st.exception(exc)

    with tabs[2]:
        st.subheader("🔄 Smart Revision")

        # Topics whose scheduled review date has arrived.
        due = due_revisions(student_id)

        if due:
            st.markdown("### 🔴 Revision Due Now")
            for x in due:
                mastery_value = float(x.get("mastery", 0) or 0)
                st.warning(
                    f"**{x['subject']} → {x['topic']}** · "
                    f"Mastery **{mastery_value:.0f}%** · "
                    f"Review interval **{x.get('interval_days', 1)} days**"
                )
        else:
            st.success("✅ No scheduled revision is due today.")

        # Weak topics can be practiced before their scheduled review date.
        recommendations = revision_recommendations(student_id, 10)
        due_keys = {(x["subject"], x["topic"]) for x in due}
        recommended = [
            x for x in recommendations
            if (x["subject"], x["topic"]) not in due_keys
        ]

        if recommended:
            st.markdown("### 🟡 Recommended Revision")
            st.caption(
                "These topics are recommended because their mastery is lower "
                "or they contain repeated mistakes, even if their scheduled "
                "review date has not arrived yet."
            )
            for x in recommended[:5]:
                mastery_value = float(x.get("mastery_score", 0) or 0)
                mistakes = int(x.get("repeated_mistakes", 0) or 0)
                st.write(
                    f"📚 **{x['subject']} → {x['topic']} → "
                    f"{x.get('concept') or 'General'}** · "
                    f"Mastery **{mastery_value:.0f}%** · "
                    f"Repeated mistakes **{mistakes}**"
                )

        # Build a unique list of topics that can be practiced now.
        candidates = []
        seen = set()
        for item in due + recommended:
            key = (
                item["subject"],
                item["topic"],
                item.get("concept", ""),
            )
            if key not in seen:
                seen.add(key)
                candidates.append(item)

        if candidates:
            labels = []
            for item in candidates:
                mastery_value = float(
                    item.get("mastery", item.get("mastery_score", 0)) or 0
                )
                labels.append(
                    f"{item['subject']} → {item['topic']} → "
                    f"{item.get('concept') or 'General'} "
                    f"({mastery_value:.0f}%)"
                )

            selected_index = st.selectbox(
                "Choose a topic to revise",
                range(len(candidates)),
                format_func=lambda i: labels[i],
                key="smart_revision_topic",
            )
            selected = candidates[selected_index]

            mastery_value = float(
                selected.get("mastery", selected.get("mastery_score", 0)) or 0
            )
            if mastery_value < 40:
                revision_difficulty = "Easy"
            elif mastery_value < 70:
                revision_difficulty = "Medium"
            else:
                revision_difficulty = "Hard"

            st.info(
                f"Recommended difficulty for this revision: **{revision_difficulty}**"
            )

            rev_count = st.slider("Number of questions", 5, 30, 10, key="revision_count")
            rev_limit_sec = time_limit_input("revision", rev_count)

            if st.button(
                "🚀 Start Smart Revision",
                type="primary",
                key="start_smart_revision",
            ):
                try:
                    subject = selected["subject"]
                    topic = selected["topic"]
                    concept = selected.get("concept") or ""

                    db_index, metadata = load_database_index()
                    context, sources = "", []
                    if db_index is not None:
                        context, sources = build_rag_context(
                            f"{subject} {topic} {concept}".strip(),
                            metadata,
                            db_index,
                            8,
                        )

                    revision_request = (
                        f"Create a targeted revision quiz for the student. "
                        f"Focus on {topic} and {concept or 'the main concepts'}. "
                        f"The student's current mastery is {mastery_value:.0f}%. "
                        f"Use {revision_difficulty} difficulty. Reinforce common "
                        f"mistakes and important concepts. Use only the provided "
                        f"learning context when source material is available."
                    )

                    ctx = create_context(
                        revision_request,
                        subject,
                        topic,
                        revision_difficulty,
                        context,
                    )
                    questions = orch.practice_request(ctx, rev_count)

                    if not questions:
                        st.warning(
                            "No reliable revision questions could be generated. "
                            "Try again or check the learning material."
                        )
                    else:
                        st.session_state.quiz = new_quiz(questions, subject, topic, revision_difficulty, sources, rev_limit_sec, revision=True)
                        st.session_state.revision_quiz_created = True
                        st.success(
                            "✅ Revision quiz created. Open **Practice → Current Quiz** to start."
                        )
                        st.rerun()
                except Exception as exc:
                    st.error(
                        "Could not create the Smart Revision quiz. "
                        "Please check your database material and Groq configuration."
                    )
                    if st.session_state.get("debug_mode"):
                        st.exception(exc)
        else:
            st.info(
                "Complete at least one quiz to build your personalized "
                "revision recommendations."
            )

        upcoming = upcoming_revisions(student_id, 10)
        if upcoming:
            st.divider()
            st.markdown("### 🟢 Upcoming Revision")
            for x in upcoming:
                mastery_value = float(x.get("mastery", 0) or 0)
                st.write(
                    f"📅 **{x['subject']} → {x['topic']}** · "
                    f"Mastery **{mastery_value:.0f}%** · "
                    f"Next review: **{x['next_review']}**"
                )

    with tabs[3]:
        mistakes = recent_mistakes(student_id, 10)
        if not mistakes:
            st.info("No recorded mistakes yet.")
        else:
            for m in mistakes:
                with st.expander(m["question_text"][:100]):
                    st.write(f"Your answer: {m['wrong_answer']}")
                    st.write(f"Correct: {m['correct_answer']}")
                    st.write(m.get("explanation") or "Review this concept from your source material.")
            if st.button("Teach Me My Mistakes", type="primary"):
                context = "\n\n".join(
                    f"Question: {m['question_text']}\nWrong: {m['wrong_answer']}\nCorrect: {m['correct_answer']}\nExplanation: {m.get('explanation', '')}"
                    for m in mistakes
                )
                ctx = create_context(
                    "Teach me my recent mistakes step by step.",
                    mistakes[0]["subject"],
                    mistakes[0]["topic"],
                    "Medium",
                    context,
                )
                answer = orch.tutor_request(ctx)
                st.markdown(answer)
                save_tutor_turn(ctx.request, answer, ctx.subject, ctx.topic)

    with tabs[4]:
        weak = weak_topics(student_id, 1)
        if st.button("Generate Flashcards"):
            topic = weak[0]["topic"] if weak else "General study skills"
            subject = weak[0]["subject"] if weak else "General"
            ctx = create_context(
                f"Generate 8 flashcards for {topic}", subject, topic, "Medium", ""
            )
            from groq_service import generate_json
            data = generate_json(
                f"Generate 8 educational flashcards for {ctx.subject} {ctx.topic}. "
                f"Return a JSON object with a cards array. Each card must have front and back. "
                f"Suitable for {ctx.level}."
            )
            cards = data if isinstance(data, list) else data.get("cards", [])
            st.session_state.flashcards = cards
        cards = st.session_state.get("flashcards", [])
        if cards:
            for i, card in enumerate(cards):
                with st.expander(f"Card {i + 1}: {card.get('front', '')}"):
                    st.write(card.get("back", ""))


# -----------------------------------------------------------------------------
# Exam mode
# -----------------------------------------------------------------------------
def render_exam() -> None:
    hero("🎓 Exam Mode", "Timed practice with no explanations until submission. The timer is enforced: the exam is submitted automatically when time runs out.")
    subject = st.selectbox("Subject", ["Biology", "Chemistry", "Physics", "English"], key="exam_subject")
    topic = st.text_input("Topic", key="exam_topic")
    count = st.slider("Questions", 10, 100, 30, key="exam_count")
    difficulty = st.selectbox("Difficulty", ["Adaptive", "Easy", "Medium", "Hard"], key="exam_diff")
    negative = st.checkbox("Negative marking", value=False)
    minutes = st.number_input("Time limit (minutes)", 1, QUIZ_MAX_MINUTES, 30, key="exam_minutes")
    st.caption(f"Suggestion: about {suggested_minutes(count)} minutes for {count} questions.")

    if st.button("Generate Exam", type="primary"):
        actual = difficulty if difficulty != "Adaptive" else difficulty_for_topic(student_id, subject, topic or "General")
        db_index, metadata = load_database_index()
        context, _ = ("", [])
        if db_index is not None:
            context, _ = build_rag_context(f"{subject} {topic}", metadata, db_index, 10)
        ctx = create_context(f"Create an exam for {subject} {topic}", subject, topic, actual, context)
        questions = orch.practice_request(ctx, count)
        if not questions:
            st.warning("No reliable exam questions were generated.")
        else:
            st.session_state.exam = new_quiz(questions, subject, topic, actual, [], minutes_to_seconds(minutes), negative=negative, exam=True)
            st.rerun()

    if st.session_state.get("exam"):
        st.divider()
        render_quiz_taker("exam")


# -----------------------------------------------------------------------------
# AI Tutor — persistent chatbot
# -----------------------------------------------------------------------------
def load_tutor_history() -> None:
    if st.session_state.tutor_messages:
        return
    sessions = get_agent_sessions(student_id, "Tutor Agent", 30)
    messages = []
    for row in reversed(sessions):
        if row.get("user_input"):
            messages.append({"role": "user", "content": row["user_input"]})
        if row.get("output"):
            messages.append({"role": "assistant", "content": row["output"]})
    st.session_state.tutor_messages = messages


def render_tutor() -> None:
    hero(
        "🧑‍🏫 AI Tutor",
        "A persistent chatbot that remembers both your questions and the tutor's previous teaching across sessions.",
    )

    c1, c2 = st.columns(2)
    with c1:
        subject = st.text_input("Subject", key="tutor_subject")
    with c2:
        topic = st.text_input("Topic", key="tutor_topic")
    level = st.selectbox(
        "Tutor level",
        ["Beginner", "Intermediate", "MDCAT", "University", "Advanced"],
        key="tutor_level",
        index=2 if (student.get("level") or "MDCAT") == "MDCAT" else 0,
    )

    load_tutor_history()
    for message in st.session_state.tutor_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Ask your tutor anything...")
    if prompt:
        prompt = prompt.strip()
        st.session_state.tutor_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        try:
            db_index, metadata = load_database_index()
            context = ""
            if db_index is not None:
                context, _ = build_rag_context(
                    f"{subject} {topic} {prompt}", metadata, db_index, 6
                )

            memories = LongTermMemory(student_id).retrieve(
                f"{subject} {topic} {prompt}", top_k=8
            )
            ctx = AgentContext(
                student_id,
                prompt,
                subject,
                topic,
                level,
                "Medium",
                context,
            )
            answer = orch.tutor_request(ctx)

            with st.chat_message("assistant"):
                st.markdown(answer)

            st.session_state.tutor_messages.append({"role": "assistant", "content": answer})
            save_tutor_turn(prompt, answer, subject, topic)
        except Exception as exc:
            st.error("Prep AI could not complete the tutor response. Check your Groq configuration and try again.")
            if st.session_state.get("debug_mode"):
                st.exception(exc)


# -----------------------------------------------------------------------------
# Voice Tutor — speak a question, hear the answer
# -----------------------------------------------------------------------------
VOICE_STYLE_HINT = (
    "VOICE MODE: your answer will be read aloud. Reply in at most 120 words, in plain "
    "spoken sentences. No markdown, no bullet points, no tables, no symbols."
)


def render_voice_tutor() -> None:
    hero(
        "🎙️ Voice Tutor",
        "Ask your question by speaking. The tutor answers in text and voice, using your documents and long-term memory.",
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        subject = st.text_input("Subject", key="voice_subject")
    with c2:
        topic = st.text_input("Topic", key="voice_topic")
    with c3:
        language = st.selectbox("Voice language", ["English", "Urdu"], key="voice_language")
    level = student.get("level") or "MDCAT"

    load_tutor_history()
    for message in st.session_state.tutor_messages[-6:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    recording = st.audio_input("🎤 Click the mic, ask your question, then click stop")
    fresh_audio = False

    if recording is not None:
        audio_bytes = recording.getvalue()
        digest = hashlib.md5(audio_bytes).hexdigest()

        # Streamlit reruns the script often; only process each recording once.
        if digest != st.session_state.get("voice_last_digest"):
            st.session_state.voice_last_digest = digest
            try:
                with st.spinner("Listening..."):
                    question = transcribe(audio_bytes, language)
                if not question:
                    st.warning("I could not hear anything. Please try again, closer to the microphone.")
                else:
                    st.session_state.tutor_messages.append({"role": "user", "content": question})
                    with st.chat_message("user"):
                        st.markdown(question)

                    with st.spinner("Thinking..."):
                        db_index, metadata = load_database_index()
                        context = ""
                        if db_index is not None:
                            context, _ = build_rag_context(f"{subject} {topic} {question}", metadata, db_index, 6)
                        ctx = AgentContext(
                            student_id, question, subject, topic, level, "Medium", context,
                            style_hint=VOICE_STYLE_HINT,
                        )
                        answer = orch.tutor_request(ctx)

                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    st.session_state.tutor_messages.append({"role": "assistant", "content": answer})
                    save_tutor_turn(question, answer, subject, topic)

                    try:
                        with st.spinner("Speaking..."):
                            st.session_state.voice_audio = speak(answer, language)
                        fresh_audio = True
                    except Exception:
                        st.session_state.voice_audio = None
                        st.warning("The answer is ready, but text-to-speech failed (check your internet connection).")
            except Exception as exc:
                st.error("Voice Tutor could not finish. Check your Groq key and try again.")
                if st.session_state.get("debug_mode"):
                    st.exception(exc)

    if st.session_state.get("voice_audio"):
        st.caption("🔊 Tutor's voice answer")
        st.audio(st.session_state.voice_audio, format="audio/mp3", autoplay=fresh_audio)


# -----------------------------------------------------------------------------
# Research Agent
# -----------------------------------------------------------------------------
def render_research() -> None:
    hero(
        "🔎 Research Agent",
        "Research current topics using DuckDuckGo and store the research conversation in your long-term memory.",
    )
    topic = st.text_area("Research topic", placeholder="Example: Recent advances in photovoltaic cell efficiency")
    if st.button("Research", type="primary") and topic.strip():
        try:
            ctx = AgentContext(student_id, topic.strip(), level="University")
            result = orch.research_request(ctx)
            st.markdown(result["answer"])
            st.subheader("Web Sources")
            for r in result["sources"]:
                st.write(f"**{r['title']}** — {r['url']}")
                st.caption(r["snippet"])
            st.download_button(
                "📥 Download Research Report as PDF",
                data=research_to_pdf("Prep AI — Research Report", result["answer"], result["sources"]),
                file_name="prep_ai_research_report.pdf",
                mime="application/pdf",
                key="download_research_report",
            )
            save_research_memory(topic.strip(), result["answer"])
            st.success("Research request and response saved to long-term memory.")
        except Exception as exc:
            st.error("Prep AI could not complete the research request. Check your Groq/DDGS configuration.")
            if st.session_state.get("debug_mode"):
                st.exception(exc)


# -----------------------------------------------------------------------------
# Study plan / memory / history / settings
# -----------------------------------------------------------------------------
def render_plan() -> None:
    hero("📅 Study Plan", "Build a weekly plan using your weak topics and exam goal.")
    current_student = get_student(student_id)
    exam_name = st.text_input("Exam name", value=current_student.get("exam_name") or "MDCAT")
    exam_date = st.date_input("Exam date", value=date.today())
    hours = st.number_input("Hours per day", 0.5, 12.0, 1.0, step=0.5)
    subjects = st.multiselect(
        "Subjects",
        ["Biology", "Chemistry", "Physics", "English"],
        default=["Biology", "Chemistry", "Physics", "English"],
    )
    goals = st.text_area("Goals")
    if st.button("Create My Study Plan", type="primary"):
        plan = orch.planner.create_plan(student_id, exam_name, exam_date.isoformat(), hours, subjects, goals)
        save_plan(student_id, exam_name, exam_date.isoformat(), plan)
        update_student(student_id, exam_name=exam_name, exam_date=exam_date.isoformat(), daily_minutes=int(hours * 60))
        st.session_state.study_plan = plan
    for day in st.session_state.get("study_plan", []):
        st.markdown(f"### {day.get('day', 'Day')}")
        for session in day.get("sessions", []):
            st.write(
                f"• {session.get('subject', '')} — {session.get('topic', '')} — "
                f"{session.get('minutes', 0)} min — {session.get('activity', '')}"
            )


def render_memory() -> None:
    hero(
        "🧠 Long-Term Memory",
        f"All memory below belongs only to {st.session_state.student_name} ({student_id}).",
    )
    memory = LongTermMemory(student_id)
    notice = st.session_state.pop("memory_reset_notice", None)
    if notice:
        st.success(
            f"Memory reset: {notice['memories_removed']} memories removed"
            + (f" and {notice['conversations_removed']} saved conversations deleted." if notice["conversations_removed"] else ". Saved conversations were kept.")
        )
    st.markdown("### 📊 Memory usage")
    render_memory_meter(memory, detailed=True)
    items = memory.recent(50)
    sessions = get_agent_sessions(student_id, None, 50)

    tabs = st.tabs(["Semantic Memory", "Tutor Chat", "Research Agent", "All Agent Sessions"])

    with tabs[0]:
        if not items:
            st.info("No long-term memories yet.")
        else:
            for m in items:
                with st.expander(
                    f"{m['memory_type']} · {m.get('subject', '')} · {m.get('topic', '')}"
                ):
                    st.write(m["content"])
                    st.caption(
                        f"Created: {m.get('created_at', '')} · "
                        f"Confidence: {m.get('confidence', 0):.2f}"
                    )

    with tabs[1]:
        tutor_sessions = [x for x in sessions if x.get("agent_name") == "Tutor Agent"]
        if not tutor_sessions:
            st.info("No tutor conversations saved yet.")
        else:
            for row in tutor_sessions:
                with st.expander(f"Tutor conversation · {row.get('created_at', '')}"):
                    st.markdown("**Student:**")
                    st.write(row.get("user_input", ""))
                    st.markdown("**AI Tutor:**")
                    st.write(row.get("output", ""))

    with tabs[2]:
        research_sessions = [x for x in sessions if x.get("agent_name") == "Research Agent"]
        if not research_sessions:
            st.info("No research conversations saved yet.")
        else:
            for row in research_sessions:
                with st.expander(f"Research · {row.get('created_at', '')}"):
                    st.markdown("**Research request:**")
                    st.write(row.get("user_input", ""))
                    st.markdown("**Research response:**")
                    st.write(row.get("output", ""))

    with tabs[3]:
        if not sessions:
            st.info("No agent sessions saved yet.")
        else:
            st.dataframe(sessions, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### 🗑 Reset my memory")
    with st.container(border=True):
        st.write("Deletes everything the tutor has remembered about you (the memories listed above) and their search index. "
                 "Your quiz results, mastery scores and revision schedule are **not** affected.")
        st.checkbox("Also delete my saved Tutor and Research chat history", key="reset_incl_chats")
        st.checkbox("I understand this cannot be undone", key="reset_confirm")

        def _reset_memory_now() -> None:
            result = reset_memory(student_id, bool(st.session_state.get("reset_incl_chats")))
            st.session_state.tutor_messages = []
            st.session_state.memory_reset_notice = result
            st.session_state.reset_confirm = False
            st.session_state.reset_incl_chats = False

        st.button("Reset my memory now", type="primary", disabled=not st.session_state.get("reset_confirm", False), on_click=_reset_memory_now)


def render_history() -> None:
    hero("📈 History", "Review quiz performance and mistakes for the current student profile.")
    rows = history(student_id)
    if rows:
        charts.show(charts.score_trend(list(reversed(rows))))
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No quiz history yet.")
    st.subheader("Recent Mistakes")
    mistakes = recent_mistakes(student_id)
    if mistakes:
        st.dataframe(mistakes, use_container_width=True, hide_index=True)


def render_settings() -> None:
    hero(
        "⚙️ Settings",
        "Choose the Groq model and learning preferences. Settings are saved for this exact Student ID.",
    )
    prefs = get_preferences(student_id)
    saved_model = prefs.get("llm_model") or st.session_state.get("llm_model", DEFAULT_GROQ_MODEL)
    saved_color = prefs.get("ui_color") or st.session_state.get("ui_color", "Blue")
    if saved_model not in GROQ_MODELS:
        saved_model = DEFAULT_GROQ_MODEL
    if saved_color not in UI_COLORS:
        saved_color = "Blue"

    with st.form("settings_form"):
        model = st.selectbox(
            "LLM Model",
            GROQ_MODELS,
            index=GROQ_MODELS.index(saved_model),
            format_func=lambda value: {
                "openai/gpt-oss-120b": "GPT-OSS 120B — Best quality",
                "openai/gpt-oss-20b": "GPT-OSS 20B — Faster / lower cost",
            }.get(value, value),
        )
        color = st.selectbox("UI Color", list(UI_COLORS), index=list(UI_COLORS).index(saved_color))
        difficulty = st.selectbox(
            "Preferred difficulty",
            ["Easy", "Medium", "Hard", "Adaptive"],
            index=["Easy", "Medium", "Hard", "Adaptive"].index(prefs.get("preferred_difficulty", "Medium"))
            if prefs.get("preferred_difficulty", "Medium") in ["Easy", "Medium", "Hard", "Adaptive"]
            else 1,
        )
        language_options = ["English", "Urdu", "English + Urdu"]
        language = st.selectbox(
            "Language",
            language_options,
            index=language_options.index(prefs.get("preferred_language", "English"))
            if prefs.get("preferred_language", "English") in language_options
            else 0,
        )
        explanation_options = ["Concise", "Detailed", "Step-by-step"]
        explanation = st.selectbox(
            "Explanation style",
            explanation_options,
            index=explanation_options.index(prefs.get("explanation_style", "Detailed"))
            if prefs.get("explanation_style", "Detailed") in explanation_options
            else 1,
        )
        learning_options = ["Examples + Practice", "Theory first", "Questions first"]
        learning = st.selectbox(
            "Learning style",
            learning_options,
            index=learning_options.index(prefs.get("learning_style", "Examples + Practice"))
            if prefs.get("learning_style", "Examples + Practice") in learning_options
            else 0,
        )
        submitted = st.form_submit_button("Save Settings", type="primary")

    if submitted:
        update_preferences(
            student_id,
            llm_model=model,
            ui_color=color,
            preferred_difficulty=difficulty,
            preferred_language=language,
            explanation_style=explanation,
            learning_style=learning,
        )
        st.session_state.llm_model = model
        st.session_state.ui_color = color
        st.success("Settings saved for this student.")
        st.rerun()

    st.markdown("### 🎨 Appearance")
    presets = list(THEME_PRESETS)
    saved_preset = st.session_state.get("theme_preset", "Default")
    if saved_preset not in presets:
        saved_preset = "Default"
    saved_bg = st.session_state.get("theme_bg") or "#FFFFFF"
    saved_text = st.session_state.get("theme_text") or "#111827"
    with st.form("appearance_form"):
        preset = st.selectbox("Theme", presets, index=presets.index(saved_preset))
        pc1, pc2 = st.columns(2)
        bg_pick = pc1.color_picker("Background colour (used with Custom)", saved_bg)
        text_pick = pc2.color_picker("Text colour (used with Custom)", saved_text)
        save_look = st.form_submit_button("Save Appearance", type="primary")
        reset_look = st.form_submit_button("Reset to default look")
    st.caption("Tables use the app's base theme. If a colour combination is hard to read, choose a preset or press Reset.")

    if reset_look:
        update_preferences(student_id, theme_preset="Default", theme_bg="", theme_text="")
        _load_theme_prefs({"theme_preset": "Default"})
        st.rerun()
    if save_look:
        pair = (bg_pick, text_pick) if preset == "Custom" else THEME_PRESETS.get(preset)
        level, message = validate_theme(*pair) if pair else ("ok", "")
        if level == "block":
            st.error(message)
        else:
            if level == "warn":
                st.warning(message)
            update_preferences(
                student_id, theme_preset=preset,
                theme_bg=(pair[0] if pair else ""), theme_text=(pair[1] if pair else ""),
            )
            _load_theme_prefs({"theme_preset": preset, "theme_bg": pair[0] if pair else "", "theme_text": pair[1] if pair else ""})
            st.rerun()

    st.info(f"**Student:** {st.session_state.student_name} · **ID:** `{student_id}` · **Groq model:** `{st.session_state.llm_model}`")

    try:
        from groq_service import get_client
        from config import get_secret
        if get_client(get_secret("GROQ_API_KEY") or ""):
            st.success("Groq API key detected.")
        else:
            st.warning("GROQ_API_KEY is missing. Add it to Streamlit Secrets.")
    except Exception:
        st.warning("Unable to initialize the Groq client.")


routes = {
    "Dashboard": render_dashboard,
    "Learn": render_learn,
    "Practice": render_practice,
    "Exam": render_exam,
    "Published Quizzes": render_published_quizzes,
    "Tutor Dashboard": lambda: render_tutor_home() if st.session_state.is_tutor else st.error("Tutor access is required."),
    "AI Tutor": render_tutor,
    "Voice Tutor": render_voice_tutor,
    "Research Agent": render_research,
    "Merit Calculator": lambda: render_merit(student_id, orch),
    "Path Finder": lambda: render_pathfinder(student_id, orch),
    "Study Plan": render_plan,
    "Memory": render_memory,
    "History": render_history,
    "Settings": render_settings,
}

routes[page]()

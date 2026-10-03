from __future__ import annotations

import hashlib
import os
import time
from datetime import date

import streamlit as st

from adaptive import difficulty_for_topic, next_best_action
from agent_system import AgentContext, Orchestrator
from config import DEFAULT_GROQ_MODEL, GROQ_MODELS, UI_COLORS, get_student_id, get_secret
from db import (
    add_achievement,
    achievements,
    dashboard_stats,
    due_revisions,
    weak_strong_areas,
    revision_recommendations,
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
    create_shared_quiz, get_shared_quiz, start_shared_quiz, finish_shared_quiz,
    all_students_performance, student_performance,
)
from documents import chunk_pages, download_drive, extract_document
from groq_service import grounded_answer
from memory import LongTermMemory, memory_prompt
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
if "shared_quiz_answers" not in st.session_state:
    st.session_state.shared_quiz_answers = {}


# -----------------------------------------------------------------------------
# First-run student profile
# -----------------------------------------------------------------------------
def render_profile_setup() -> None:
    st.markdown(
        """
        <div style="max-width:760px;margin:4rem auto 1rem auto;text-align:center;">
            <h1>🎓 Welcome to Prep AI V4</h1>
            <p style="font-size:1.1rem;color:#6b7280;">
                First create your student profile. Your dashboard, learning history,
                tutor conversations, research memories, quiz performance and adaptive
                recommendations will be linked to this Student ID.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

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

        st.session_state.student_id = sid
        st.session_state.student_name = name
        st.session_state.profile_complete = True
        st.session_state.llm_model = DEFAULT_GROQ_MODEL
        st.session_state.ui_color = "Blue"
        st.session_state.background_color = "#FFFFFF"
        st.session_state.text_color = "#111827"
        st.session_state.quiz = None
        st.session_state.quiz_answers = {}
        st.session_state.last_quiz_result = None
        st.session_state.personal_chunks = []
        st.session_state.personal_index = None
        st.session_state.personal_info = []
        st.session_state.tutor_messages = []
        ensure_student(sid, name)
        update_student(sid, name=name, level=level)
        prefs = get_preferences(sid)
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
if "background_color" not in st.session_state:
    st.session_state.background_color = preferences.get("background_color") or "#FFFFFF"
if "text_color" not in st.session_state:
    st.session_state.text_color = preferences.get("text_color") or "#111827"

apply_theme(UI_COLORS[st.session_state.ui_color], st.session_state.background_color, st.session_state.text_color)


# -----------------------------------------------------------------------------
# Sidebar / identity
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🎓 Prep AI V4")
    st.markdown(f"**Student:** {st.session_state.student_name}")
    st.caption(f"Student ID: `{student_id}`")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Learn",
            "Practice",
            "Exam",
            "AI Tutor",
            "Voice Tutor",
            "Research Agent",
            "Study Plan",
            "Memory",
            "History",
            "Shared Quiz",
            "Tutor Dashboard",
            "Settings",
        ],
        label_visibility="collapsed",
    )

    if st.button("Change Student Profile"):
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
# Dashboard
# -----------------------------------------------------------------------------
def render_dashboard() -> None:
    hero(
        f"Welcome, {st.session_state.student_name} 👋",
        "Your adaptive learning dashboard is personalized to this Student ID.",
    )
    stats = dashboard_stats(student_id)
    cols = st.columns(4)
    cols[0].metric("Overall Mastery", f"{stats['overall']}%", help="Average mastery across your topics. See 'How is mastery calculated?' below.")
    cols[1].metric("Questions Answered", stats["attempted"], help=f"{stats['skipped']} skipped question(s) are not counted here." if stats["skipped"] else None)
    cols[2].metric("Accuracy", f"{stats['accuracy']}%")
    cols[3].metric("Revision Due", stats["revision_due"])

    chart_rows = [{"Metric": "Correct", "Value": stats["correct"]}, {"Metric": "Incorrect", "Value": stats["incorrect"]}, {"Metric": "Skipped", "Value": stats["skipped"]}]
    st.markdown("### 📊 Progress Overview")
    st.bar_chart(chart_rows, x="Metric", y="Value", horizontal=True)
    mastery_rows = weak_strong_areas(student_id, 100)["weak"] + weak_strong_areas(student_id, 100)["developing"] + weak_strong_areas(student_id, 100)["strong"]
    if mastery_rows:
        st.markdown("### 📈 Topic Mastery")
        st.bar_chart({x["topic"]: float(x["mastery_score"]) for x in mastery_rows[:12]})

    action = next_best_action(student_id)
    st.markdown("### 🎯 Next Best Action")
    st.info(f"**{action['title']}** — {action['reason']}")
    if action["topic"]:
        st.write(f"Recommended: **{action['subject']} → {action['topic']}**")

    if 0 < stats["attempted"] < 10:
        st.caption(f"ℹ️ Based on only {stats['attempted']} answered question(s) - scores become more reliable as you practice.")

    areas = weak_strong_areas(student_id, 5)

    def _show(items: list[dict]) -> None:
        for x in items:
            st.write(
                f"**{x['subject']} → {x['topic']}** — {float(x['mastery_score']):.0f}% "
                f"({x['label']} · {x['attempts']} answered · {float(x['accuracy']):.0f}% correct)"
            )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### ⚠️ Weak Areas")
        if areas["weak"]:
            _show(areas["weak"])
        elif stats["topics_tracked"]:
            st.success("No weak areas right now. Keep practicing!")
        else:
            st.info("Complete a quiz to build your learning profile.")
    with c2:
        st.markdown("### 🧠 Strong Areas")
        if areas["strong"]:
            _show(areas["strong"])
        elif stats["topics_tracked"]:
            st.info("No topic has reached 75% with enough answers yet.")
        else:
            st.info("Strong areas will appear after quiz attempts.")

    if areas["developing"]:
        with st.expander(f"📈 Developing topics ({len(areas['developing'])}) — between 60% and 74%"):
            _show(sorted(areas["developing"], key=lambda r: r["mastery_score"]))
    if areas["building"]:
        with st.expander(f"🧪 Not enough data yet ({len(areas['building'])}) — fewer than 3 answers"):
            for x in areas["building"]:
                st.write(f"{x['subject']} → {x['topic']} — {x['attempts']} answered so far")

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
    memories = LongTermMemory(student_id).recent(5)
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
    time_limit = st.number_input("Quiz time limit (minutes)", 1, 180, 20, key="personal_time")
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
                    st.session_state.quiz = {
                        "questions": questions,
                        "subject": "Personalized Material",
                        "topic": topic,
                        "difficulty": actual,
                        "sources": results,
                        "time_limit_minutes": int(time_limit),
                        "started": time.time(),
                    }
                    st.session_state.quiz_answers = {}
                    st.session_state.last_quiz_result = None
                    st.success("Quiz created. Open Practice → Current Quiz.")
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
        time_limit = st.number_input("Quiz time limit (minutes)", 1, 180, 20, key="db_time")
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
                        st.session_state.quiz = {
                            "questions": questions,
                            "subject": subject,
                            "topic": topic,
                            "difficulty": actual,
                            "sources": results,
                            "time_limit_minutes": int(time_limit),
                            "started": time.time(),
                        }
                        st.session_state.quiz_answers = {}
                        st.session_state.last_quiz_result = None
                        st.success("Quiz created. Open Practice → Current Quiz to take it.")
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
            st.info("No active quiz. Start one from Learn or Practice My Weak Topics.")
        else:
            questions = quiz["questions"]
            st.write(f"**{quiz['subject']} → {quiz['topic']}** · {quiz['difficulty']} · {len(questions)} questions")
            if quiz.get("time_limit_minutes"):
                elapsed = int(time.time() - quiz.get("started", time.time()))
                remaining = max(0, int(quiz["time_limit_minutes"] * 60) - elapsed)
                st.warning(f"⏱️ Time remaining: {remaining // 60}:{remaining % 60:02d}")
                if remaining <= 0:
                    st.error("Time is up. Submit the quiz now.")
            for i, q in enumerate(questions):
                st.markdown(f"### Q{i + 1}. {q.get('question', '')}")
                options = q.get("options") or {}
                choice = st.radio(
                    "Choose an answer",
                    list(options.keys()),
                    format_func=lambda k, q=q: f"{k}. {q['options'][k]}",
                    key=f"quiz_choice_{i}",
                    index=None,
                )
                if choice:
                    st.session_state.quiz_answers[i] = choice

            if st.button("Submit Quiz", type="primary"):
                try:
                    answers = dict(st.session_state.quiz_answers)
                    quiz_id = record_quiz(
                        student_id,
                        quiz["subject"],
                        quiz["topic"],
                        questions,
                        answers,
                        quiz["difficulty"],
                    )
                    correct = sum(
                        answers.get(i) == str(q.get("answer", "")).strip()
                        for i, q in enumerate(questions)
                    )
                    st.session_state.last_quiz_id = quiz_id
                    st.session_state.last_quiz_result = {
                        "questions": questions,
                        "answers": answers,
                        "correct": correct,
                    }
                    if correct == len(questions):
                        add_achievement(student_id, "perfect_quiz", "Perfect Quiz")
                    st.success(f"Quiz submitted: {correct}/{len(questions)} correct.")
                except Exception as exc:
                    st.error("Prep AI could not save this quiz result. Please try again.")
                    if st.session_state.get("debug_mode"):
                        st.exception(exc)

            result = st.session_state.get("last_quiz_result")
            if result:
                st.markdown("## Quiz Result")
                for i, q in enumerate(result["questions"]):
                    selected = result["answers"].get(i, "Skipped")
                    correct_answer = str(q.get("answer", "")).strip()
                    mark = "✅ Correct" if selected == correct_answer else "❌ Incorrect"
                    st.markdown(f"### Q{i + 1}. {q.get('question', '')}")
                    st.write(f"**Result:** {mark}")
                    st.write(f"**Your answer:** {selected}")
                    st.write(f"**Correct answer:** {correct_answer}")
                    st.write(f"**Explanation:** {q.get('explanation', 'No explanation provided.')}")

                st.download_button(
                    "Download Quiz PDF",
                    questions_to_pdf("Prep AI Quiz", questions),
                    "prep_ai_quiz.pdf",
                    "application/pdf",
                )

    with tabs[1]:
        weak = weak_topics(student_id, 10)
        if not weak:
            st.info("No topic is below 75% right now. Complete more quizzes to keep your profile up to date.")
        else:
            for x in weak:
                st.write(
                    f"**{x['subject']} → {x['topic']} → {x['concept']}** · "
                    f"{float(x['mastery_score']):.0f}% · mistakes {x['repeated_mistakes']}"
                )

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
                    questions = orch.practice_request(ctx, 10)
                    if not questions:
                        st.warning("Prep AI could not generate reliable questions for this weak topic. Try again or check your database material.")
                    else:
                        st.session_state.quiz = {
                            "questions": questions,
                            "subject": x["subject"],
                            "topic": x["topic"],
                            "difficulty": difficulty,
                            "sources": sources,
                        }
                        st.session_state.quiz_answers = {}
                        st.session_state.last_quiz_result = None
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
                    questions = orch.practice_request(ctx, 10)

                    if not questions:
                        st.warning(
                            "No reliable revision questions could be generated. "
                            "Try again or check the learning material."
                        )
                    else:
                        st.session_state.quiz = {
                            "questions": questions,
                            "subject": subject,
                            "topic": topic,
                            "difficulty": revision_difficulty,
                            "sources": sources,
                            "revision": True,
                        }
                        st.session_state.quiz_answers = {}
                        st.session_state.last_quiz_result = None
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
    hero("🎓 Exam Mode", "Timed practice with no explanations until submission.")
    subject = st.selectbox("Subject", ["Biology", "Chemistry", "Physics", "English"], key="exam_subject")
    topic = st.text_input("Topic", key="exam_topic")
    count = st.slider("Questions", 10, 100, 30, key="exam_count")
    difficulty = st.selectbox("Difficulty", ["Adaptive", "Easy", "Medium", "Hard"], key="exam_diff")
    negative = st.checkbox("Negative marking", value=False)
    minutes = st.number_input("Time limit (minutes)", 5, 180, 30)

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
            st.session_state.exam = {
                "questions": questions,
                "subject": subject,
                "topic": topic,
                "difficulty": actual,
                "negative": negative,
                "minutes": minutes,
                "started": time.time(),
            }
            st.session_state.exam_answers = {}
            st.rerun()

    exam = st.session_state.get("exam")
    if exam:
        elapsed = int(time.time() - exam["started"])
        st.warning(f"Time used: {elapsed // 60}:{elapsed % 60:02d} / {exam['minutes']} minutes")
        for i, q in enumerate(exam["questions"]):
            options = q.get("options") or {}
            choice = st.radio(
                f"Q{i + 1}. {q.get('question', '')}",
                list(options.keys()),
                format_func=lambda k, q=q: f"{k}. {q['options'][k]}",
                key=f"exam_{i}",
                index=None,
            )
            if choice:
                st.session_state.exam_answers[i] = choice

        if st.button("Submit Exam", type="primary"):
            answers = dict(st.session_state.exam_answers)
            score = sum(
                answers.get(i) == str(q.get("answer", "")).strip()
                for i, q in enumerate(exam["questions"])
            )
            wrong = sum(
                bool(answers.get(i)) and answers.get(i) != str(q.get("answer", "")).strip()
                for i, q in enumerate(exam["questions"])
            )
            skipped = len(exam["questions"]) - len(answers)
            final = score - (wrong * 0.25 if exam["negative"] else 0)
            st.success(
                f"Exam result: {final:.2f}/{len(exam['questions'])} · "
                f"Correct {score} · Incorrect {wrong} · Skipped {skipped}"
            )
            record_quiz(student_id, exam["subject"], exam["topic"], exam["questions"], answers, exam["difficulty"])
            st.session_state.exam = None


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
    usage = memory.usage()
    st.markdown("### 💾 Memory Usage")
    c1, c2 = st.columns(2)
    c1.metric("Memory records", f"{usage['items']} / {usage['max_items']}")
    c2.metric("Storage", f"{usage['bytes']/1024:.1f} KB / {usage['max_bytes']/1024:.0f} KB")
    st.progress(usage['percent_items'] / 100, text=f"Record utilization: {usage['percent_items']:.1f}%")
    st.progress(usage['percent_bytes'] / 100, text=f"Storage utilization: {usage['percent_bytes']:.1f}%")
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

    if st.button("Clear My Long-Term Memory"):
        memory.clear()
        st.session_state.tutor_messages = []
        st.success("Long-term semantic memory cleared. Agent session history is retained separately for audit/history.")
        st.rerun()


def render_history() -> None:
    hero("📈 History", "Review quiz performance and mistakes for the current student profile.")
    rows = history(student_id)
    if rows:
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
    saved_bg = prefs.get("background_color") or st.session_state.get("background_color", "#FFFFFF")
    saved_text = prefs.get("text_color") or st.session_state.get("text_color", "#111827")
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
        background_color = st.color_picker("Background theme color", saved_bg)
        text_color = st.color_picker("Text color", saved_text)
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
            background_color=background_color,
            text_color=text_color,
        )
        st.session_state.llm_model = model
        st.session_state.ui_color = color
        st.session_state.background_color = background_color
        st.session_state.text_color = text_color
        st.success("Settings saved for this student.")
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



def render_shared_quiz() -> None:
    hero("👥 Shared Quiz", "Create one question set and let multiple students take it concurrently with individual timing and performance records.")
    tabs = st.tabs(["Create", "Join"])
    with tabs[0]:
        subject = st.selectbox("Subject", ["Biology", "Chemistry", "Physics", "English"], key="shared_subject")
        topic = st.text_input("Topic", key="shared_topic")
        count = st.number_input("Number of questions", 1, 100, 20, key="shared_count")
        minutes = st.number_input("Time limit (minutes)", 1, 180, 30, key="shared_minutes")
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard", "Adaptive"], key="shared_diff")
        title = st.text_input("Quiz title", value="Shared Quiz", key="shared_title")
        if st.button("Create Shared Quiz", type="primary"):
            actual = difficulty if difficulty != "Adaptive" else difficulty_for_topic(student_id, subject, topic or "General")
            db_index, metadata = load_database_index(); context = ""
            if db_index is not None:
                context, _ = build_rag_context(f"{subject} {topic}", metadata, db_index, 10)
            questions = orch.practice_request(create_context("Create shared quiz", subject, topic, actual, context), int(count))
            if not questions:
                st.error("Could not generate reliable questions.")
            else:
                import secrets
                code = secrets.token_hex(3).upper()
                create_shared_quiz(student_id, title, subject, topic, actual, questions, int(minutes), code)
                st.success(f"Shared quiz created. Join code: {code}")
                st.code(code)

    with tabs[1]:
        code = st.text_input("Quiz join code", max_chars=12, key="join_code").strip().upper()
        if code:
            sq = get_shared_quiz(code)
            if not sq:
                st.error("Shared quiz not found or inactive.")
            else:
                started = start_shared_quiz(sq["id"], student_id)
                from datetime import datetime
                elapsed = max(0, int((datetime.utcnow() - datetime.fromisoformat(started)).total_seconds()))
                remaining = max(0, int(sq["time_limit_minutes"] * 60) - elapsed)
                st.info(f"{sq['title']} · {len(sq['questions'])} questions · {sq['time_limit_minutes']} minutes")
                st.warning(f"Time remaining: {remaining // 60}:{remaining % 60:02d}")
                answers = {}
                for i, q in enumerate(sq["questions"]):
                    opts = q.get("options") or {}
                    choice = st.radio(f"Q{i + 1}. {q.get('question', '')}", list(opts), format_func=lambda k, q=q: f"{k}. {q['options'][k]}", key=f"shared_{sq['id']}_{i}", index=None)
                    if choice:
                        answers[i] = choice
                if st.button("Submit Shared Quiz", type="primary"):
                    quiz_id = record_quiz(student_id, sq["subject"], sq["topic"], sq["questions"], answers, sq["difficulty"], shared_quiz_id=sq["id"])
                    finish_shared_quiz(sq["id"], student_id)
                    correct = sum(answers.get(i) == str(q.get("answer", "")).strip() for i, q in enumerate(sq["questions"]))
                    st.success(f"Submitted: {correct}/{len(sq['questions'])} correct.")


def render_tutor_dashboard() -> None:
    hero("👨‍🏫 Tutor Dashboard", "Review individual student performance, mastery, quiz history and mistakes.")
    expected = get_secret("TUTOR_ACCESS_CODE", "") or ""
    if expected and st.session_state.get("tutor_authenticated") is not True:
        code = st.text_input("Tutor access code", type="password")
        if st.button("Unlock Tutor Dashboard"):
            if code == expected:
                st.session_state.tutor_authenticated = True
                st.rerun()
            else:
                st.error("Invalid tutor access code.")
        return
    rows = all_students_performance()
    if not rows:
        st.info("No student performance data yet.")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)
    ids = [r["student_id"] for r in rows]
    selected = st.selectbox("Inspect student", ids, format_func=lambda x: next((f"{r['name']} ({x})" for r in rows if r["student_id"] == x), x))
    perf = student_performance(selected)
    stats = perf["stats"]
    c = st.columns(4)
    c[0].metric("Mastery", f"{stats['overall']}%")
    c[1].metric("Accuracy", f"{stats['accuracy']}%")
    c[2].metric("Answered", stats["attempted"])
    c[3].metric("Quizzes", len(perf["history"]))
    if perf["mastery"]:
        st.markdown("### Topic Mastery")
        st.bar_chart({x["topic"]: float(x["mastery_score"]) for x in perf["mastery"][:15]})
    if perf["history"]:
        st.markdown("### Quiz History")
        st.dataframe(perf["history"], use_container_width=True, hide_index=True)
    if perf["mistakes"]:
        with st.expander("Recent mistakes"):
            st.dataframe(perf["mistakes"], use_container_width=True, hide_index=True)

routes = {
    "Dashboard": render_dashboard,
    "Learn": render_learn,
    "Practice": render_practice,
    "Exam": render_exam,
    "AI Tutor": render_tutor,
    "Voice Tutor": render_voice_tutor,
    "Research Agent": render_research,
    "Study Plan": render_plan,
    "Memory": render_memory,
    "History": render_history,
    "Shared Quiz": render_shared_quiz,
    "Tutor Dashboard": render_tutor_dashboard,
    "Settings": render_settings,
}

routes[page]()

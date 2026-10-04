"""Tutor dashboard: class overview, per-student drill-down and shared-quiz results.

Only performance data is shown. Students' private tutor chats, research and
memory contents are deliberately NOT visible here.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

import charts
from db import (
    list_shared_quizzes,
    roster,
    set_shared_quiz_open,
    shared_quiz_results,
    student_quiz_trend,
    subject_breakdown,
    topic_mastery,
    weak_strong_areas,
    recent_mistakes,
)
from quiz_runtime import clock
from ui import hero


def _csv(rows: list[dict]) -> bytes:
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def render_student_performance() -> None:
    """Class overview + one-student drill-down."""
    rows = roster()
    tabs = st.tabs(["Class overview", "Individual student"])

    # ---------------------------------------------------------------- overview
    with tabs[0]:
        if not rows:
            st.info("No students have joined yet. Students appear here after they create a profile.")
        else:
            active = [r for r in rows if r["quizzes"]]
            c = st.columns(4)
            c[0].metric("Students", len(rows))
            c[1].metric("Taken ≥1 quiz", len(active))
            c[2].metric("Class avg score", f"{sum(r['avg_score'] for r in active) / len(active):.0f}%" if active else "—")
            c[3].metric("Need attention", sum(1 for r in active if r["mastery"] < 60))

            if active:
                ranked = sorted(active, key=lambda r: r["avg_score"], reverse=True)
                charts.show(charts.simple_bars([r["name"] for r in ranked], [r["avg_score"] for r in ranked], "Average quiz score per student", "Avg score %"))

            table = [{
                "Student": r["name"], "ID": r["student_id"], "Quizzes": r["quizzes"], "Avg score %": r["avg_score"],
                "Answered": r["questions_answered"], "Accuracy %": r["accuracy"], "Mastery %": r["mastery"],
                "Revision due": r["revision_due"], "Timed out": r["timeouts"], "Last active": str(r["last_active"] or "")[:16].replace("T", " "),
            } for r in rows]
            st.dataframe(table, hide_index=True)
            st.download_button("📥 Download class report (CSV)", _csv(table), "class_performance.csv", "text/csv", key="dl_class_csv")

    # ------------------------------------------------------------- one student
    with tabs[1]:
        if not rows:
            st.info("No students yet.")
        else:
            labels = {r["student_id"]: f"{r['name']} ({r['student_id']})" for r in rows}
            sid = st.selectbox("Choose a student", list(labels), format_func=lambda k: labels[k], key="tutor_pick_student")
            info = next(r for r in rows if r["student_id"] == sid)
            trend = student_quiz_trend(sid)

            c = st.columns(4)
            c[0].metric("Quizzes taken", info["quizzes"])
            c[1].metric("Questions answered", info["questions_answered"])
            c[2].metric("Revision due", info["revision_due"])
            c[3].metric("Timed out", info["timeouts"])

            g1, g2, g3 = st.columns(3)
            with g1:
                charts.show(charts.ring(info["mastery"], "Overall mastery"))
            with g2:
                charts.show(charts.ring(info["accuracy"], "Accuracy", color="#16A34A"))
            with g3:
                tot_c = sum(t["correct"] for t in trend)
                tot_i = sum(t["incorrect"] for t in trend)
                tot_s = sum(t["skipped"] for t in trend)
                charts.show(charts.donut({"Correct": tot_c, "Incorrect": tot_i, "Skipped": tot_s}, charts.OUTCOME_COLORS, "All answers", center_text=str(tot_c + tot_i + tot_s)))

            if trend:
                charts.show(charts.score_trend(trend))
            else:
                st.info("This student has not completed a quiz yet.")

            topics = topic_mastery(sid)
            if topics:
                charts.show(charts.mastery_bars(topics))
            subs = subject_breakdown(sid)
            if subs:
                charts.show(charts.simple_bars([s["subject"] for s in subs], [s["accuracy"] for s in subs], "Accuracy by subject", "Accuracy %"))

            areas = weak_strong_areas(sid, 5)
            if areas["weak"]:
                st.markdown("**Needs help with:** " + ", ".join(f"{a['subject']} → {a['topic']}" for a in areas["weak"]))
            mistakes = recent_mistakes(sid, 5)
            if mistakes:
                with st.expander("Most recent mistakes"):
                    for m in mistakes:
                        st.write(f"**{m['topic']}** — {m['question_text'][:140]}")

            if trend:
                hist = [{
                    "When": str(t["created_at"])[:16].replace("T", " "), "Subject": t["subject"], "Topic": t["topic"],
                    "Score %": round(t["score"], 1), "Correct": t["correct"], "Wrong": t["incorrect"], "Skipped": t["skipped"],
                    "Time used": clock(t["time_taken_sec"]) if t["time_taken_sec"] is not None else "—",
                    "Time limit": clock(t["time_limit_sec"]) if t["time_limit_sec"] else "None",
                    "Timed out": "Yes" if t["timed_out"] else "", "Shared quiz": t["shared_code"] or "",
                } for t in reversed(trend)]
                with st.expander("Full quiz history"):
                    st.dataframe(hist, hide_index=True)
                st.download_button("📥 Download this student's history (CSV)", _csv(hist), f"{sid}_history.csv", "text/csv", key="dl_student_csv")



def render_quiz_results() -> None:
    """Results of every published quiz (tutor only)."""
    if True:
        quizzes = list_shared_quizzes()
        if not quizzes:
            st.info("No quizzes have been published yet. Create one in Create Quiz, then publish it.")
        else:
            labels = {q["code"]: f"{q['code']} · {q['title']} · {q['attempts']} attempt(s)" for q in quizzes}
            code = st.selectbox("Shared quiz", list(labels), format_func=lambda k: labels[k], key="tutor_pick_shared")
            q = next(x for x in quizzes if x["code"] == code)
            st.caption(f"Created by {q.get('creator_name') or q['created_by']} · {q['subject']} → {q['topic']} · {q['difficulty']} · time limit {clock(q['time_limit_sec']) if q['time_limit_sec'] else 'none'} · {'OPEN' if q['is_open'] else 'CLOSED'}")
            results = shared_quiz_results(code)
            if results:
                c1, c2 = st.columns([1, 2])
                with c1:
                    done = sum(1 for r in results if not r["timed_out"])
                    charts.show(charts.donut({"Finished": done, "Timed out": len(results) - done}, {"Finished": "#16A34A", "Timed out": "#DC2626"}, "Completion", center_text=str(len(results))))
                with c2:
                    charts.show(charts.simple_bars([r["name"] for r in results], [round(r["score"], 1) for r in results], "Score per student", "Score %"))
                table = [{"Rank": i + 1, "Student": r["name"], "ID": r["student_id"], "Score %": round(r["score"], 1), "Correct": r["correct"], "Wrong": r["incorrect"], "Skipped": r["skipped"],
                          "Time": clock(r["time_taken_sec"]) if r["time_taken_sec"] is not None else "—", "Timed out": "Yes" if r["timed_out"] else ""} for i, r in enumerate(results)]
                st.dataframe(table, hide_index=True)
                st.download_button("📥 Download results (CSV)", _csv(table), f"quiz_{code}_results.csv", "text/csv", key=f"dl_quiz_{code}")
            else:
                st.info("Nobody has submitted this quiz yet.")
            if q["is_open"]:
                if st.button("Close this quiz (no new students)", key=f"close_{code}"):
                    set_shared_quiz_open(code, False)
                    st.rerun()
            else:
                if st.button("Re-open this quiz", key=f"open_{code}"):
                    set_shared_quiz_open(code, True)
                    st.rerun()


def render_tutor_dashboard() -> None:
    """Performance-only view (kept for backwards compatibility)."""
    hero("👩‍🏫 Tutor Dashboard", "Check the performance of every student and of every published quiz.")
    tabs = st.tabs(["Student performance", "Quiz results"])
    with tabs[0]:
        render_student_performance()
    with tabs[1]:
        render_quiz_results()

"""Streamlit pages for the Merit Aggregate Agent and the Path Finder agent.

Kept out of app.py so that file stays readable. app.py only calls render_merit() / render_pathfinder().
"""
from __future__ import annotations

import csv
import io
import json
from typing import Any

import streamlit as st

import merit_service as ms
import pathfinder_service as pf
from db import merit_results, save_agent_session, save_merit_result, weak_strong_areas
from memory import LongTermMemory
from pdf_export import text_to_pdf
from rag import hybrid_search, load_database_index
from ui import hero

BAND_ICON = {"Safe": "🟢 Safe", "Target": "🟡 Target", "Reach": "🟠 Reach", "Unlikely": "🔴 Unlikely"}
STATUS_ICON = {"meets": "✅", "does not meet": "❌", "cannot check": "❔", "pass": "✅", "fail": "❌", "unknown": "❔"}
PROVINCES = ["Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan", "Islamabad (ICT)", "Azad Kashmir", "Gilgit-Baltistan", "Other"]


# ============================================================================ helpers
def _confidence_badge(level: str) -> str:
    return {"high": "🟢 high", "medium": "🟡 medium", "medium-high": "🟢 medium-high", "low": "🔴 low", "user-provided": "🔵 user-provided"}.get(level, level)


def _rows_to_csv(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _cite_line(ids: list[str], evidence: list[dict[str, Any]]) -> str:
    by_id = {e["id"]: e for e in evidence}
    parts = []
    for i in ids:
        e = by_id.get(i)
        if not e:
            continue
        label = pf.TRUST_NOTE.get(e["source_type"], e["source_type"] or e["kind"])
        link = f" - {e['url']}" if e.get("url") else ""
        parts.append(f"[{i}] {e['title']} ({label}){link}")
    return "Sources: " + " | ".join(parts) if parts else ""


# ============================================================================ MERIT AGENT PAGE
def render_merit(student_id: str, orch: Any) -> None:
    hero("🧮 Merit Aggregate Agent",
         "Calculate your aggregate for MDCAT, NUMS, ECAT, NUST NET, NTS-based universities or your own formula, then compare it with previous years' closing merits.")

    formulas = ms.load_formulas()
    options: dict[str, dict[str, Any] | None] = {f["name"]: f for f in formulas}
    custom_label = "Custom: type the weights from your prospectus"
    options[custom_label] = None
    choice = st.selectbox("Exam / merit formula", list(options), key="merit_formula_choice")

    formula: dict[str, Any] | None = options[choice]
    if formula is None:
        st.caption("Use this for any university/NTS-based test. Weights must add up to 100.")
        c1, c2, c3, c4 = st.columns(4)
        w_m = c1.number_input("Matric weight %", 0.0, 100.0, 10.0, 1.0, key="cw_m")
        w_f = c2.number_input("FSc weight %", 0.0, 100.0, 40.0, 1.0, key="cw_f")
        w_t = c3.number_input("Test weight %", 0.0, 100.0, 50.0, 1.0, key="cw_t")
        t_tot = c4.number_input("Test total marks", 1.0, 2000.0, 100.0, 1.0, key="cw_tt")
        try:
            formula = ms.custom_formula(w_m, w_f, w_t, t_tot, name=f"Custom ({w_m:g}/{w_f:g}/{w_t:g})")
        except ValueError as exc:
            st.error(str(exc))
            return
    else:
        weights = " + ".join(f"{c['label']} {c['weight']:g}%" for c in formula["components"])
        st.caption(f"**Weights:** {weights}  ·  **Authority:** {formula['authority']}")
        if formula["confidence"] in {"low"} or "conflict" in formula["status"]:
            st.warning(f"⚠️ Formula confidence: **{formula['confidence']}**. {formula['verify_note']}")
        else:
            st.info(f"Formula confidence: **{_confidence_badge(formula['confidence'])}**. {formula['verify_note']}")
        with st.expander("Formula sources"):
            for s in formula["sources"]:
                st.write(f"• [{s['type']}] {s['title']} — {s['url']}")

    # ---- inputs
    st.subheader("Your marks")
    marks: dict[str, tuple[float, float]] = {}
    fid = formula["id"]
    for comp in formula["components"]:
        if float(comp["weight"]) <= 0:
            continue
        c1, c2 = st.columns(2)
        obtained = c1.number_input(f"{comp['label']}: obtained", 0.0, 5000.0, 0.0, 1.0, key=f"m_{fid}_{comp['key']}_o")
        total = c2.number_input(f"{comp['label']}: total", 1.0, 5000.0, float(comp.get("default_total", 100)), 1.0, key=f"m_{fid}_{comp['key']}_t")
        if comp.get("hint"):
            st.caption(f"ℹ️ {comp['hint']}")
        marks[comp["key"]] = (obtained, total)

    rows_all = ms.load_closing_merits()
    programs = ms.available_programs(rows_all, formula["exam"])
    c1, c2, c3 = st.columns(3)
    program = c1.selectbox("Program (for recommendations)", ["(any)"] + programs, key="merit_program") if programs else ""
    program = "" if program == "(any)" else program
    quota = c2.text_input("Quota", "Open Merit", key="merit_quota")
    province = c3.selectbox("Province filter", ["(all)", "Punjab", "Sindh", "Khyber Pakhtunkhwa", "Balochistan", "Islamabad (ICT)"], key="merit_province")
    province = "" if province == "(all)" else province

    if st.button("Calculate merit & recommend", type="primary", key="merit_go"):
        try:
            result, elig = orch.merit.calculate(formula, marks, program or None)
            recs = orch.merit.recommend(result["aggregate"], formula, program or None, quota, province or None)
            st.session_state.merit_state = {
                "formula": formula, "marks": marks, "result": result, "elig": elig, "recs": recs,
                "program": program, "quota": quota, "province": province, "explanation": "",
            }
            save_merit_result(student_id, formula["exam"], formula["id"], formula["name"], program, {k: list(v) for k, v in marks.items()}, result["aggregate"])
            mem = LongTermMemory(student_id)
            mem.add(f"Merit calculation: {formula['name']} aggregate {result['aggregate']:.2f}%" + (f" for {program}" if program else ""), "merit_result", "", formula["exam"], importance=0.7, confidence=1.0)
        except ValueError as exc:
            st.error(str(exc))
            st.session_state.pop("merit_state", None)
        except Exception as exc:
            st.error("The merit calculation could not be completed.")
            if st.session_state.get("debug_mode"):
                st.exception(exc)

    state = st.session_state.get("merit_state")
    if not state or state["formula"]["id"] != formula["id"]:
        _merit_data_tools(formula, programs, orch)
        return

    result, elig, recs = state["result"], state["elig"], state["recs"]
    st.divider()
    m1, m2 = st.columns([1, 2])
    m1.metric("Your aggregate", f"{result['aggregate']:.4f}%")
    m2.dataframe(
        [{"Component": c["label"], "Marks": f"{c['obtained']:g}/{c['total']:g}", "Percent": f"{c['percent']:.2f}%", "Weight": f"{c['weight']:g}%", "Contribution": f"{c['contribution']:.4f}"} for c in result["components"]],
        hide_index=True, use_container_width=True,
    )
    if elig:
        st.markdown("**Eligibility checks (only rules that have a recorded source)**")
        for e in elig:
            st.write(f"{STATUS_ICON[e['status']]} {e['check']} — {e['detail']} _(confidence: {e['confidence']})_")

    # ---- recommendations
    st.subheader("Recommendations from previous years' merit")
    if recs:
        table = [{
            "Band": BAND_ICON[r["band"]], "Institution": r["institution"], "City": r["city"], "Program": r["program"],
            "Expected cutoff": r["expected_cutoff"], "Latest": f"{r['latest_cutoff']} ({r['latest_year']})",
            "Your margin": r["margin"], "Years used": ", ".join(map(str, r["years_used"])),
            "List": r["list_type"], "Data confidence": _confidence_badge(r["confidence"]), "Caution": r["notes"],
        } for r in recs]
        st.dataframe(table, hide_index=True, use_container_width=True)
        st.caption("Band rule: expected cutoff = latest closing merit + average yearly change; Safe = above it by a buffer, Target = within the buffer, Reach = up to 3 buffers below. Past merits never guarantee future merits. Always check the official merit lists.")
        with st.expander("Where this data comes from"):
            for r in recs:
                st.write(f"**{r['institution']}** — {', '.join(r['source_types'])}: " + ", ".join(r["sources"]))
        d1, d2 = st.columns(2)
        d1.download_button("📥 Recommendations (CSV)", _rows_to_csv(table), "merit_recommendations.csv", "text/csv", key="merit_csv")
        report = "\n".join([f"Aggregate: {result['aggregate']:.4f}% ({result['formula_name']})", ""] + [f"{r['band']}: {r['institution']} - expected cutoff {r['expected_cutoff']} (latest {r['latest_cutoff']} in {r['latest_year']}, {r['list_type']} list, data confidence {r['confidence']}) {r['notes']}" for r in recs] + ["", "Past merits do not guarantee future merits. Verify the formula and merit lists in the official prospectus."])
        d2.download_button("📥 Report (PDF)", text_to_pdf("Prep AI - Merit Report", report), "merit_report.pdf", "application/pdf", key="merit_pdf")
    else:
        st.info(f"No closing-merit data is loaded for **{formula['exam']}**" + (f" / {state['program']}" if state["program"] else "") + f" / {state['quota']}. I will not guess cutoffs: add official data below (CSV upload or the verified web finder).")

    # ---- what do I need?
    st.subheader("What test score do I need?")
    names = ["Custom target"] + [f"{r['institution']} (safe target {r['expected_cutoff'] + r['buffer']:.2f})" for r in recs]
    pick = st.selectbox("Target", names, key="merit_target_pick")
    default_target = min(100.0, round(result["aggregate"] + 2, 2))
    if pick != "Custom target":
        default_target = min(100.0, round(recs[names.index(pick) - 1]["expected_cutoff"] + recs[names.index(pick) - 1]["buffer"], 2))
    target = st.number_input("Target aggregate %", 0.0, 100.0, float(default_target), 0.1, key=f"merit_target_{names.index(pick)}")
    try:
        test_comp = next(c for c in state["formula"]["components"] if c["key"] == "test")
        need = ms.required_test_score(state["formula"], {k: v for k, v in state["marks"].items() if k != "test"}, target, state["marks"]["test"][1])
        if need["already_met"]:
            st.success("Your Matric/FSc contribution alone already reaches this target.")
        elif not need["feasible"]:
            st.error(f"Even a perfect test score cannot reach {target:g}% with these Matric/FSc marks (you would need {need['needed_percent']:.1f}%).")
        else:
            st.success(f"You need about **{need['needed_marks']:g} / {need['test_total']:g}** ({need['needed_percent']:.1f}%) in the {test_comp['label']}.")
    except Exception as exc:
        st.warning(f"Could not compute the required score: {exc}")

    # ---- AI explanation
    if st.button("🤖 Explain my result (AI)", key="merit_explain"):
        try:
            with st.spinner("Writing your explanation..."):
                mems = orch.memory_agent.retrieve("merit admission goals")
                text = orch.merit.explain(result, recs, elig, state["formula"], mems)
            st.session_state.merit_state["explanation"] = text
            save_agent_session(student_id, "Merit Aggregate Agent", f"{result['formula_name']} aggregate {result['aggregate']:.2f}", text)
        except Exception as exc:
            st.error("The AI explanation failed. The calculation above is still valid. Check your Groq key.")
            if st.session_state.get("debug_mode"):
                st.exception(exc)
    if st.session_state.merit_state.get("explanation"):
        st.markdown(st.session_state.merit_state["explanation"])

    _merit_data_tools(formula, programs, orch)


def _merit_data_tools(formula: dict[str, Any], programs: list[str], orch: Any) -> None:
    with st.expander("📚 Closing-merit data: add official numbers"):
        st.caption("Recommendations are only as good as this data. Seed data comes with a source and a trust level for every row. Add official merit-list numbers here.")
        st.download_button("Download CSV template", ms.template_csv(), "closing_merit_template.csv", "text/csv", key="merit_tpl")
        up = st.file_uploader("Upload a closing-merit CSV", type=["csv"], key="merit_upload")
        if up is not None:
            rows, errors = ms.parse_closing_csv(up.getvalue())
            for e in errors[:10]:
                st.warning(e)
            if rows:
                st.write(f"{len(rows)} valid row(s) found.")
                st.dataframe(rows[:20], hide_index=True, use_container_width=True)
                if st.button("Save these rows", key="merit_save_upload"):
                    st.success(f"Added {ms.save_user_closing_rows(rows)} new row(s).")

        st.markdown("**Find on the web (AI-assisted, every number is verified against the page text)**")
        q = st.text_input("Search", f"{formula['exam']} closing merit last year", key="merit_web_q")
        if st.button("Search & extract", key="merit_web_go") and q.strip():
            try:
                with st.spinner("Searching and verifying..."):
                    accepted, rejected = orch.merit.find_closing_merits(q.strip(), formula["exam"])
                st.session_state.merit_found = accepted
                if rejected:
                    st.caption(f"{len(rejected)} extracted row(s) were rejected because the number or name was not in the source text.")
            except Exception as exc:
                st.error("Web extraction failed (needs internet and a Groq key).")
                if st.session_state.get("debug_mode"):
                    st.exception(exc)
        found = st.session_state.get("merit_found") or []
        if found:
            st.write("Verified rows. Review them, they are saved as unverified website data unless the source is official:")
            st.dataframe(found, hide_index=True, use_container_width=True)
            if st.button("Save verified rows", key="merit_save_found"):
                st.success(f"Added {ms.save_user_closing_rows(found)} new row(s).")
                st.session_state.merit_found = []


# ============================================================================ PATH FINDER PAGE
def _prefill_from_merit(student_id: str) -> dict[str, float]:
    try:
        last = merit_results(student_id, 1)
        if not last:
            return {}
        marks = last[0]["marks"]
        out = {}
        if "matric" in marks and marks["matric"][1]:
            out["matric_pct"] = round(100 * marks["matric"][0] / marks["matric"][1], 1)
        if "fsc" in marks and marks["fsc"][1]:
            out["fsc_pct"] = round(100 * marks["fsc"][0] / marks["fsc"][1], 1)
        if "test" in marks and marks["test"][1]:
            out["test_pct"] = round(100 * marks["test"][0] / marks["test"][1], 1)
            out["test_exam"] = last[0]["exam"]
        return out
    except Exception:
        return {}


def _doc_chunks(query: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    try:
        if st.session_state.get("personal_index") is not None and st.session_state.get("personal_chunks"):
            chunks += hybrid_search(query, st.session_state.personal_chunks, st.session_state.personal_index, 3)
        index, meta = load_database_index()
        if index is not None and meta:
            chunks += hybrid_search(query, meta, index, 3)
    except Exception:
        pass
    return chunks


def render_pathfinder(student_id: str, orch: Any) -> None:
    hero("🧭 Path Finder",
         "Career paths, programs, eligibility checks, scholarships, accreditation and a roadmap, grounded in verified sources. Missing information is reported, never invented.")
    kb = pf.load_kb()
    pre = _prefill_from_merit(student_id)

    with st.form("pf_form"):
        c1, c2 = st.columns(2)
        level = c1.selectbox("Where are you now?", pf.LEVELS)
        stream = c2.selectbox("Intermediate stream", pf.STREAMS)
        interests = st.multiselect("What interests you?", list(kb["interest_map"]))
        c1, c2, c3 = st.columns(3)
        matric = c1.number_input("Matric / O-Level %", 0.0, 100.0, float(pre.get("matric_pct", 0.0)), 0.1, help="0 = not available")
        fsc = c2.number_input("FSc / A-Level % (so far)", 0.0, 100.0, float(pre.get("fsc_pct", 0.0)), 0.1, help="0 = not available")
        test_pct = c3.number_input("Best entry-test % (if any)", 0.0, 100.0, float(pre.get("test_pct", 0.0)), 0.1, help="0 = not available")
        c1, c2, c3 = st.columns(3)
        province = c1.selectbox("Province / domicile", PROVINCES)
        income = c2.number_input("Family monthly income (PKR)", 0, 5_000_000, 0, 5000, help="0 = prefer not to say. Used only for scholarship checks.")
        budget = c3.number_input("Yearly fee budget (PKR)", 0, 5_000_000, 0, 10_000, help="0 = no limit given")
        goals = st.text_area("Your goals / constraints", placeholder="Example: I want a government medical college, but I would like an engineering backup in Lahore.")
        c1, c2, c3 = st.columns(3)
        use_web = c1.checkbox("Search trusted websites", True)
        use_docs = c2.checkbox("Use my uploaded documents", True)
        use_signals = c3.checkbox("Use my Prep AI strengths", True)
        go = st.form_submit_button("Find my path", type="primary")

    if go:
        profile = {
            "level": level, "stream": stream, "interests": interests, "matric_pct": matric or None, "fsc_pct": fsc or None,
            "test_pct": test_pct or None, "province": province, "family_income": income or None, "annual_budget": budget or None, "goals": goals.strip(),
        }
        signals = ""
        if use_signals:
            try:
                areas = weak_strong_areas(student_id, 3)
                strong = ", ".join(f"{a['subject']} - {a['topic']}" for a in areas["strong"])
                weak = ", ".join(f"{a['subject']} - {a['topic']}" for a in areas["weak"])
                signals = f"strong topics: {strong or 'none yet'}; weak topics: {weak or 'none yet'}"
            except Exception:
                signals = ""
        try:
            with st.spinner("Checking verified data, documents and trusted sources..."):
                query = f"{' '.join(interests)} {stream} admission eligibility fee scholarship merit"
                chunks = _doc_chunks(query) if use_docs else []
                memories = orch.memory_agent.retrieve(f"career education goals {goals}")
                result = orch.pathfinder.run(profile, memories, chunks, signals, use_web)
            st.session_state.pf_state = {"profile": profile, "result": result}
            summary = result["ai"].get("summary") or "Path Finder run (no AI summary)."
            save_agent_session(student_id, "Path Finder", json.dumps({k: v for k, v in profile.items() if k != "family_income"}, ensure_ascii=False)[:800], summary)
            LongTermMemory(student_id).add(f"Path Finder: stream {stream}; interests {', '.join(interests) or 'not stated'}; goals {goals.strip()[:200] or 'not stated'}", "pathfinder_profile", "", "career", importance=0.8, confidence=1.0)
        except Exception as exc:
            st.error("Path Finder could not finish. Check your internet/Groq configuration.")
            if st.session_state.get("debug_mode"):
                st.exception(exc)

    state = st.session_state.get("pf_state")
    if not state:
        return
    result, profile = state["result"], state["profile"]
    evidence = result["evidence"]

    if result["web_error"]:
        st.warning(result["web_error"])
    if result["llm_error"]:
        st.warning(f"The AI writing step failed ({result['llm_error']}). The verified checks below are still complete.")
    if result["dropped"]:
        st.caption(f"🛡️ {result['dropped']} AI statement(s) were removed because they cited no valid source.")

    tabs = st.tabs(["Summary & careers", "Programs & eligibility", "Scholarships", "Accreditation", "Roadmap", "Evidence & gaps", "Ask Path Finder"])

    with tabs[0]:
        ai = result["ai"]
        if ai["summary"]:
            st.info(ai["summary"])
        if ai["career_paths"]:
            for cp in ai["career_paths"]:
                with st.expander(f"🎯 {cp['title']}", expanded=True):
                    st.write(cp["why_it_fits"])
                    for i, s in enumerate(cp["steps"], 1):
                        st.write(f"{i}. {s}")
                    st.caption(_cite_line(cp["evidence_ids"], evidence))
        else:
            st.write("No AI career narrative could be grounded. Showing the verified career pathways instead:")
            for c in result["careers_kb"]:
                st.markdown(f"**{c['title']}** _( {pf.TRUST_NOTE.get(c['source_type'], c['source_type'])} )_")
                for s in c["steps"]:
                    st.write(f"- {s}")

    with tabs[1]:
        if not result["programs"]:
            st.info("No program matched your stream/interests in the verified data. Pick an interest above.")
        for p in result["programs"]:
            with st.expander(f"🎓 {p['name']}  ·  {p['field']}  ·  match: {p['match_reason']}", expanded=True):
                st.caption(f"Entry tests: {', '.join(p.get('entry_tests', [])) or 'varies by university'} · data: {pf.TRUST_NOTE.get(p['source_type'], p['source_type'])} · confidence {p['confidence']}")
                for c in result["eligibility"][p["id"]]:
                    st.write(f"{STATUS_ICON[c['status']]} **{c['rule']}** — {c['detail']} _(confidence: {c['confidence']})_")
                if p.get("notes"):
                    st.caption(p["notes"])
                if p.get("merit_formula_ids"):
                    st.caption("Merit formulas available in the Merit Calculator: " + ", ".join(p["merit_formula_ids"]))
                for s in p.get("sources", []):
                    st.caption(f"Source: {s['title']} — {s['url']}")

    with tabs[2]:
        for s in result["scholarships"]:
            with st.expander(f"💰 {s['name']} ({s['provider']})", expanded=True):
                for c in s["checks"]:
                    st.write(f"{STATUS_ICON[c['status']]} {c['rule']} — {c['detail']} _(confidence: {c['confidence']})_")
                st.markdown("**Requirements**")
                for r in s["requirements"]:
                    st.write(f"- {r}")
                st.caption(f"Apply: {s['apply_url']} · data: {pf.TRUST_NOTE.get(s['source_type'], s['source_type'])}. Always confirm the current cycle, amounts and deadline with the university financial aid office.")
                for src in s["sources"]:
                    st.caption(f"Source: {src['title']} — {src['url']}")
        st.info("Only scholarships present in the verified data are listed here. Other scholarships appear under 'Evidence & gaps' if trusted web results mention them: confirm each one on its official page.")

    with tabs[3]:
        st.write("Never apply to a program before confirming its recognition. Checks for your matched programs:")
        for r in result["regulators"]:
            link = f" — {r['url']}" if r.get("url") else ""
            st.write(f"🛡️ **{r['name']}** (for: {', '.join(r['for_programs'])})")
            st.write(f"{r['check']}{link}")
            st.caption(f"Data: {pf.TRUST_NOTE.get(r['source_type'], r['source_type'])}. Specific accredited-program lists are not stored here: check the official site.")

    with tabs[4]:
        ai = result["ai"]
        if ai["roadmap"]:
            for phase in ai["roadmap"]:
                st.markdown(f"#### {phase['phase']}")
                for a in phase["actions"]:
                    st.write(f"- {a['action']}")
                    st.caption(_cite_line(a["evidence_ids"], evidence))
        else:
            st.write("No AI roadmap could be grounded.")
        st.markdown("#### ✅ Verification checklist (always valid)")
        for step in result["verification_steps"]:
            st.write(f"- {step['action']}")
        report = "\n".join(
            [ai.get("summary", "")] + [f"Career: {c['title']} - " + "; ".join(c["steps"]) for c in ai["career_paths"]]
            + [f"{ph['phase']}: " + "; ".join(a["action"] for a in ph["actions"]) for ph in ai["roadmap"]]
            + ["Verify: " + s["action"] for s in result["verification_steps"]] + ["", "Generated by Prep AI Path Finder. Confirm every requirement with the official university/regulator."]
        )
        st.download_button("📥 Download roadmap (PDF)", text_to_pdf("Prep AI - Path Finder Roadmap", report), "path_finder_roadmap.pdf", "application/pdf", key="pf_pdf")

    with tabs[5]:
        ai = result["ai"]
        st.markdown("#### Not available / you must check")
        for m in ai["missing_information"] or ["Fees, seats, deadlines and each university's exact formula are not in the verified data: check the official prospectus."]:
            st.write(f"- {m}")
        if ai["follow_up_questions"]:
            st.markdown("#### Questions that would sharpen the advice")
            for q in ai["follow_up_questions"]:
                st.write(f"- {q}")
        if ai["web_findings"]:
            st.markdown("#### Live web findings")
            for f in ai["web_findings"]:
                st.write(f"- {f['finding']}")
                st.caption(_cite_line(f["evidence_ids"], evidence))
        with st.expander(f"All evidence used ({len(evidence)})"):
            for e in evidence:
                st.write(f"**[{e['id']}]** {e['title']} — _{pf.TRUST_NOTE.get(e['source_type'], e['source_type'] or e['kind'])}_")
                st.caption((e["text"][:300] + ("..." if len(e["text"]) > 300 else "")) + (f"  {e['url']}" if e["url"] else ""))

    with tabs[6]:
        q = st.text_input("Ask a follow-up question", placeholder="Example: What does the HEC need-based scholarship require?", key="pf_q")
        if st.button("Ask", key="pf_ask") and q.strip():
            try:
                with st.spinner("Checking the sources..."):
                    ans = orch.pathfinder.ask(q.strip(), profile, evidence)
                if ans["not_found"]:
                    st.warning(ans["answer"] or "This is not in the available sources.")
                else:
                    st.markdown(ans["answer"])
                st.caption(_cite_line(ans["evidence_ids"], evidence))
                save_agent_session(student_id, "Path Finder", q.strip(), ans["answer"])
            except Exception as exc:
                st.error("Path Finder could not answer. Check your Groq key.")
                if st.session_state.get("debug_mode"):
                    st.exception(exc)

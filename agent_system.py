from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive import next_best_action
from groq_service import generate_json, generate_text
from memory import LongTermMemory, memory_prompt
from web_search import search_web
import merit_service as ms
import pathfinder_service as pf


@dataclass
class AgentContext:
    student_id: str
    request: str
    subject: str = ""
    topic: str = ""
    level: str = "MDCAT"
    difficulty: str = "Medium"
    rag_context: str = ""
    style_hint: str = ""  # extra instruction, e.g. "keep it short for voice"


class MemoryAgent:
    def __init__(self, memory: LongTermMemory):
        self.memory = memory

    def retrieve(self, request: str) -> list[dict[str, Any]]:
        return self.memory.retrieve(request, top_k=5)

    def remember_learning_event(self, text: str, subject: str = "", topic: str = "") -> None:
        # Keep long-term memory selective; store only meaningful learning events.
        if len(text.strip()) >= 20:
            self.memory.add(text, "learning", subject, topic, importance=0.8, confidence=0.8)


class TutorAgent:
    name = "Tutor Agent"

    def run(self, ctx: AgentContext, memories: list[dict[str, Any]]) -> str:
        return generate_text(f"""You are Prep AI Tutor, a persistent educational chatbot.
Maintain continuity with the student's previous learning memories.
Teach the student using a Socratic, adaptive approach.
Student level: {ctx.level}
Subject: {ctx.subject}
Topic: {ctx.topic}
Request: {ctx.request}

{memory_prompt(memories)}

RAG CONTEXT:
{(ctx.rag_context or "No document context was supplied.")[:12000]}

Start with a concise diagnostic question if the student asks to learn a concept. If they ask for an explanation, explain simply first, then give an MDCAT-level explanation, an example, and one follow-up question. Do not invent source-grounded facts outside the RAG context when the context is required.
{ctx.style_hint}""" )


class AssessmentAgent:
    name = "Assessment Agent"

    def generate_mcqs(self, ctx: AgentContext, memories: list[dict[str, Any]], count: int = 10) -> list[dict[str, Any]]:
        prompt = f"""Create exactly {count} high-quality {ctx.level}-level MCQs.
Subject: {ctx.subject}
Topic: {ctx.topic}
Difficulty target: {ctx.difficulty}
Use only the RAG context for factual content. Student memories may guide emphasis but are not evidence.
Avoid duplicates. Exactly one option must be correct.
Return a JSON object with a "questions" array. Each item must have: question, options (A/B/C/D), answer (A/B/C/D), explanation, concept, difficulty.

MEMORIES:
{memory_prompt(memories)[:3000]}

RAG CONTEXT:
{ctx.rag_context[:12000]}
"""
        data = generate_json(prompt)
        items = data if isinstance(data, list) else data.get("questions", [])
        return [q for q in items if isinstance(q, dict)][:count]

    def validate(self, questions: list[dict[str, Any]], context: str, topic: str) -> list[dict[str, Any]]:
        valid = []
        seen = set()
        for q in questions:
            text = str(q.get("question", "")).strip()
            options = q.get("options") or {}
            answer = str(q.get("answer", "")).strip().upper()
            if not text or text.lower() in seen or set(options.keys()) != {"A", "B", "C", "D"} or answer not in options:
                continue
            if context and not any(word in context.lower() for word in text.lower().split()[:4] if len(word) > 4):
                # Do not reject too aggressively; LLM phrasing can differ.
                pass
            seen.add(text.lower())
            valid.append(q)
        return valid


class PlannerAgent:
    name = "Planner Agent"

    def recommend(self, student_id: str) -> dict[str, str]:
        return next_best_action(student_id)

    def create_plan(self, student_id: str, exam_name: str, exam_date: str, hours_per_day: float, subjects: list[str], goals: str) -> list[dict[str, Any]]:
        weak = __import__("db").weak_topics(student_id, 8)
        weak_text = ", ".join(f"{x['subject']} - {x['topic']}" for x in weak)
        prompt = f"""Create a practical 7-day study plan for a student.
Exam: {exam_name}
Exam date: {exam_date}
Hours/day: {hours_per_day}
Subjects: {', '.join(subjects)}
Student goals: {goals}
Weak topics from learning profile: {weak_text or 'none yet'}
Return a JSON object with a "days" array containing 7 days. Each day has day and sessions (subject, topic, minutes, activity).
Prioritize weak topics and include MCQ practice and revision."""
        data = generate_json(prompt)
        return data if isinstance(data, list) else data.get("days", [])


class ResearchAgent:
    name = "Research Agent"

    def run(self, ctx: AgentContext) -> dict[str, Any]:
        results = search_web(ctx.request, max_results=5)
        source_text = "\n".join(f"- {r['title']}: {r['snippet']} ({r['url']})" for r in results)
        answer = generate_text(f"""You are a research agent. Answer this request using the web research below.
Request: {ctx.request}
Prefer trusted educational/government/medical sources. Clearly distinguish web research from student material.

WEB RESULTS:
{source_text}""")
        return {"answer": answer, "sources": results}


class MeritAgent:
    """Merit Aggregate Agent: the maths and the college bands are deterministic; AI only explains."""

    name = "Merit Aggregate Agent"

    def calculate(self, formula: dict[str, Any], marks: dict[str, tuple[float, float]], program: str | None = None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        return ms.calculate_aggregate(formula, marks), ms.eligibility_checks(formula, marks, program)

    def recommend(self, aggregate: float, formula: dict[str, Any], program: str | None, quota: str, province: str | None) -> list[dict[str, Any]]:
        rows = ms.load_closing_merits()
        return ms.recommend(aggregate, rows, exam=formula["exam"], program=program or None, quota=quota, province=province or None)

    def explain(self, result: dict[str, Any], recs: list[dict[str, Any]], elig: list[dict[str, Any]], formula: dict[str, Any], memories: list[dict[str, Any]]) -> str:
        facts = ms.merit_summary_for_llm(result, recs, elig)
        return generate_text(f"""You are the Merit Aggregate Agent of Prep AI. Explain the result below to a student in simple, encouraging language.
RULES: use ONLY the numbers in FACTS. Do not add, change or estimate any merit, cutoff, seat count or deadline. Say clearly that past merits do not guarantee future merits, that the formula must be confirmed in the official prospectus (formula confidence: {formula.get('confidence', 'unknown')}; note: {formula.get('verify_note', '')}), and mention any caution in the notes (first-list data, unverified websites, single year).
Give: 1) what the aggregate means, 2) which colleges are Safe/Target/Reach and why, 3) two practical next steps. Keep it under 250 words.
{memory_prompt(memories)}

FACTS:
{facts}""", max_completion_tokens=1200)

    def find_closing_merits(self, query: str, exam: str) -> tuple[list[dict[str, Any]], list[str]]:
        """Web search -> AI extraction -> every row must be literally present in its source snippet."""
        results = search_web(query, max_results=6)
        if not results:
            return [], []
        blocks = "\n".join(f"[{i}] {r['title']} <{r['url']}>: {r['snippet']}" for i, r in enumerate(results))
        data = generate_json(f"""Extract previous-year CLOSING MERIT rows from the search results below.
Only extract numbers that literally appear in the text. Do not calculate, round or guess. If a value is missing, skip that row.
Return JSON: {{"rows": [{{"source_index": 0, "year": 2025, "exam": "{exam}", "program": "MBBS", "institution": "...", "city": "", "province": "", "quota": "Open Merit", "list_type": "first|final|unknown", "closing_merit": 93.1}}]}}
If nothing can be extracted return {{"rows": []}}.

RESULTS:
{blocks}""")
        accepted: list[dict[str, Any]] = []
        rejected: list[str] = []
        for raw in (data.get("rows", []) if isinstance(data, dict) else []):
            try:
                src = results[int(raw.get("source_index"))]
            except (TypeError, ValueError, IndexError):
                rejected.append(f"{raw.get('institution', '?')}: no valid source_index")
                continue
            raw = {**raw, "source_title": src["title"], "source_url": src["url"],
                   "source_type": "official" if src.get("trusted") == "True" else "aggregator",
                   "note": "AI-extracted from a search snippet and number-verified. Check the official merit list."}
            ok, bad = ms.verify_extracted_rows([raw], f"{src['title']} {src['snippet']}")
            accepted += ok
            rejected += bad
        return accepted, rejected


class PathFinderAgent:
    """Path Finder: deterministic checks + cited evidence; the LLM may only write cited sections."""

    name = "Path Finder"

    def run(self, profile: dict[str, Any], memories: list[dict[str, Any]], doc_chunks: list[dict[str, Any]], learning_signals: str = "", use_web: bool = True) -> dict[str, Any]:
        kb = pf.load_kb()
        programs = pf.match_programs(kb, profile)
        eligibility = {p["id"]: pf.check_program_eligibility(p, profile) for p in programs}
        scholarships = pf.check_scholarships(kb, profile)
        regulators = pf.regulator_checks(kb, programs)
        careers = pf.careers_for(kb, programs)

        web_results: list[dict[str, str]] = []
        web_error = ""
        if use_web and programs:
            try:
                names = " ".join(p["name"].split(" (")[0] for p in programs[:3])
                web_results = search_web(f"{names} admission eligibility Pakistan {profile.get('province', '')} scholarship", max_results=5, trusted_only=True)
            except Exception as exc:  # web is optional: the verified parts still work
                web_error = f"Live web search failed ({type(exc).__name__}); showing verified data only."

        evidence = pf.build_evidence(kb, programs, scholarships, regulators, careers, doc_chunks, web_results)
        result: dict[str, Any] = {
            "programs": programs, "eligibility": eligibility, "scholarships": scholarships, "regulators": regulators,
            "careers_kb": careers, "evidence": evidence, "web_error": web_error, "dropped": 0, "llm_error": "",
            "verification_steps": pf.verification_roadmap(regulators),
            "ai": {"summary": "", "career_paths": [], "roadmap": [], "web_findings": [], "missing_information": [], "follow_up_questions": []},
        }
        try:
            data = generate_json(pf.build_prompt(profile, evidence, learning_signals, memory_prompt(memories)))
            result["ai"], result["dropped"] = pf.validate_llm_output(data, evidence)
        except Exception as exc:
            result["llm_error"] = str(exc)
        return result

    def ask(self, question: str, profile: dict[str, Any], evidence: list[dict[str, Any]]) -> dict[str, Any]:
        data = generate_json(pf.build_qa_prompt(question, profile, evidence))
        return pf.validate_qa(data, evidence)


class Orchestrator:
    """Lightweight multi-agent orchestrator. Deterministic routing keeps the app understandable."""

    def __init__(self, student_id: str):
        memory = LongTermMemory(student_id)
        self.memory_agent = MemoryAgent(memory)
        self.tutor = TutorAgent()
        self.assessment = AssessmentAgent()
        self.planner = PlannerAgent()
        self.research = ResearchAgent()
        self.merit = MeritAgent()
        self.pathfinder = PathFinderAgent()

    def tutor_request(self, ctx: AgentContext) -> str:
        return self.tutor.run(ctx, self.memory_agent.retrieve(ctx.request))

    def research_request(self, ctx: AgentContext) -> dict[str, Any]:
        return self.research.run(ctx)

    def practice_request(self, ctx: AgentContext, count: int = 10) -> list[dict[str, Any]]:
        memories = self.memory_agent.retrieve(f"{ctx.subject} {ctx.topic} weaknesses mistakes")
        questions = self.assessment.generate_mcqs(ctx, memories, count)
        return self.assessment.validate(questions, ctx.rag_context, ctx.topic)

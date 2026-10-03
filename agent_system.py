from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from adaptive import next_best_action
from groq_service import generate_json, generate_text
from memory import LongTermMemory, memory_prompt
from web_search import search_web


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


class Orchestrator:
    """Lightweight multi-agent orchestrator. Deterministic routing keeps the app understandable."""

    def __init__(self, student_id: str):
        memory = LongTermMemory(student_id)
        self.memory_agent = MemoryAgent(memory)
        self.tutor = TutorAgent()
        self.assessment = AssessmentAgent()
        self.planner = PlannerAgent()
        self.research = ResearchAgent()

    def tutor_request(self, ctx: AgentContext) -> str:
        return self.tutor.run(ctx, self.memory_agent.retrieve(ctx.request))

    def research_request(self, ctx: AgentContext) -> dict[str, Any]:
        return self.research.run(ctx)

    def practice_request(self, ctx: AgentContext, count: int = 10) -> list[dict[str, Any]]:
        memories = self.memory_agent.retrieve(f"{ctx.subject} {ctx.topic} weaknesses mistakes")
        questions = self.assessment.generate_mcqs(ctx, memories, count)
        return self.assessment.validate(questions, ctx.rag_context, ctx.topic)

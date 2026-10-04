"""Path Finder logic (no Streamlit): grounding is enforced HERE, not just requested in a prompt.

Pipeline
  1. deterministic matching + eligibility checks from the knowledge base (no AI)
  2. evidence pack: knowledge records (K), student's documents (D), trusted web results (W)
  3. the LLM may only write careers / roadmap / web findings, and must cite evidence ids
  4. validate_llm_output() drops anything without a valid citation and reports what was dropped
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
KB_PATH = ROOT / "knowledge" / "pathfinder_kb.json"

STREAMS = ["Pre-Medical", "Pre-Engineering", "ICS (Computer Science)", "I.Com (Commerce)", "Humanities / Arts", "A-Levels", "Other"]
LEVELS = ["Matric / O-Level (finished or finishing)", "FSc / A-Level (in progress)", "FSc / A-Level (finished)", "Undergraduate student", "Graduate"]
TRUST_NOTE = {
    "official": "official source", "news": "news report", "university-page": "university page",
    "official+secondary": "official + secondary sources", "secondary": "secondary website",
    "aggregator": "unverified website", "general-knowledge": "general guidance - verify",
}


def load_kb() -> dict[str, Any]:
    return json.loads(KB_PATH.read_text(encoding="utf-8"))


def _by_id(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {x["id"]: x for x in items}


# ----------------------------------------------------------------------------- 1. deterministic part
def match_programs(kb: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Programs that fit the student's stream or stated interests (never invents programs)."""
    programs = kb["programs"]
    wanted: set[str] = set()
    for label in profile.get("interests", []):
        wanted.update(kb.get("interest_map", {}).get(label, []))
    stream = profile.get("stream", "")
    matched = []
    for p in programs:
        stream_fit = bool(stream) and stream in p.get("streams", [])
        interest_fit = p["id"] in wanted
        if stream_fit or interest_fit:
            matched.append({**p, "match_reason": ("stream" if stream_fit else "") + (" + " if stream_fit and interest_fit else "") + ("interest" if interest_fit else "")})
    if not matched and not wanted and not stream:
        matched = [{**p, "match_reason": "no profile given"} for p in programs]
    return matched


def _check(rule: str, status: str, detail: str, confidence: str) -> dict[str, str]:
    return {"rule": rule, "status": status, "detail": detail, "confidence": confidence}


def check_program_eligibility(program: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, str]]:
    checks: list[dict[str, str]] = []
    stream = profile.get("stream", "")
    if program.get("streams"):
        needed = " / ".join(program["streams"])
        if not stream:
            checks.append(_check(f"Stream: {needed}", "cannot check", "your stream was not provided", program.get("stream_source", "unknown")))
        elif stream in program["streams"]:
            checks.append(_check(f"Stream: {needed}", "meets", f"you selected {stream}", program.get("stream_source", "unknown")))
        else:
            checks.append(_check(f"Stream: {needed}", "does not meet", f"you selected {stream}. Ask the university whether bridging or another route exists.", program.get("stream_source", "unknown")))
    else:
        checks.append(_check("Stream requirement", "cannot check", program.get("streams_note", "Varies by university: check the prospectus."), "not in verified data"))

    for rule in program.get("rules", []):
        if rule["type"] == "min_fsc_percent":
            value = profile.get("fsc_pct")
            subject = "FSc / intermediate percentage"
        elif rule["type"] == "min_test_percent":
            value = profile.get("test_pct")
            subject = "entry-test percentage"
        else:
            continue
        label = f"{rule['text']} [{rule.get('scope', '')}]"
        if value in (None, ""):
            checks.append(_check(label, "cannot check", f"your {subject} was not provided", rule.get("confidence", "unknown")))
        else:
            ok = float(value) >= float(rule["value"])
            checks.append(_check(label, "meets" if ok else "does not meet", f"you have {float(value):.1f}%, rule needs {float(rule['value']):g}%", rule.get("confidence", "unknown")))
    return checks


def check_scholarships(kb: dict[str, Any], profile: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for s in kb.get("scholarships", []):
        checks = []
        for rule in s.get("rules", []):
            if rule["type"] == "max_monthly_family_income":
                income = profile.get("family_income")
                if income in (None, "", 0):
                    checks.append(_check(rule["text"], "cannot check", "family income was not provided", rule.get("confidence", "unknown")))
                else:
                    ok = float(income) <= float(rule["value"])
                    checks.append(_check(rule["text"], "meets" if ok else "does not meet", f"you entered PKR {float(income):,.0f} per month, limit quoted is PKR {float(rule['value']):,.0f}", rule.get("confidence", "unknown")))
        out.append({**s, "checks": checks})
    return out


def regulator_checks(kb: dict[str, Any], programs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    regs = _by_id(kb["regulators"])
    rows: dict[str, dict[str, Any]] = {}
    hec = regs.get("reg_hec")
    if hec:
        rows[hec["id"]] = {**hec, "for_programs": ["All programs"]}
    for p in programs:
        reg = regs.get(p.get("regulator_id", ""))
        if reg:
            row = rows.setdefault(reg["id"], {**reg, "for_programs": []})
            row["for_programs"].append(p["name"])
    return list(rows.values())


def careers_for(kb: dict[str, Any], programs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = {p["field"] for p in programs}
    return [c for c in kb.get("careers", []) if fields & set(c.get("field", []))]


def verification_roadmap(regulators: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Safe, source-free steps: they only tell the student WHERE to verify (no facts asserted)."""
    steps = []
    for r in regulators:
        link = f" ({r['url']})" if r.get("url") else ""
        steps.append({"action": r["check"] + link, "basis": f"{r['name']}"})
    return steps


# ----------------------------------------------------------------------------- 2. evidence pack
def build_evidence(
    kb: dict[str, Any],
    programs: list[dict[str, Any]],
    scholarships: list[dict[str, Any]],
    regulators: list[dict[str, Any]],
    careers: list[dict[str, Any]],
    doc_chunks: list[dict[str, Any]] | None = None,
    web_results: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    ev: list[dict[str, Any]] = []

    def add(prefix: str, kind: str, title: str, text: str, url: str = "", source_type: str = "", confidence: str = "") -> None:
        n = sum(1 for e in ev if e["id"].startswith(prefix)) + 1
        ev.append({"id": f"{prefix}{n}", "kind": kind, "title": title, "text": text, "url": url, "source_type": source_type, "confidence": confidence})

    for p in programs:
        rules = "; ".join(f"{r['text']} ({r.get('scope','')}, confidence {r.get('confidence','?')})" for r in p.get("rules", [])) or "none recorded"
        text = (f"Program: {p['name']} | field: {p['field']} | streams: {', '.join(p.get('streams', [])) or p.get('streams_note', 'varies')} | "
                f"entry tests: {', '.join(p.get('entry_tests', [])) or 'varies'} | rules: {rules} | notes: {p.get('notes') or '-'}")
        add("K", "knowledge", f"Program: {p['name']}", text, (p.get("sources") or [{}])[0].get("url", ""), p.get("source_type", ""), p.get("confidence", ""))
    for s in scholarships:
        text = (f"Scholarship: {s['name']} | provider: {s['provider']} | level: {s['level']} | requirements: " + "; ".join(s.get("requirements", [])))
        add("K", "knowledge", f"Scholarship: {s['name']}", text, s.get("apply_url", ""), s.get("source_type", ""), s.get("confidence", ""))
    for r in regulators:
        add("K", "knowledge", f"Regulator: {r['name']}", f"{r['name']} covers {', '.join(r['covers'])}. What to verify: {r['check']}", r.get("url", ""), r.get("source_type", ""), r.get("confidence", ""))
    for c in careers:
        add("K", "knowledge", f"Career: {c['title']}", f"Career path: {c['title']}. Typical steps: " + " -> ".join(c["steps"]), "", c.get("source_type", ""), c.get("confidence", ""))
    for ch in (doc_chunks or [])[:5]:
        add("D", "document", f"{ch.get('filename', 'Document')} p.{ch.get('page') or 'N/A'}", str(ch.get("text", ""))[:700], "", "student-document", "")
    for w in (web_results or [])[:6]:
        add("W", "web", w.get("title", "Web result"), str(w.get("snippet", ""))[:400], w.get("url", ""), "trusted-web" if w.get("trusted") == "True" else "web", "")
    return ev


def evidence_text(evidence: list[dict[str, Any]], max_chars: int = 9000) -> str:
    lines = []
    for e in evidence:
        tag = TRUST_NOTE.get(e["source_type"], e["source_type"] or "n/a")
        lines.append(f"[{e['id']}] ({e['kind']}; {tag}) {e['title']}: {e['text']}" + (f" <{e['url']}>" if e["url"] else ""))
    return "\n".join(lines)[:max_chars]


# ----------------------------------------------------------------------------- 3. prompt
def build_prompt(profile: dict[str, Any], evidence: list[dict[str, Any]], learning_signals: str = "", memories: str = "") -> str:
    return f"""You are Path Finder, an education and career guidance agent for Pakistani students.

STRICT GROUNDING RULES
- Use ONLY the EVIDENCE below. Do not use outside knowledge about universities, fees, merit, seats, deadlines or scholarships.
- If the student needs something that is not in EVIDENCE, add it to "missing_information" instead of guessing.
- Every career path, roadmap action and web finding MUST list evidence_ids that exist in EVIDENCE.
- Never invent numbers, dates, university names or links. Do not copy the student's marks into claims about eligibility: eligibility is checked separately.
- Evidence tagged "general guidance - verify" must be worded as guidance to verify with the official body.
- Student's learning signals and memories only help personalise the explanation; they are NOT evidence of eligibility or admission.

STUDENT PROFILE
{json.dumps(profile, ensure_ascii=False)}

LEARNING SIGNALS (soft, from Prep AI): {learning_signals or 'none'}
MEMORIES: {memories[:1200] or 'none'}

EVIDENCE
{evidence_text(evidence)}

Return ONE JSON object with exactly these keys:
{{"summary": "2-3 sentences about the student's situation, no new facts",
 "career_paths": [{{"title": "...", "why_it_fits": "...", "steps": ["..."], "evidence_ids": ["K1"]}}],
 "roadmap": [{{"phase": "Next 30 days | 3-6 months | 6-12 months", "actions": [{{"action": "...", "evidence_ids": ["K1"]}}]}}],
 "web_findings": [{{"finding": "...", "evidence_ids": ["W1"]}}],
 "missing_information": ["what the student must check or provide"],
 "follow_up_questions": ["up to 3 questions that would sharpen the advice"]}}"""


# ----------------------------------------------------------------------------- 4. validation
def _valid_ids(ids: Any, valid: set[str]) -> list[str]:
    if not isinstance(ids, list):
        return []
    return [str(i) for i in ids if str(i) in valid]


def validate_llm_output(data: Any, evidence: list[dict[str, Any]]) -> tuple[dict[str, Any], int]:
    """Keep only items that cite at least one real evidence id. Returns (clean, dropped_count)."""
    valid = {e["id"] for e in evidence}
    dropped = 0
    if not isinstance(data, dict):
        return {"summary": "", "career_paths": [], "roadmap": [], "web_findings": [], "missing_information": [], "follow_up_questions": []}, 0

    clean: dict[str, Any] = {"summary": str(data.get("summary", ""))[:700]}

    paths = []
    for item in data.get("career_paths", []) or []:
        ids = _valid_ids(item.get("evidence_ids"), valid) if isinstance(item, dict) else []
        if ids and item.get("title"):
            paths.append({"title": str(item["title"]), "why_it_fits": str(item.get("why_it_fits", "")),
                          "steps": [str(s) for s in (item.get("steps") or [])][:6], "evidence_ids": ids})
        else:
            dropped += 1
    clean["career_paths"] = paths

    roadmap = []
    for phase in data.get("roadmap", []) or []:
        if not isinstance(phase, dict):
            dropped += 1
            continue
        actions = []
        for a in phase.get("actions", []) or []:
            ids = _valid_ids(a.get("evidence_ids"), valid) if isinstance(a, dict) else []
            if ids and a.get("action"):
                actions.append({"action": str(a["action"]), "evidence_ids": ids})
            else:
                dropped += 1
        if actions:
            roadmap.append({"phase": str(phase.get("phase", "")), "actions": actions})
    clean["roadmap"] = roadmap

    findings = []
    for f in data.get("web_findings", []) or []:
        ids = _valid_ids(f.get("evidence_ids"), valid) if isinstance(f, dict) else []
        if ids and f.get("finding"):
            findings.append({"finding": str(f["finding"]), "evidence_ids": ids})
        else:
            dropped += 1
    clean["web_findings"] = findings

    clean["missing_information"] = [str(x) for x in (data.get("missing_information") or [])][:10]
    clean["follow_up_questions"] = [str(x) for x in (data.get("follow_up_questions") or [])][:3]
    return clean, dropped


def build_qa_prompt(question: str, profile: dict[str, Any], evidence: list[dict[str, Any]]) -> str:
    return f"""You are Path Finder. Answer the student's question using ONLY the EVIDENCE.
If the evidence does not contain the answer, set "not_found" to true and say what to check and where. Never guess numbers, dates, fees or merit.
Return JSON: {{"answer": "...", "evidence_ids": ["K1"], "not_found": false}}

STUDENT PROFILE: {json.dumps(profile, ensure_ascii=False)}
QUESTION: {question}

EVIDENCE
{evidence_text(evidence, 8000)}"""


def validate_qa(data: Any, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    valid = {e["id"] for e in evidence}
    if not isinstance(data, dict):
        return {"answer": "I could not produce a grounded answer. Please check the official source.", "evidence_ids": [], "not_found": True}
    ids = _valid_ids(data.get("evidence_ids"), valid)
    not_found = bool(data.get("not_found"))
    answer = str(data.get("answer", "")).strip()
    if not ids and not not_found:
        return {"answer": "I could not ground that answer in the available sources, so I will not guess. Please check the official university/regulator website.", "evidence_ids": [], "not_found": True}
    return {"answer": answer, "evidence_ids": ids, "not_found": not_found}

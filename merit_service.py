"""Merit aggregate engine for Prep AI (no Streamlit, no LLM: pure and testable).

Design rule: formulas and past closing merits are DATA (knowledge/ folder), each with a
source and a confidence level. Nothing here guesses a number that is not in that data.
"""
from __future__ import annotations

import csv
import io
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
KNOWLEDGE_DIR = ROOT / "knowledge"
FORMULAS_PATH = KNOWLEDGE_DIR / "merit_formulas.json"
SEED_CLOSING_CSV = KNOWLEDGE_DIR / "closing_merit.csv"
USER_CLOSING_CSV = ROOT / "data" / "closing_merit_user.csv"

CLOSING_FIELDS = [
    "year", "exam", "program", "institution", "city", "province", "quota", "list_type",
    "closing_merit", "source_title", "source_url", "source_type", "note",
]
SOURCE_RANK = {"official": 3, "news": 2, "aggregator": 1, "secondary": 1, "user": 2}
BAND_ORDER = {"Safe": 0, "Target": 1, "Reach": 2, "Unlikely": 3}


# ----------------------------------------------------------------------------- formulas
def load_formulas() -> list[dict[str, Any]]:
    data = json.loads(FORMULAS_PATH.read_text(encoding="utf-8"))
    formulas = data.get("formulas", [])
    for f in formulas:
        total = sum(float(c["weight"]) for c in f["components"])
        if abs(total - 100) > 1e-6:
            raise ValueError(f"Formula {f.get('id')} weights add up to {total}, not 100.")
    return formulas


def get_formula(formula_id: str) -> dict[str, Any]:
    for f in load_formulas():
        if f["id"] == formula_id:
            return f
    raise KeyError(f"Unknown formula: {formula_id}")


def custom_formula(matric: float, fsc: float, test: float, test_total: float = 100, name: str = "Custom formula") -> dict[str, Any]:
    """Build a formula from weights the student copied from a prospectus."""
    weights = {"matric": float(matric), "fsc": float(fsc), "test": float(test)}
    if any(w < 0 for w in weights.values()):
        raise ValueError("Weights cannot be negative.")
    if abs(sum(weights.values()) - 100) > 1e-6:
        raise ValueError(f"Weights must add up to 100 (currently {sum(weights.values()):g}).")
    return {
        "id": "custom", "exam": "Custom", "name": name, "authority": "Student-entered (from prospectus)",
        "applies_to": "Any", "program_family": "General",
        "components": [
            {"key": "matric", "label": "Matric / SSC", "weight": weights["matric"], "default_total": 1100},
            {"key": "fsc", "label": "FSc / HSSC", "weight": weights["fsc"], "default_total": 1100},
            {"key": "test", "label": "Entry test", "weight": weights["test"], "default_total": float(test_total)},
        ],
        "eligibility": [], "status": "student-entered", "confidence": "user-provided",
        "sources": [], "verify_note": "Weights were typed in by the student.",
    }


# ----------------------------------------------------------------------------- calculation
def percent(obtained: float, total: float, label: str = "marks") -> float:
    if total is None or float(total) <= 0:
        raise ValueError(f"{label}: total marks must be greater than 0.")
    if obtained is None or float(obtained) < 0:
        raise ValueError(f"{label}: obtained marks cannot be negative.")
    if float(obtained) > float(total):
        raise ValueError(f"{label}: obtained marks ({obtained:g}) cannot exceed total marks ({total:g}).")
    return 100.0 * float(obtained) / float(total)


def calculate_aggregate(formula: dict[str, Any], marks: dict[str, tuple[float, float]]) -> dict[str, Any]:
    """marks = {'matric': (obtained, total), 'fsc': (...), 'test': (...)}"""
    parts = []
    aggregate = 0.0
    for c in formula["components"]:
        weight = float(c["weight"])
        if weight <= 0:
            continue
        if c["key"] not in marks:
            raise ValueError(f"Missing marks for {c['label']}.")
        obtained, total = marks[c["key"]]
        p = percent(obtained, total, c["label"])
        contribution = p * weight / 100.0
        aggregate += contribution
        parts.append({
            "key": c["key"], "label": c["label"], "obtained": float(obtained), "total": float(total),
            "percent": round(p, 4), "weight": weight, "contribution": round(contribution, 4),
        })
    return {"aggregate": round(aggregate, 4), "components": parts, "formula_id": formula["id"], "formula_name": formula["name"]}


def required_test_score(formula: dict[str, Any], marks: dict[str, tuple[float, float]], target: float, test_total: float) -> dict[str, Any]:
    """What test marks are needed to reach a target aggregate, given Matric and FSc."""
    fixed = 0.0
    test_weight = 0.0
    for c in formula["components"]:
        if c["key"] == "test":
            test_weight = float(c["weight"])
            continue
        if float(c["weight"]) <= 0:
            continue
        o, t = marks[c["key"]]
        fixed += percent(o, t, c["label"]) * float(c["weight"]) / 100.0
    if test_weight <= 0:
        raise ValueError("This formula has no entry-test component.")
    needed_pct = (float(target) - fixed) * 100.0 / test_weight
    return {
        "target": float(target), "fixed_part": round(fixed, 4), "needed_percent": round(needed_pct, 2),
        "needed_marks": round(max(0.0, needed_pct) / 100.0 * float(test_total), 1), "test_total": float(test_total),
        "feasible": needed_pct <= 100.0, "already_met": needed_pct <= 0.0,
    }


def eligibility_checks(formula: dict[str, Any], marks: dict[str, tuple[float, float]], program: str | None = None) -> list[dict[str, Any]]:
    """Check only the numeric rules that are written (with a source) in the formula data."""
    out = []
    for rule in formula.get("eligibility", []):
        if rule.get("program") and program and rule["program"].lower() != program.lower():
            continue
        if rule.get("type") != "min_percent":
            continue
        key = rule["component"]
        if key not in marks:
            status, detail = "unknown", "marks not provided"
        else:
            p = percent(*marks[key], label=key)
            ok = p >= float(rule["value"])
            status = "pass" if ok else "fail"
            detail = f"you have {p:.2f}%, rule needs {float(rule['value']):g}%"
        out.append({"check": rule["text"], "status": status, "detail": detail, "confidence": rule.get("confidence", "unknown")})
    return out


# ----------------------------------------------------------------------------- closing-merit data
def _clean_row(raw: dict[str, Any]) -> dict[str, Any]:
    row = {k: (str(raw.get(k, "") or "").strip()) for k in CLOSING_FIELDS}
    row["year"] = int(float(row["year"]))
    row["closing_merit"] = float(row["closing_merit"])
    if not (0 <= row["closing_merit"] <= 100):
        raise ValueError("closing_merit must be between 0 and 100")
    row["quota"] = row["quota"] or "Open Merit"
    row["list_type"] = (row["list_type"] or "final").lower()
    if row["list_type"] not in {"first", "final", "unknown"}:
        row["list_type"] = "unknown"
    row["source_type"] = (row["source_type"] or "user").lower()
    if not row["institution"] or not row["exam"]:
        raise ValueError("institution and exam are required")
    return row


def parse_closing_csv(data: bytes | str) -> tuple[list[dict[str, Any]], list[str]]:
    text = data.decode("utf-8-sig") if isinstance(data, bytes) else data
    reader = csv.DictReader(io.StringIO(text))
    missing = {"year", "exam", "institution", "closing_merit"} - set(reader.fieldnames or [])
    if missing:
        return [], [f"Missing required column(s): {', '.join(sorted(missing))}"]
    rows, errors = [], []
    for i, raw in enumerate(reader, start=2):
        try:
            rows.append(_clean_row(raw))
        except Exception as exc:  # row-level problems are reported, not fatal
            errors.append(f"Row {i}: {exc}")
    return rows, errors


def template_csv() -> str:
    sample = {
        "year": 2025, "exam": "ECAT", "program": "Electrical Engineering", "institution": "Example University",
        "city": "Lahore", "province": "Punjab", "quota": "Open Merit", "list_type": "final", "closing_merit": 80.5,
        "source_title": "Official merit list (name it)", "source_url": "https://example.edu.pk/merit.pdf",
        "source_type": "official", "note": "Delete this sample row",
    }
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CLOSING_FIELDS)
    w.writeheader()
    w.writerow(sample)
    return buf.getvalue()


def load_closing_merits(include_user: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    paths = [SEED_CLOSING_CSV] + ([USER_CLOSING_CSV] if include_user else [])
    for path in paths:
        if path.exists():
            rows.extend(parse_closing_csv(path.read_bytes())[0])
    return rows


def save_user_closing_rows(rows: list[dict[str, Any]]) -> int:
    """Append rows to the student's own data file (de-duplicated). Returns rows added."""
    USER_CLOSING_CSV.parent.mkdir(parents=True, exist_ok=True)
    existing = parse_closing_csv(USER_CLOSING_CSV.read_bytes())[0] if USER_CLOSING_CSV.exists() else []
    key = lambda r: (r["year"], r["exam"].lower(), r["program"].lower(), r["institution"].lower(), r["quota"].lower(), r["list_type"])  # noqa: E731
    seen = {key(r) for r in existing}
    added = 0
    for raw in rows:
        r = _clean_row(raw)
        if key(r) in seen:
            continue
        existing.append(r)
        seen.add(key(r))
        added += 1
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=CLOSING_FIELDS)
    w.writeheader()
    for r in existing:
        w.writerow({k: r.get(k, "") for k in CLOSING_FIELDS})
    USER_CLOSING_CSV.write_text(buf.getvalue(), encoding="utf-8")
    return added


def verify_extracted_rows(rows: list[dict[str, Any]], source_text: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Anti-hallucination gate for AI-extracted merits: the number AND the institution name
    must literally appear in the source text, otherwise the row is rejected."""
    text = source_text.lower().replace("per cent", "%").replace("percent", "%")
    accepted, rejected = [], []
    for raw in rows:
        try:
            r = _clean_row(raw)
        except Exception as exc:
            rejected.append(f"{raw.get('institution', '?')}: {exc}")
            continue
        v = r["closing_merit"]
        candidates = {f"{v:g}", f"{v:.1f}", f"{v:.2f}", f"{v:.3f}", f"{v:.4f}"}
        number_found = any(re.search(rf"(?<![\d.]){re.escape(c)}(?!\d)", text) for c in candidates)
        words = [w for w in re.findall(r"[a-z]{4,}", r["institution"].lower()) if w not in {"medical", "college", "university", "institute", "engineering", "technology"}]
        name_found = (not words) or any(w in text for w in words)
        if number_found and name_found:
            r["source_type"] = r["source_type"] if r["source_type"] != "user" else "secondary"
            accepted.append(r)
        else:
            why = "number not found in source" if not number_found else "institution name not found in source"
            rejected.append(f"{r['institution']} {v}: {why}")
    return accepted, rejected


# ----------------------------------------------------------------------------- recommendations
def _norm(s: Any) -> str:
    return " ".join(str(s or "").lower().split())


def recommend(
    aggregate: float,
    rows: list[dict[str, Any]],
    *,
    exam: str,
    program: str | None = None,
    quota: str = "Open Merit",
    province: str | None = None,
    max_years: int = 3,
) -> list[dict[str, Any]]:
    """Compare an aggregate with previous closing merits. Transparent maths, no AI.

    expected cutoff = latest closing merit + average yearly change (clamped to +-3)
    buffer          = max(0.75, half of the spread between the years used)
    Safe   : aggregate >= expected + buffer
    Target : within +-buffer of expected
    Reach  : up to 3 buffers below expected
    Unlikely: further below
    """
    pool = [r for r in rows if _norm(r["exam"]) == _norm(exam) and _norm(r["quota"]) == _norm(quota)]
    if program:
        pool = [r for r in pool if _norm(r["program"]) == _norm(program)]
    if province:
        pool = [r for r in pool if _norm(r["province"]) == _norm(province)]

    groups: dict[tuple, list[dict[str, Any]]] = {}
    for r in pool:
        groups.setdefault((_norm(r["institution"]), _norm(r["program"])), []).append(r)

    results = []
    for items in groups.values():
        # Do not mix first-list and final-list cutoffs in one trend: prefer 'final' when present.
        list_types = {r["list_type"] for r in items}
        chosen = "final" if "final" in list_types else ("first" if "first" in list_types else sorted(list_types)[0])
        series_rows = [r for r in items if r["list_type"] == chosen]
        by_year: dict[int, dict[str, Any]] = {}
        for r in series_rows:  # one value per year: keep the best-quality source
            cur = by_year.get(r["year"])
            if cur is None or SOURCE_RANK.get(r["source_type"], 0) > SOURCE_RANK.get(cur["source_type"], 0):
                by_year[r["year"]] = r
        years = sorted(by_year, reverse=True)[:max_years]
        used = [by_year[y] for y in years]  # newest first
        latest = used[0]
        values = [u["closing_merit"] for u in used]

        trend = 0.0
        if len(used) >= 2:
            oldest = used[-1]
            span = max(1, latest["year"] - oldest["year"])
            trend = (latest["closing_merit"] - oldest["closing_merit"]) / span
            trend = max(-3.0, min(3.0, trend))
        expected = latest["closing_merit"] + trend
        spread = max(values) - min(values)
        buffer = max(0.75, spread / 2.0)
        margin = float(aggregate) - expected

        if margin >= buffer:
            band = "Safe"
        elif margin >= -buffer:
            band = "Target"
        elif margin >= -3 * buffer:
            band = "Reach"
        else:
            band = "Unlikely"

        best_source = max((u["source_type"] for u in used), key=lambda s: SOURCE_RANK.get(s, 0))
        trusted = SOURCE_RANK.get(best_source, 0) >= 2          # official / news / user-supplied
        if not trusted:
            confidence = "low"                                   # unverified websites never rate above low
        elif len(used) >= 2:
            confidence = "medium-high"
        else:
            confidence = "medium"

        notes = []
        if chosen == "first":
            notes.append("Based on FIRST-list cutoffs; final lists usually close lower.")
        if all(u["source_type"] in {"aggregator", "secondary"} for u in used):
            notes.append("Unverified website figures: check the official merit list.")
        if len(used) == 1:
            notes.append("Only one year of data: no trend available.")

        results.append({
            "institution": latest["institution"], "city": latest["city"], "province": latest["province"],
            "program": latest["program"], "quota": latest["quota"], "band": band,
            "expected_cutoff": round(expected, 2), "latest_year": latest["year"], "latest_cutoff": latest["closing_merit"],
            "trend_per_year": round(trend, 2), "buffer": round(buffer, 2), "margin": round(margin, 2),
            "years_used": sorted(years), "list_type": chosen, "confidence": confidence,
            "sources": sorted({u["source_url"] for u in used if u["source_url"]}),
            "source_types": sorted({u["source_type"] for u in used}), "notes": " ".join(notes),
        })

    results.sort(key=lambda r: (BAND_ORDER[r["band"]], -r["expected_cutoff"]))
    return results


def available_programs(rows: list[dict[str, Any]], exam: str) -> list[str]:
    return sorted({r["program"] for r in rows if _norm(r["exam"]) == _norm(exam) and r["program"]})


def merit_summary_for_llm(result: dict[str, Any], recs: list[dict[str, Any]], elig: list[dict[str, Any]]) -> str:
    """Compact, number-only facts for the explanation prompt."""
    lines = [f"Formula: {result['formula_name']}", f"Aggregate: {result['aggregate']:.4f}%"]
    for c in result["components"]:
        lines.append(f"- {c['label']}: {c['percent']:.2f}% x {c['weight']:g}% = {c['contribution']:.4f}")
    for e in elig:
        lines.append(f"Eligibility check: {e['check']} -> {e['status']} ({e['detail']})")
    for r in recs[:12]:
        lines.append(
            f"College: {r['institution']} | band {r['band']} | expected cutoff {r['expected_cutoff']} | "
            f"years {r['years_used']} | list {r['list_type']} | confidence {r['confidence']} | {r['notes']}"
        )
    if not recs:
        lines.append("No closing-merit data is loaded for this exam/program/quota.")
    return "\n".join(lines)

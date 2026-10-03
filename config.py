from __future__ import annotations

from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MEMORY_DIR = DATA_DIR / "memory"
MEMORY_MAX_ITEMS = 500
MEMORY_MAX_BYTES = 5 * 1024 * 1024
FAISS_DIR = ROOT / "faiss_index"
DB_PATH = DATA_DIR / "prep_ai.db"

ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md"}
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Current Groq production models. GPT-OSS 120B is the default for best quality;
# 20B is a faster/lower-cost alternative.
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]
DEFAULT_GROQ_MODEL = GROQ_MODELS[0]
UI_COLORS = {
    "Blue": "#2563EB",
    "Green": "#16A34A",
    "Purple": "#7C3AED",
    "Orange": "#EA580C",
    "Red": "#DC2626",
}

from mastery_model import MASTERY_LABELS  # noqa: E402,F401  (kept here for backwards compatibility)

for path in (DATA_DIR, MEMORY_DIR, FAISS_DIR):
    path.mkdir(parents=True, exist_ok=True)


def get_secret(name: str, default: str | None = None) -> str | None:
    """Read a Streamlit secret without exposing it."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return value


def get_student_id() -> str:
    raw = st.session_state.get("student_id", "student_001")
    safe = "".join(ch for ch in str(raw) if ch.isalnum() or ch in "_-." )
    return safe[:64] or "student_001"

from __future__ import annotations

from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MEMORY_DIR = DATA_DIR / "memory"
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

# --- Long-term memory limits (feature: memory usage meter) -------------------
MEMORY_MAX_ITEMS = 5000         # hard cap of stored memories per student (meter runs 0 to 5000)
MEMORY_WARN_RATIO = 0.80        # show a warning from this fill level

# --- Quiz timer / size limits -------------------------------------------------
QUIZ_MIN_QUESTIONS, QUIZ_MAX_QUESTIONS = 5, 100
QUIZ_MAX_MINUTES = 300          # 0 minutes always means "no time limit"
SHARED_CODE_LENGTH = 6

# --- Theme presets (background, text). "Default" = leave Streamlit's own theme alone.
THEME_PRESETS = {
    "Default": None,
    "Light": ("#FFFFFF", "#111827"),
    "Dark": ("#0E1117", "#F3F4F6"),
    "Sepia": ("#F4ECD8", "#3B2F1E"),
    "Midnight Blue": ("#0B1730", "#E6EDF7"),
    "Custom": None,
}
MIN_CONTRAST_BLOCK = 3.0        # refuse to save unreadable colour pairs
MIN_CONTRAST_WARN = 4.5         # WCAG AA for normal text

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


def get_tutor_code() -> str:
    """Tutor access code from secrets / environment. Empty string = tutor role disabled."""
    import os
    return str(get_secret("TUTOR_ACCESS_CODE") or os.environ.get("TUTOR_ACCESS_CODE") or "").strip()

from __future__ import annotations

import json
from typing import Any

import streamlit as st
from groq import Groq

from config import DEFAULT_GROQ_MODEL, GROQ_MODELS, get_secret


class GroqServiceError(RuntimeError):
    """Safe, user-facing Groq configuration/API error."""


def _clean(value: str | None) -> str:
    return str(value or "").strip()


@st.cache_resource(show_spinner=False)
def get_client(api_key: str) -> Groq | None:
    """Create one cached Groq client for the supplied API key."""
    key = _clean(api_key)
    if not key:
        return None
    return Groq(api_key=key)


def _api_key() -> str:
    return _clean(get_secret("GROQ_API_KEY"))


def current_model() -> str:
    selected = _clean(st.session_state.get("llm_model"))
    # Prevent an old Gemini/deprecated model value from reaching Groq.
    return selected if selected in GROQ_MODELS else DEFAULT_GROQ_MODEL


def _compact(text: str, max_chars: int = 14000) -> str:
    """Keep RAG prompts small enough for Groq free/developer token limits."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[Context truncated for reliability.]"


def generate_text(
    prompt: str,
    model: str | None = None,
    json_output: bool = False,
    max_completion_tokens: int = 3500,
) -> str:
    key = _api_key()
    if not key:
        raise GroqServiceError(
            "GROQ_API_KEY is missing. Add GROQ_API_KEY to Streamlit Cloud → Settings → Secrets."
        )

    selected_model = _clean(model) if model else current_model()
    if selected_model not in GROQ_MODELS:
        selected_model = DEFAULT_GROQ_MODEL

    client = get_client(key)
    if client is None:
        raise GroqServiceError("Groq client could not be initialized. Check GROQ_API_KEY.")

    # GPT-OSS supports reasoning_effort. Low keeps educational requests fast and
    # avoids unnecessary token consumption on Groq's token limits.
    kwargs: dict[str, Any] = {
        "model": selected_model,
        "messages": [{"role": "user", "content": _compact(prompt)}],
        "max_completion_tokens": max_completion_tokens,
        "reasoning_effort": "low",
    }

    if json_output:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = client.chat.completions.create(**kwargs)
    except Exception as exc:
        message = str(exc).lower()
        if "401" in message or "authentication" in message or "invalid api key" in message:
            raise GroqServiceError("Groq API key is invalid or expired.") from exc
        if "403" in message or "permission" in message or "blocked" in message:
            raise GroqServiceError(
                f"The Groq model '{selected_model}' is not permitted for this API project. "
                "Choose another model in Settings."
            ) from exc
        if "429" in message or "rate limit" in message or "too many requests" in message:
            raise GroqServiceError(
                "Groq rate limit reached. Reduce the number of questions or wait a moment and try again."
            ) from exc
        if "413" in message or "too large" in message or "context" in message and "limit" in message:
            raise GroqServiceError(
                "The learning context is too large for the Groq request. Try a more specific topic or fewer questions."
            ) from exc
        raise GroqServiceError(
            f"Groq request failed with model '{selected_model}'. Check the Groq API key, model setting, and account limits."
        ) from exc

    text = response.choices[0].message.content or ""
    if not text.strip():
        raise GroqServiceError("Groq returned an empty response. Please try again.")
    return text.strip()


def generate_json(prompt: str, model: str | None = None) -> Any:
    raw = generate_text(prompt, model=model, json_output=True, max_completion_tokens=5000)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        start = raw.find("[")
        end = raw.rfind("]")
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise GroqServiceError("Groq returned an invalid JSON response. Please try again.")


def grounded_answer(question: str, context: str, memories: str = "") -> str:
    prompt = f"""You are Prep AI, a source-grounded educational assistant.
Answer the student's request using ONLY the provided learning context when the request is about the supplied material.
If the answer is not available in the context, clearly say it is not available in the provided material instead of inventing it.
Use student memories only to personalize explanation style, not as factual evidence.

STUDENT REQUEST:
{question}

LONG-TERM MEMORY:
{_compact(memories, 3500)}

LEARNING CONTEXT:
{_compact(context, 12000)}
"""
    return generate_text(prompt, max_completion_tokens=2500)


def test_groq_connection(model: str | None = None) -> tuple[bool, str]:
    """Small diagnostic request used by Settings."""
    try:
        answer = generate_text(
            "Reply with exactly: Prep AI Groq connection successful.",
            model=model,
            max_completion_tokens=100,
        )
        return True, answer
    except GroqServiceError as exc:
        return False, str(exc)

"""Voice helpers for Prep AI: speech-to-text (Groq Whisper) and text-to-speech (gTTS)."""
from __future__ import annotations

import re
from io import BytesIO

from groq_service import GroqServiceError, get_client, _api_key

STT_MODEL = "whisper-large-v3-turbo"  # fast + cheap; use "whisper-large-v3" for best accuracy
LANG_CODES = {"English": "en", "Urdu": "ur"}


def transcribe(audio_bytes: bytes, language: str = "English") -> str:
    """Turn recorded audio into text using Groq's Whisper model."""
    client = get_client(_api_key())
    if client is None:
        raise GroqServiceError("GROQ_API_KEY is missing. Add it to your Streamlit secrets.")
    try:
        result = client.audio.transcriptions.create(
            file=("question.wav", audio_bytes),
            model=STT_MODEL,
            language=LANG_CODES.get(language, "en"),
            temperature=0.0,
        )
    except Exception as exc:
        raise GroqServiceError("Speech-to-text failed. Check your Groq key and try again.") from exc
    return (getattr(result, "text", "") or "").strip()


def clean_for_speech(text: str, max_chars: int = 1200) -> str:
    """Remove markdown symbols so the voice doesn't read '**' or '#' aloud."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)          # code blocks
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)         # [label](url) -> label
    text = re.sub(r"[*_`#>|~]", "", text)                        # markdown symbols
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)          # bullets
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def speak(text: str, language: str = "English") -> bytes:
    """Turn text into MP3 bytes using gTTS (free, needs internet)."""
    from gtts import gTTS  # imported here so the app still starts if gTTS isn't installed

    spoken = clean_for_speech(text)
    if not spoken:
        return b""
    buffer = BytesIO()
    gTTS(text=spoken, lang=LANG_CODES.get(language, "en")).write_to_fp(buffer)
    return buffer.getvalue()

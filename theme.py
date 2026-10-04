"""Pure helpers for the user-selectable theme (background + text colour).

No Streamlit imports here so everything can be unit-tested.
"""
from __future__ import annotations

import re

from config import MIN_CONTRAST_BLOCK, MIN_CONTRAST_WARN, THEME_PRESETS

_HEX = re.compile(r"^#[0-9a-fA-F]{6}$")


def is_hex(value: str | None) -> bool:
    return bool(value and _HEX.match(value))


def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _hex(rgb: tuple[float, float, float]) -> str:
    return "#%02X%02X%02X" % tuple(max(0, min(255, round(c))) for c in rgb)


def _lin(c: int) -> float:
    c = c / 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def luminance(h: str) -> float:
    r, g, b = _rgb(h)
    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio between two #RRGGBB colours (1.0 .. 21.0)."""
    la, lb = luminance(a), luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return round((hi + 0.05) / (lo + 0.05), 2)


def mix(a: str, b: str, amount: float) -> str:
    """Blend colour `a` towards colour `b` by `amount` (0..1)."""
    ra, rb = _rgb(a), _rgb(b)
    return _hex(tuple(x + (y - x) * amount for x, y in zip(ra, rb)))


def resolve_theme(preset: str | None, bg: str | None, text: str | None) -> tuple[str, str] | None:
    """Return (background, text) to apply, or None for Streamlit's own theme."""
    if preset == "Custom":
        if is_hex(bg) and is_hex(text):
            return bg.upper(), text.upper()
        return None
    pair = THEME_PRESETS.get(preset or "Default")
    return pair if pair else None


def validate_theme(bg: str, text: str) -> tuple[str, str]:
    """Return (level, message). level is 'ok', 'warn' or 'block'."""
    if not (is_hex(bg) and is_hex(text)):
        return "block", "Colours must be in #RRGGBB format."
    ratio = contrast_ratio(bg, text)
    if ratio < MIN_CONTRAST_BLOCK:
        return "block", f"These colours are too similar to read (contrast {ratio}:1, minimum {MIN_CONTRAST_BLOCK}:1). Please pick a lighter background with darker text, or the reverse."
    if ratio < MIN_CONTRAST_WARN:
        return "warn", f"Readable but low contrast ({ratio}:1). {MIN_CONTRAST_WARN}:1 or more is recommended."
    return "ok", f"Good contrast ({ratio}:1)."


def build_css(primary: str, bg: str | None = None, text: str | None = None) -> str:
    """CSS for the app. With bg/text it re-colours the page, sidebar, inputs, buttons and alerts."""
    base = f"""
    :root {{ --prep-primary: {primary}; }}
    .prep-hero {{padding: 1.2rem 1.4rem; border-radius: 18px; background: linear-gradient(135deg,#111827,#1f2937); color:white; margin-bottom:1rem; border-left: 5px solid var(--prep-primary);}}
    .prep-card {{padding:1rem; border:1px solid rgba(128,128,128,.25); border-radius:16px; background:rgba(128,128,128,.05); margin-bottom:.7rem;}}
    .small-muted {{color:#6b7280; font-size:.9rem;}}
    div.stButton > button[kind="primary"] {{background-color: var(--prep-primary); border-color: var(--prep-primary);}}
    """
    if not (is_hex(bg) and is_hex(text)):
        return base
    field = mix(bg, text, 0.08)      # inputs / cards: a touch towards the text colour
    side = mix(bg, text, 0.04)
    border = mix(bg, text, 0.25)
    return base + f"""
    /* ---- user theme: background {bg}, text {text} ---- */
    .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ background-color: {bg} !important; color: {text} !important; }}
    [data-testid="stHeader"] {{ background-color: {bg} !important; }}
    [data-testid="stSidebar"], [data-testid="stSidebar"] > div {{ background-color: {side} !important; }}
    .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stCaptionContainer"],
    .stApp [data-testid="stMetricValue"], .stApp [data-testid="stMetricLabel"], .stApp summary,
    .stApp [data-baseweb="tab"] {{ color: {text} !important; }}
    .stApp .small-muted {{ color: {mix(text, bg, 0.35)} !important; }}
    /* keep the dark hero banner readable whatever the page colours are */
    .stApp .prep-hero, .stApp .prep-hero * {{ color: #ffffff !important; }}
    /* form fields */
    .stApp input, .stApp textarea, .stApp [data-baseweb="select"] > div, .stApp [data-baseweb="input"],
    .stApp [data-baseweb="textarea"], .stApp [data-baseweb="base-input"] {{ background-color: {field} !important; color: {text} !important; border-color: {border} !important; }}
    .stApp [data-baseweb="select"] *, .stApp [data-baseweb="input"] * {{ color: {text} !important; }}
    div[data-baseweb="popover"] ul, div[data-baseweb="popover"] li, div[data-baseweb="menu"] {{ background-color: {field} !important; color: {text} !important; }}
    /* buttons: neutral by default, accent colour for primary */
    .stApp button {{ background-color: {field}; color: {text}; border-color: {border}; }}
    .stApp button p, .stApp button span {{ color: inherit !important; }}
    .stApp button[role="tab"] {{ background-color: transparent !important; border-color: transparent; }}
    .stApp button[kind="primary"], .stApp button[data-testid="stBaseButton-primary"], .stApp button[data-testid="stBaseButton-primaryFormSubmit"] {{ background-color: {primary} !important; border-color: {primary} !important; color: #ffffff !important; }}
    /* alerts / expanders: one neutral surface so text is always readable */
    [data-testid="stAlert"] {{ background-color: {field} !important; border-left: 4px solid {primary}; }}
    [data-testid="stExpander"] details {{ background-color: {field} !important; border-color: {border} !important; }}
    """

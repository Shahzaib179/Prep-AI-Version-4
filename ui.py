from __future__ import annotations

import streamlit as st

from theme import build_css


def apply_theme(primary_color: str = "#2563EB", bg: str | None = None, text: str | None = None) -> None:
    """Inject the app CSS. bg/text (hex) are the user's optional page and text colours."""
    st.markdown(f"<style>{build_css(primary_color, bg, text)}</style>", unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="prep-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def source_cards(sources: list[dict]) -> None:
    if not sources:
        return
    st.subheader("Retrieved Sources")
    for i, s in enumerate(sources, 1):
        with st.expander(f"Source {i}: {s.get('filename', 'Source')} — Page {s.get('page') or 'N/A'}"):
            st.caption(f"Chunk {s.get('chunk_id', 'N/A')} · hybrid score {float(s.get('hybrid_score', 0)):.3f}")
            st.write(s.get("text", ""))

"""Chart helpers (Altair - already installed with Streamlit, no new dependency).

Every function returns an Altair chart; call ``show(chart)`` to render it.
Colours follow the student's chosen theme (read from the session, never from a
shared global, so one user's theme can never leak into another user's session).
"""
from __future__ import annotations

from typing import Any, Iterable

import altair as alt
import pandas as pd
import streamlit as st

from config import UI_COLORS
from theme import mix

BAND_COLORS = {"Weak": "#DC2626", "Developing": "#F59E0B", "Strong": "#16A34A", "Mastered": "#0EA5E9"}
OUTCOME_COLORS = {"Correct": "#16A34A", "Incorrect": "#DC2626", "Skipped": "#9CA3AF"}
_NEUTRAL_TEXT = "#7A8494"   # readable on both light and dark Streamlit themes


def _palette() -> dict[str, str]:
    pair = st.session_state.get("theme_pair")
    if pair:
        bg, text = pair
        return {"text": text, "track": mix(bg, text, 0.15), "grid": mix(bg, text, 0.18)}
    return {"text": _NEUTRAL_TEXT, "track": "#7A849433", "grid": "#7A849433"}


def _accent() -> str:
    return UI_COLORS.get(st.session_state.get("ui_color", "Blue"), "#2563EB")


def _style(chart: Any) -> Any:
    p = _palette()
    return (
        chart.configure_view(stroke=None)
        .configure_axis(labelColor=p["text"], titleColor=p["text"], gridColor=p["grid"], domainColor=p["grid"], tickColor=p["grid"])
        .configure_legend(labelColor=p["text"], titleColor=p["text"], orient="bottom", symbolType="circle")
        .configure_title(color=p["text"], fontSize=14, anchor="middle")
        .properties(background="transparent")
    )


def show(chart: Any) -> None:
    """Render a chart across Streamlit versions (use_container_width -> width='stretch')."""
    try:
        st.altair_chart(chart, theme=None, width="stretch")
    except TypeError:
        st.altair_chart(chart, theme=None, use_container_width=True)


def ring(percent: float, title: str = "", color: str | None = None, size: int = 170, center_text: str | None = None) -> Any:
    """Circular progress gauge (donut) with the value in the middle."""
    p = _palette()
    pct = max(0.0, min(100.0, float(percent or 0)))
    df = pd.DataFrame({"part": ["done", "rest"], "v": [pct, 100 - pct], "order": [0, 1]})
    arc = alt.Chart(df).mark_arc(innerRadius=size * 0.32, outerRadius=size * 0.46, stroke=None).encode(
        theta=alt.Theta("v:Q", stack=True),
        order=alt.Order("order:Q"),
        color=alt.Color("part:N", legend=None, scale=alt.Scale(domain=["done", "rest"], range=[color or _accent(), p["track"]])),
        tooltip=alt.value(None),
    )
    label = center_text if center_text is not None else f"{pct:.0f}%"
    txt = alt.Chart(pd.DataFrame({"t": [label]})).mark_text(fontSize=size * 0.16, fontWeight="bold", color=p["text"]).encode(text="t:N")
    return _style(alt.layer(arc, txt).properties(width=size, height=size, title=title))


def donut(parts: dict[str, float], colors: dict[str, str] | None = None, title: str = "", size: int = 190, center_text: str | None = None) -> Any:
    """Pie/donut of named parts, e.g. {'Correct': 7, 'Incorrect': 2, 'Skipped': 1}. Zero parts are hidden."""
    p = _palette()
    items = [(k, float(v)) for k, v in parts.items() if v and float(v) > 0]
    if not items:
        items = [("No data", 1.0)]
        colors = {"No data": p["track"]}
    df = pd.DataFrame({"label": [k for k, _ in items], "value": [v for _, v in items]})
    df["share"] = (df["value"] / df["value"].sum() * 100).round(1)
    domain = list(df["label"])
    rng = [(colors or {}).get(k, c) for k, c in zip(domain, ["#2563EB", "#F59E0B", "#9CA3AF", "#7C3AED", "#0EA5E9"] * 3)]
    arc = alt.Chart(df).mark_arc(innerRadius=size * 0.28, outerRadius=size * 0.46, stroke=None).encode(
        theta=alt.Theta("value:Q", stack=True),
        color=alt.Color("label:N", scale=alt.Scale(domain=domain, range=rng), legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("value:Q", title="Count"), alt.Tooltip("share:Q", title="Share %")],
    )
    layers = [arc]
    if center_text:
        layers.append(alt.Chart(pd.DataFrame({"t": [center_text]})).mark_text(fontSize=size * 0.14, fontWeight="bold", color=p["text"]).encode(text="t:N"))
    return _style(alt.layer(*layers).properties(width=size, height=size, title=title))


def mastery_bars(rows: Iterable[dict[str, Any]], title: str = "Topic mastery") -> Any:
    """Horizontal bars, one per topic, coloured by band, with 60% / 75% guide lines."""
    p = _palette()
    data = []
    for r in rows:
        score = float(r.get("mastery_score", 0) or 0)
        band = "Mastered" if score >= 90 else "Strong" if score >= 75 else "Developing" if score >= 60 else "Weak"
        data.append({"topic": f"{r.get('subject', '')} → {r.get('topic', '')}", "score": round(score, 1), "band": band,
                     "answered": int(r.get("attempts", 0) or 0), "enough": bool(r.get("enough_data", True))})
    df = pd.DataFrame(data, columns=["topic", "score", "band", "answered", "enough"])
    height = max(120, 34 * len(df) + 50)
    base = alt.Chart(df).encode(y=alt.Y("topic:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)))
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        x=alt.X("score:Q", scale=alt.Scale(domain=[0, 100]), title="Mastery %"),
        color=alt.Color("band:N", scale=alt.Scale(domain=list(BAND_COLORS), range=list(BAND_COLORS.values())), legend=alt.Legend(title=None)),
        opacity=alt.condition("datum.enough", alt.value(1.0), alt.value(0.45)),
        tooltip=[alt.Tooltip("topic:N", title="Topic"), alt.Tooltip("score:Q", title="Mastery %"), alt.Tooltip("band:N", title="Level"), alt.Tooltip("answered:Q", title="Answered")],
    )
    labels = base.mark_text(align="left", dx=4, color=p["text"], fontSize=11).encode(x="score:Q", text=alt.Text("score:Q", format=".0f"))
    guides = alt.Chart(pd.DataFrame({"x": [60, 75]})).mark_rule(strokeDash=[4, 4], color=p["grid"]).encode(x="x:Q")
    return _style(alt.layer(bars, labels, guides).properties(height=height, title=title))


def score_trend(attempts: list[dict[str, Any]], title: str = "Quiz score over time") -> Any:
    """Line + points: score of each quiz in the order it was taken."""
    p = _palette()
    df = pd.DataFrame([{"n": i + 1, "score": float(a.get("score", 0) or 0), "topic": a.get("topic", ""), "when": str(a.get("created_at", ""))[:16].replace("T", " ")} for i, a in enumerate(attempts)],
                      columns=["n", "score", "topic", "when"])
    base = alt.Chart(df).encode(x=alt.X("n:Q", title="Quiz number", axis=alt.Axis(tickMinStep=1, format="d")), y=alt.Y("score:Q", scale=alt.Scale(domain=[0, 100]), title="Score %"))
    line = base.mark_line(color=_accent(), strokeWidth=2.5)
    pts = base.mark_point(color=_accent(), filled=True, size=70).encode(tooltip=[alt.Tooltip("n:Q", title="Quiz #"), alt.Tooltip("score:Q", title="Score %", format=".0f"), alt.Tooltip("topic:N", title="Topic"), alt.Tooltip("when:N", title="When")])
    goal = alt.Chart(pd.DataFrame({"y": [75]})).mark_rule(strokeDash=[4, 4], color=p["grid"]).encode(y="y:Q")
    return _style(alt.layer(line, pts, goal).properties(height=240, title=title))


def simple_bars(labels: list[str], values: list[float], title: str = "", value_title: str = "Value", domain: tuple[float, float] | None = (0, 100), color: str | None = None) -> Any:
    df = pd.DataFrame({"label": labels, "value": values})
    y = alt.Y("value:Q", title=value_title, scale=alt.Scale(domain=list(domain)) if domain else alt.Undefined)
    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, color=color or _accent()).encode(
        x=alt.X("label:N", sort=None, title=None, axis=alt.Axis(labelAngle=-30, labelLimit=140)), y=y,
        tooltip=[alt.Tooltip("label:N", title=""), alt.Tooltip("value:Q", title=value_title)])
    return _style(chart.properties(height=240, title=title))

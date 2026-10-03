from __future__ import annotations

from io import BytesIO
from xml.sax.saxutils import escape

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak


def _pdf_document() -> tuple[BytesIO, SimpleDocTemplate, object]:
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
        title="Prep AI",
        author="Prep AI",
    )
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="CenterTitle", parent=styles["Title"], alignment=TA_CENTER))
    return buffer, doc, styles


def _safe(text: object) -> str:
    """Escape text so &, < and > in AI output cannot break ReportLab paragraphs."""
    return escape(str(text or "")).replace("\n", "<br/>")


def questions_to_pdf(title: str, questions: list[dict]) -> bytes:
    """Create a readable PDF containing generated MCQs and their answers/explanations."""
    buffer, doc, styles = _pdf_document()
    story = [Paragraph(_safe(title), styles["CenterTitle"]), Spacer(1, 16)]

    for i, q in enumerate(questions, start=1):
        story.append(Paragraph(f"Q{i}. {_safe(q.get('question', ''))}", styles["Heading3"]))
        options = q.get("options") or {}
        for key in ("A", "B", "C", "D"):
            if key in options:
                story.append(Paragraph(f"<b>{key}.</b> {_safe(options[key])}", styles["BodyText"]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Correct Answer:</b> {_safe(q.get('answer', 'Not provided'))}", styles["BodyText"]))
        if q.get("explanation"):
            story.append(Paragraph(f"<b>Explanation:</b> {_safe(q['explanation'])}", styles["BodyText"]))
        if q.get("concept"):
            story.append(Paragraph(f"<b>Concept:</b> {_safe(q['concept'])}", styles["Small"]))
        story.append(Spacer(1, 12))

    doc.build(story)
    return buffer.getvalue()


def text_to_pdf(title: str, text: str, subtitle: str | None = None) -> bytes:
    """Create a PDF from an explanation, tutor answer, or other text response."""
    buffer, doc, styles = _pdf_document()
    story = [Paragraph(_safe(title), styles["CenterTitle"]), Spacer(1, 12)]
    if subtitle:
        story.append(Paragraph(_safe(subtitle), styles["Small"]))
        story.append(Spacer(1, 10))

    # Preserve paragraphs while allowing ReportLab to wrap long AI responses.
    paragraphs = str(text or "").split("\n")
    for paragraph in paragraphs:
        if paragraph.strip():
            story.append(Paragraph(_safe(paragraph.strip()), styles["BodyText"]))
            story.append(Spacer(1, 7))

    doc.build(story)
    return buffer.getvalue()


def research_to_pdf(title: str, response: str, sources: list[dict] | None = None) -> bytes:
    """Create a research report PDF containing the answer and web sources."""
    buffer, doc, styles = _pdf_document()
    story = [Paragraph(_safe(title), styles["CenterTitle"]), Spacer(1, 16)]

    story.append(Paragraph("Research Response", styles["Heading2"]))
    for paragraph in str(response or "").split("\n"):
        if paragraph.strip():
            story.append(Paragraph(_safe(paragraph.strip()), styles["BodyText"]))
            story.append(Spacer(1, 7))

    if sources:
        story.append(Spacer(1, 10))
        story.append(Paragraph("Web Sources", styles["Heading2"]))
        for i, source in enumerate(sources, start=1):
            title_text = source.get("title", "Source")
            url = source.get("url", "")
            snippet = source.get("snippet", "")
            story.append(Paragraph(f"<b>{i}. {_safe(title_text)}</b>", styles["BodyText"]))
            if url:
                story.append(Paragraph(_safe(url), styles["Small"]))
            if snippet:
                story.append(Paragraph(_safe(snippet), styles["Small"]))
            story.append(Spacer(1, 8))

    doc.build(story)
    return buffer.getvalue()

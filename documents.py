from __future__ import annotations

from io import BytesIO
from pathlib import Path
import hashlib
import os
import re
import shutil
import tempfile
from typing import Any

import gdown
from docx import Document
from pypdf import PdfReader

from config import ALLOWED_EXTENSIONS


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(data: bytes, filename: str, source_path: str = "") -> list[dict[str, Any]]:
    reader = PdfReader(BytesIO(data))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append({"text": text, "filename": filename, "page": number, "source_path": source_path or filename})
    return pages


def extract_docx(data: bytes, filename: str, source_path: str = "") -> list[dict[str, Any]]:
    doc = Document(BytesIO(data))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = clean_text("\n".join(paragraphs))
    return [{"text": text, "filename": filename, "page": None, "source_path": source_path or filename}] if text else []


def extract_txt(data: bytes, filename: str, source_path: str = "") -> list[dict[str, Any]]:
    text = data.decode("utf-8", errors="replace")
    text = clean_text(text)
    return [{"text": text, "filename": filename, "page": None, "source_path": source_path or filename}] if text else []


def extract_md(data: bytes, filename: str, source_path: str = "") -> list[dict[str, Any]]:
    text = data.decode("utf-8", errors="replace")
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"[#*_>`]", " ", text)
    text = clean_text(text)
    return [{"text": text, "filename": filename, "page": None, "source_path": source_path or filename}] if text else []


def extract_document(data: bytes, filename: str, source_path: str = "") -> list[dict[str, Any]]:
    ext = Path(filename).suffix.lower().lstrip(".")
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext or '[missing extension]'}")
    if ext == "pdf":
        return extract_pdf(data, filename, source_path)
    if ext == "docx":
        return extract_docx(data, filename, source_path)
    if ext == "txt":
        return extract_txt(data, filename, source_path)
    return extract_md(data, filename, source_path)


def chunk_pages(pages: list[dict[str, Any]], chunk_size: int = 900, overlap: int = 150) -> list[dict[str, Any]]:
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    chunks: list[dict[str, Any]] = []
    chunk_id = 0
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_size)
            piece = text[start:end].strip()
            if piece:
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": piece,
                    "filename": page["filename"],
                    "page": page.get("page"),
                    "source_path": page.get("source_path", page["filename"]),
                })
                chunk_id += 1
            if end >= len(text):
                break
            start = max(0, end - overlap)
    return chunks


def extract_uploaded_files(uploaded_files: list[Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pages: list[dict[str, Any]] = []
    info: list[dict[str, Any]] = []
    for f in uploaded_files:
        data = f.getvalue()
        filename = os.path.basename(f.name)
        extracted = extract_document(data, filename, filename)
        pages.extend(extracted)
        info.append({"filename": filename, "bytes": len(data), "pages": len(extracted), "hash": file_hash(data)})
    return pages, info


def download_drive(url: str) -> tuple[list[tuple[str, bytes]], str]:
    """Download a public Drive file/folder using gdown, then scan supported files."""
    url = url.strip()
    if not url:
        return [], ""
    temp_dir = tempfile.mkdtemp(prefix="prep_ai_drive_")
    try:
        output = Path(temp_dir) / "drive_download"
        try:
            result = gdown.download_folder(url, output=str(output), quiet=True, use_cookies=False)
            paths = [Path(p) for p in result] if result else []
        except Exception:
            result_path = gdown.download(url=url, output=None, quiet=True, use_cookies=False)
            if result_path:
                downloaded = Path(result_path)
                target = Path(temp_dir) / downloaded.name
                if downloaded.resolve() != target.resolve():
                    shutil.move(str(downloaded), str(target))
                paths = [target]
            else:
                paths = []
        files: list[tuple[str, bytes]] = []
        for path in paths:
            if path.is_file() and path.suffix.lower().lstrip(".") in ALLOWED_EXTENSIONS:
                files.append((path.name, path.read_bytes()))
        if not files:
            for path in Path(temp_dir).rglob("*"):
                if path.is_file() and path.suffix.lower().lstrip(".") in ALLOWED_EXTENSIONS:
                    files.append((path.name, path.read_bytes()))
        return files, temp_dir
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

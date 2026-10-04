from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL, FAISS_DIR


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_texts(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 384), dtype="float32")
    model = get_embedding_model()
    return model.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).astype("float32")


def build_index(chunks: list[dict[str, Any]]) -> tuple[faiss.IndexFlatIP, np.ndarray]:
    vectors = embed_texts([c["text"] for c in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    if len(vectors):
        index.add(vectors)
    return index, vectors


def important_words(text: str) -> set[str]:
    stop = {"the", "and", "for", "with", "from", "this", "that", "what", "where", "when", "which", "into", "about", "how", "why", "are", "is", "of", "to", "in", "a", "an", "on"}
    words = re.findall(r"[a-zA-Z0-9][a-zA-Z0-9_-]{2,}", text.lower())
    return {w for w in words if w not in stop}


def hybrid_search(query: str, chunks: list[dict[str, Any]], index: faiss.Index, top_k: int = 6) -> list[dict[str, Any]]:
    if not chunks or index.ntotal == 0:
        return []
    q = embed_texts([query])
    distances, indices = index.search(q, min(max(top_k * 3, top_k), index.ntotal))
    q_words = important_words(query)
    candidates: dict[int, dict[str, Any]] = {}
    for rank, idx in enumerate(indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        c = dict(chunks[int(idx)])
        semantic = float(distances[0][rank])
        c_words = important_words(c["text"])
        keyword = len(q_words & c_words) / max(1, len(q_words))
        c["semantic_score"] = semantic
        c["keyword_score"] = keyword
        c["hybrid_score"] = 0.75 * semantic + 0.25 * keyword
        candidates[int(idx)] = c
    return sorted(candidates.values(), key=lambda x: x["hybrid_score"], reverse=True)[:top_k]


def load_database_index() -> tuple[faiss.Index | None, list[dict[str, Any]]]:
    index_path = FAISS_DIR / "database.faiss"
    meta_path = FAISS_DIR / "metadata.json"
    if not index_path.exists() or not meta_path.exists():
        return None, []
    index = faiss.read_index(str(index_path))
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if isinstance(metadata, dict):
        metadata = metadata.get("chunks", metadata.get("metadata", []))
    return index, metadata


def filter_database_chunks(metadata: list[dict[str, Any]], subject: str, topic: str = "") -> list[dict[str, Any]]:
    subject_l = subject.lower().strip()
    topic_l = topic.lower().strip()
    result = []
    for item in metadata:
        item_subject = str(item.get("subject", "")).lower()
        item_topic = str(item.get("topic", "")).lower()
        if subject_l and item_subject and item_subject != subject_l:
            continue
        if topic_l and topic_l not in (item_topic + " " + str(item.get("text", ""))).lower():
            continue
        result.append(item)
    return result if result else metadata

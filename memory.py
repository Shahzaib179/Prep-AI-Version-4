from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from config import MEMORY_DIR, MEMORY_MAX_ITEMS, MEMORY_WARN_RATIO
from db import connect, delete_agent_sessions
from rag import embed_texts, get_embedding_model


class LongTermMemory:
    """Structured SQLite memory + semantic FAISS memory per student."""

    def __init__(self, student_id: str):
        self.student_id = student_id
        self.root = MEMORY_DIR / student_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "memory.faiss"
        self.meta_path = self.root / "metadata.json"
        self._load()

    def _load(self) -> None:
        if self.index_path.exists() and self.meta_path.exists():
            try:
                index = faiss.read_index(str(self.index_path))
                metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
                if isinstance(metadata, list) and index.ntotal == len(metadata):
                    self.index = index
                    self.metadata = metadata
                    return
            except Exception:
                pass
        self.index = None
        self.metadata: list[dict[str, Any]] = []

    def _save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, content: str, memory_type: str = "learning", subject: str = "", topic: str = "", importance: float = 0.7, confidence: float = 0.8) -> int | None:
        content = content.strip()
        if len(content) < 12:
            return None
        # Avoid exact duplicates.
        if any(m.get("content", "").strip().lower() == content.lower() for m in self.metadata):
            return None
        # Keep the store inside its limit: make room by dropping the least valuable memories.
        self._make_room(1)
        now = datetime.utcnow().isoformat()
        with connect() as con:
            cur = con.execute("INSERT INTO memories(student_id,memory_type,content,subject,topic,importance,confidence,created_at,updated_at,last_used_at) VALUES(?,?,?,?,?,?,?,?,?,?)", (self.student_id, memory_type, content, subject, topic, importance, confidence, now, now, now))
            memory_id = int(cur.lastrowid)
        vector = embed_texts([content])
        if self.index is None:
            self.index = faiss.IndexFlatIP(vector.shape[1])
        self.index.add(vector)
        self.metadata.append({"memory_id": memory_id, "content": content, "memory_type": memory_type, "subject": subject, "topic": topic, "importance": importance, "confidence": confidence, "created_at": now})
        self._save()
        return memory_id

    # ------------------------------------------------------------------ usage / limits
    def count(self) -> int:
        with connect() as con:
            return int(con.execute("SELECT COUNT(*) FROM memories WHERE student_id=? AND is_active=1", (self.student_id,)).fetchone()[0])

    def usage(self) -> dict[str, Any]:
        """How much of this student's memory allowance is in use."""
        with connect() as con:
            used = int(con.execute("SELECT COUNT(*) FROM memories WHERE student_id=? AND is_active=1", (self.student_id,)).fetchone()[0])
            text_bytes = int(con.execute("SELECT COALESCE(SUM(LENGTH(CAST(content AS BLOB))),0) FROM memories WHERE student_id=? AND is_active=1", (self.student_id,)).fetchone()[0])
            by_type = {r[0] or "other": int(r[1]) for r in con.execute("SELECT memory_type, COUNT(*) FROM memories WHERE student_id=? AND is_active=1 GROUP BY memory_type ORDER BY 2 DESC", (self.student_id,)).fetchall()}
        disk = sum(p.stat().st_size for p in (self.index_path, self.meta_path) if p.exists())
        ratio = min(1.0, used / MEMORY_MAX_ITEMS) if MEMORY_MAX_ITEMS else 0.0
        return {
            "used": used, "limit": MEMORY_MAX_ITEMS, "free": max(0, MEMORY_MAX_ITEMS - used),
            "percent": round(ratio * 100, 1), "text_bytes": text_bytes, "disk_bytes": disk,
            "by_type": by_type, "near_limit": ratio >= MEMORY_WARN_RATIO, "full": used >= MEMORY_MAX_ITEMS,
        }

    def _make_room(self, needed: int = 1) -> int:
        """Evict the lowest-importance (then oldest) memories until `needed` slots are free."""
        over = self.count() + needed - MEMORY_MAX_ITEMS
        if over <= 0:
            return 0
        with connect() as con:
            victims = [int(r[0]) for r in con.execute(
                "SELECT id FROM memories WHERE student_id=? AND is_active=1 ORDER BY importance ASC, created_at ASC, id ASC LIMIT ?",
                (self.student_id, over)).fetchall()]
            if victims:
                marks = ",".join("?" * len(victims))
                con.execute(f"DELETE FROM memories WHERE student_id=? AND id IN ({marks})", (self.student_id, *victims))
        self._drop_vectors(set(victims))
        return len(victims)

    def _drop_vectors(self, memory_ids: set[int]) -> None:
        """Remove the vectors belonging to `memory_ids` and rewrite the index."""
        if not memory_ids or self.index is None or not self.metadata:
            return
        keep = [i for i, m in enumerate(self.metadata) if int(m.get("memory_id", -1)) not in memory_ids]
        if len(keep) == len(self.metadata):
            return
        if keep:
            vectors = np.ascontiguousarray(self.index.reconstruct_n(0, self.index.ntotal)[keep], dtype="float32")
            new_index = faiss.IndexFlatIP(vectors.shape[1])
            new_index.add(vectors)
            self.index = new_index
        else:
            self.index = None
            if self.index_path.exists():
                self.index_path.unlink()
        self.metadata = [self.metadata[i] for i in keep]
        self._save()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = embed_texts([query])
        scores, ids = self.index.search(q, min(top_k, self.index.ntotal))
        result = []
        for score, idx in zip(scores[0], ids[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            item = dict(self.metadata[int(idx)])
            item["score"] = float(score)
            result.append(item)
        return result

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        with connect() as con:
            rows = con.execute("SELECT * FROM memories WHERE student_id=? AND is_active=1 ORDER BY id DESC LIMIT ?", (self.student_id, limit)).fetchall()
        return [dict(r) for r in rows]

    def clear(self) -> None:
        with connect() as con:
            con.execute("DELETE FROM memories WHERE student_id=?", (self.student_id,))
        self.index = None
        self.metadata = []
        # Remove the old vector index too. Otherwise a later app run could
        # reload stale vectors with an empty metadata list.
        if self.index_path.exists():
            self.index_path.unlink()
        self._save()


def memory_prompt(memories: list[dict[str, Any]]) -> str:
    if not memories:
        return "No long-term student memories were found."
    lines = ["Relevant long-term student memories:"]
    for m in memories:
        lines.append(f"- [{m.get('memory_type','learning')}] {m.get('content','')}")
    return "\n".join(lines)


def reset_memory(student_id: str, include_conversations: bool = False) -> dict[str, int]:
    """User-initiated reset.

    Always clears the semantic long-term memory (database rows + vector index).
    With ``include_conversations`` it also deletes the saved Tutor / Research chat history.
    Quiz history, mastery and revision data are never touched here.
    """
    memory = LongTermMemory(student_id)
    removed = memory.count()
    memory.clear()
    chats = delete_agent_sessions(student_id) if include_conversations else 0
    return {"memories_removed": removed, "conversations_removed": chats}

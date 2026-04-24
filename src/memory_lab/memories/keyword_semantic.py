from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .base import MemoryBackend, MemoryItem
from ..utils import normalize_ws


def _score(query: str, text: str) -> int:
    q = set(normalize_ws(query).lower().split())
    t = set(normalize_ws(text).lower().split())
    if not q or not t:
        return 0
    return len(q.intersection(t))


@dataclass
class KeywordSemanticMemory(MemoryBackend):
    """
    Semantic memory fallback without embeddings.
    Stores documents in JSON and retrieves by simple keyword overlap.
    """

    name: str = "keyword_semantic"
    path: Path = Path("semantic_memory.json")

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def _save(self, rows: List[Dict[str, Any]]) -> None:
        self._ensure_parent()
        self.path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    def read(self, user_id: str, query: str, k: int = 5) -> List[MemoryItem]:
        rows = [r for r in self._load() if r.get("user_id") == user_id]
        scored: List[Tuple[int, Dict[str, Any]]] = []
        for r in rows:
            txt = str(r.get("text") or "")
            scored.append((_score(query, txt), r))
        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[MemoryItem] = []
        for s, r in scored[: max(1, k)]:
            if s <= 0:
                continue
            out.append(MemoryItem(kind="semantic", text=str(r.get("text") or ""), metadata=dict(r.get("metadata") or {})))
        return out

    def write(self, user_id: str, items: List[MemoryItem]) -> None:
        rows = self._load()
        for it in items:
            if it.kind != "semantic":
                continue
            rows.append(
                {
                    "user_id": user_id,
                    "text": it.text,
                    "metadata": dict(it.metadata or {}),
                }
            )
        self._save(rows)


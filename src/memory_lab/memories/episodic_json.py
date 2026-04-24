from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from .base import MemoryBackend, MemoryItem


@dataclass
class JsonEpisodicMemory(MemoryBackend):
    """
    Episodic memory stored as JSONL (append-only) for "experience recall".
    """

    name: str = "episodic_json"
    path: Path = Path("episodic_memory.jsonl")

    def _ensure_parent(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def read(self, user_id: str, query: str, k: int = 5) -> List[MemoryItem]:
        if not self.path.exists():
            return []

        # Naive retrieval: last-k episodes for that user (good enough for lab).
        rows: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("user_id") == user_id:
                    rows.append(obj)

        rows = rows[-k:]
        return [
            MemoryItem(
                kind="episode",
                text=r.get("text", ""),
                metadata={"ts": r.get("ts"), "session_id": r.get("session_id")},
            )
            for r in rows
            if r.get("text")
        ]

    def write(self, user_id: str, items: List[MemoryItem]) -> None:
        self._ensure_parent()
        now = datetime.utcnow().isoformat() + "Z"
        with self.path.open("a", encoding="utf-8") as f:
            for it in items:
                if it.kind != "episode":
                    continue
                row = {
                    "ts": now,
                    "user_id": user_id,
                    "session_id": it.metadata.get("session_id"),
                    "text": it.text,
                    "metadata": it.metadata,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


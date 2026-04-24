from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class MemoryItem:
    kind: str
    text: str
    metadata: Dict[str, Any]


class MemoryBackend:
    name: str

    def read(self, user_id: str, query: str, k: int = 5) -> List[MemoryItem]:
        raise NotImplementedError

    def write(self, user_id: str, items: List[MemoryItem]) -> None:
        raise NotImplementedError


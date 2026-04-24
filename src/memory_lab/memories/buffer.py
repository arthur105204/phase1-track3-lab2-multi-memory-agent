from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .base import MemoryBackend, MemoryItem


@dataclass
class BufferMemory(MemoryBackend):
    """
    Short-term memory (ConversationBufferMemory-like).
    We store the last N messages for the current session.
    """

    name: str = "buffer"
    max_messages: int = 12
    _by_session: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)

    def set_session_messages(self, session_id: str, messages: List[Dict[str, str]]) -> None:
        self._by_session[session_id] = list(messages)[-self.max_messages :]

    def read(self, user_id: str, query: str, k: int = 5) -> List[MemoryItem]:
        # buffer reads are handled by passing messages directly to LLM;
        # this is a lightweight adapter for metrics/debug.
        return []

    def write(self, user_id: str, items: List[MemoryItem]) -> None:
        # no-op: buffer is updated via set_session_messages()
        return


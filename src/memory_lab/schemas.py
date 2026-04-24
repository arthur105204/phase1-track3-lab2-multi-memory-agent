from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


Role = Literal["system", "user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str


Intent = Literal["preference", "fact", "episode", "semantic", "buffer_only"]


@dataclass
class RouterDecision:
    intent: Intent
    reasons: List[str]


@dataclass
class ContextStats:
    max_tokens: int
    estimated_tokens_before: int
    estimated_tokens_after: int
    trimmed_layers: List[str]


@dataclass
class MemoryReadResult:
    items: List[Dict[str, Any]]


@dataclass
class MemoryWriteResult:
    writes: List[Dict[str, Any]]
    skipped: bool = False
    reason: Optional[str] = None


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .utils import estimate_tokens, join_nonempty


LAYER_ORDER = [
    # Highest-level guidance (kept longest)
    "system",
    "task",
    "user",
    "memory",
    # Retrieved docs / RAG results
    "retrieval",
    # Tool outputs / API responses
    "tool",
    # Safety / policy guardrails (never removed)
    "policy",
]


EVICT_ORDER = [
    # Trim “from bottom up”, but policy is last and never removed.
    "tool",
    "retrieval",
    "memory",
    "user",
    "task",
    "system",
]


@dataclass
class LayeredContext:
    system: str = ""
    task: str = ""
    user: str = ""
    memory: str = ""
    retrieval: str = ""
    tool: str = ""
    policy: str = ""

    def as_prompt_blocks(self) -> List[Tuple[str, str]]:
        return [
            ("system", self.system),
            ("task", self.task),
            ("user", self.user),
            ("memory", self.memory),
            ("retrieval", self.retrieval),
            ("tool", self.tool),
            ("policy", self.policy),
        ]

    def estimated_tokens(self) -> int:
        return sum(estimate_tokens(v) for _, v in self.as_prompt_blocks() if v and v.strip())


def build_system_context() -> str:
    return join_nonempty(
        [
            "You are a helpful AI assistant.",
            "Be concise when the user requests concise answers.",
        ]
    )


def build_policy_context() -> str:
    return join_nonempty(
        [
            "Policy: Follow safety rules. Do not provide disallowed content.",
            "If user preference conflicts with policy, policy wins.",
        ]
    )


def trim_to_budget(ctx: LayeredContext, max_tokens: int) -> Tuple[LayeredContext, Dict[str, int], List[str]]:
    before = ctx.estimated_tokens()
    trimmed: List[str] = []

    if before <= max_tokens:
        return ctx, {"before": before, "after": before}, trimmed

    # Copy to mutate
    mutable = LayeredContext(**{k: getattr(ctx, k) for k in LAYER_ORDER})

    for layer in EVICT_ORDER:
        if mutable.estimated_tokens() <= max_tokens:
            break
        value = getattr(mutable, layer)
        if value and value.strip():
            setattr(mutable, layer, "")
            trimmed.append(layer)

    after = mutable.estimated_tokens()
    return mutable, {"before": before, "after": after}, trimmed


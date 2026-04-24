from __future__ import annotations

import re
from typing import Iterable, List, Tuple


def estimate_tokens(text: str) -> int:
    # Cheap heuristic: ~4 chars/token in English, slightly different for VI,
    # but good enough for trimming decisions.
    if not text:
        return 0
    return max(1, len(text) // 4)


def join_nonempty(lines: Iterable[str], sep: str = "\n") -> str:
    return sep.join([x for x in lines if x and x.strip()])


def simple_sentence_split(text: str) -> List[str]:
    parts = re.split(r"(?<=[\.\?\!])\s+", (text or "").strip())
    return [p.strip() for p in parts if p and p.strip()]


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


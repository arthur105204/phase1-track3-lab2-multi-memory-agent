from __future__ import annotations

import re
from typing import List

from .schemas import RouterDecision


def route_intent(user_text: str, enable_memory: bool) -> RouterDecision:
    if not enable_memory:
        return RouterDecision(intent="buffer_only", reasons=["memory disabled"])

    t = (user_text or "").lower()
    reasons: List[str] = []

    # Preference cues
    if any(p in t for p in ["tôi thích", "toi thich", "không thích", "khong thich", "tôi muốn", "toi muon", "prefer", "ngắn gọn", "dài", "qua dai"]):
        reasons.append("preference cue")
        return RouterDecision(intent="preference", reasons=reasons)

    # Experience/episode cues
    if any(p in t for p in ["hay bị", "hay bi", "bị rối", "bi roi", "confused", "hôm trước", "hom truoc", "lần trước", "lan truoc"]):
        reasons.append("episode cue")
        return RouterDecision(intent="episode", reasons=reasons)

    # Fact/profile cues
    if any(
        p in t
        for p in [
            "tôi đang",
            "toi dang",
            "tôi biết",
            "toi biet",
            "profile",
            "tôi dùng",
            "toi dung",
            "dị ứng",
            "di ung",
            "allergy",
        ]
    ):
        reasons.append("fact cue")
        return RouterDecision(intent="fact", reasons=reasons)

    # Semantic memory: explicit "ghi nhớ" or definitional questions
    if any(p in t for p in ["ghi nhớ", "ghi nho", "là gì", "la gi", "dùng để", "dung de"]):
        reasons.append("semantic cue")
        return RouterDecision(intent="semantic", reasons=reasons)

    return RouterDecision(intent="semantic", reasons=["default semantic"])


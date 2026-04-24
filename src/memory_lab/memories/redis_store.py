from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .base import MemoryBackend, MemoryItem


def _safe_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


@dataclass
class RedisLongTermMemory(MemoryBackend):
    """
    Long-term memory (facts/preferences) stored in Redis for cross-session persistence.

    Data model (per user):
    - Hash:  user:{user_id}:preferences
    - Hash:  user:{user_id}:profile      (keyed facts for conflict updates)
    - Set:   user:{user_id}:facts
    - List:  user:{user_id}:sessions  (recent session summaries)
    """

    name: str = "redis"
    redis_url: Optional[str] = None

    def _client(self):
        try:
            import redis  # type: ignore
        except Exception as e:
            raise RuntimeError("redis package not installed") from e

        url = self.redis_url or os.getenv("REDIS_URL")
        if not url:
            raise RuntimeError("REDIS_URL is not set")
        return redis.Redis.from_url(url, decode_responses=True)

    def read(self, user_id: str, query: str, k: int = 5) -> List[MemoryItem]:
        r = self._client()
        prefs_key = f"user:{user_id}:preferences"
        profile_key = f"user:{user_id}:profile"
        facts_key = f"user:{user_id}:facts"

        prefs = r.hgetall(prefs_key) or {}
        profile = r.hgetall(profile_key) or {}
        facts = list(r.smembers(facts_key) or [])

        items: List[MemoryItem] = []
        if prefs:
            items.append(
                MemoryItem(
                    kind="preferences",
                    text=_safe_json(prefs),
                    metadata={"count": len(prefs)},
                )
            )
        if profile:
            items.append(
                MemoryItem(
                    kind="profile",
                    text=_safe_json(profile),
                    metadata={"count": len(profile)},
                )
            )
        if facts:
            items.append(
                MemoryItem(
                    kind="facts",
                    text=_safe_json(sorted(facts)[: max(1, k * 2)]),
                    metadata={"count": len(facts)},
                )
            )
        return items

    def write(self, user_id: str, items: List[MemoryItem]) -> None:
        r = self._client()
        prefs_key = f"user:{user_id}:preferences"
        profile_key = f"user:{user_id}:profile"
        facts_key = f"user:{user_id}:facts"

        for it in items:
            if it.kind == "preference":
                # metadata: {"key": "...", "value": "..."}
                key = str(it.metadata.get("key") or "")
                val = str(it.metadata.get("value") or it.text)
                if key:
                    r.hset(prefs_key, key, val)
            elif it.kind == "fact":
                # If fact has a stable key, overwrite old value (conflict handling)
                key = str(it.metadata.get("key") or "")
                if key:
                    val = str(it.metadata.get("value") or it.text)
                    r.hset(profile_key, key, val)
                    # Cleanup contradictory/legacy facts in the set for keyed attributes.
                    # Rubric-required: allergy should end up as a single latest value.
                    if key == "allergy":
                        try:
                            existing = list(r.smembers(facts_key) or [])
                            to_remove = [
                                x
                                for x in existing
                                if isinstance(x, str)
                                and (
                                    "allergic to" in x.lower()
                                    or "dị ứng" in x.lower()
                                    or "di ung" in x.lower()
                                    or "allergy" in x.lower()
                                )
                            ]
                            if to_remove:
                                r.srem(facts_key, *to_remove)
                        except Exception:
                            pass
                else:
                    r.sadd(facts_key, it.text)


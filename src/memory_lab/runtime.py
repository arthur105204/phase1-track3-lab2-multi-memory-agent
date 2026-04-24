from __future__ import annotations

import json
import os
import random
import time
from urllib.error import HTTPError, URLError
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class RuntimeMode(str, Enum):
    mock = "mock"
    llm = "llm"


@dataclass
class LLMConfig:
    api_key: Optional[str]
    model: str
    base_url: str
    temperature: float = 0.2
    timeout_s: int = 60

    @staticmethod
    def from_env() -> "LLMConfig":
        return LLMConfig(
            api_key=os.getenv("LLM_API_KEY"),
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
            timeout_s=int(os.getenv("LLM_TIMEOUT_S", "60")),
        )


def _openai_chat_completions(
    cfg: LLMConfig, messages: List[Dict[str, str]]
) -> Tuple[str, Dict[str, Any]]:
    url = cfg.base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
    }

    headers = {
        "Content-Type": "application/json",
    }
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    last_err: Optional[Exception] = None
    # Retry transient gateway errors
    for attempt in range(1, 4):
        try:
            with urllib.request.urlopen(req, timeout=cfg.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
                data = json.loads(raw)
            break
        except HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(1.5 * attempt)
                continue
            raise
        except URLError as e:
            last_err = e
            if attempt < 4:
                time.sleep(1.5 * attempt)
                continue
            raise
        except Exception as e:
            last_err = e
            raise
    else:
        raise RuntimeError(f"LLM request failed after retries: {last_err}")

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage") or {}
    return content, {"usage": usage, "raw": data}


def generate_assistant_reply(
    mode: RuntimeMode,
    llm_cfg: LLMConfig,
    messages: List[Dict[str, str]],
) -> Tuple[str, Dict[str, Any]]:
    if mode == RuntimeMode.llm:
        if not llm_cfg.api_key:
            raise RuntimeError("Missing LLM_API_KEY for mode=llm")
        return _openai_chat_completions(llm_cfg, messages)

    # mock mode: deterministic-ish echo + tiny “helpful” behavior
    last_user = ""
    for m in reversed(messages):
        if m.get("role") == "user":
            last_user = m.get("content", "")
            break
    canned = [
        "Mình đã hiểu. Đây là câu trả lời mock để kiểm tra pipeline.",
        "Mock response: mình sẽ trả lời ngắn gọn theo yêu cầu.",
        "Mock response: mình sẽ gợi ý các bước thực hành.",
    ]
    reply = f"{random.choice(canned)}\n\nUser said: {last_user[:400]}"
    return reply, {"usage": {"total_tokens": 0}, "raw": {"mock": True}}


from __future__ import annotations

from dataclasses import asdict
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.graph import END, StateGraph

from .context import LayeredContext, build_policy_context, build_system_context, trim_to_budget
from .extractor import extract_key_facts
from .memories import (
    BufferMemory,
    ChromaSemanticMemory,
    JsonEpisodicMemory,
    KeywordSemanticMemory,
    RedisLongTermMemory,
)
from .memories.base import MemoryItem
from .router import route_intent
from .runtime import LLMConfig, RuntimeMode, generate_assistant_reply
from .schemas import ContextStats, RouterDecision
from .utils import join_nonempty


GraphState = Dict[str, Any]


def build_agent(
    runtime_mode: RuntimeMode,
    llm_config: LLMConfig,
    enable_memory: bool,
    user_id: str,
    max_context_tokens: int,
    work_dir: Path,
):
    buffer = BufferMemory()
    redis_mem = RedisLongTermMemory()
    episodic = JsonEpisodicMemory(path=work_dir / "episodic_memory.jsonl")
    semantic_backend = os.getenv("SEMANTIC_BACKEND", "chroma").lower().strip()
    if semantic_backend == "keyword":
        semantic_mem = KeywordSemanticMemory(path=work_dir / "semantic_memory.json")
    else:
        semantic_mem = ChromaSemanticMemory(persist_dir=work_dir / "chroma_db")

    def _route(state: GraphState) -> GraphState:
        messages = state.get("messages") or []
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break
        decision = route_intent(last_user, enable_memory=enable_memory)
        state["router"] = asdict(decision)
        state["last_user_text"] = last_user
        return state

    def _read_memory(state: GraphState) -> GraphState:
        decision = (state.get("router") or {}).get("intent", "buffer_only")
        query = state.get("last_user_text", "")
        reads: Dict[str, Any] = {}

        if not enable_memory or decision == "buffer_only":
            state["memory_reads"] = reads
            return state

        def safe_read(name: str, fn):
            try:
                items = fn()
                reads[name] = [asdict(x) for x in items]
            except Exception as e:
                reads[name] = {"error": str(e)}

        # Always available: episodic JSON
        if decision in {"episode"}:
            safe_read("episodic_json", lambda: episodic.read(user_id, query, k=5))
        if decision in {"semantic"}:
            safe_read(semantic_mem.name, lambda: semantic_mem.read(user_id, query, k=5))
        if decision in {"preference", "fact"}:
            safe_read("redis", lambda: redis_mem.read(user_id, query, k=5))

        state["memory_reads"] = reads
        return state

    def _generate(state: GraphState) -> GraphState:
        session_id = state.get("session_id", "session")
        turn_index = int(state.get("turn_index") or 0)
        messages = list(state.get("messages") or [])
        buffer.set_session_messages(session_id, messages)

        router = state.get("router") or {}
        reads = state.get("memory_reads") or {}

        # Build layered context
        ctx = LayeredContext(
            system=build_system_context(),
            task="Task: Help the user. Use memory if relevant.",
            user=f"UserID: {user_id}",
            memory=join_nonempty(
                [
                    "Recalled memory:",
                    join_nonempty([str(reads)], sep="\n"),
                ]
            )
            if reads
            else "",
            retrieval="",
            tool="",
            policy=build_policy_context(),
        )

        trimmed_ctx, stats, trimmed_layers = trim_to_budget(ctx, max_tokens=max_context_tokens)

        prompt_messages: List[Dict[str, str]] = []
        # Put policy and system in system role to maximize adherence
        sys_text = join_nonempty(
            [
                trimmed_ctx.policy,
                trimmed_ctx.system,
                trimmed_ctx.task,
                trimmed_ctx.user,
                trimmed_ctx.memory,
                trimmed_ctx.retrieval,
                trimmed_ctx.tool,
            ]
        )
        prompt_messages.append({"role": "system", "content": sys_text})
        prompt_messages.extend(messages)

        assistant, meta = generate_assistant_reply(runtime_mode, llm_config, prompt_messages)
        messages.append({"role": "assistant", "content": assistant})
        state["messages"] = messages

        state["context_stats"] = {
            "max_tokens": max_context_tokens,
            "estimated_tokens_before": stats["before"],
            "estimated_tokens_after": stats["after"],
            "trimmed_layers": trimmed_layers,
        }
        state["llm_usage"] = meta.get("usage") or {}
        return state

    def _write_memory(state: GraphState) -> GraphState:
        if not enable_memory:
            state["memory_writes"] = {"skipped": True, "reason": "memory disabled"}
            return state

        session_id = state.get("session_id", "session")
        router_intent = (state.get("router") or {}).get("intent", "buffer_only")
        user_text = state.get("last_user_text") or ""

        items = extract_key_facts(runtime_mode, llm_config, user_text, session_id=session_id)
        writes: Dict[str, Any] = {"items": [asdict(x) for x in items]}

        def safe_write(name: str, fn):
            try:
                fn()
                writes[name] = "ok"
            except Exception as e:
                writes[name] = {"error": str(e)}

        # Persist strategy: write depending on extracted item kinds (not only router)
        redis_items = [it for it in items if it.kind in {"preference", "fact"}]
        episode_items = [it for it in items if it.kind == "episode"]
        semantic_items = [it for it in items if it.kind == "semantic"]

        if redis_items:
            safe_write("redis", lambda: redis_mem.write(user_id, redis_items))
        if episode_items:
            safe_write("episodic_json", lambda: episodic.write(user_id, episode_items))
        if semantic_items:
            safe_write(semantic_mem.name, lambda: semantic_mem.write(user_id, semantic_items))

        state["memory_writes"] = writes
        return state

    graph = StateGraph(GraphState)
    graph.add_node("route", _route)
    graph.add_node("read_memory", _read_memory)
    graph.add_node("generate", _generate)
    graph.add_node("write_memory", _write_memory)

    graph.set_entry_point("route")
    graph.add_edge("route", "read_memory")
    graph.add_edge("read_memory", "generate")
    graph.add_edge("generate", "write_memory")
    graph.add_edge("write_memory", END)

    return graph.compile()


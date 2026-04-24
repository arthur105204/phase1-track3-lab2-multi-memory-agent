from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .utils import estimate_tokens


def _compute_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_turns = 0
    total_prompt_tokens = 0
    total_trimmed_layers = 0
    memory_read_events = 0
    memory_hit_events = 0

    for conv in records:
        gold_keys = set(conv.get("gold_memory_keys") or [])
        for t in conv.get("trace", []):
            total_turns += 1
            ctx = t.get("context_stats") or {}
            total_prompt_tokens += int(ctx.get("estimated_tokens_after") or 0)
            total_trimmed_layers += len(ctx.get("trimmed_layers") or [])

            reads = t.get("memory_reads") or {}
            if reads:
                memory_read_events += 1
                # naive “hit”: any gold key appears in retrieved memory stringified
                blob = json.dumps(reads, ensure_ascii=False).lower()
                if any(k.lower() in blob for k in gold_keys):
                    memory_hit_events += 1

    avg_ctx_tokens = (total_prompt_tokens / total_turns) if total_turns else 0.0
    hit_rate = (memory_hit_events / memory_read_events) if memory_read_events else 0.0

    return {
        "total_conversations": len(records),
        "total_turns": total_turns,
        "avg_context_tokens": avg_ctx_tokens,
        "avg_trimmed_layers_per_turn": (total_trimmed_layers / total_turns) if total_turns else 0.0,
        "memory_read_events": memory_read_events,
        "memory_hit_rate": hit_rate,
    }


def write_json_report(out_dir: Path, records: List[Dict[str, Any]], metadata: Dict[str, Any]) -> Dict[str, Any]:
    report = {
        "metadata": {
            **(metadata or {}),
            "total": len(records),
        },
        "metrics": _compute_metrics(records),
        "records": records,
    }
    path = out_dir / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def render_markdown_report(out_dir: Path, report: Dict[str, Any]) -> None:
    m = report.get("metrics") or {}
    lines = []
    lines.append("# Multi-Memory Agent Benchmark Report")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Total conversations: **{m.get('total_conversations', 0)}**")
    lines.append(f"- Total turns: **{m.get('total_turns', 0)}**")
    lines.append(f"- Avg context tokens (est): **{m.get('avg_context_tokens', 0):.1f}**")
    lines.append(f"- Avg trimmed layers/turn: **{m.get('avg_trimmed_layers_per_turn', 0):.2f}**")
    lines.append(f"- Memory hit rate (naive): **{m.get('memory_hit_rate', 0)*100:.1f}%**")
    lines.append("")
    lines.append("## Conversations")
    for rec in report.get("records", []):
        lines.append(f"### {rec.get('id')} — {rec.get('title')}")
        for t in rec.get("trace", []):
            idx = t.get("turn_index")
            router = (t.get("router") or {}).get("intent")
            lines.append(f"- Turn {idx}: intent={router}")
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


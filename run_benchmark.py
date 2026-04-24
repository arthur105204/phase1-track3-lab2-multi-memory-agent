from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import typer
from rich.console import Console

from src.memory_lab.agent import build_agent
from src.memory_lab.reporting import render_markdown_report, write_json_report
from src.memory_lab.runtime import LLMConfig, RuntimeMode

app = typer.Typer(add_completion=False)
console = Console()


@dataclass
class RunConfig:
    mode: RuntimeMode
    dataset_path: Path
    out_dir: Path
    enable_memory: bool
    user_id: str
    max_context_tokens: int


def _load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _expand_turn(turn: Dict[str, Any]) -> str:
    text = turn.get("user", "")
    repeat = int(turn.get("repeat") or 0)
    if repeat > 0:
        padding = str(turn.get("padding") or "X")
        text = text + (" " + padding) * repeat
    return text


@app.command()
def main(
    mode: Literal["mock", "llm"] = typer.Option("mock", help="Runtime mode: mock|llm"),
    dataset: str = typer.Option("data/bench_conversations.json", help="Path to benchmark dataset JSON"),
    out_dir: str = typer.Option("outputs/run", help="Output directory"),
    enable_memory: bool = typer.Option(True, help="Enable full memory stack (on/off)"),
    user_id: str = typer.Option("user_001", help="User id for cross-session memory"),
    max_context_tokens: int = typer.Option(1800, help="Approx token budget for prompt context"),
):
    # Load .env automatically so users don't need to export variables manually.
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(override=False)
    except Exception:
        pass

    cfg = RunConfig(
        mode=RuntimeMode(mode),
        dataset_path=Path(dataset),
        out_dir=Path(out_dir),
        enable_memory=enable_memory,
        user_id=user_id,
        max_context_tokens=max_context_tokens,
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_obj = _load_dataset(cfg.dataset_path)

    llm_cfg = LLMConfig.from_env()
    agent = build_agent(
        runtime_mode=cfg.mode,
        llm_config=llm_cfg,
        enable_memory=cfg.enable_memory,
        user_id=cfg.user_id,
        max_context_tokens=cfg.max_context_tokens,
        work_dir=cfg.out_dir,
    )

    console.print(f"[bold]Dataset[/bold]: {cfg.dataset_path}")
    console.print(f"[bold]Mode[/bold]: {cfg.mode.value} | [bold]Memory[/bold]: {cfg.enable_memory}")

    run_records: List[Dict[str, Any]] = []
    for conv in dataset_obj:
        conv_id = conv["id"]
        session_id = f"{conv_id}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        turns = conv.get("turns", [])

        messages: List[Dict[str, str]] = []
        conv_trace: List[Dict[str, Any]] = []
        for i, turn in enumerate(turns):
            user_text = _expand_turn(turn)
            messages.append({"role": "user", "content": user_text})

            result = agent.invoke(
                {
                    "session_id": session_id,
                    "messages": messages,
                    "turn_index": i,
                }
            )
            assistant_text = result["messages"][-1]["content"]
            messages.append({"role": "assistant", "content": assistant_text})

            conv_trace.append(
                {
                    "turn_index": i,
                    "user": user_text,
                    "assistant": assistant_text,
                    "router": result.get("router", {}),
                    "memory_reads": result.get("memory_reads", {}),
                    "memory_writes": result.get("memory_writes", {}),
                    "context_stats": result.get("context_stats", {}),
                }
            )

        run_records.append(
            {
                "id": conv_id,
                "title": conv.get("title"),
                "enable_memory": cfg.enable_memory,
                "trace": conv_trace,
                "gold_memory_keys": conv.get("gold_memory_keys", []),
            }
        )

    report = write_json_report(cfg.out_dir, run_records, metadata={"mode": cfg.mode.value})
    render_markdown_report(cfg.out_dir, report)
    console.print(f"[green]Wrote report[/green]: {cfg.out_dir / 'report.json'}")
    console.print(f"[green]Wrote report[/green]: {cfg.out_dir / 'report.md'}")


if __name__ == "__main__":
    app()


"""Command-line entry point."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__, ui
from .agent import Agent
from .ollama_client import OllamaClient, OllamaError

DEFAULT_MODEL = os.environ.get("COLLAMA_MODEL", "qwen2.5-coder")
DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="collama",
        description="A Claude Code / Codex-style coding agent powered by Ollama.",
    )
    p.add_argument("-p", "--prompt", help="One-shot prompt; print response and exit.")
    p.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL}).")
    p.add_argument("--host", default=DEFAULT_HOST, help=f"Ollama host (default: {DEFAULT_HOST}).")
    p.add_argument("-C", "--cwd", default=".", help="Workspace root (default: cwd).")
    p.add_argument("--yolo", action="store_true", help="Auto-approve all tool calls (skip prompts).")
    p.add_argument("-t", "--temperature", type=float, default=0.2)
    p.add_argument("-V", "--version", action="version", version=f"collama {__version__}")
    return p.parse_args(argv)


HELP_TEXT = """\
Slash commands:
  /help              show this help
  /tools             list tools the model can call
  /model <name>      switch model
  /clear             reset conversation history
  /yolo              toggle auto-approve for tool calls
  /exit, /quit       leave
"""


def repl(agent: Agent) -> int:
    ui.banner(agent.model, str(agent.ctx.root))
    while True:
        try:
            line = input(ui.color("\n› ", ui.BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue

        if line.startswith("/"):
            parts = line.split(maxsplit=1)
            cmd = parts[0][1:].lower()
            arg = parts[1] if len(parts) > 1 else ""
            if cmd in ("exit", "quit"):
                return 0
            if cmd == "help":
                print(HELP_TEXT)
                continue
            if cmd == "tools":
                from .tools import TOOLS
                for n in TOOLS:
                    print(f"  - {n}")
                continue
            if cmd == "model":
                if not arg:
                    ui.info(f"current model: {agent.model}")
                else:
                    agent.model = arg
                    ui.info(f"switched to {arg}")
                continue
            if cmd == "clear":
                agent.reset()
                ui.info("history cleared")
                continue
            if cmd == "yolo":
                agent.ctx.yolo = not agent.ctx.yolo
                ui.info(f"yolo: {'ON' if agent.ctx.yolo else 'OFF'}")
                continue
            ui.warn(f"unknown command: /{cmd}")
            continue

        try:
            agent.turn(line)
        except KeyboardInterrupt:
            ui.warn("interrupted")
            continue


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    root = Path(args.cwd).resolve()
    if not root.is_dir():
        ui.error(f"workspace not a directory: {root}")
        return 2

    client = OllamaClient(host=args.host)

    try:
        models = client.list_models()
    except OllamaError as e:
        ui.error(str(e))
        ui.warn("Is `ollama serve` running?")
        return 1

    if args.model not in models and models:
        ui.warn(f"model '{args.model}' not found locally. Available: {', '.join(models[:8])}")
        ui.warn(f"Pull it with: ollama pull {args.model}")

    agent = Agent(
        client=client,
        model=args.model,
        root=root,
        yolo=args.yolo,
        temperature=args.temperature,
    )

    if args.prompt:
        agent.turn(args.prompt)
        return 0
    return repl(agent)


if __name__ == "__main__":
    sys.exit(main())

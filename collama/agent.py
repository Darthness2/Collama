"""Main agent loop: send messages to Ollama, handle tool calls, repeat."""
from __future__ import annotations

import json
from pathlib import Path

from . import ui
from .ollama_client import OllamaClient, OllamaError
from .tools import TOOL_SCHEMAS, ToolContext, dispatch

SYSTEM_PROMPT = """You are Collama, a terminal-based coding assistant running on the user's machine via Ollama.

You help the user read, edit, and create files, run commands, and answer questions about their codebase. You have tools available — use them. Don't ask the user to paste file contents; read files yourself with read_file. Don't guess at code; verify with grep / read_file first.

Operating principles:
- Be concise. The user is in a terminal; long answers are noise.
- Prefer edit_file over write_file for existing files (it's safer — exact replacement).
- When making changes, briefly confirm what you did and any follow-ups the user should run (tests, lints).
- Never invent file paths. Use list_dir / grep to discover them.
- One step at a time: call a tool, see the result, then decide.
- When the task is complete, give a short final answer and stop calling tools.
"""

MAX_TOOL_ITERATIONS = 25


class Agent:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        root: Path,
        yolo: bool = False,
        temperature: float = 0.2,
    ):
        self.client = client
        self.model = model
        self.ctx = ToolContext(root=root, yolo=yolo)
        self.temperature = temperature
        self.messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _options(self) -> dict:
        return {"temperature": self.temperature}

    def _summarize_args(self, name: str, args: dict) -> str:
        if name in ("read_file", "write_file", "edit_file"):
            return str(args.get("path", ""))
        if name == "list_dir":
            return str(args.get("path", "."))
        if name == "grep":
            return f"/{args.get('pattern', '')}/  in {args.get('path', '.')}"
        if name == "run_bash":
            cmd = str(args.get("command", ""))
            return cmd if len(cmd) < 80 else cmd[:77] + "…"
        return ""

    def turn(self, user_input: str) -> str:
        """Run one user-turn: send message, loop through tool calls, return final text."""
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(MAX_TOOL_ITERATIONS):
            try:
                msg = self.client.chat(
                    model=self.model,
                    messages=self.messages,
                    tools=TOOL_SCHEMAS,
                    options=self._options(),
                )
            except OllamaError as e:
                ui.error(str(e))
                return ""

            self.messages.append(msg)

            tool_calls = msg.get("tool_calls") or []
            content = (msg.get("content") or "").strip()

            if content and not tool_calls:
                ui.assistant(content)
                return content

            if content and tool_calls:
                ui.assistant(content)

            if not tool_calls:
                return content

            for call in tool_calls:
                fn = call.get("function", {}) or {}
                name = fn.get("name", "")
                raw_args = fn.get("arguments", {})
                if isinstance(raw_args, str):
                    try:
                        args = json.loads(raw_args) if raw_args else {}
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = raw_args or {}

                ui.tool_call(name, self._summarize_args(name, args))
                result = dispatch(name, args, self.ctx)
                ok = not result.startswith("ERROR")
                first_line = result.splitlines()[0] if result else ""
                ui.tool_result(first_line[:160], ok=ok)

                self.messages.append({
                    "role": "tool",
                    "name": name,
                    "content": result,
                })

        ui.warn(f"hit tool-call limit ({MAX_TOOL_ITERATIONS}); stopping.")
        return ""

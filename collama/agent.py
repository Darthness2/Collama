"""Main agent loop: send messages to Ollama, handle tool calls, repeat."""
from __future__ import annotations

import json
from pathlib import Path

from . import ui
from .ollama_client import OllamaClient, OllamaError, ToolsUnsupportedError
from .tools import ToolContext, all_tool_schemas, dispatch

def build_system_prompt(workspace: Path, home: Path, tools_enabled: bool = True) -> str:
    base = f"""You are Collama, a terminal-based coding assistant running on the user's machine via Ollama.

You help the user read, edit, and create files, run commands, query GitHub, and answer questions about their codebase.

Environment:
- Workspace (cwd): {workspace}
- User home dir:  {home}
- OS user can reach files anywhere on the filesystem.

Filesystem access:
- Your file/dir/grep/bash tools accept relative paths (resolved against the workspace), absolute paths (e.g. /etc/hosts), and ~-paths (e.g. ~/Documents). The home dir above is what ~ expands to.
- You can — and should — read and edit files OUTSIDE the workspace when the user asks. The workspace is just the default for relative paths; it is NOT a sandbox.
- When the user starts a NEW project, create a new top-level directory under the home dir (e.g. {home}/<project-name>/) and put all the project files inside it. Don't nest a new project inside the current workspace unless the user explicitly asks.
- Be careful with destructive operations. Always confirm intent before deleting or overwriting outside the workspace.

Operating principles:
- Be concise. The user is in a terminal; long answers are noise.
- Don't ask the user to paste file contents — read files yourself with read_file.
- Don't guess at code — verify with grep / read_file first.
- Prefer edit_file over write_file for existing files (safer — exact replacement, and the user sees a diff).
- When making changes, briefly confirm what you did and any follow-ups the user should run (tests, lints).
- One step at a time: call a tool, see the result, then decide.
- When the task is complete, give a short final answer and stop calling tools.
"""
    if not tools_enabled:
        base += (
            "\nNOTE: This Ollama model does not support tool calling, so file/shell tools "
            "are DISABLED for this session. You can still discuss code and answer questions, "
            "but you cannot read or edit files yourself. Ask the user to paste contents when needed."
        )
    return base

MAX_TOOL_ITERATIONS = 25


class Agent:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        root: Path,
        yolo: bool = False,
        temperature: float = 0.2,
        on_turn_complete=None,
        tools_enabled: bool = True,
        on_tools_disabled=None,
    ):
        self.client = client
        self.model = model
        self.ctx = ToolContext(root=root, yolo=yolo)
        self.temperature = temperature
        self.tools_enabled = tools_enabled
        self.on_turn_complete = on_turn_complete  # called after each user turn
        self.on_tools_disabled = on_tools_disabled  # called the first time tool support fails
        self.messages: list[dict] = [{"role": "system", "content": self._system_prompt()}]

    def _system_prompt(self) -> str:
        return build_system_prompt(self.ctx.root, Path.home(), tools_enabled=self.tools_enabled)

    def _refresh_system_prompt(self) -> None:
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = self._system_prompt()
        else:
            self.messages.insert(0, {"role": "system", "content": self._system_prompt()})

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self._system_prompt()}]

    def load_messages(self, messages: list[dict]) -> None:
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": self._system_prompt()}] + list(messages)
        self.messages = list(messages)

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
        if name.startswith("gh_") or name == "github_api":
            bits = []
            for k in ("repo", "path", "number", "query", "method"):
                if k in args:
                    bits.append(f"{k}={args[k]}")
            return " ".join(bits)
        return ""

    def turn(self, user_input: str) -> str:
        """Run one user-turn: send message, loop through tool calls, return final text."""
        self.messages.append({"role": "user", "content": user_input})

        for _ in range(MAX_TOOL_ITERATIONS):
            try:
                msg = self.client.chat(
                    model=self.model,
                    messages=self.messages,
                    tools=all_tool_schemas() if self.tools_enabled else None,
                    options=self._options(),
                )
            except ToolsUnsupportedError:
                ui.warn(
                    f"model '{self.model}' does not support tool calls — "
                    "disabling file/shell tools for this session."
                )
                self.tools_enabled = False
                self._refresh_system_prompt()
                if self.on_tools_disabled:
                    try:
                        self.on_tools_disabled(self)
                    except Exception:
                        pass
                continue
            except OllamaError as e:
                ui.error(str(e))
                return ""

            self.messages.append(msg)

            tool_calls = msg.get("tool_calls") or []
            content = (msg.get("content") or "").strip()

            if content and not tool_calls:
                ui.assistant(content)
                if self.on_turn_complete:
                    try:
                        self.on_turn_complete(self)
                    except Exception:
                        pass
                return content

            if content and tool_calls:
                ui.assistant(content)

            if not tool_calls:
                if self.on_turn_complete:
                    try:
                        self.on_turn_complete(self)
                    except Exception:
                        pass
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

"""Main agent loop: send messages to Ollama, handle tool calls, repeat."""
from __future__ import annotations

import json
import re
from pathlib import Path

_TOOL_TAG_RX = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL)

# DeepSeek-Coder native tool-call format (uses unicode-bar special tokens).
# Example:
#   <｜tool▁call▁begin｜>function<｜tool▁sep｜>write_file
#   ```json
#   {"path": "main.py", "content": "..."}
#   ```
#   <｜tool▁call▁end｜>
# Outer bracket char: ｜ (U+FF5C) or plain |. Inner word-separator: ▁ (U+2581).
_DS_BAR = r"[｜|]"
_DEEPSEEK_CALL_RX = re.compile(
    rf"<{_DS_BAR}tool▁call▁begin{_DS_BAR}>"
    rf"\s*(?:function\s*<{_DS_BAR}tool▁sep{_DS_BAR}>\s*)?"
    rf"([A-Za-z_][\w-]*)"
    rf".*?```(?:json)?\s*(\{{.*?\}})\s*```"
    rf".*?<{_DS_BAR}tool▁call▁end{_DS_BAR}>",
    re.DOTALL,
)

# Hallucinated tool *outputs* — we never emit these; strip and reprimand.
# Stripped in two passes: nuke wrapped <begin>...<end> blocks first (greedy
# enough to span fake content), then any stray begin/end tokens that remain.
_DEEPSEEK_OUTPUTS_BLOCK_RX = re.compile(
    rf"<{_DS_BAR}tool▁outputs?▁begin{_DS_BAR}>.*?<{_DS_BAR}tool▁outputs?▁end{_DS_BAR}>",
    re.DOTALL,
)
_DEEPSEEK_STRAY_TOKEN_RX = re.compile(
    rf"<{_DS_BAR}tool▁outputs?▁(?:begin|end){_DS_BAR}>"
)


class _Outputs:
    @staticmethod
    def search(text: str) -> bool:
        return bool(_DEEPSEEK_OUTPUTS_BLOCK_RX.search(text) or _DEEPSEEK_STRAY_TOKEN_RX.search(text))

    @staticmethod
    def strip(text: str) -> str:
        text = _DEEPSEEK_OUTPUTS_BLOCK_RX.sub("", text)
        text = _DEEPSEEK_STRAY_TOKEN_RX.sub("", text)
        return text


_DEEPSEEK_OUTPUTS_RX = _Outputs  # back-compat alias used elsewhere


def _extract_tool_call(content: str):
    """Return (name, args_dict, span_start, span_end) or None."""
    m = _TOOL_TAG_RX.search(content)
    if m:
        try:
            payload = json.loads(m.group(1))
            name = payload["name"]
            args = payload.get("arguments") or payload.get("args") or {}
            if not isinstance(args, dict):
                return None
            return name, args, m.start(), m.end()
        except (json.JSONDecodeError, KeyError, TypeError):
            return None
    m = _DEEPSEEK_CALL_RX.search(content)
    if m:
        try:
            args = json.loads(m.group(2))
            if not isinstance(args, dict):
                return None
            return m.group(1), args, m.start(), m.end()
        except json.JSONDecodeError:
            return None
    return None

from . import ui
from .ollama_client import OllamaClient, OllamaError, ToolsUnsupportedError
from .tools import ToolContext, all_tool_schemas, dispatch

TEXT_TOOL_PROTOCOL = """
=== TEXT TOOL PROTOCOL (use this — your runtime does NOT support native tool calls) ===
You DO have full access to the user's filesystem and shell, but you must invoke tools by emitting a tag.

To call a tool, emit EXACTLY one tag and STOP. Use either of these formats:

  Format A (preferred):
    <tool>{"name":"TOOL_NAME","arguments":{...}}</tool>

  Format B (DeepSeek native, also accepted):
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>TOOL_NAME
    ```json
    {...arguments...}
    ```
    <｜tool▁call▁end｜>

After the tag, STOP. I — the harness — will run the tool and reply with the actual result. You then continue (call another tool, or write your final answer).
When the task is complete, write your final answer as plain text with NO tool tag.

CRITICAL — DO NOT FAKE TOOL OUTPUTS. Never produce <｜tool▁output▁…｜> or <｜tool▁outputs▁…｜> blocks; those are MINE to emit, not yours. If you write fake outputs, your changes did NOT actually happen and the user will see no files. Just emit the call tag and wait.

Available tools (same names as native function calling):
- read_file({"path": str, "start_line"?: int, "end_line"?: int})
- write_file({"path": str, "content": str})
- edit_file({"path": str, "old_string": str, "new_string": str, "replace_all"?: bool})
- list_dir({"path"?: str})
- grep({"pattern": str, "path"?: str, "case_insensitive"?: bool})
- run_bash({"command": str, "timeout"?: int})
- set_workspace({"path": str, "create"?: bool})
- gh_whoami({}), gh_list_repos({...}), gh_get_repo({"repo": str}),
  gh_get_file({"repo": str, "path": str, "ref"?: str}),
  gh_list_issues({"repo": str, ...}), gh_create_issue({"repo": str, "title": str, "body"?: str}),
  gh_list_pulls({"repo": str, ...}), gh_get_pull({"repo": str, "number": int}),
  gh_search_code({"query": str}), github_api({"method"?: str, "path": str, "body"?: object})

IMPORTANT: You CAN see the user's files. Use read_file / list_dir / grep instead of saying you cannot.
"""


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
- When the user starts a NEW project, follow this recipe EXACTLY:
    1. Pick a project dir under the home dir, e.g. {home}/<project-name>/
    2. Call set_workspace with that path and create=true.
    3. From that point on, all relative file paths land inside the project. write_file("main.py", ...) now writes to {home}/<project-name>/main.py — NOT to {workspace}/main.py.
  Do NOT keep using the previous workspace for the new project. Do NOT write files using paths like "src/main.py" without first calling set_workspace, or your files will land in the wrong directory.
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
        base += "\n" + TEXT_TOOL_PROTOCOL
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

    def _finish(self, content: str) -> str:
        if self.on_turn_complete:
            try:
                self.on_turn_complete(self)
            except Exception:
                pass
        return content

    def _run_text_protocol(self) -> str:
        """Loop using the <tool>{json}</tool> text protocol — for models without native tool calls."""
        for _ in range(MAX_TOOL_ITERATIONS):
            try:
                msg = self.client.chat(
                    model=self.model,
                    messages=self.messages,
                    tools=None,
                    options=self._options(),
                )
            except OllamaError as e:
                ui.error(str(e))
                return ""

            raw = (msg.get("content") or "")

            # Strip any fabricated tool *outputs* the model invented. We never
            # emit these; the model is hallucinating success.
            had_fake_outputs = _Outputs.search(raw)
            cleaned = _Outputs.strip(raw).strip()

            # Save the cleaned message, not the fabricated one.
            msg["content"] = cleaned
            self.messages.append(msg)

            extracted = _extract_tool_call(cleaned)

            if not extracted:
                if had_fake_outputs:
                    ui.warn("model fabricated tool outputs — telling it to retry properly.")
                    self.messages.append({
                        "role": "user",
                        "content": (
                            "STOP. You fabricated tool output blocks; that is not allowed. "
                            "I am the harness — only I can produce tool outputs. "
                            "If you need to call a tool, emit ONE call and STOP. "
                            "Acceptable formats:\n"
                            "  <tool>{\"name\":\"NAME\",\"arguments\":{...}}</tool>\n"
                            "  …or your native <｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME ```json {...} ```<｜tool▁call▁end｜>\n"
                            "Then wait for my response. Do NOT write tool_outputs yourself."
                        ),
                    })
                    continue
                if cleaned:
                    ui.assistant(cleaned)
                return self._finish(cleaned)

            name, args, start, end = extracted
            preamble = cleaned[:start].strip()
            if preamble:
                ui.assistant(preamble)

            ui.tool_call(name, self._summarize_args(name, args))
            result = dispatch(name, args, self.ctx)
            ok = not result.startswith("ERROR")
            first_line = result.splitlines()[0] if result else ""
            ui.tool_result(first_line[:160], ok=ok)

            self.messages.append({
                "role": "user",
                "content": f"Tool result for {name}:\n{result}",
            })

        ui.warn(f"hit tool-call limit ({MAX_TOOL_ITERATIONS}); stopping.")
        return ""

    def turn(self, user_input: str) -> str:
        """Run one user-turn: send message, loop through tool calls, return final text."""
        self.messages.append({"role": "user", "content": user_input})

        if not self.tools_enabled:
            return self._run_text_protocol()

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
                    f"model '{self.model}' does not support native tool calls — "
                    "switching to text-protocol tools for this session."
                )
                self.tools_enabled = False
                self._refresh_system_prompt()
                if self.on_tools_disabled:
                    try:
                        self.on_tools_disabled(self)
                    except Exception:
                        pass
                return self._run_text_protocol()
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

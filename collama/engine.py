"""QueryEngine — the core agent loop, decoupled from any UI.

Mirrors Claude Code's architecture:

    submitMessage(prompt) ──> Iterator[Message]
        ├── fetch_system_prompt_parts()   assemble system prompt
        ├── process_user_input()          handle /commands (no-op here)
        ├── query()                       main agent loop
        │     ├── auto_compact()          context compression
        │     └── run_tools()              tool orchestration
        └── yield Message                  stream to consumer

The REPL consumes events for rendering. A headless caller can consume the
same stream programmatically — same engine, two front-ends.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

from . import ui
from .config import get_value, set_value
from .ollama_client import OllamaClient, OllamaError, ToolsUnsupportedError
from .services.compact import auto_compact
from .state import AppState
from .tasks import TaskKind, new_id
from .tools import ToolContext, all_tool_schemas, dispatch


MAX_TOOL_ITERATIONS = 25
LOOP_THRESHOLD = 3
COMPACT_TOKENS = 12000
COMPACT_KEEP_RECENT = 12


# ---------------------------------------------------------------- events ----

EventKind = Literal[
    "system",      # system prompt installed (data: {prompt})
    "user",        # user message submitted (data: {text})
    "thinking",    # model wrote <think>...</think> (data: {text})
    "plan",        # model wrote <plan>...</plan>   (data: {steps})
    "narration",   # mid-turn text alongside a tool call (data: {text})
    "assistant",   # final text answer for this turn (data: {text})
    "tool_call",   # about to dispatch a tool (data: {id,name,args,summary})
    "tool_result", # tool returned (data: {id,name,result,ok,first_line})
    "warn",        # advisory (loop, fakes) (data: {text})
    "error",       # call/loop failed (data: {text})
    "compact",     # context compacted (data: {before,after})
    "done",        # turn finished (data: {text})
]


@dataclass
class Message:
    """One streamed event from the engine."""
    kind: EventKind
    data: dict = field(default_factory=dict)
    task_id: str | None = None


# ---------------------------------------------------------------- regexes ----

_TOOL_TAG_RX = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL)
_PLAN_RX = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)
_THINK_RX = re.compile(r"<think(?:ing)?>(.*?)</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_FENCE_RX = re.compile(r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
_DS_BAR = r"[｜|]"
_DEEPSEEK_CALL_RX = re.compile(
    rf"<{_DS_BAR}tool▁call▁begin{_DS_BAR}>"
    rf"\s*(?:function\s*<{_DS_BAR}tool▁sep{_DS_BAR}>\s*)?"
    rf"([A-Za-z_][\w-]*)"
    rf".*?```(?:json)?\s*(\{{.*?\}})\s*```"
    rf".*?<{_DS_BAR}tool▁call▁end{_DS_BAR}>",
    re.DOTALL,
)
_DEEPSEEK_OUTPUTS_BLOCK_RX = re.compile(
    rf"<{_DS_BAR}tool▁outputs?▁begin{_DS_BAR}>.*?<{_DS_BAR}tool▁outputs?▁end{_DS_BAR}>",
    re.DOTALL,
)
_DEEPSEEK_STRAY_TOKEN_RX = re.compile(
    rf"<{_DS_BAR}tool▁outputs?▁(?:begin|end){_DS_BAR}>"
)


# ---------------------------------------------------------------- helpers ----

def _looks_like_call(obj):
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("tool") or obj.get("function")
    if not isinstance(name, str) or not name:
        return None
    args = obj.get("arguments")
    if args is None:
        args = obj.get("args") or obj.get("parameters") or {}
    if not isinstance(args, dict):
        return None
    return name, args


def _try_bare_json(text: str):
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, end = decoder.raw_decode(text[i:])
        except (json.JSONDecodeError, ValueError):
            continue
        call = _looks_like_call(obj)
        if not call:
            continue
        name, args = call
        return name, args, i, i + end
    return None


def _extract_tool_call(content: str):
    m = _TOOL_TAG_RX.search(content)
    if m:
        try:
            payload = json.loads(m.group(1))
            call = _looks_like_call(payload)
            if call:
                return call[0], call[1], m.start(), m.end()
        except json.JSONDecodeError:
            pass
    m = _DEEPSEEK_CALL_RX.search(content)
    if m:
        try:
            args = json.loads(m.group(2))
            if isinstance(args, dict):
                return m.group(1), args, m.start(), m.end()
        except json.JSONDecodeError:
            pass
    for m in _FENCE_RX.finditer(content):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        call = _looks_like_call(obj)
        if call:
            return call[0], call[1], m.start(), m.end()
    bare = _try_bare_json(content)
    if bare:
        return bare
    return None


def _extract_plan(text: str) -> tuple[list[str], str]:
    m = _PLAN_RX.search(text)
    if not m:
        return [], text
    body = m.group(1).strip()
    steps: list[str] = []
    for raw in body.splitlines():
        s = raw.strip()
        if not s:
            continue
        s = re.sub(r"^(?:[-*•]|\d+[.)])\s*", "", s)
        if s:
            steps.append(s)
    cleaned = (text[:m.start()] + text[m.end():]).strip()
    return steps, cleaned


def _extract_thinking(text: str) -> tuple[list[str], str]:
    blocks: list[str] = []
    def _consume(m):
        blocks.append(m.group(1).strip())
        return ""
    cleaned = _THINK_RX.sub(_consume, text).strip()
    return blocks, cleaned


def _strip_fakes(text: str) -> tuple[bool, str]:
    had = bool(_DEEPSEEK_OUTPUTS_BLOCK_RX.search(text) or _DEEPSEEK_STRAY_TOKEN_RX.search(text))
    out = _DEEPSEEK_OUTPUTS_BLOCK_RX.sub("", text)
    out = _DEEPSEEK_STRAY_TOKEN_RX.sub("", out)
    return had, out


def _summarize_args(name: str, args: dict) -> str:
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


# ---------------------------------------------------------------- prompt ----

def fetch_system_prompt_parts(state: AppState) -> str:
    workspace = state.workspace
    home = state.home
    tools_enabled = state.tools_enabled

    base = f"""You are Collama, a terminal-based coding assistant running on the user's machine via Ollama.

You help the user read, edit, and create files, run commands, query GitHub, and answer questions about their codebase.

Environment:
- Workspace (cwd): {workspace}
- User home dir:  {home}
- OS user can reach files anywhere on the filesystem.

Filesystem access:
- Your file/dir/grep/bash tools accept relative paths (resolved against the workspace), absolute paths, and ~-paths.
- You can — and should — read and edit files OUTSIDE the workspace when the user asks. The workspace is just the default for relative paths; it is NOT a sandbox.
- IMPORTANT — local first, GitHub second: when the user mentions a project, repo, or directory name (e.g. "use meteteoman/Market", "in my snake project"), it is almost certainly a LOCAL clone or directory on this machine, NOT a remote GitHub repo. ALWAYS check locally first with list_dir on {home}/<name>. Only call gh_* tools when the user explicitly says "on GitHub".
- After list_dir on a directory OUTSIDE the current workspace, you MUST do ONE of these before reading files inside it:
    a) call set_workspace with that directory (preferred), OR
    b) use absolute paths for every read_file / edit_file / grep call.
  Never use relative paths after listing a different directory; they will resolve against the OLD workspace and 404.
- When the user starts a NEW project, follow this recipe EXACTLY:
    1. Pick a project dir under the home dir, e.g. {home}/<project-name>/
    2. Call set_workspace with that path and create=true.
    3. From that point on, all relative file paths land inside the project.

Operating principles:
- Plan first. For ANY non-trivial task, open your reply with a numbered plan in this exact format:

    <plan>
    1. first step
    2. second step
    </plan>

  Keep steps short. After the plan, immediately call the first tool. Skip the plan only for trivial single-tool questions.
- Be concise. The user is in a terminal; long answers are noise.
- Don't ask the user to paste file contents — read files yourself.
- Don't guess at code — verify with grep / read_file first.
- Prefer edit_file over write_file for existing files.
- One step at a time: call a tool, see the result, then decide.
- When the task is complete, give a short final answer and stop calling tools.
- If you want to think out loud privately, wrap it in <think>...</think>.
"""
    if not tools_enabled:
        base += """
=== TEXT TOOL PROTOCOL (use this — your runtime does NOT support native tool calls) ===
You DO have full filesystem and shell access. To call a tool, emit ONE call and STOP. Any of these formats works:

  Format A (preferred):
    <tool>{"name":"TOOL_NAME","arguments":{...}}</tool>

  Format B (DeepSeek native):
    <｜tool▁call▁begin｜>function<｜tool▁sep｜>TOOL_NAME
    ```json
    {...arguments...}
    ```
    <｜tool▁call▁end｜>

  Format C (fenced JSON):
    ```json
    {"name": "TOOL_NAME", "arguments": {...}}
    ```

After the call, STOP. Do NOT write tool_outputs yourself; the harness emits those.

Available tools: read_file, write_file, edit_file, list_dir, grep, run_bash, set_workspace,
gh_whoami, gh_list_repos, gh_get_repo, gh_get_file, gh_list_issues, gh_create_issue,
gh_list_pulls, gh_get_pull, gh_search_code, github_api.
"""
    return base


# ---------------------------------------------------------------- engine ----


class QueryEngine:
    """Stateful, event-emitting agent loop."""

    def __init__(
        self,
        client: OllamaClient,
        state: AppState,
        model: str,
        temperature: float = 0.2,
        config: dict | None = None,
    ) -> None:
        self.client = client
        self.state = state
        self.model = model
        self.temperature = temperature
        self.config = config  # mutable; used to persist tools_supported per model
        self.messages: list[dict] = [{"role": "system", "content": fetch_system_prompt_parts(state)}]
        self._recent_calls: list[tuple[str, str]] = []
        self._plan_shown_this_turn = False

    # ---- public API ----

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": fetch_system_prompt_parts(self.state)}]

    def load_messages(self, messages: list[dict]) -> None:
        if not messages or messages[0].get("role") != "system":
            messages = [{"role": "system", "content": fetch_system_prompt_parts(self.state)}] + list(messages)
        self.messages = list(messages)

    def refresh_system_prompt(self) -> None:
        prompt = fetch_system_prompt_parts(self.state)
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = prompt
        else:
            self.messages.insert(0, {"role": "system", "content": prompt})

    def submit_message(self, prompt: str) -> Iterator[Message]:
        """The main entry point. Yields a stream of Message events."""
        self._plan_shown_this_turn = False
        self._recent_calls = []

        # processUserInput would normally handle /commands here. The REPL does
        # that before reaching us, so pass-through.
        yield Message("user", {"text": prompt})

        self.messages.append({"role": "user", "content": prompt})

        # autoCompact before we send.
        report = auto_compact(
            self.messages,
            max_tokens=COMPACT_TOKENS,
            keep_recent=COMPACT_KEEP_RECENT,
        )
        if report.triggered:
            yield Message("compact", {"before": report.before, "after": report.after})

        # main query() loop
        for _ in range(MAX_TOOL_ITERATIONS):
            try:
                with ui.Spinner("thinking"):
                    msg = self.client.chat(
                        model=self.model,
                        messages=self.messages,
                        tools=all_tool_schemas() if self.state.tools_enabled else None,
                        options={"temperature": self.temperature},
                    )
            except ToolsUnsupportedError:
                # First-time learning: switch this session and persist.
                yield Message("warn", {"text": f"model '{self.model}' lacks native tool support — switching to text-protocol fallback."})
                self.state.update(tools_enabled=False)
                self.refresh_system_prompt()
                if self.config is not None:
                    set_value(self.config, f"models.{self.model}.tools_supported", False)
                continue
            except OllamaError as e:
                yield Message("error", {"text": str(e)})
                return

            raw = msg.get("content") or ""
            had_fakes, cleaned = _strip_fakes(raw)
            cleaned = cleaned.strip()
            msg["content"] = cleaned
            self.messages.append(msg)

            if had_fakes:
                yield Message("warn", {"text": "model fabricated tool outputs — asking it to retry."})
                self.messages.append({
                    "role": "user",
                    "content": (
                        "STOP. Tool outputs are mine to emit. Send a single tool call "
                        "(<tool>{...}</tool> or fenced JSON) and STOP. Do NOT write tool_outputs."
                    ),
                })
                continue

            # Strip <think> and <plan>; emit them as their own events.
            thinks, content = _extract_thinking(cleaned)
            for t in thinks:
                if t:
                    yield Message("thinking", {"text": t})
            steps, content = _extract_plan(content)
            content = content.strip()
            if steps and not self._plan_shown_this_turn:
                yield Message("plan", {"steps": steps})
                self._plan_shown_this_turn = True

            calls, narration = self._extract_calls(msg, content)

            if not calls:
                if narration:
                    yield Message("assistant", {"text": narration})
                yield Message("done", {"text": narration})
                return

            if narration:
                yield Message("narration", {"text": narration})

            yield from self._run_tools(calls)

        yield Message("error", {"text": f"hit tool-call limit ({MAX_TOOL_ITERATIONS}); stopping."})

    # ---- internals ----

    def _extract_calls(self, msg: dict, content: str):
        native = msg.get("tool_calls") or []
        if native:
            calls: list[tuple[str, dict, str]] = []
            for c in native:
                fn = c.get("function", {}) or {}
                name = fn.get("name", "")
                raw = fn.get("arguments", {})
                if isinstance(raw, str):
                    try:
                        args = json.loads(raw) if raw else {}
                    except json.JSONDecodeError:
                        args = {}
                else:
                    args = raw or {}
                calls.append((name, args, "tool"))
            return calls, content
        if not content:
            return [], content
        ext = _extract_tool_call(content)
        if not ext:
            return [], content
        name, args, start, end = ext
        narration = (content[:start] + content[end:]).strip()
        return [(name, args, "user")], narration

    def _check_loop(self, name: str, args: dict) -> bool:
        key = (name, json.dumps(args, sort_keys=True, default=str))
        self._recent_calls.append(key)
        run = 1
        for prev in reversed(self._recent_calls[:-1]):
            if prev == key:
                run += 1
            else:
                break
        self._recent_calls = self._recent_calls[-12:]
        if run >= LOOP_THRESHOLD:
            self._recent_calls = []
            self.messages.append({
                "role": "user",
                "content": (
                    f"You called {name} with the same arguments {run} times in a row "
                    f"and got the same result. Stop repeating. Try a different approach: "
                    f"a different path, different pattern, list_dir the parent, set_workspace, "
                    f"or summarize and ask a clarifying question."
                ),
            })
            return True
        return False

    def _run_tools(self, calls):
        ctx = ToolContext(**self.state.to_tool_ctx_kwargs())
        for name, args, role in calls:
            if self._check_loop(name, args):
                yield Message("warn", {"text": f"loop detected on {name} — steered."})
                continue
            tid = new_id(TaskKind.BASH if name == "run_bash" else TaskKind.TOOL)
            yield Message("tool_call", {
                "id": tid, "name": name, "args": args,
                "summary": _summarize_args(name, args),
            })
            try:
                result = dispatch(name, args, ctx)
            except Exception as e:
                result = f"ERROR: {type(e).__name__}: {e}"
            ok = not result.startswith("ERROR")
            first_line = result.splitlines()[0] if result else ""
            yield Message("tool_result", {
                "id": tid, "name": name, "result": result,
                "ok": ok, "first_line": first_line,
            })

            # Reflect any state changes that tools made (e.g. set_workspace).
            if name == "set_workspace" and ok:
                self.state.update(workspace=ctx.root)
                self.refresh_system_prompt()

            # File-history bookkeeping for read/write/edit.
            if name in ("read_file", "write_file", "edit_file") and ok:
                self.state.push_file(str(args.get("path", "")))

            if role == "tool":
                self.messages.append({"role": "tool", "name": name, "content": result})
            else:
                self.messages.append({
                    "role": "user",
                    "content": f"Tool result for {name}:\n{result}",
                })

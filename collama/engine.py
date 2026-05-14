"""QueryEngine — the core agent loop, decoupled from any UI.

Implements the Claude-Code-style data flow:

    USER INPUT ──> processUserInput() ──> UserMessage
       │
       ▼
    fetchSystemPromptParts() ─── tools, env, CLAUDE.md memory
       │
       ▼
    recordTranscript() ─── persist user message to JSONL
       │
       ▼
    ┌─> manage_context() ─── snip + collapse + autoCompact
    │   normalizeMessagesForAPI()
    │   │
    │   ▼
    │   Ollama (streaming) ─── /api/chat
    │   │
    │   ▼
    │   stream events ─── delta → done
    │   │
    │   ├─ text/plan/think ──> yield Message(...) to consumer
    │   │
    │   └─ tool_use?
    │       │
    │       ▼
    │   StreamingToolExecutor
    │       │  (partition: concurrent-safe vs serial)
    │       ▼
    │   canUseTool() ─── permission check
    │       │
    │       ├─ DENY ──> append tool_result(error), continue loop
    │       │
    │       └─ ALLOW ──> tool.call() ──> append tool_result, recordTranscript()
    │       │
    └───────┘
       │
       ▼ (no more tool calls)
    yield Message("done", {text, usage})
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

from . import ui
from .background import BackgroundExecutor
from .config import set_value
from .ollama_client import OllamaClient, OllamaError, ToolsUnsupportedError
from .permissions import CONCURRENT_SAFE, Resolver, auto_deny_resolver, can_use_tool
from .services.compact import BOUNDARY_MARKER, manage_context
from .services.transcript import record as record_transcript
from .state import AppState
from .tasks import TaskGraph, TaskKind, new_id
from .teams import TeamRegistry
from .tools import ToolContext, all_tool_schemas, dispatch


MAX_TOOL_ITERATIONS = 1000
LOOP_THRESHOLD = 3
COMPACT_TOKENS = 12000
COMPACT_KEEP_RECENT = 12


# ---------------------------------------------------------------- events ----

EventKind = Literal[
    "system", "user", "thinking", "plan", "narration", "delta", "assistant",
    "tool_call", "tool_denied", "tool_result",
    "warn", "error", "compact", "done",
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


# ---------------------------------------------------------------- s05: memory ---

def _load_claude_md(workspace: Path, home: Path) -> str:
    """Lazily collect CLAUDE.md / .collama.md memory files from workspace and parents."""
    seen: set[Path] = set()
    chunks: list[str] = []
    cur = workspace.resolve()
    home = home.resolve()
    while True:
        for name in ("CLAUDE.md", ".collama.md", "AGENTS.md"):
            p = cur / name
            if p.exists() and p not in seen:
                seen.add(p)
                try:
                    text = p.read_text(errors="replace")
                except OSError:
                    continue
                chunks.append(f"--- {p} ---\n{text.strip()}")
        if cur == cur.parent or cur == home.parent:
            break
        cur = cur.parent
    return "\n\n".join(chunks)


# ---------------------------------------------------------------- prompt ----

def fetch_system_prompt_parts(state: AppState) -> str:
    """Assemble the system prompt: base instructions + tool guidance + memory."""
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
- Tools accept relative paths (resolved against the workspace), absolute paths, and ~-paths.
- The workspace is NOT a sandbox; read/edit files anywhere when asked.
- LOCAL FIRST: when the user mentions a project or "owner/repo"-style name, treat it as a LOCAL directory under {home}/<name>. Only call gh_* tools when the user explicitly says "on GitHub".
- After list_dir on a directory OUTSIDE the current workspace, EITHER call set_workspace to it OR use absolute paths for follow-up reads. Never use relative paths after listing a different directory.
- For NEW projects: pick {home}/<project-name>/ → call set_workspace with create=true → write files relatively.

Operating principles:
- Plan first. For non-trivial tasks, open with a numbered plan inside <plan>...</plan>. Then call the first tool.
- Be concise. Don't ask for file contents — read them. Don't guess code — verify.
- Prefer edit_file over write_file for existing files.
- One step at a time: call a tool, observe, decide.
- Wrap private reasoning in <think>...</think>.
"""
    if not tools_enabled:
        base += """
=== TEXT TOOL PROTOCOL (your runtime does NOT support native tool calls) ===
You DO have full filesystem access. Call a tool by emitting ONE call and STOPPING. Any of these works:

  <tool>{"name":"TOOL_NAME","arguments":{...}}</tool>
  <｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME ```json {...} ```<｜tool▁call▁end｜>
  ```json
  {"name":"TOOL_NAME","arguments":{...}}
  ```

After the call, STOP. The harness emits tool outputs; you must NOT.
Available: read_file, write_file, edit_file, list_dir, grep, run_bash, set_workspace,
gh_whoami, gh_list_repos, gh_get_repo, gh_get_file, gh_list_issues, gh_create_issue,
gh_list_pulls, gh_get_pull, gh_search_code, github_api.
"""

    # s05: KNOWLEDGE ON DEMAND — append CLAUDE.md / .collama.md / AGENTS.md from workspace and parents.
    memory = _load_claude_md(workspace, home)
    if memory:
        base += "\n\n=== PROJECT MEMORY (from CLAUDE.md / AGENTS.md / .collama.md) ===\n" + memory

    return base


def process_user_input(raw: str) -> dict:
    """Parse user input into a UserMessage. Slash commands are handled by the
    REPL before reaching here, so for now this is a passthrough."""
    return {"role": "user", "content": raw}


def normalize_messages_for_api(messages: list[dict]) -> list[dict]:
    """Strip UI-only fields and keep only what Ollama expects."""
    keep_keys = {"role", "content", "tool_calls", "name", "tool_call_id"}
    out: list[dict] = []
    for m in messages:
        cleaned = {k: v for k, v in m.items() if k in keep_keys}
        # Filter out our internal compact-boundary marker — it's just for our bookkeeping.
        if cleaned.get("role") == "system" and BOUNDARY_MARKER in (cleaned.get("content") or ""):
            continue
        out.append(cleaned)
    return out


# ---------------------------------------------------------------- executor --

class StreamingToolExecutor:
    """Partition and dispatch tool calls.

    Concurrent-safe (read-only) tools run in a small thread pool; mutating
    tools run serially in the order the model emitted them. Each call goes
    through canUseTool() before dispatch; denials emit a warn + a synthetic
    tool_result so the model can recover.
    """

    def __init__(
        self,
        state: AppState,
        resolver: Resolver,
        max_workers: int = 4,
        engine: object | None = None,
        background: object | None = None,
        tasks: object | None = None,
        teams: object | None = None,
    ) -> None:
        self.state = state
        self.resolver = resolver
        self.max_workers = max_workers
        self.engine = engine
        self.background = background
        self.tasks = tasks
        self.teams = teams

    def execute(self, calls: list[tuple[str, dict, str]]) -> Iterator[Message]:
        if not calls:
            return
        concurrent_calls = [c for c in calls if c[0] in CONCURRENT_SAFE]
        serial_calls = [c for c in calls if c[0] not in CONCURRENT_SAFE]

        # Concurrent first: read tools that don't mutate.
        if len(concurrent_calls) > 1:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                futures = [pool.submit(self._run_one, c) for c in concurrent_calls]
                for fut in as_completed(futures):
                    yield from fut.result()
        else:
            for c in concurrent_calls:
                yield from self._run_one(c)

        for c in serial_calls:
            yield from self._run_one(c)

    def _run_one(self, call: tuple[str, dict, str]) -> list[Message]:
        name, args, role = call
        events: list[Message] = []
        tid = new_id(TaskKind.BASH if name == "run_bash" else TaskKind.TOOL)
        events.append(Message("tool_call", {
            "id": tid, "name": name, "args": args, "role": role,
            "summary": _summarize_args(name, args),
        }, task_id=tid))

        allowed, reason = can_use_tool(name, args, self.state, self.resolver)
        if not allowed:
            err = f"ERROR: permission denied ({reason})"
            events.append(Message("tool_denied", {"id": tid, "name": name, "reason": reason}, task_id=tid))
            events.append(Message("tool_result", {
                "id": tid, "name": name, "result": err,
                "ok": False, "first_line": err, "role": role,
            }, task_id=tid))
            return events

        ctx = ToolContext(
            **self.state.to_tool_ctx_kwargs(),
            state=self.state,
            engine=self.engine,
            background=self.background,
            tasks=self.tasks,
            teams=self.teams,
        )
        try:
            result = dispatch(name, args, ctx)
        except Exception as e:
            result = f"ERROR: {type(e).__name__}: {e}"
        ok = not result.startswith("ERROR")
        first_line = result.splitlines()[0] if result else ""
        events.append(Message("tool_result", {
            "id": tid, "name": name, "result": result,
            "ok": ok, "first_line": first_line, "role": role,
            "ctx_root": str(ctx.root),
        }, task_id=tid))
        return events


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
        session_id: str | None = None,
        permission_resolver: Resolver | None = None,
        stream: bool = True,
    ) -> None:
        self.client = client
        self.state = state
        self.model = model
        self.temperature = temperature
        self.config = config
        self.session_id = session_id
        self.stream = stream
        self.permission_resolver: Resolver = permission_resolver or auto_deny_resolver
        self.task_graph = TaskGraph()
        self.background = BackgroundExecutor(tasks=self.task_graph)
        self.teams = TeamRegistry()
        self.executor = StreamingToolExecutor(
            state, self.permission_resolver,
            engine=self, background=self.background,
            tasks=self.task_graph, teams=self.teams,
        )
        self.messages: list[dict] = [{"role": "system", "content": fetch_system_prompt_parts(state)}]
        self._recent_calls: list[tuple[str, str]] = []
        self._plan_shown_this_turn = False
        self._usage = {"input": 0, "output": 0, "ms": 0}

    # ---- public API ----

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": fetch_system_prompt_parts(self.state)}]
        self._usage = {"input": 0, "output": 0, "ms": 0}

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
        """Drive one user-turn through the full pipeline."""
        self._plan_shown_this_turn = False
        self._recent_calls = []

        # processUserInput()
        user_msg = process_user_input(prompt)
        yield Message("user", {"text": prompt})

        # recordTranscript()
        record_transcript(self.session_id or "", "user", prompt)

        self.messages.append(user_msg)

        # Manage context BEFORE sending: snip + collapse + autoCompact.
        for r in manage_context(
            self.messages,
            max_tokens=COMPACT_TOKENS,
            keep_recent=COMPACT_KEEP_RECENT,
            summarize_with_model=self._summarize_with_model,
        ):
            if r.triggered:
                yield Message("compact", {
                    "strategy": r.strategy, "before": r.before, "after": r.after,
                })

        # main query() loop
        for _ in range(MAX_TOOL_ITERATIONS):
            # Drain background notifications BEFORE the next API call so the
            # model sees finished bash_async / agent_call_async results.
            for note in self.background.drain_notifications():
                inject = (
                    f"[background] {note['kind']} {note['id']} "
                    f"finished ({note['status']}): {note['label']}\n{note['result']}"
                )
                self.messages.append({"role": "user", "content": inject})
                yield Message("warn", {"text": f"background {note['id']} {note['status']} — injected into context"})
            try:
                msg, usage = self._chat_once(yield_deltas=self.stream)
                if isinstance(msg, _StreamGen):
                    # streaming generator: spinner runs during prompt-eval
                    # (no tokens yet), then stops the instant the first token
                    # arrives so the user sees generation happening live.
                    final_msg = None
                    spinner = ui.Spinner("thinking")
                    spinner.start()
                    got_first = False
                    try:
                        for kind, payload in msg.iter():
                            if kind == "delta":
                                if not got_first:
                                    spinner.stop()
                                    got_first = True
                                yield Message("delta", {"text": payload})
                            elif kind == "done":
                                final_msg = payload
                    finally:
                        spinner.stop()
                    if final_msg is None:
                        yield Message("error", {"text": "stream ended without 'done'"})
                        return
                    msg = final_msg["message"]
                    usage = {
                        "input": final_msg.get("prompt_eval_count", 0),
                        "output": final_msg.get("eval_count", 0),
                        "ms": final_msg.get("total_duration_ns", 0) // 1_000_000,
                    }
            except ToolsUnsupportedError:
                yield Message("warn", {"text": f"model '{self.model}' lacks native tool support — switching to text-protocol fallback."})
                self.state.update(tools_enabled=False)
                self.refresh_system_prompt()
                if self.config is not None:
                    set_value(self.config, f"models.{self.model}.tools_supported", False)
                continue
            except OllamaError as e:
                yield Message("error", {"text": str(e)})
                return

            self._usage["input"] += usage.get("input", 0)
            self._usage["output"] += usage.get("output", 0)
            self._usage["ms"] += usage.get("ms", 0)

            raw = msg.get("content") or ""
            had_fakes, cleaned = _strip_fakes(raw)
            cleaned = cleaned.strip()
            msg["content"] = cleaned
            self.messages.append(msg)

            # Record the assistant turn.
            record_transcript(
                self.session_id or "", "assistant", cleaned,
                tool_calls=msg.get("tool_calls") or [],
            )

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
                yield Message("done", {
                    "text": narration,
                    "usage": dict(self._usage),
                    "session_id": self.session_id,
                })
                return

            # In streaming mode the narration was ALREADY shown live as delta
            # tokens — emitting it again as a narration event double-prints it
            # ('● ...' then '▪ ...'). Only emit the narration event for the
            # non-streaming path, which has no deltas.
            if narration and not self.stream:
                yield Message("narration", {"text": narration})

            yield from self._execute_and_record(calls)

        yield Message("error", {"text": f"hit tool-call limit ({MAX_TOOL_ITERATIONS}); stopping."})

    # ---- internals ----

    def _chat_once(self, *, yield_deltas: bool):
        """Returns (msg-or-stream, usage_dict). If yield_deltas, returns a
        _StreamGen that the caller iterates; otherwise returns a fully-assembled
        message."""
        api_messages = normalize_messages_for_api(self.messages)
        tools = all_tool_schemas() if self.state.tools_enabled else None

        if yield_deltas and hasattr(self.client, "chat_stream_assembled"):
            gen = self.client.chat_stream_assembled(
                model=self.model,
                messages=api_messages,
                tools=tools,
                options={"temperature": self.temperature},
            )
            return _StreamGen(gen), {}

        with ui.Spinner("thinking"):
            msg = self.client.chat(
                model=self.model,
                messages=api_messages,
                tools=tools,
                options={"temperature": self.temperature},
            )
        # Non-streaming endpoint doesn't expose usage in our wrapper today.
        return msg, {}

    def _summarize_with_model(self, middle: list[dict]) -> str | None:
        """Used by autoCompact to LLM-summarize older messages."""
        if not middle:
            return None
        try:
            short = []
            for m in middle:
                content = (m.get("content") or "")[:1500]
                short.append({"role": m.get("role"), "content": content})
            req = [
                {"role": "system", "content": "Summarize the following conversation as a tight bullet list of decisions made, files touched, and outstanding goals. <=200 words. No preamble."},
                {"role": "user", "content": json.dumps(short)},
            ]
            res = self.client.chat(
                model=self.model, messages=req, tools=None,
                options={"temperature": 0.0},
            )
            return (res.get("content") or "").strip() or None
        except Exception:
            return None

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

    def _execute_and_record(self, calls):
        """Run calls through the executor; reflect tool effects into state and
        record results into messages and the on-disk transcript."""
        # Drop calls that loop.
        cleaned_calls = []
        for name, args, role in calls:
            if self._check_loop(name, args):
                yield Message("warn", {"text": f"loop detected on {name} — steered."})
                continue
            cleaned_calls.append((name, args, role))

        for ev in self.executor.execute(cleaned_calls):
            yield ev
            if ev.kind == "tool_result":
                d = ev.data
                name = d["name"]
                result = d["result"]
                role = d.get("role", "user")

                # Reflect state changes from set_workspace.
                if name == "set_workspace" and d.get("ok"):
                    new_root = Path(d.get("ctx_root") or self.state.workspace)
                    self.state.update(workspace=new_root)
                    self.refresh_system_prompt()
                # File-history bookkeeping.
                if name in ("read_file", "write_file", "edit_file") and d.get("ok"):
                    # path lives in the corresponding tool_call event; we lost it here,
                    # but we can pull it from message history (last call with same id).
                    pass

                if role == "tool":
                    self.messages.append({"role": "tool", "name": name, "content": result})
                else:
                    self.messages.append({
                        "role": "user",
                        "content": f"Tool result for {name}:\n{result}",
                    })
                record_transcript(self.session_id or "", "tool", result, name=name)


# ---------------------------------------------------------------- helpers ----

class _StreamGen:
    """Tiny wrapper so submit_message can detect 'this is a streaming generator'."""
    def __init__(self, gen):
        self.gen = gen
    def iter(self):
        return self.gen

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
LOOP_THRESHOLD = 4  # tightened from 2 — at 2 a single retry tripped it,
# producing constant "steering hard" warnings on perfectly normal model
# behavior. 4 identical results in a turn is a genuine sign of being stuck.
ARGS_LOOP_THRESHOLD = 3  # consecutive identical-arg calls — a stronger signal
# than result repetition, so a slightly lower bar is reasonable.
LOOP_ABORT_THRESHOLD = 16  # hard turn-kill. Way above LOOP_THRESHOLD so we
# only abort on genuine runaway repetition, not normal recovery. Forcing the
# user to type /retry on every minor loop was the #1 friction point.
COMPACT_TOKENS = 12000
COMPACT_KEEP_RECENT = 12


# ---------------------------------------------------------------- events ----

EventKind = Literal[
    "system", "user", "thinking", "plan", "narration", "delta", "assistant",
    "tool_call", "tool_denied", "tool_result",
    "info", "warn", "error", "compact", "done",
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
    if name in ("read_file", "write_file", "edit_file", "multi_edit"):
        return str(args.get("path", ""))
    if name == "replace_lines":
        path = str(args.get("path", ""))
        start, end = args.get("start_line"), args.get("end_line")
        return f"{path}:{start}-{end}" if start is not None else path
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


# ---------------------------------------------------------------- effort ----

# The effort dial steers how much work the model puts into a turn. Ollama
# models have no native "reasoning effort" knob, so we steer it through the
# system prompt — the one lever that reliably changes a local model's
# thoroughness. Changeable live with /effort; persisted in config.
EFFORT_LEVELS = ("low", "medium", "high")

_EFFORT_GUIDANCE = {
    "low": (
        "Optimize for SPEED and minimalism. Do the smallest amount of work "
        "that satisfies the request:\n"
        "- Make the most direct change; don't refactor or polish beyond what "
        "was asked.\n"
        "- Investigate only as much as you must — a couple of reads, not a "
        "survey of the codebase.\n"
        "- Skip extra verification / test runs unless the user asked for them.\n"
        "- Keep your final answer to one or two sentences."
    ),
    "medium": (
        "Balance speed and rigor (the default):\n"
        "- Investigate enough to be confident, then act.\n"
        "- Verify the change when it's cheap to do so — re-run the failing "
        "command, or read back the region you edited.\n"
        "- Keep answers concise but complete."
    ),
    "high": (
        "Optimize for CORRECTNESS and thoroughness. Spend the extra effort:\n"
        "- Explore the relevant code broadly before editing — understand "
        "callers, edge cases, and related files, not just the first match.\n"
        "- Consider failure modes and edge cases; handle errors explicitly.\n"
        "- After editing, VERIFY: run the relevant command or test and confirm "
        "it passes, and re-read the changed region to be sure it's correct.\n"
        "- Prefer a complete, robust fix over a quick patch — but still act, "
        "then confirm; don't narrate endlessly."
    ),
}


def _effort_section(effort: str | None) -> str:
    level = (effort or "medium").lower()
    if level not in EFFORT_LEVELS:
        level = "medium"
    return f"\n\n=== EFFORT LEVEL: {level.upper()} ===\n{_EFFORT_GUIDANCE[level]}\n"


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
- Plan ONCE per turn. Emit <plan>...</plan> at the START of your first
  reply on a non-trivial task — never before every tool call. After the
  initial plan, just call tools.
- As you work through the plan, emit a step marker before starting each
  step's actions, in this exact form:

      <step 2/4>

  (where the first number is the current step and the second is total).
  The harness renders this as a visible "▸ step 2 of 4" header so the
  user can see live which step you're on. Emit ONE marker per step.
- Be concise. Don't ask for file contents — read them. Don't guess code
  — verify.
- One step at a time for INVESTIGATION (read, grep, observe, decide).
  But when you EDIT, batch it: collect every change you intend for a
  file and apply them together in one multi_edit call. Do not
  read-edit-read-edit one line at a time — that is slow and loops.
- Wrap private reasoning in <think>...</think>.
- DO NOT re-read a file you've already read this turn. The previous read
  is still in your context — scroll back and use it. Reading the same
  file twice wastes tokens and is a clear sign you are stuck. (The
  harness will return a CACHED nudge anyway.)
- NEVER refuse a coding task. Local file/code work is always appropriate.
  If you can't act, ask one specific clarifying question — do not write
  "I can't assist with that."

How to use the tools:
- "Slash commands" mentioned by the user (e.g. /ask, /help, --verbose,
  npm run X) are command/feature NAMES, NOT file paths. Don't call
  read_file with "/ask" as the path. Instead, search the codebase
  with grep for where the command is registered.
- read_file: read in ranges if a file is huge. Default returns 1..end
  but for files > ~300 lines pass start_line/end_line so you don't drown
  in tokens.
- BATCH YOUR EDITS. When a task changes several places in one file
  (renames, color/constant swaps, multi-site refactors), do NOT make
  one edit_file call per spot. Read the file ONCE, work out EVERY
  change, then send them all in a SINGLE multi_edit call:
      multi_edit(path, edits=[
        {{"old_string": "...", "new_string": "..."}},
        {{"old_string": "...", "new_string": "..."}},
        ... every change for this file ...
      ])
  multi_edit is atomic — if one edit doesn't match, nothing is written
  and the harness names the failing edit so you fix just that one.
  Prefer one multi_edit of 8 changes over 8 edit_file calls. Use plain
  edit_file only for a genuine single-spot change.
- edit_file / multi_edit rules:
    * old_string must match the file EXACTLY, including indentation.
      Copy text DIRECTLY from a recent read_file output — do not retype
      from memory or paraphrase.
    * If old_string isn't found, the harness shows the closest matching
      region. Use THAT exact text as your new old_string.
    * If edits keep failing to match on the same file, STOP retrying.
      Call replace_lines(path, start_line, end_line, new_content) — a
      surgical line-range edit that doesn't depend on string matching,
      so it sidesteps every encoding/quote/indent issue. Use grep to
      find the line numbers first, then call replace_lines.
    * Do NOT escape into run_bash to do edits with a Python script. The
      harness can't track those edits for /undo, and small models tend
      to write read-only `f.read()` scripts that look like progress but
      change nothing.
- write_file: use for new files, or when edit_file fails twice on an
  existing file. Always read the existing file first so you preserve
  the parts you aren't changing.
- grep / glob: use these to FIND code by content/pattern. ripgrep is
  used automatically when available.
- @path mentions: the user can write @path/to/file in their prompt and
  the harness will inline that file's contents — if you see "--- @path
  ---" sections, the file is already in your context, don't re-read it.

ACT, DON'T NARRATE. On debug/fix tasks:
- Your final output MUST be one of: (a) an edit_file/write_file call
  that addresses the request, (b) a run_bash that exercises a hypothesis,
  or (c) a SINGLE specific clarifying question. NEVER end a fix request
  with a generic summary of the code followed by "what would you like
  to do?" — the user already told you what they want.
- "I now understand the code, here's what it does" is NOT a fix. Use
  that understanding to make the change in the same turn.
- After at most 3 exploration tool calls on a fix task, you must either
  ACT or ASK one specific question. Open-ended summarizing is failure.
- DO NOT paste code blocks in your final answer. The user sees the edit
  directly via edit_file output ('✓ edited /path/file  +5 -2'); they do
  NOT need a markdown ``` block showing the same code. Final answers on
  a fix should be ONE sentence describing what changed and any verify
  step ('Fixed: timeout wraps _ticker_exists with 5s budget. Try /ask
  AAPL again.'). Save code blocks only for genuine explanations the
  user asked for ("show me the function").

Debugging discipline — when a command fails or code misbehaves:
1. READ THE ACTUAL ERROR. run_bash marks results PASS/FAIL and, on
   failure, appends a '↳ ...' hint with the error type and the most
   likely file:line. Trust that location.
2. OPEN THE EXACT SITE. read_file that file around the reported line
   before changing anything. Never patch a file you haven't just read.
3. STATE A HYPOTHESIS in ONE line, then make the smallest fix. Don't
   scatter speculative changes.
4. VERIFY. After the fix, re-run the same command (or test) and
   confirm it now reports PASS. If it still fails, the hypothesis was
   wrong — re-read the new error, don't keep guessing.
5. If you call the same tool with the same arguments and get the same
   result twice, STOP repeating — change approach or ask the user.
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

    # Effort dial — steer how much exploration / verification / rigor the
    # model applies this turn (changeable live with /effort).
    base += _effort_section(getattr(state, "effort", "medium"))

    # s05: KNOWLEDGE ON DEMAND — append CLAUDE.md / .collama.md / AGENTS.md from workspace and parents.
    memory = _load_claude_md(workspace, home)
    if memory:
        base += "\n\n=== PROJECT MEMORY (from CLAUDE.md / AGENTS.md / .collama.md) ===\n" + memory

    return base


_MENTION_RX = re.compile(r"(?<![\w/.])@([A-Za-z0-9_./\\-]+(?::\d+(?:-\d+)?)?)")


def _resolve_mention(token: str, workspace: Path, home: Path) -> tuple[Path, int | None, int | None] | None:
    """Resolve an `@token` to (path, start_line?, end_line?). Returns None if
    the path doesn't exist."""
    line_start: int | None = None
    line_end: int | None = None
    path_part = token
    # Support @path:N or @path:N-M
    if ":" in token:
        head, tail = token.rsplit(":", 1)
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", tail)
        if m:
            path_part = head
            line_start = int(m.group(1))
            line_end = int(m.group(2)) if m.group(2) else line_start
    import os as _os
    expanded = _os.path.expanduser(_os.path.expandvars(path_part))
    p = Path(expanded)
    if not p.is_absolute():
        p = workspace / p
    if p.exists() and p.is_file():
        return p, line_start, line_end
    return None


def process_user_input(raw: str, workspace: Path | None = None, home: Path | None = None) -> dict:
    """Parse user input into a UserMessage.

    Expands `@path/to/file` mentions (optionally `@path:N` or `@path:N-M`) by
    inlining the referenced file's contents — drops a couple of tool calls
    off most fix tasks and anchors the model to the right code immediately.
    """
    if not workspace or "@" not in raw:
        return {"role": "user", "content": raw}
    attachments: list[str] = []
    seen: set[Path] = set()
    for m in _MENTION_RX.finditer(raw):
        resolved = _resolve_mention(m.group(1), workspace, home or Path.home())
        if resolved is None:
            continue
        p, s, e = resolved
        if p in seen:
            continue
        seen.add(p)
        try:
            text = p.read_text(errors="replace")
        except OSError as exc:
            attachments.append(f"--- @{p} (read failed: {exc}) ---")
            continue
        lines = text.splitlines()
        lo = (s - 1) if s else 0
        hi = e if e else len(lines)
        lo = max(0, lo)
        hi = min(len(lines), hi)
        selected = lines[lo:hi]
        numbered = "\n".join(f"{lo + i + 1:>5}  {ln}" for i, ln in enumerate(selected))
        range_hdr = f":{s}-{e}" if s else ""
        attachments.append(f"--- @{p}{range_hdr}  ({len(lines)} lines total) ---\n{numbered}")
    if not attachments:
        return {"role": "user", "content": raw}
    body = raw + "\n\n" + "\n\n".join(attachments)
    return {"role": "user", "content": body}


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
                futures = [pool.submit(self._run_one_collect, c) for c in concurrent_calls]
                for fut in as_completed(futures):
                    yield from fut.result()
        else:
            for c in concurrent_calls:
                yield from self._run_one(c)

        for c in serial_calls:
            yield from self._run_one(c)

    def _run_one(self, call: tuple[str, dict, str]):
        """Generator: yields tool_call IMMEDIATELY, runs dispatch with a live
        spinner showing what's happening, then yields tool_result.

        Yielding tool_call first means the '▸ name  summary' line appears the
        moment the tool starts — not after it finishes — so you can see what
        the agent is doing in real time. The spinner has a 150 ms grace
        period so fast tools don't flash.
        """
        name, args, role = call
        tid = new_id(TaskKind.BASH if name == "run_bash" else TaskKind.TOOL)
        summary = _summarize_args(name, args)
        yield Message("tool_call", {
            "id": tid, "name": name, "args": args, "role": role,
            "summary": summary,
        }, task_id=tid)

        allowed, reason = can_use_tool(name, args, self.state, self.resolver)
        if not allowed:
            err = f"ERROR: permission denied ({reason})"
            yield Message("tool_denied", {"id": tid, "name": name, "reason": reason}, task_id=tid)
            yield Message("tool_result", {
                "id": tid, "name": name, "result": err,
                "ok": False, "first_line": err, "role": role,
            }, task_id=tid)
            return

        ctx = ToolContext(
            **self.state.to_tool_ctx_kwargs(),
            state=self.state,
            engine=self.engine,
            background=self.background,
            tasks=self.tasks,
            teams=self.teams,
            read_cache=getattr(self.engine, "_read_cache", None),
        )

        # Spinner labelled with what we're doing — appears only if dispatch
        # takes long enough to matter (>150ms thanks to Spinner's initial wait).
        # Tools with their own spinner (run_bash) just override the label.
        spin_label = f"{name}  {summary[:60]}" if summary else name
        with ui.Spinner(spin_label):
            try:
                result = dispatch(name, args, ctx)
            except Exception as e:
                result = f"ERROR: {type(e).__name__}: {e}"

        ok = not result.startswith("ERROR")
        first_line = result.splitlines()[0] if result else ""
        yield Message("tool_result", {
            "id": tid, "name": name, "result": result,
            "ok": ok, "first_line": first_line, "role": role,
            "ctx_root": str(ctx.root),
        }, task_id=tid)

    def _run_one_collect(self, call):
        """List-returning variant for the ThreadPoolExecutor path — workers
        can't safely yield across threads, so they materialize the events
        and the main thread drains them in order."""
        return list(self._run_one(call))


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
        compact_schemas: bool = True,
    ) -> None:
        self.client = client
        self.state = state
        self.model = model
        self.temperature = temperature
        self.compact_schemas = compact_schemas
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
        self._recent_results: list[tuple[str, str]] = []
        self._abort_turn = False
        self._plan_shown_this_turn = False
        self._usage = {"input": 0, "output": 0, "ms": 0}
        # Per-turn file-read cache; reset at the start of every submit_message.
        self._read_cache: dict = {}

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
        self._recent_results = []
        self._abort_turn = False
        self._read_cache = {}
        # Reset per-turn state: edit-failure map AND files_read set, so cache
        # nudges fire afresh per user message.
        resets = {}
        if self.state.edit_fails:
            resets["edit_fails"] = {}
        if self.state.files_read:
            resets["files_read"] = set()
        if resets:
            self.state.update(**resets)

        # processUserInput()
        user_msg = process_user_input(prompt, workspace=self.state.workspace, home=self.state.home)
        yield Message("user", {"text": prompt})

        # recordTranscript()
        record_transcript(self.session_id or "", "user", prompt)

        self.messages.append(user_msg)

        # Manage context BEFORE sending: snip + collapse + autoCompact.
        # Pre-check token budget so we can tell the user we're about to
        # compact (otherwise compaction looks invisible until it finishes,
        # and the LLM-summarization variant was slow enough to feel like
        # a hang).
        _pre = sum(len(str(m.get("content") or "")) // 4 for m in self.messages)
        will_compact = _pre > COMPACT_TOKENS
        if will_compact:
            yield Message("info", {
                "text": f"compacting context (~{_pre:,} → ≤{COMPACT_TOKENS:,} tokens)…"
            })
        # LLM-summarization is slow (it's an extra Ollama round-trip on
        # the model that's already struggling). The deterministic bulletize
        # fallback is plenty for context tracking, so we default to that.
        # Power users can flip ollama.llm_summarize on if they want richer
        # summaries at the cost of speed.
        summarize_fn = (
            self._summarize_with_model
            if (self.config and self.config.get("ollama", {}).get("llm_summarize"))
            else None
        )
        for r in manage_context(
            self.messages,
            max_tokens=COMPACT_TOKENS,
            keep_recent=COMPACT_KEEP_RECENT,
            summarize_with_model=summarize_fn,
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
                    # Escalating labels tell the user WHY thinking is taking
                    # this long. After prompt-eval starts, the Ollama API
                    # gives no progress signal, so all we can do is hint at
                    # likely causes by elapsed time.
                    spinner = ui.Spinner(
                        "thinking",
                        escalations=[
                            (10.0, "thinking (large prompt — Ollama is in prompt-eval)"),
                            (30.0, "still thinking — model may not fully fit in VRAM, check /diag"),
                            (60.0, "still thinking — try /new for smaller context, or Ctrl+C to abort"),
                        ],
                    )
                    spinner.start()
                    # Heartbeat for AFTER the first token has arrived. The
                    # spinner above covers prompt-eval; this covers Qwen-style
                    # silent <think> phases between visible tokens that would
                    # otherwise look like a freeze.
                    watchdog = ui.SilenceWatchdog()
                    got_first = False
                    try:
                        for kind, payload in msg.iter():
                            if kind == "delta":
                                if not got_first:
                                    spinner.stop()
                                    watchdog.start()
                                    got_first = True
                                else:
                                    watchdog.ping()
                                yield Message("delta", {"text": payload})
                            elif kind == "done":
                                final_msg = payload
                    except OllamaError as e:
                        # The chunked stream broke mid-response (common behind
                        # proxies that mangle chunked transfers). Fall back to
                        # a single non-streaming request for this turn.
                        spinner.stop()
                        watchdog.stop()
                        yield Message("warn", {
                            "text": f"streaming connection broke ({e}); retrying without streaming."
                        })
                        msg = self._chat_nonstream()
                        if msg is None:
                            yield Message("error", {"text": "non-streaming retry also failed"})
                            return
                        usage = {}
                        final_msg = "RETRIED"  # sentinel: msg is already the message dict
                    finally:
                        spinner.stop()
                        watchdog.stop()
                    if final_msg is None:
                        # chat_stream_assembled always synthesizes a 'done'
                        # now, so this only happens if the iterator was
                        # exhausted with zero chunks (broken socket before
                        # any response). Treat as a no-output turn.
                        yield Message("warn", {
                            "text": "stream produced no chunks — Ollama may have crashed; try /retry"
                        })
                        return
                    if final_msg != "RETRIED":
                        if final_msg.get("truncated"):
                            yield Message("warn", {
                                "text": "response was truncated (stream closed early — Ollama crashed or was killed); using what arrived"
                            })
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

            # Lightweight model probe: tally native vs salvaged tool calls so
            # we can build a per-model behavior profile and (later) auto-tune
            # things like which JSON format to suggest in the system prompt.
            native_tc = len(msg.get("tool_calls") or [])
            salvaged_tc = 1 if (not native_tc and _extract_tool_call(cleaned)) else 0
            if self.config is not None and (native_tc or salvaged_tc):
                profile_key = f"models.{self.model}.profile"
                profile = dict(self.config.get("models", {}).get(self.model, {}).get("profile", {}) or {})
                profile["native_tool_calls"] = int(profile.get("native_tool_calls", 0)) + native_tc
                profile["salvaged_tool_calls"] = int(profile.get("salvaged_tool_calls", 0)) + salvaged_tc
                profile["last_emit_native"] = bool(native_tc)
                set_value(self.config, profile_key, profile)

            if had_fakes:
                yield Message("warn", {"text": "model fabricated tool outputs — asking it to retry."})
                # role=system so this never shows in /resume replay (and the
                # model treats it as an instruction, not user content).
                self.messages.append({
                    "role": "system",
                    "content": (
                        "STOP. Tool outputs are mine to emit. Send a single tool call "
                        "(<tool>{...}</tool> or fenced JSON) and STOP. Do NOT write tool_outputs."
                    ),
                })
                continue

            thinks, content = _extract_thinking(cleaned)
            # In streaming mode the <think> block was already rendered live
            # (dim italic, with the ◦ marker); emitting a thinking event
            # would re-render it as a panel. Only emit for the non-streaming
            # path where there's no live render.
            if not self.stream:
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
            elif narration and self.stream:
                # Streaming mode: emit assistant so the renderer can fall
                # back to the static panel when the stream was invisible
                # (model wrapped everything in <plan>/<think>). The
                # renderer no-ops this if visible deltas already streamed.
                yield Message("assistant", {"text": narration})

            yield from self._execute_and_record(calls)

            if self._abort_turn:
                # Salvage: surface the model's last substantive narration so
                # the user actually sees the diagnosis the model worked out
                # before it got stuck in a re-read loop. Way better than
                # 'Stopped' with no content.
                salvaged = self._last_assistant_narration()
                if salvaged:
                    yield Message("assistant", {"text": salvaged})
                yield Message("done", {
                    "text": "Stopped: the model was stuck repeating the same tool call. "
                            "Try /retry, rephrase the request, or /new for a fresh context.",
                    "usage": dict(self._usage),
                    "session_id": self.session_id,
                })
                return

        yield Message("error", {"text": f"hit tool-call limit ({MAX_TOOL_ITERATIONS}); stopping."})

    # ---- internals ----

    def _chat_once(self, *, yield_deltas: bool):
        """Returns (msg-or-stream, usage_dict). If yield_deltas, returns a
        _StreamGen that the caller iterates; otherwise returns a fully-assembled
        message."""
        api_messages = normalize_messages_for_api(self.messages)
        tools = all_tool_schemas(self.state.tool_groups, compact=self.compact_schemas) if self.state.tools_enabled else None

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

    def _chat_nonstream(self) -> dict | None:
        """Single non-streaming request — used as a fallback when a streamed
        response breaks mid-flight (chunk parse errors, proxy interference)."""
        api_messages = normalize_messages_for_api(self.messages)
        tools = all_tool_schemas(self.state.tool_groups, compact=self.compact_schemas) if self.state.tools_enabled else None
        try:
            with ui.Spinner("retrying (no stream)"):
                return self.client.chat(
                    model=self.model,
                    messages=api_messages,
                    tools=tools,
                    options={"temperature": self.temperature},
                )
        except OllamaError:
            return None

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
        if run >= ARGS_LOOP_THRESHOLD:
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

    def _last_assistant_narration(self) -> str:
        """Find the most recent assistant message in this turn whose content
        is substantive narration (not a bare tool call, not steer/control
        text, not stripped <plan>/<think>). Used to salvage the model's last
        useful thought when a loop forces us to abort the turn."""
        for m in reversed(self.messages):
            if m.get("role") != "assistant":
                continue
            content = (m.get("content") or "").strip()
            # Strip any leftover <plan>/<think> blocks for a cleaner readout.
            content = _PLAN_RX.sub("", content)
            content = _THINK_RX.sub("", content).strip()
            if not content:
                continue
            # Skip messages that are nothing but a JSON tool-call (no prose).
            if content.startswith("{") and content.endswith("}"):
                continue
            return content
        return ""

    def _result_loop_count(self, name: str, result: str) -> int:
        """Track repeated identical (tool, result) pairs this turn — the
        robust loop signal. The model often varies args slightly (path
        slashes, optional start_line) so arg-based detection misses it, but
        an identical RESULT coming back means it's genuinely going in
        circles. Returns how many times this exact (name, result) has been
        seen so far this turn."""
        key = (name, (result or "")[:240])
        self._recent_results.append(key)
        self._recent_results = self._recent_results[-30:]
        return self._recent_results.count(key)

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

                # Result-based loop detection: same (tool, result) coming back
                # repeatedly means the model is stuck — args-based detection
                # misses this because the model varies args slightly.
                #
                # BUT: a SUCCESSFUL edit/write is progress by definition, even
                # though its result string ("OK: edited f.py +1 -1") collides
                # with the previous successful edit's string. Counting those as
                # a loop falsely aborts a turn that's making many small, real
                # changes. Only loop-check mutations when they FAILED.
                _mutating = name in ("edit_file", "write_file", "replace_lines",
                                     "multi_edit", "notebook_edit")
                if _mutating and d.get("ok"):
                    continue
                # CACHED read_file results are internal no-ops — they already
                # nudge the model not to re-read. Counting them as a loop
                # double-punishes the same situation and surfaces a noisy
                # "steering hard" warning for nothing.
                if result.startswith("[CACHED") or (d.get("first_line") or "").startswith("[CACHED"):
                    continue
                seen = self._result_loop_count(name, result)
                if seen == LOOP_THRESHOLD:
                    yield Message("warn", {
                        "text": f"loop: {name} returned the same result {seen}× — steering hard"
                    })
                    # role=system so this never surfaces in /resume replay,
                    # and so the model treats it as a runtime directive.
                    self.messages.append({
                        "role": "system",
                        "content": (
                            f"STOP. You have called {name} and received the EXACT SAME "
                            f"result {seen} times. You are stuck in a loop and making no "
                            f"progress. Do NOT call {name} again. You already have the "
                            f"information you need. Either: (a) make the change the user "
                            f"asked for RIGHT NOW using edit_file or write_file, or (b) if "
                            f"something is genuinely missing, give the user a short final "
                            f"answer explaining what you need. Acting or answering is "
                            f"mandatory on your next turn — no more reads."
                        ),
                    })
                elif seen >= LOOP_ABORT_THRESHOLD:
                    yield Message("warn", {
                        "text": f"loop unbroken after {seen} identical results — ending turn"
                    })
                    self._abort_turn = True
                    return


# ---------------------------------------------------------------- helpers ----

class _StreamGen:
    """Tiny wrapper so submit_message can detect 'this is a streaming generator'."""
    def __init__(self, gen):
        self.gen = gen
    def iter(self):
        return self.gen

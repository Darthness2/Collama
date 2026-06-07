"""Thin compatibility wrapper around QueryEngine.

The real loop now lives in `collama.engine.QueryEngine`. This shim keeps the
old `Agent` API alive for any external caller (tests, scripts) and renders
the QueryEngine event stream to the terminal — which is exactly what the
REPL also does, just inlined here for backwards compatibility.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

from . import ui
from .engine import Message, QueryEngine
from .ollama_client import OllamaClient
from .state import AppState
from .tools import ToolContext  # re-exported so existing imports keep working


class Agent:
    def __init__(
        self,
        client: OllamaClient,
        model: str,
        root: Path,
        yolo: bool = False,
        temperature: float = 0.2,
        on_turn_complete: Optional[Callable] = None,
        tools_enabled: bool = True,
        on_tools_disabled: Optional[Callable] = None,
        config: Optional[dict] = None,
        stream: bool = True,
        compact_schemas: bool = True,
        effort: str = "medium",
    ) -> None:
        self.state = AppState(
            workspace=root,
            home=Path.home(),
            yolo=yolo,
            tools_enabled=tools_enabled,
            effort=effort,
        )
        from .permissions import terminal_plan_resolver, terminal_resolver
        self.engine = QueryEngine(
            client=client,
            state=self.state,
            model=model,
            temperature=temperature,
            config=config,
            permission_resolver=terminal_resolver,
            plan_resolver=terminal_plan_resolver,
            stream=stream,  # stream tokens live so generation is visible
            compact_schemas=compact_schemas,
        )
        self.client = client
        self.on_turn_complete = on_turn_complete
        self.on_tools_disabled = on_tools_disabled

        if on_tools_disabled is not None:
            def _watch(state, key, val):
                if key == "tools_enabled" and val is False:
                    try:
                        on_tools_disabled(self)
                    except Exception:
                        pass
            self.state.subscribe(_watch)

    # --- legacy-shaped properties ---
    @property
    def model(self) -> str:
        return self.engine.model

    @model.setter
    def model(self, value: str) -> None:
        self.engine.model = value

    @property
    def messages(self) -> list[dict]:
        return self.engine.messages

    @property
    def tools_enabled(self) -> bool:
        return self.state.tools_enabled

    @tools_enabled.setter
    def tools_enabled(self, value: bool) -> None:
        self.state.update(tools_enabled=value)
        self.engine.refresh_system_prompt()

    @property
    def ctx(self) -> ToolContext:
        # Build on demand so external mutation lands in state.
        return ToolContext(**self.state.to_tool_ctx_kwargs())

    # --- legacy methods ---
    def reset(self) -> None:
        self.engine.reset()

    def load_messages(self, messages: list[dict]) -> None:
        self.engine.load_messages(messages)

    # Back-compat: callers still using the old private name on Agent.
    def refresh_system_prompt(self) -> None:
        self.engine.refresh_system_prompt()

    _refresh_system_prompt = refresh_system_prompt

    def turn(self, user_input: str) -> str:
        """Run a turn, rendering events to the terminal as they stream.

        Ctrl+C cleanly aborts the in-flight turn: the engine generator is
        closed (which shuts the streaming HTTP connection), spinners are
        stopped, and we return to the prompt with whatever was produced so
        far — the process is not killed.
        """
        rs = _RenderState()
        # Sticky bottom status bar: live elapsed time + rough token tally
        # (output streamed this turn + cumulative context size). Seeded
        # with the pre-turn context size; refreshed after each chat round
        # below so the 'ctx ~N' figure tracks the conversation growing.
        status = ui.StatusBar()
        status.start(ctx_tokens=self.engine.approx_context_tokens())
        gen = self.engine.submit_message(user_input)
        try:
            for msg in gen:
                if msg.kind == "delta":
                    status.add_output_text(msg.data.get("text") or "")
                elif msg.kind == "done":
                    # Refresh the context number once the turn settles —
                    # tool results and the assistant reply have all been
                    # appended to engine.messages by now.
                    status.set_ctx_tokens(self.engine.approx_context_tokens())
                render_event(msg, rs)
        except KeyboardInterrupt:
            gen.close()
            ui.stop_all_spinners()
            status.stop()
            print()
            ui.warn("turn interrupted — back to prompt")
        finally:
            status.stop()
            if self.on_turn_complete:
                try:
                    self.on_turn_complete(self)
                except Exception:
                    pass
        return rs.final_text

    def stream(self, user_input: str) -> Iterator[Message]:
        """Direct access to the event stream (skip terminal rendering)."""
        return self.engine.submit_message(user_input)


@dataclass
class _RenderState:
    """Per-turn rendering bookkeeping passed through render_event."""
    final_text: str = ""
    streaming: bool = False        # currently mid-stream (deltas arriving)
    streamed_any: bool = False     # streamed at least one delta this assistant msg
    streamed_visible: bool = False # stream emitted non-dim, non-empty content
    md: object | None = None       # ui.StreamMarkdown when streaming
    # Pending tool_call event waiting for its tool_result. We defer drawing
    # '▸ name  summary' until the result arrives so CACHED read_file events
    # can suppress BOTH lines (the user shouldn't see internal recovery).
    pending_tool_call: tuple[str, str] | None = None
    # Collapsible-tool run: consecutive calls of the same simple tool
    # (read_file, list_dir, …) — or any run of file edits — fold into ONE
    # line that updates in place ("▸ read 5 files", "▸ edited 5 files").
    run_cat: str | None = None     # run category, or None. "edit" for edits.
    run_count: int = 0             # calls in the current run
    run_fail: int = 0              # how many of them failed
    run_summary: str = ""          # first call's summary (shown when count == 1)
    run_tool: str = ""             # first call's tool name (edit runs, count 1)
    run_adds: int = 0              # total lines added across an edit run
    run_dels: int = 0              # total lines removed across an edit run


# File-mutating tools — collapse into one "▸ edited N files  +A -D" run.
_MUTATING_TOOLS = {"write_file", "edit_file", "multi_edit", "replace_lines"}

# Tools whose consecutive calls collapse into a single tally line.
# Maps tool name -> (verb, singular noun, plural noun).
_RUN_CATEGORIES: dict[str, tuple[str, str, str]] = {
    "read_file": ("read", "file", "files"),
    "list_dir":  ("listed", "directory", "directories"),
    "grep":      ("ran", "search", "searches"),
    "glob":      ("ran", "glob", "globs"),
    "run_bash":  ("ran", "command", "commands"),
}


def _finalize_run(rs: _RenderState) -> None:
    """Close the current collapsible-tool run. The run line is already on
    screen (committed with a trailing newline), so this only resets state."""
    rs.run_cat = None
    rs.run_count = 0
    rs.run_fail = 0
    rs.run_summary = ""
    rs.run_tool = ""
    rs.run_adds = 0
    rs.run_dels = 0


def _parse_diff_stats(result: str) -> tuple[int, int]:
    """Pull the trailing '+adds -dels' counts out of an edit tool result."""
    import re as _re
    m = _re.search(r"\+(\d+)\s+-(\d+)", result or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def _single_edit_line(name: str, summary: str, ok: bool,
                      first_line: str, adds: int, dels: int) -> str:
    """One-line render of a single file edit: '▸ write_file  ~/path  +9 -0'."""
    accent = ui.TEAL_BRIGHT if ok else ui.ERR
    line = ui.color("  ▸ ", accent) + ui.color(name, accent)
    if summary:
        line += ui.color("  " + ui.tilde(summary), ui.MUTED if ok else ui.ERR)
    if ok:
        line += ("  " + ui.color(f"+{adds}", ui.OK)
                 + " " + ui.color(f"-{dels}", ui.ERR))
    else:
        err = first_line.split("ERROR:", 1)[-1].strip()
        line += ui.color(f"  ✗ {err}", ui.ERR)
    return line


def _edit_run_line(rs: _RenderState) -> str:
    """Collapsed render of 2+ edits: '▸ edited 5 files  +363 -266'."""
    txt = f"edited {rs.run_count} files"
    line = ui.color("  ▸ ", ui.TEAL_BRIGHT) + ui.color(txt, ui.TEAL_BRIGHT)
    line += ("  " + ui.color(f"+{rs.run_adds}", ui.OK)
             + " " + ui.color(f"-{rs.run_dels}", ui.ERR))
    if rs.run_fail:
        line += ui.color(f"  ({rs.run_fail} failed)", ui.ERR)
    return line


def _run_line(rs: _RenderState) -> str:
    """Build the single line that represents the current read/list/grep run."""
    verb, _singular, plural = _RUN_CATEGORIES[rs.run_cat]  # type: ignore[index]
    failed = rs.run_fail > 0
    accent = ui.ERR if failed else ui.TEAL_BRIGHT
    if rs.run_count == 1:
        # Looks like a normal tool_call line: "▸ read_file  ~/path".
        # run_bash is a deliberate exception — the command itself is noise.
        # Whether it succeeded is the only thing worth showing, and the
        # accent color already carries that signal (red on fail, teal on ok).
        line = ui.color("  ▸ ", accent) + ui.color(rs.run_cat or "", accent)
        if rs.run_cat == "run_bash":
            return line
        detail = ui.tilde(rs.run_summary)
        if detail:
            line += ui.color(f"  {detail}", ui.ERR if failed else ui.MUTED)
        return line
    txt = f"{verb} {rs.run_count} {plural}"
    line = ui.color("  ▸ ", accent) + ui.color(txt, accent)
    if failed:
        line += ui.color(f"  ({rs.run_fail} failed)", ui.ERR)
    return line


def _end_stream_line(rs: _RenderState) -> None:
    if rs.streaming:
        if rs.md is not None:
            rs.md.flush()  # type: ignore[attr-defined]
            rs.streamed_visible = bool(getattr(rs.md, "visible_emitted", False))
        sys.stdout.write("\n")
        sys.stdout.flush()
        rs.streaming = False
        rs.md = None


def _stream_emit(s: str) -> None:
    sys.stdout.write(s)
    sys.stdout.flush()


def render_event(event: Message, rs: _RenderState) -> None:
    """Render a single QueryEngine event to the terminal, updating `rs`."""
    k, d = event.kind, event.data

    # Any non-delta event ends an in-progress streamed line first.
    if k != "delta":
        _end_stream_line(rs)

    # Any event that isn't part of a tool run closes an open run. tool_call
    # is exempt (it belongs to the run); tool_result manages the run itself.
    if k not in ("tool_call", "tool_result"):
        _finalize_run(rs)

    if k == "thinking":
        ui.thinking(d["text"])
    elif k == "plan":
        ui.plan(d["steps"])
    elif k == "plan_review":
        # Approve-then-execute: show the plan as a highlighted card. The
        # interactive approve/reject prompt follows (terminal_plan_resolver).
        ui.panel(d["text"], title="PLAN — approve to apply", markdown=True)
    elif k == "narration":
        print(ui.color("  ▪ ", ui.TEAL_DIM) + ui.color(d["text"], ui.MUTED))
    elif k == "delta":
        # Live token stream — buffer per-line and render markdown so **bold**,
        # *italic*, `code`, and # headers come out styled instead of as
        # literal characters.
        if not rs.streaming:
            rs.md = ui.StreamMarkdown(
                emit=_stream_emit,
                first_prefix=ui.color("  ● ", ui.TEAL_BRIGHT),
                cont_prefix="    ",
                # Live <think> rendering: dim italic, '◦' marker, so the
                # model's internal reasoning is visible while it thinks.
                dim_first_prefix=ui.color("  ◦ ", ui.SOFT),
                dim_cont_prefix=ui.color("    ", ui.SOFT),
            )
            rs.streaming = True
        rs.md.feed(d["text"])  # type: ignore[union-attr]
        rs.streamed_any = True
    elif k == "assistant":
        rs.final_text = d["text"]
        # Skip the panel only when the stream actually showed visible text.
        # If the model wrapped its entire response in <plan>/<think> tags,
        # the live stream emitted nothing the user could read — fall back
        # to the static panel so the answer isn't invisible.
        if rs.streamed_any and rs.streamed_visible:
            rs.streamed_any = False
            rs.streamed_visible = False
        else:
            rs.streamed_any = False
            rs.streamed_visible = False
            if d["text"].strip():
                ui.assistant(d["text"])
    elif k == "tool_call":
        # Defer the ▸ line — we may want to suppress it entirely if the
        # result is a cache nudge (pure internal recovery).
        rs.pending_tool_call = (d["name"], d["summary"])
    elif k == "tool_result":
        name = d.get("name") or ""
        result = d.get("result") or ""
        first_line = d.get("first_line") or ""
        ok = bool(d["ok"])
        # If the result is a CACHED nudge, drop BOTH the tool_call line and
        # the result line. The user already saw the file the first time
        # around; redundant reads are internal-only.
        if first_line.startswith("[CACHED"):
            rs.pending_tool_call = None
            return
        # Pull the deferred tool_call summary (we now know the result is real).
        summary = ""
        if rs.pending_tool_call is not None:
            _, summary = rs.pending_tool_call
            rs.pending_tool_call = None

        # Collapsible tools (read_file, list_dir, grep, glob, run_bash) fold
        # consecutive calls into one self-updating "▸ read 5 files" line.
        if name in _RUN_CATEGORIES:
            is_tty = sys.stdout.isatty()
            if rs.run_cat == name and is_tty:
                # Continue the run: bump counts, rewrite the line in place.
                rs.run_count += 1
                rs.run_fail += 0 if ok else 1
                sys.stdout.write("\033[A\r\033[2K" + _run_line(rs) + "\n")
                sys.stdout.flush()
            else:
                # Start a fresh run (also the non-TTY path: one line per call).
                _finalize_run(rs)
                rs.run_cat = name
                rs.run_count = 1
                rs.run_fail = 0 if ok else 1
                rs.run_summary = summary
                print(_run_line(rs))
            return

        # File-mutating tools collapse into one "▸ edited N files +A -D" run,
        # exactly like read_file. A lone edit shows "▸ write_file ~/path +9 -0".
        if name in _MUTATING_TOOLS:
            ok = bool(d["ok"])
            adds, dels = _parse_diff_stats(result) if ok else (0, 0)
            is_tty = sys.stdout.isatty()
            if rs.run_cat == "edit" and is_tty:
                rs.run_count += 1
                rs.run_fail += 0 if ok else 1
                rs.run_adds += adds
                rs.run_dels += dels
                sys.stdout.write("\033[A\r\033[2K" + _edit_run_line(rs) + "\n")
                sys.stdout.flush()
            else:
                _finalize_run(rs)
                rs.run_cat = "edit"
                rs.run_count = 1
                rs.run_fail = 0 if ok else 1
                rs.run_adds = adds
                rs.run_dels = dels
                print(_single_edit_line(name, summary, ok, first_line, adds, dels))
            return

        # Truly non-collapsible tool — close any run, render normally.
        _finalize_run(rs)
        ui.tool_call(name, summary)
        ui.tool_result(first_line[:160], ok=d["ok"])
    elif k == "info":
        ui.info(d["text"])
    elif k == "warn":
        ui.warn(d["text"])
    elif k == "error":
        ui.error(d["text"])
    elif k == "compact":
        strategy = d.get("strategy", "compact")
        ui.info(f"context {strategy}: {d['before']} → {d['after']} approx tokens")
    elif k == "tool_denied":
        # Flush any pending tool_call line so the denial reads sensibly.
        if rs.pending_tool_call is not None:
            pname, psummary = rs.pending_tool_call
            ui.tool_call(pname, psummary)
            rs.pending_tool_call = None
        ui.warn(f"permission denied: {d['name']} ({d['reason']})")
    elif k == "done":
        usage = d.get("usage", {})
        if any(usage.values()):
            print(ui.color(
                f"  ↳ tokens in/out {usage.get('input', 0)}/{usage.get('output', 0)}"
                f"  · {usage.get('ms', 0)}ms",
                ui.SOFT,
            ))

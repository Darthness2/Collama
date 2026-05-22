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
    ) -> None:
        self.state = AppState(
            workspace=root,
            home=Path.home(),
            yolo=yolo,
            tools_enabled=tools_enabled,
        )
        from .permissions import terminal_resolver
        self.engine = QueryEngine(
            client=client,
            state=self.state,
            model=model,
            temperature=temperature,
            config=config,
            permission_resolver=terminal_resolver,
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
        gen = self.engine.submit_message(user_input)
        try:
            for msg in gen:
                render_event(msg, rs)
        except KeyboardInterrupt:
            gen.close()
            ui.stop_all_spinners()
            print()
            ui.warn("turn interrupted — back to prompt")
        finally:
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
    # (read_file, list_dir, …) fold into ONE line that updates in place —
    # "▸ read 5 files" instead of ten ▸/✓ pairs.
    run_cat: str | None = None     # tool name of the current run, or None
    run_count: int = 0             # calls in the current run
    run_fail: int = 0              # how many of them failed
    run_summary: str = ""          # first call's summary (shown when count == 1)


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


def _run_line(rs: _RenderState) -> str:
    """Build the single line that represents the current run."""
    verb, _singular, plural = _RUN_CATEGORIES[rs.run_cat]  # type: ignore[index]
    failed = rs.run_fail > 0
    accent = ui.ERR if failed else ui.TEAL_BRIGHT
    if rs.run_count == 1:
        # Looks like a normal tool_call line: "▸ read_file  ~/path".
        line = ui.color("  ▸ ", accent) + ui.color(rs.run_cat or "", accent)
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

        # Non-collapsible tool — close any run, then render normally.
        _finalize_run(rs)
        ui.tool_call(name, summary)
        # Special-case file edits: show only the file + colored +adds/-dels.
        if d["ok"] and name in ("write_file", "edit_file", "multi_edit"):
            import re as _re_edit
            m = _re_edit.match(r"OK:\s+(\w+)\s+(.+?)\s+\+(\d+)\s+-(\d+)\s*$", result)
            m_multi = _re_edit.match(
                r"OK:\s+applied\s+(\d+)\s+edits?\s+to\s+(.+?)\s+\+(\d+)\s+-(\d+)", result
            )
            if m_multi:
                n_edits, path, adds, dels = m_multi.groups()
                print(
                    ui.color("    ✓", ui.OK)
                    + " " + ui.color(f"multi_edit ({n_edits})", ui.TEAL_BRIGHT)
                    + " " + ui.color(path, ui.SURFACE)
                    + "  " + ui.color(f"+{adds}", ui.OK)
                    + " " + ui.color(f"-{dels}", ui.ERR)
                )
                return
            if m:
                op, path, adds, dels = m.group(1), m.group(2), m.group(3), m.group(4)
                mark = ui.color("    ✓", ui.OK)
                print(
                    mark + " " + ui.color(op, ui.TEAL_BRIGHT)
                    + " " + ui.color(path, ui.SURFACE)
                    + "  " + ui.color(f"+{adds}", ui.OK)
                    + " " + ui.color(f"-{dels}", ui.ERR)
                )
                return
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

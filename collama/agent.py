"""Thin compatibility wrapper around QueryEngine.

The real loop now lives in `collama.engine.QueryEngine`. This shim keeps the
old `Agent` API alive for any external caller (tests, scripts) and renders
the QueryEngine event stream to the terminal — which is exactly what the
REPL also does, just inlined here for backwards compatibility.
"""
from __future__ import annotations

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
            stream=False,  # REPL keeps a single spinner; no in-line deltas
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

    def turn(self, user_input: str) -> str:
        """Run a turn, rendering events to the terminal as they stream."""
        final = ""
        for msg in self.engine.submit_message(user_input):
            final = render_event(msg, final)
        if self.on_turn_complete:
            try:
                self.on_turn_complete(self)
            except Exception:
                pass
        return final

    def stream(self, user_input: str) -> Iterator[Message]:
        """Direct access to the event stream (skip terminal rendering)."""
        return self.engine.submit_message(user_input)


def render_event(event: Message, final_text: str) -> str:
    """Render a single QueryEngine event to the terminal. Returns updated final text."""
    k, d = event.kind, event.data
    if k == "thinking":
        ui.thinking(d["text"])
    elif k == "plan":
        ui.plan(d["steps"])
    elif k == "narration":
        print(ui.color("  ▪ ", ui.TEAL_DIM) + ui.color(d["text"], ui.MUTED))
    elif k == "assistant":
        ui.assistant(d["text"])
        final_text = d["text"]
    elif k == "tool_call":
        ui.tool_call(d["name"], d["summary"])
    elif k == "tool_result":
        ui.tool_result(d["first_line"][:160], ok=d["ok"])
    elif k == "warn":
        ui.warn(d["text"])
    elif k == "error":
        ui.error(d["text"])
    elif k == "compact":
        strategy = d.get("strategy", "compact")
        ui.info(f"context {strategy}: {d['before']} → {d['after']} approx tokens")
    elif k == "tool_denied":
        ui.warn(f"permission denied: {d['name']} ({d['reason']})")
    elif k == "delta":
        # SDK consumers can render incrementally; the REPL ignores deltas.
        pass
    elif k == "done":
        usage = d.get("usage", {})
        if any(usage.values()):
            print(ui.color(
                f"  ↳ tokens in/out {usage.get('input', 0)}/{usage.get('output', 0)}"
                f"  · {usage.get('ms', 0)}ms",
                ui.SOFT,
            ))
    return final_text

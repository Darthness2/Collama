"""Input prompt with slash-command auto-completion.

Uses prompt_toolkit if available — gives a real popup as you type `/`.
Falls back to readline tab-completion, then plain input().
"""
from __future__ import annotations

# (name, hint) pairs shown in the popup.
SLASH_COMMANDS: list[tuple[str, str]] = [
    ("/help", "show available commands"),
    ("/tools", "list tools the model can call"),
    ("/groups", "show/change which tool groups are sent to the model"),
    ("/tools-on", "force native tool calls for this model"),
    ("/tools-off", "force text-protocol tool fallback for this model"),
    ("/cd", "show or change the workspace directory"),
    ("/tasks", "list persistent tasks"),
    ("/jobs", "list background jobs"),
    ("/wt", "show worktree stack"),
    ("/teams", "list teams and teammates"),
    ("/tick", "coordinator tick — process mailboxes"),
    ("/plan", "toggle plan mode (read-only)"),
    ("/todo", "view or modify the session todo list"),
    ("/brief", "list briefs, or print one"),
    ("/stream", "toggle token streaming on/off"),
    ("/insecure", "toggle SSL verification (school/corp MITM proxies)"),
    ("/diag", "print model / workspace / tools / github status"),
    ("/model", "show or switch model (saved)"),
    ("/preset", "show/save/clear per-model presets"),
    ("/models", "list installed Ollama models"),
    ("/host", "show or change Ollama host (saved)"),
    ("/config", "show current config (token redacted)"),
    ("/set", "set a config value: /set <key> <value>"),
    ("/login", "/login github <token>"),
    ("/logout", "/logout github"),
    ("/whoami", "show authenticated GitHub user"),
    ("/clear", "reset conversation history"),
    ("/diff", "show file edits this session"),
    ("/undo", "revert the most recent file edit"),
    ("/retry", "re-run your last message"),
    ("/new", "start a new conversation"),
    ("/resume", "list/resume saved conversations"),
    ("/sessions", "list saved conversations"),
    ("/save", "force-save / set title of current conversation"),
    ("/rename", "rename the current conversation"),
    ("/delete", "delete a saved conversation"),
    ("/yolo", "toggle auto-approve for tool calls"),
    ("/exit", "leave Collama"),
    ("/quit", "leave Collama"),
]


def _build_pt_session():
    """Try to build a prompt_toolkit PromptSession with a slash completer.

    Returns (session, error). On success error is None; on failure session
    is None and error explains why (so the caller can tell the user instead
    of silently degrading to readline).
    """
    try:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.completion import Completer, Completion
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.shortcuts import CompleteStyle
        from prompt_toolkit.styles import Style
    except ImportError as e:
        return None, f"prompt_toolkit not installed ({e}). Run: pip install prompt_toolkit"
    except Exception as e:  # pragma: no cover
        return None, f"prompt_toolkit import failed: {type(e).__name__}: {e}"

    from .config import config_dir

    class SlashCompleter(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if not text.startswith("/"):
                return
            if " " in text:
                return
            for name, hint in SLASH_COMMANDS:
                if name.startswith(text):
                    yield Completion(
                        name,
                        start_position=-len(text),
                        display=name,
                        display_meta=hint,
                    )

    history_path = config_dir() / "history"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    style = Style.from_dict({
        "completion-menu":               "bg:#0b3b3a #b8e6e1",
        "completion-menu.completion":    "bg:#0b3b3a #b8e6e1",
        "completion-menu.completion.current": "bg:#1abc9c #002b2b bold",
        "completion-menu.meta":          "bg:#0b3b3a #5fbfb5",
        "completion-menu.meta.current":  "bg:#1abc9c #002b2b",
        "scrollbar.background":          "bg:#0b3b3a",
        "scrollbar.button":              "bg:#1abc9c",
    })

    # NOTE: AutoSuggestFromHistory was previously enabled here. It shows
    # greyed-out "ghost" text past the cursor matching prior inputs, which
    # users mistook for their typed text disappearing. Disabled by default.
    try:
        session = PromptSession(
            completer=SlashCompleter(),
            complete_while_typing=True,          # show the popup as you type
            complete_style=CompleteStyle.MULTI_COLUMN,
            reserve_space_for_menu=8,            # always leave room for the menu
            history=FileHistory(str(history_path)),
            style=style,
        )
    except Exception as e:  # pragma: no cover
        return None, f"prompt_toolkit session build failed: {type(e).__name__}: {e}"
    return session, None


def _install_readline_fallback() -> bool:
    try:
        import readline
    except ImportError:
        return False

    names = [c[0] for c in SLASH_COMMANDS]

    def completer(text, state):
        if not text.startswith("/"):
            return None
        matches = [n for n in names if n.startswith(text)]
        return matches[state] if state < len(matches) else None

    readline.set_completer(completer)
    readline.parse_and_bind("tab: complete")
    # Don't break command words on '/'.
    readline.set_completer_delims(" \t\n")
    return True


class Prompt:
    def __init__(self) -> None:
        self._pt, self._pt_error = _build_pt_session()
        if self._pt is None:
            self._readline = _install_readline_fallback()
        else:
            self._readline = False

    @property
    def backend(self) -> str:
        """Which input backend is active: 'prompt_toolkit' gives the live
        slash-command popup; 'readline' only does TAB completion; 'plain'
        has neither (prompt_toolkit not installed)."""
        if self._pt is not None:
            return "prompt_toolkit"
        if self._readline:
            return "readline"
        return "plain"

    @property
    def status_note(self) -> str | None:
        """A human-readable note when the live popup isn't available."""
        if self._pt is not None:
            return None
        reason = self._pt_error or "prompt_toolkit unavailable"
        if self._readline:
            return (f"slash-command popup OFF — {reason}. "
                    f"TAB still completes /commands.")
        return f"slash-command popup OFF — {reason}."

    def ask(self, prompt: str) -> str:
        if self._pt is not None:
            # prompt_toolkit has its own renderer; on Windows it treats raw
            # ANSI escapes in the prompt string as literal text. Wrapping
            # with ANSI(...) tells it to parse our escapes.
            try:
                from prompt_toolkit.formatted_text import ANSI
                return self._pt.prompt(ANSI(prompt))
            except Exception:
                return self._pt.prompt(prompt)
        # plain input(); readline (if installed) decorates it with TAB-completion
        return input(prompt)

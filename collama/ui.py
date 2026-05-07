"""Terminal UI helpers — teal palette, blocky panels, no external deps."""
from __future__ import annotations

import os
import shutil
import sys
import textwrap
import threading
import time

# ANSI base
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
ITALIC = "\033[3m"

# Standard 16-color codes (kept for fallbacks)
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
GRAY = "\033[90m"

# Teal palette (256-color codes — degrade gracefully if unsupported).
TEAL = "\033[38;5;43m"          # main accent
TEAL_BRIGHT = "\033[38;5;49m"   # highlights
TEAL_DIM = "\033[38;5;30m"      # subtle borders
SURFACE = "\033[38;5;252m"      # body text
MUTED = "\033[38;5;245m"        # secondary
SOFT = "\033[38;5;240m"         # very muted
WARN = "\033[38;5;215m"         # soft amber
ERR = "\033[38;5;203m"          # soft coral
OK = "\033[38;5;78m"            # mint


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def color(text: str, c: str) -> str:
    if not _supports_color():
        return text
    return f"{c}{text}{RESET}"


# ---------- screen / sizing ----------

def clear_screen() -> None:
    if not sys.stdout.isatty():
        return
    sys.stdout.write("\033[3J\033[2J\033[H")
    sys.stdout.flush()


def width() -> int:
    try:
        return min(shutil.get_terminal_size((80, 24)).columns, 120)
    except OSError:
        return 80


def _strip_ansi(s: str) -> str:
    out = []
    i = 0
    while i < len(s):
        if s[i] == "\033":
            while i < len(s) and s[i] != "m":
                i += 1
            i += 1
            continue
        out.append(s[i])
        i += 1
    return "".join(out)


def _vlen(s: str) -> int:
    return len(_strip_ansi(s))


def _pad(s: str, w: int) -> str:
    return s + " " * max(0, w - _vlen(s))


# ---------- panels / boxes ----------

# Box characters
_BX = {
    "single":  ("┌", "┐", "└", "┘", "─", "│"),
    "double":  ("╔", "╗", "╚", "╝", "═", "║"),
    "round":   ("╭", "╮", "╰", "╯", "─", "│"),
    "thick":   ("┏", "┓", "┗", "┛", "━", "┃"),
}


def panel(body: str | list[str], title: str = "", style: str = "round",
          color_c: str = TEAL_DIM, title_c: str = TEAL_BRIGHT,
          inner_pad: int = 1) -> None:
    """Print a bordered panel containing wrapped body text."""
    tl, tr, bl, br, h, v = _BX[style]
    w = width()
    inner = w - 2 - inner_pad * 2

    # body lines (allow either a string or a list of pre-formatted lines)
    if isinstance(body, str):
        lines: list[str] = []
        for raw in body.splitlines() or [""]:
            if not raw.strip():
                lines.append("")
                continue
            # preserve ANSI by wrapping the visible text only
            lines.extend(textwrap.wrap(raw, width=inner) or [""])
    else:
        lines = list(body)

    # title bar
    if title:
        title_visible = f" {title} "
        bar_len = max(0, w - 2 - _vlen(title_visible))
        top = (color(tl + h * 2, color_c) + color(title_visible, title_c) +
               color(h * (bar_len - 2) + tr, color_c))
    else:
        top = color(tl + h * (w - 2) + tr, color_c)

    bottom = color(bl + h * (w - 2) + br, color_c)

    print(top)
    for line in lines:
        body_text = " " * inner_pad + _pad(line, inner) + " " * inner_pad
        print(color(v, color_c) + body_text + color(v, color_c))
    print(bottom)


def hr(char: str = "─", c: str = TEAL_DIM) -> None:
    print(color(char * width(), c))


# ---------- spinner ----------

_SPIN_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


class Spinner:
    """Tiny non-blocking status spinner. Use as a context manager.

    Renders as:    ⠋ thinking…   (0.4s)
    On stop, clears the line so the next print is clean.
    """

    def __init__(self, label: str = "thinking", color_c: str = TEAL_BRIGHT) -> None:
        self.label = label
        self.color_c = color_c
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def set_label(self, label: str) -> None:
        self.label = label

    def start(self) -> None:
        if not sys.stdout.isatty() or self._thread is not None:
            return
        self._stop.clear()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        # Clear the spinner line.
        sys.stdout.write("\r\033[2K")
        sys.stdout.flush()

    def _run(self) -> None:
        i = 0
        while not self._stop.is_set():
            frame = _SPIN_FRAMES[i % len(_SPIN_FRAMES)]
            elapsed = time.monotonic() - self._t0
            timer = f"({elapsed:0.1f}s)"
            line = (
                "  "
                + color(frame, self.color_c)
                + " "
                + color(self.label + "…", MUTED)
                + "  "
                + color(timer, SOFT)
            )
            sys.stdout.write("\r\033[2K" + line)
            sys.stdout.flush()
            i += 1
            # Wait in small increments so .stop() reacts quickly.
            self._stop.wait(0.08)

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


# ---------- log helpers ----------

def info(msg: str) -> None:
    print(color("  ℹ ", TEAL) + color(msg, SURFACE))


def warn(msg: str) -> None:
    print(color("  ⚠ ", WARN) + color(msg, WARN))


def error(msg: str) -> None:
    print(color("  ✖ ", ERR) + color(msg, ERR), file=sys.stderr)


def assistant(msg: str) -> None:
    """Render a final assistant answer in a soft panel."""
    panel(msg, title="answer", style="round", color_c=TEAL_DIM, title_c=TEAL_BRIGHT)


def thinking(msg: str) -> None:
    """Render <think>…</think> content dim-italic in a panel."""
    rendered = color(msg, MUTED + ITALIC) if _supports_color() else msg
    panel(rendered, title="thinking", style="round", color_c=SOFT, title_c=MUTED)


def plan(items: list[str]) -> None:
    """Render a numbered plan in a panel."""
    if not items:
        return
    body = []
    for i, step in enumerate(items, 1):
        body.append(color(f"  {i:>2}.", TEAL) + " " + color(step, SURFACE))
    panel(body, title="plan", style="thick", color_c=TEAL, title_c=TEAL_BRIGHT)


def tool_call(name: str, summary: str) -> None:
    head = color("  ▸ ", TEAL_BRIGHT) + color(name, TEAL_BRIGHT)
    if summary:
        head += color(f"  {summary}", MUTED)
    print(head)


def tool_result(summary: str, ok: bool = True) -> None:
    mark = color("    ✓", OK) if ok else color("    ✗", ERR)
    print(mark + color(f" {summary}", MUTED))


# ---------- banner ----------

# Block-letter rendering of "COLLAMA". 5 rows, ~8 cols per glyph.
_GLYPHS = {
    "C": [
        " ████ ",
        "██  ██",
        "██    ",
        "██  ██",
        " ████ ",
    ],
    "O": [
        " ████ ",
        "██  ██",
        "██  ██",
        "██  ██",
        " ████ ",
    ],
    "L": [
        "██    ",
        "██    ",
        "██    ",
        "██    ",
        "██████",
    ],
    "A": [
        " ████ ",
        "██  ██",
        "██████",
        "██  ██",
        "██  ██",
    ],
    "M": [
        "██   ██",
        "███ ███",
        "███████",
        "██ █ ██",
        "██   ██",
    ],
    " ": [
        "  ",
        "  ",
        "  ",
        "  ",
        "  ",
    ],
}


def _render_logo(text: str = "COLLAMA") -> list[str]:
    rows = ["", "", "", "", ""]
    for ch in text.upper():
        glyph = _GLYPHS.get(ch, _GLYPHS[" "])
        for i, line in enumerate(glyph):
            rows[i] += line + "  "
    return [r.rstrip() for r in rows]


def banner(model: str, cwd: str, tools_enabled: bool = True) -> None:
    clear_screen()
    w = width()
    tl, tr, bl, br, h, v = _BX["double"]

    # Top double border
    print(color(tl + h * (w - 2) + tr, TEAL))

    # Empty padding row
    print(color(v, TEAL) + " " * (w - 2) + color(v, TEAL))

    # Logo, centered, in bright teal
    for row in _render_logo("COLLAMA"):
        pad = max(0, (w - 2 - len(row)) // 2)
        line = " " * pad + row
        line = _pad(line, w - 2)
        print(color(v, TEAL) + color(line, TEAL_BRIGHT) + color(v, TEAL))

    # Tagline row
    tagline = "a local terminal coding agent · powered by ollama"
    pad = max(0, (w - 2 - len(tagline)) // 2)
    print(color(v, TEAL) + color(_pad(" " * pad + tagline, w - 2), MUTED) + color(v, TEAL))

    # Empty padding row
    print(color(v, TEAL) + " " * (w - 2) + color(v, TEAL))

    # Bottom double border
    print(color(bl + h * (w - 2) + br, TEAL))

    # Status panel under the banner
    badge = "tools: native" if tools_enabled else "tools: text-protocol fallback"
    badge_c = OK if tools_enabled else WARN
    rows = [
        color("model    ", MUTED) + color(model, TEAL_BRIGHT),
        color("cwd      ", MUTED) + color(cwd, SURFACE),
        color("status   ", MUTED) + color(badge, badge_c),
        color("hint     ", MUTED) + color("type / for commands · /exit to quit", TEAL_DIM),
    ]
    panel(rows, title="session", style="thick", color_c=TEAL_DIM, title_c=TEAL_BRIGHT)

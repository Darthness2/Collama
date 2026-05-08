"""Terminal UI helpers — teal palette, blocky panels, no external deps."""
from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap
import threading
import time


# ----- Windows ANSI passthrough --------------------------------------------
# Older Windows shells don't process ANSI escape sequences by default; the
# escapes leak as raw text (`^[[38;5;49m`). Enable Virtual Terminal Processing
# on stdout/stderr so our color/cursor codes work in cmd.exe / PowerShell /
# Windows Terminal without needing colorama.

def _enable_windows_ansi() -> None:
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        for handle_id in (-11, -12):  # STD_OUTPUT_HANDLE, STD_ERROR_HANDLE
            handle = kernel32.GetStdHandle(handle_id)
            if not handle or handle == ctypes.c_void_p(-1).value:
                continue
            mode = ctypes.c_ulong()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                kernel32.SetConsoleMode(
                    handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING,
                )
    except Exception:
        pass


_enable_windows_ansi()


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


# ---------- markdown rendering ----------

_MD_FENCE_RX = re.compile(r"```([A-Za-z0-9_+\-]*)\n(.*?)```", re.DOTALL)
_MD_INLINE_CODE_RX = re.compile(r"`([^`\n]+)`")
_MD_BOLD_RX = re.compile(r"\*\*([^*\n]+)\*\*")
_MD_ITALIC_AST_RX = re.compile(r"(?<![*\w])\*([^*\n]+?)\*(?!\w)")
_MD_ITALIC_UND_RX = re.compile(r"(?<!\w)_([^_\n]+?)_(?!\w)")
_MD_HEADER_RX = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_MD_BULLET_RX = re.compile(r"^(\s*)[-*]\s+", re.MULTILINE)
_MD_LINK_RX = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")


def render_markdown(text: str) -> str:
    """Convert a small subset of CommonMark to ANSI-styled text.

    Handles fenced code blocks, inline code, **bold**, *italic*/_italic_,
    headers (#), bullets, and inline links. Anything fancier passes through
    as plain text.
    """
    if not _supports_color():
        return text

    # Fenced code blocks first — pull them out and replace with placeholders
    # so we don't rewrite their interior with bold/italic rules.
    placeholders: list[str] = []

    def _stash_block(m: re.Match) -> str:
        body = m.group(2)
        rendered_lines = []
        for line in body.splitlines():
            rendered_lines.append(color("  │ ", TEAL_DIM) + color(line, TEAL_BRIGHT))
        rendered = "\n".join(rendered_lines)
        placeholders.append(rendered)
        return f"\x00BLOCK{len(placeholders) - 1}\x00"

    out = _MD_FENCE_RX.sub(_stash_block, text)

    def _stash_inline(m: re.Match) -> str:
        rendered = color(m.group(1), TEAL_BRIGHT + BOLD)
        placeholders.append(rendered)
        return f"\x00INL{len(placeholders) - 1}\x00"

    out = _MD_INLINE_CODE_RX.sub(_stash_inline, out)

    # Headers (one per line).
    def _header(m: re.Match) -> str:
        level = len(m.group(1))
        body = m.group(2)
        if level == 1:
            return color("━━ " + body + " ━━", TEAL_BRIGHT + BOLD)
        if level == 2:
            return color("●  " + body, TEAL + BOLD)
        return color("·  " + body, TEAL_DIM + BOLD)

    out = _MD_HEADER_RX.sub(_header, out)

    # Bullets.
    out = _MD_BULLET_RX.sub(lambda m: m.group(1) + color("• ", TEAL), out)

    # Bold and italic. (Order matters — bold first.)
    out = _MD_BOLD_RX.sub(lambda m: color(m.group(1), BOLD), out)
    out = _MD_ITALIC_AST_RX.sub(lambda m: color(m.group(1), ITALIC), out)
    out = _MD_ITALIC_UND_RX.sub(lambda m: color(m.group(1), ITALIC), out)

    # Inline links: [text](url) → text (url, dimmed).
    out = _MD_LINK_RX.sub(
        lambda m: color(m.group(1), TEAL_BRIGHT) + color(f" ({m.group(2)})", SOFT),
        out,
    )

    # Restore stashed code.
    def _restore(m: re.Match) -> str:
        return placeholders[int(m.group(1))]

    out = re.sub(r"\x00BLOCK(\d+)\x00", _restore, out)
    out = re.sub(r"\x00INL(\d+)\x00", _restore, out)
    return out


def _wrap_visible(text: str, width: int) -> list[str]:
    """Wrap `text` to `width` *visible* columns, leaving ANSI escapes intact."""
    out: list[str] = []
    for raw in text.splitlines() or [""]:
        if not raw.strip():
            out.append("")
            continue
        if _vlen(raw) <= width:
            out.append(raw)
            continue
        # Greedy word-wrap; break on spaces. Tracks visible length only.
        words = raw.split(" ")
        line = ""
        for w in words:
            wlen = _vlen(w)
            if not line:
                line = w
                continue
            if _vlen(line) + 1 + wlen <= width:
                line = line + " " + w
            else:
                out.append(line)
                line = w
        if line:
            out.append(line)
    return out


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
          inner_pad: int = 1, markdown: bool = False) -> None:
    """Print a bordered panel containing wrapped body text.

    If `markdown=True`, the body is run through render_markdown() and
    wrapped with a visible-width-aware wrapper that preserves ANSI escapes.
    """
    tl, tr, bl, br, h, v = _BX[style]
    w = width()
    inner = w - 2 - inner_pad * 2

    # body lines (allow either a string or a list of pre-formatted lines)
    if isinstance(body, str):
        if markdown:
            rendered = render_markdown(body)
            lines = _wrap_visible(rendered, inner)
        else:
            lines = []
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

# Track any live spinner so we can force-stop it before reading user input.
_active_spinners: list["Spinner"] = []


def stop_all_spinners() -> None:
    for s in list(_active_spinners):
        try:
            s.stop()
        except Exception:
            pass


def prepare_for_input() -> None:
    """Call right before reading user input: stop spinners, show cursor, flush."""
    stop_all_spinners()
    if sys.stdout.isatty():
        sys.stdout.write("\033[?25h")  # show cursor
        sys.stdout.flush()


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
        _active_spinners.append(self)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join()
        self._thread = None
        if self in _active_spinners:
            _active_spinners.remove(self)
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
    """Render a final assistant answer in a soft panel, with markdown styling."""
    panel(msg, title="answer", style="round", color_c=TEAL_DIM, title_c=TEAL_BRIGHT,
          markdown=True)


def thinking(msg: str) -> None:
    """Render <think>…</think> content dim-italic in a panel, with markdown."""
    panel(msg, title="thinking", style="round", color_c=SOFT, title_c=MUTED,
          markdown=True)


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

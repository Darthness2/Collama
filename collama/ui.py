"""Terminal UI helpers — teal palette, no external deps for output."""
from __future__ import annotations

import os
import shutil
import sys

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
TEAL_DIM = "\033[38;5;30m"      # subtle
TEAL_BG = "\033[48;5;23m"       # bg accent for the banner
SURFACE = "\033[38;5;245m"      # body text in dim mode (light gray)
MUTED = "\033[38;5;240m"        # secondary muted text
WARN = "\033[38;5;215m"         # soft amber, fits the teal palette
ERR = "\033[38;5;203m"          # soft coral red
OK = "\033[38;5;78m"            # mint green


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def color(text: str, c: str) -> str:
    if not _supports_color():
        return text
    return f"{c}{text}{RESET}"


# ---------- screen control ----------

def clear_screen() -> None:
    """Clear the terminal and scrollback so prior output is hidden."""
    if not sys.stdout.isatty():
        return
    # \033[2J clears visible buffer; \033[3J clears scrollback (xterm/iTerm/most modern terms);
    # \033[H homes the cursor.
    sys.stdout.write("\033[3J\033[2J\033[H")
    sys.stdout.flush()


def width() -> int:
    try:
        return shutil.get_terminal_size((80, 24)).columns
    except OSError:
        return 80


def hr(char: str = "─", c: str = TEAL_DIM) -> None:
    print(color(char * width(), c))


# ---------- log helpers ----------

def info(msg: str) -> None:
    print(color("ℹ ", TEAL) + color(msg, SURFACE))


def warn(msg: str) -> None:
    print(color("⚠ ", WARN) + color(msg, WARN))


def error(msg: str) -> None:
    print(color("✖ ", ERR) + color(msg, ERR), file=sys.stderr)


def assistant(msg: str) -> None:
    print(color("● ", TEAL_BRIGHT) + color(msg, SURFACE))


def tool_call(name: str, summary: str) -> None:
    print(color("  ▸ ", TEAL) + color(name, TEAL_BRIGHT) + color(f"  {summary}", MUTED))


def tool_result(summary: str, ok: bool = True) -> None:
    mark = color("    ✓", OK) if ok else color("    ✗", ERR)
    print(mark + color(f" {summary}", MUTED))


# ---------- banner ----------

_LOGO = [
    "  ┏━┓┏━┓╻  ╻  ┏━┓┏┳┓┏━┓",
    "  ┃  ┃ ┃┃  ┃  ┣━┫┃┃┃┣━┫",
    "  ┗━┛┗━┛┗━╸┗━╸╹ ╹╹ ╹╹ ╹",
]


def banner(model: str, cwd: str, tools_enabled: bool = True) -> None:
    clear_screen()
    w = width()

    print()
    for line in _LOGO:
        print(color(line, TEAL_BRIGHT))
    print()

    badge = "tools: on" if tools_enabled else "tools: OFF (model lacks support)"
    badge_c = OK if tools_enabled else WARN

    rows = [
        (color("model", MUTED), color(model, TEAL_BRIGHT)),
        (color("cwd",   MUTED), color(cwd, SURFACE)),
        (color("status",MUTED), color(badge, badge_c)),
        (color("hint",  MUTED), color("type / for commands · /exit to quit", TEAL_DIM)),
    ]
    label_w = max(len(_strip(l)) for l, _ in rows)
    for label, value in rows:
        pad = " " * (label_w - len(_strip(label)))
        print(f"  {label}{pad}  {value}")

    print()
    hr()


def _strip(s: str) -> str:
    """Strip ANSI for length calc."""
    out = []
    i = 0
    while i < len(s):
        if s[i] == "\033":
            # skip escape until 'm'
            while i < len(s) and s[i] != "m":
                i += 1
            i += 1
            continue
        out.append(s[i])
        i += 1
    return "".join(out)

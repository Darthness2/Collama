"""Tiny ANSI helpers — no external deps."""
import sys

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
GRAY = "\033[90m"


def _supports_color() -> bool:
    return sys.stdout.isatty()


def color(text: str, c: str) -> str:
    if not _supports_color():
        return text
    return f"{c}{text}{RESET}"


def info(msg: str) -> None:
    print(color(msg, CYAN))


def warn(msg: str) -> None:
    print(color(msg, YELLOW))


def error(msg: str) -> None:
    print(color(msg, RED), file=sys.stderr)


def assistant(msg: str) -> None:
    print(color("● ", MAGENTA) + msg)


def tool_call(name: str, summary: str) -> None:
    print(color(f"⚙ {name}", BLUE) + color(f" {summary}", GRAY))


def tool_result(summary: str, ok: bool = True) -> None:
    mark = color("  ✓", GREEN) if ok else color("  ✗", RED)
    print(mark + color(f" {summary}", GRAY))


def banner(model: str, cwd: str) -> None:
    line = "─" * 56
    print(color(line, GRAY))
    print(color("  Collama", MAGENTA) + color(f"  ({model})", GRAY))
    print(color(f"  cwd: {cwd}", GRAY))
    print(color("  /help for commands · /exit to quit", GRAY))
    print(color(line, GRAY))

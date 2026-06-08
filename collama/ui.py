"""Terminal UI helpers — teal palette, blocky panels, no external deps."""
from __future__ import annotations

import os
import random
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

# Palette — calm, cohesive 256-color set (degrades gracefully if unsupported).
# Names are kept stable so the rest of the codebase doesn't need to change.
TEAL        = "\033[38;5;73m"    # muted teal — primary accent
TEAL_BRIGHT = "\033[38;5;80m"    # bright cyan — highlights, prompt, logo
TEAL_DIM    = "\033[38;5;23m"    # dark teal — borders, rules
SURFACE     = "\033[38;5;253m"   # near-white — body text
MUTED       = "\033[38;5;246m"   # grey — secondary text
SOFT        = "\033[38;5;240m"   # faint grey — timers, hints, rules
WARN        = "\033[38;5;180m"   # warm sand — warnings
ERR         = "\033[38;5;174m"   # dusty rose — errors
OK          = "\033[38;5;108m"   # sage green — success marks


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
# A whole line that is JUST a fence marker (```lang or ``` or ~~~). Used by
# the streaming renderer, which sees one line at a time and can't use the
# multi-line _MD_FENCE_RX.
_MD_FENCE_LINE_RX = re.compile(r"^(?:`{3,}|~{3,})\s*([A-Za-z0-9_+\-]*)\s*$")
_MD_INLINE_CODE_RX = re.compile(r"`([^`\n]+)`")
_MD_BOLD_RX = re.compile(r"\*\*([^*\n]+)\*\*")
_MD_ITALIC_AST_RX = re.compile(r"(?<![*\w])\*([^*\n]+?)\*(?!\w)")
_MD_ITALIC_UND_RX = re.compile(r"(?<!\w)_([^_\n]+?)_(?!\w)")
_MD_HEADER_RX = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_MD_BULLET_RX = re.compile(r"^(\s*)[-*]\s+", re.MULTILINE)
# <step N> markers the model emits to signal progress through its plan —
# rendered as a visible 'on step N' header so the user sees what's happening.
_MD_STEP_RX = re.compile(r"<step\s+(\d+)(?:\s*/\s*(\d+))?\s*>", re.IGNORECASE)
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

    # Step markers: '<step 2>' becomes a styled '▸ step 2', '<step 2/4>'
    # becomes '▸ step 2 of 4'. The model emits these on their own line, so we
    # render them in place — injecting surrounding newlines here would, in the
    # line-at-a-time streaming renderer, leave a prefix-only line (stray
    # trailing whitespace) and a blank line around the marker.
    def _step(m: "re.Match[str]") -> str:
        n, total = m.group(1), m.group(2)
        label = f"▸ step {n} of {total}" if total else f"▸ step {n}"
        return color(label, TEAL_BRIGHT + BOLD)
    out = _MD_STEP_RX.sub(_step, out)

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


class StreamMarkdown:
    """Render streamed tokens with markdown styling, line-by-line.

    Two ways of handling tagged blocks:
      • mode='elide' (e.g. <plan>): drop the contents — the engine
        renders the block as a panel later, showing both would duplicate.
      • mode='dim' (e.g. <think>): keep the contents but render dim
        italic with a '◦' prefix so the model's internal reasoning is
        visible LIVE, clearly distinguished from the final answer
        (which uses '●'). Each thinking line gets its own visual line.
    """

    SUPPRESS_PAIRS = [
        ("<plan>",     "</plan>",     "elide"),
        ("<think>",    "</think>",    "dim"),
        ("<thinking>", "</thinking>", "dim"),
    ]

    def __init__(self, emit, first_prefix: str = "", cont_prefix: str = "",
                 dim_first_prefix: str = "", dim_cont_prefix: str = ""):
        self.emit = emit
        self.first_prefix = first_prefix
        self.cont_prefix = cont_prefix
        self.dim_first_prefix = dim_first_prefix or first_prefix
        self.dim_cont_prefix = dim_cont_prefix or cont_prefix
        self.buf = ""
        self.opened = False
        self.mid_line = False
        self.in_dim = False        # currently inside a <think> block
        self.dim_opened = False    # have we emitted any dim line in current block?
        self.in_code = False       # currently inside a ``` fenced code block
        self.suppress_until: str | None = None  # close marker if currently eliding
        self._max_open = max(len(o) for o, _, _ in self.SUPPRESS_PAIRS)
        # True iff we've emitted at least one non-dim, non-empty line.
        # The renderer falls back to the static assistant panel when this
        # stays False — covers the case where the model wraps its entire
        # response in <plan>/<think> tags and the stream looks empty.
        self.visible_emitted = False

    def _consume_prefix(self) -> str:
        """The line prefix for the next emit, advancing the first/cont state."""
        if self.mid_line:
            return ""
        if self.in_dim:
            if not self.dim_opened:
                self.dim_opened = True
                return self.dim_first_prefix
            return self.dim_cont_prefix
        if not self.opened:
            self.opened = True
            return self.first_prefix
        return self.cont_prefix

    def _emit_line(self, line: str, terminator: str) -> None:
        if not self.in_dim and line.strip():
            self.visible_emitted = True

        # Fenced code blocks. render_markdown() only catches fences in
        # whole-text mode; here we see one line at a time, so we track the
        # open/close state ourselves and render the interior as code instead
        # of leaking literal ``` and unstyled source into the answer.
        fence = None if self.mid_line else _MD_FENCE_LINE_RX.match(line.strip())
        if fence is not None:
            prefix = self._consume_prefix()
            if not self.in_code:
                self.in_code = True
                lang = fence.group(1)
                bar = color("  ┌─ " + (lang or "code"), TEAL_DIM)
            else:
                self.in_code = False
                bar = color("  └─", TEAL_DIM)
            self.emit(prefix + bar + terminator)
            self.mid_line = (terminator == "")
            return
        if self.in_code:
            prefix = self._consume_prefix()
            bar = color("  │ ", TEAL_DIM) if not self.mid_line else ""
            self.emit(prefix + bar + color(line, TEAL_BRIGHT) + terminator)
            self.mid_line = (terminator == "")
            return

        if self.in_dim:
            # Dim block — italic gray, '◦' prefix for the first line of the
            # block, indented continuation after.
            prefix = self._consume_prefix()
            # MUTED (246) is more legible than SOFT (240) on dark terminals;
            # the model's reasoning is worth reading, not squinting at.
            styled = color(line, MUTED + ITALIC) if line else ""
            self.emit(prefix + styled + terminator)
        else:
            prefix = self._consume_prefix()
            self.emit(prefix + render_markdown(line) + terminator)
        self.mid_line = (terminator == "")

    def feed(self, text: str) -> None:
        if not text:
            return
        self.buf += text
        self._drain(final=False)

    def flush(self) -> None:
        self._drain(final=True)
        if self.buf and not self.suppress_until:
            self._emit_line(self.buf, "")
        self.buf = ""

    def _drain(self, *, final: bool) -> None:
        # Loop until no more progress can be made on the buffer.
        while True:
            if self.suppress_until and not self.in_dim:
                # ELIDE mode: drop everything up to and including the close.
                close = self.suppress_until
                idx = self.buf.find(close)
                if idx < 0:
                    keep = len(close) - 1
                    if final:
                        self.buf = ""
                        self.suppress_until = None
                        return
                    if len(self.buf) > keep:
                        self.buf = self.buf[-keep:]
                    return
                self.buf = self.buf[idx + len(close):]
                self.suppress_until = None
                continue

            if self.in_dim:
                # DIM mode: keep emitting lines as dim/italic until we hit
                # the close marker. Treat the close marker as a line break.
                close = self.suppress_until or ""
                idx = self.buf.find(close) if close else -1
                if idx < 0:
                    # No close yet — emit completed lines, hold a small tail
                    # so we don't split the close marker across emits.
                    if "\n" in self.buf:
                        last_nl = self.buf.rfind("\n")
                        head = self.buf[:last_nl + 1]
                        self.buf = self.buf[last_nl + 1:]
                        for line in head.split("\n")[:-1]:
                            self._emit_line(line, "\n")
                        continue
                    if final:
                        if self.buf:
                            self._emit_line(self.buf, "")
                            self.buf = ""
                        return
                    keep = len(close) - 1 if close else 0
                    if len(self.buf) > keep:
                        emit_now = self.buf[:-keep] if keep else self.buf
                        self.buf = self.buf[-keep:] if keep else ""
                        if emit_now:
                            self._emit_line(emit_now, "")
                    return
                # Close marker found — emit any content before it (with a
                # newline so the next normal line starts fresh), then exit
                # dim mode.
                before = self.buf[:idx]
                self.buf = self.buf[idx + len(close):]
                if before:
                    parts = before.split("\n")
                    for line in parts[:-1]:
                        self._emit_line(line, "\n")
                    if parts[-1]:
                        self._emit_line(parts[-1], "\n")  # force newline at end of dim block
                elif self.mid_line:
                    self.emit("\n")
                    self.mid_line = False
                self.in_dim = False
                self.dim_opened = False
                self.suppress_until = None
                continue

            # Not suppressing — look for the earliest open marker in the buf.
            earliest = -1
            earliest_close: str | None = None
            earliest_open_len = 0
            earliest_mode = "elide"
            for o, c, mode in self.SUPPRESS_PAIRS:
                i = self.buf.find(o)
                if i >= 0 and (earliest < 0 or i < earliest):
                    earliest = i
                    earliest_close = c
                    earliest_open_len = len(o)
                    earliest_mode = mode

            if earliest >= 0:
                # Emit text before the marker as normal lines.
                before = self.buf[:earliest]
                self.buf = self.buf[earliest + earliest_open_len:]
                self.suppress_until = earliest_close
                if before:
                    parts = before.split("\n")
                    for line in parts[:-1]:
                        self._emit_line(line, "\n")
                    if parts[-1]:
                        # Force newline so dim content starts on its own line.
                        self._emit_line(parts[-1], "\n" if earliest_mode == "dim" else "")
                if earliest_mode == "dim":
                    if self.mid_line:
                        self.emit("\n")
                        self.mid_line = False
                    self.in_dim = True
                    self.dim_opened = False
                continue

            # No open marker; emit completed lines but hold a small tail to
            # catch open markers split across feed() calls.
            if "\n" in self.buf:
                last_nl = self.buf.rfind("\n")
                head = self.buf[:last_nl + 1]
                self.buf = self.buf[last_nl + 1:]
                for line in head.split("\n")[:-1]:
                    self._emit_line(line, "\n")
                continue

            # Single line, no newline yet. Hold the WHOLE partial line until
            # its newline arrives (or final flush). Emitting partials as
            # continuations breaks per-line markdown — a header / bold / fence
            # split across two feed() chunks would never match its regex.
            # Buffering the line costs at most one line of latency and makes
            # markdown rendering reliable. Any open marker was already located
            # by the search above, so holding here can't miss one.
            if final:
                if self.buf:
                    self._emit_line(self.buf, "")
                    self.buf = ""
            return


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

# Default: a calm braille spinner — a single glyph that cycles in ONE fixed
# column. No left/right drift, no width change, so it doesn't pull the eye.
# The axolotl is kept as an opt-in (COLLAMA_SPINNER=axolotl) for those who
# liked it.
_AXOLOTL_FRAMES = (
    "~(◕‿◕)~",
    "~(◕‿◕)~~",
    "~~(◕‿◕)~~",
    "~~~(◕‿◕)~",
    "~~(◕‿◕)~~",
    "~(◕‿◕)~",
    "(◕‿◕)~",
    "~(◕‿◕)~",
    "~~(◕‿◕)~~",
    "~~~(◕‿◕)~~~",
    "~~(◕‿◕)~~",
    "~(◕‿◕)~",
)

_BRAILLE_FRAMES = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def _default_frames() -> tuple[str, ...]:
    style = os.environ.get("COLLAMA_SPINNER", "braille").lower()
    if style == "axolotl":
        return _AXOLOTL_FRAMES
    return _BRAILLE_FRAMES


_SPIN_FRAMES = _default_frames()

# Collapse any whitespace run (newlines, tabs, multiple spaces) in a spinner
# label down to a single space, and strip ends. The spinner clears its line
# each frame with \r\033[2K — which only erases the CURRENT row, so a label
# containing a newline (e.g. a multi-line shell command summary) leaves the
# trailing visual rows behind every frame, accumulating as scrollback junk.
_LABEL_WS_RX = re.compile(r"\s+")


def _sanitize_label(label: str) -> str:
    return _LABEL_WS_RX.sub(" ", label).strip()


# Whimsical stand-ins for a bland "thinking…" while the model chews on a
# prompt. The Spinner rotates through a shuffled copy of these (one every
# few seconds) so a long wait feels a little more alive — purely cosmetic,
# and only used when the label is the default "thinking".
THINKING_LABEL = "thinking"
_THINKING_VERBS: tuple[str, ...] = (
    "thinking",
    "pondering",
    "cogitating",
    "ruminating",
    "musing",
    "noodling",
    "percolating",
    "marinating",
    "mulling it over",
    "scheming",
    "conjuring",
    "brewing",
    "tinkering",
    "deliberating",
    "contemplating",
    "wrangling thoughts",
    "hatching a plan",
    "connecting the dots",
    "puzzling",
    "spelunking",
    "untangling",
    "reticulating splines",
    "consulting the oracle",
    "vibing",
)

# Track any live spinner so we can force-stop it before reading user input.
_active_spinners: list["Spinner"] = []


def stop_all_spinners() -> None:
    for s in list(_active_spinners):
        try:
            s.stop()
        except Exception:
            pass


def prepare_for_input() -> None:
    """Call right before reading user input: stop spinners and status bar,
    show cursor, flush. The status bar in particular MUST come down before
    readline takes over — otherwise its scroll region clashes with the
    input prompt and the cursor lands on the wrong row."""
    stop_all_spinners()
    stop_all_status_bars()
    if sys.stdout.isatty():
        sys.stdout.write("\033[?25h")  # show cursor
        sys.stdout.flush()


class Spinner:
    """Tiny non-blocking status spinner. Use as a context manager.

    Renders as:    ⠋ thinking…   (0.4s)
    On stop, clears the line so the next print is clean.
    """

    def __init__(
        self,
        label: str = "thinking",
        color_c: str = TEAL_BRIGHT,
        escalations: list[tuple[float, str]] | None = None,
    ) -> None:
        """`escalations` is an optional list of (after_seconds, label) pairs
        applied as time passes — used by the engine to tell the user WHY
        the agent has been thinking for a while (large prompt, model loading,
        etc.) instead of just sitting on the original label."""
        self.label = _sanitize_label(label)
        self.color_c = color_c
        # Sort ascending so the loop just picks the highest matching tier.
        # Sanitize each escalation label for the same reason as the main one.
        self.escalations = sorted(
            [(t, _sanitize_label(l)) for t, l in (escalations or [])],
            key=lambda x: x[0],
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        # A plain "thinking" spinner gets the fun rotating verbs. Shuffle a
        # private copy per instance so different runs vary, but keep it stable
        # within a single spin (time-indexed below, not per-frame) so it reads
        # as words swapping every few seconds rather than flickering.
        self._verbs: list[str] = []
        if self.label == THINKING_LABEL:
            self._verbs = random.sample(_THINKING_VERBS, len(_THINKING_VERBS))

    def set_label(self, label: str) -> None:
        self.label = _sanitize_label(label)
        self._verbs = (
            random.sample(_THINKING_VERBS, len(_THINKING_VERBS))
            if self.label == THINKING_LABEL
            else []
        )

    def _whimsy_verb(self, elapsed: float) -> str:
        # New verb every ~3.5s, indexed off elapsed time so every frame in
        # that window renders the same word (no per-frame flicker).
        return self._verbs[int(elapsed // 3.5) % len(self._verbs)]

    def _current_label(self, elapsed: float) -> str:
        label = self.label
        for after, lbl in self.escalations:
            if elapsed >= after:
                label = lbl
            else:
                break
        # Still on the plain "thinking" label (no escalation has fired yet)?
        # Swap in a rotating whimsical verb. Escalation messages carry real
        # diagnostic context, so we leave those untouched.
        if self._verbs and label == self.label:
            label = self._whimsy_verb(elapsed)
        return label

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
        # 150ms grace period — if the work finishes fast (most tool dispatches
        # are sub-100ms), we never draw a frame and there's no visual flash.
        if self._stop.wait(0.15):
            return
        import shutil
        i = 0
        while not self._stop.is_set():
            frame = _SPIN_FRAMES[i % len(_SPIN_FRAMES)]
            elapsed = time.monotonic() - self._t0
            label = self._current_label(elapsed)
            # Whole seconds — a decimal ticking 10×/s is its own kind of
            # visual noise. The braille glyph already signals liveness.
            timer = f"({elapsed:0.0f}s)"
            # Build the plain (ANSI-free) line first so we can measure it and
            # hard-truncate to terminal width — a line that wraps would make
            # \r\033[2K only clear the last visual row, scrolling the rest
            # into scrollback every frame.
            raw_visible = f"  {frame} {label}…  {timer}"
            cols = shutil.get_terminal_size((80, 24)).columns
            if len(raw_visible) > cols - 1:
                styled = color(raw_visible[: max(1, cols - 2)] + "…", MUTED)
            else:
                styled = (
                    "  "
                    + color(frame, self.color_c)
                    + " "
                    + color(label + "…", MUTED)
                    + "  "
                    + color(timer, SOFT)
                )
            sys.stdout.write("\r\033[2K" + styled)
            sys.stdout.flush()
            i += 1
            # ~8 fps — quick enough to read as alive, slow enough to be calm.
            self._stop.wait(0.12)

    def __enter__(self) -> "Spinner":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


class SilenceWatchdog:
    """Background watchdog that prints a dim 'still receiving' breadcrumb
    when a streaming response has gone quiet for a while.

    Why this exists: Qwen-class reasoning models can stop emitting tokens
    for a long time while they think (<think> happens model-side, BEFORE
    any tokens are sent). Without feedback the user can't tell the
    difference between 'thinking hard' and 'Ollama died' and reaches for
    Ctrl+C — usually right before the answer would have started.

    Design notes:
    - Writes to stderr, not stdout, so the streaming text buffer is never
      overwritten mid-line. Output may visually land below an unfinished
      line; that's intentional — accuracy over prettiness.
    - Escalation tiers are spaced wide (25s, 60s, 180s, 600s) so we never
      spam the scrollback. Each tier prints once per silence stretch.
    - .ping() resets the silence counter AND the escalation index, so a
      single token between two long stalls produces two breadcrumb runs
      rather than skipping past tiers.
    """

    DEFAULT_TIERS = (25.0, 60.0, 180.0, 600.0)

    def __init__(
        self,
        label: str = "still receiving",
        tiers: tuple[float, ...] = DEFAULT_TIERS,
    ) -> None:
        self.label = label
        self.tiers = tiers
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._last = 0.0
        self._lock = threading.Lock()
        self._tier_idx = 0

    def start(self) -> None:
        if not sys.stderr.isatty() or self._thread is not None:
            return
        self._last = time.monotonic()
        self._tier_idx = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def ping(self) -> None:
        # Called on every received token.
        with self._lock:
            self._last = time.monotonic()
            self._tier_idx = 0

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(1.0):
            with self._lock:
                silent = time.monotonic() - self._last
                idx = self._tier_idx
            if idx >= len(self.tiers):
                continue
            if silent < self.tiers[idx]:
                continue
            # Dim italic line on stderr, prefixed with \n so it lands on a
            # fresh row even if stdout was mid-line.
            sys.stderr.write(
                "\n"
                + color(f"  ◦ {self.label} — {silent:0.0f}s since last token…", MUTED + ITALIC)
                + "\n"
            )
            sys.stderr.flush()
            with self._lock:
                self._tier_idx += 1


# ---------- sticky status bar (bottom row) ---------------------------------

# Active status bars so prepare_for_input() / Ctrl+C can force-tear-down
# the scroll region before any input prompt or new top-level output.
_active_status_bars: list["StatusBar"] = []


def _fmt_elapsed(secs: float) -> str:
    """Compact human-friendly duration: '4.2s', '1m12s', '5m02s'."""
    if secs < 60:
        return f"{secs:0.1f}s"
    m = int(secs // 60)
    s = int(secs - m * 60)
    return f"{m}m{s:02d}s"


class StatusBar:
    """Sticky one-line status row pinned to the bottom of the terminal.

    While installed, scrolling is constrained to rows 1..N-1 via DECSTBM
    (`\\033[1;{N-1}r`), so streaming text and tool output scroll naturally
    above while the bottom row stays put. The row is repainted on a
    ~5fps timer and shows elapsed turn time plus a rough token tally
    (output streamed this turn + cumulative context size).

    Display:
        ⏱ 4.2s  ·  ~234 tok  ·  ctx ~8,421

    Token counts during streaming are estimates (chars / 4 — the
    standard ASCII-English approximation), since Ollama only reports
    the exact eval_count at the end of the response.

    Lifecycle: start() reserves the scroll region; stop() restores the
    full screen and clears the status row. Resilient to terminal resize
    (the row count is re-read on each draw and the region updated).
    Safe to start() on a non-TTY — becomes a no-op.

    Concurrency note: the per-frame render is emitted as a single
    write() call (save-cursor + position + clear + status + restore)
    so it cannot interleave with streaming text writes in the middle of
    a frame. Drawing happens in a background thread on a 0.2s tick.
    """

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._lock = threading.Lock()
        self._ctx_tokens = 0       # cumulative context size (chars/4)
        self._out_chars = 0        # output chars streamed this turn
        self._last_rows = 0
        self._installed = False

    def start(self, ctx_tokens: int = 0) -> None:
        if not sys.stdout.isatty() or self._thread is not None:
            return
        with self._lock:
            self._t0 = time.monotonic()
            self._ctx_tokens = ctx_tokens
            self._out_chars = 0
            self._stop.clear()
        size = shutil.get_terminal_size((80, 24))
        rows = max(2, size.lines)
        self._last_rows = rows
        # Reserve the bottom row for status. Wrap the DECSTBM set in
        # save/restore-cursor because per the xterm spec it moves the
        # cursor to home (1,1) — without the wrap, streaming would start
        # from row 1 and leave a giant blank gap between the user's
        # prompt and the response.
        sys.stdout.write(
            "\033[s"                    # save cursor
            f"\033[1;{rows - 1}r"       # set scroll region
            "\033[u"                    # restore cursor
        )
        sys.stdout.flush()
        self._installed = True
        _active_status_bars.append(self)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def add_output_text(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            self._out_chars += len(text)

    def set_ctx_tokens(self, n: int) -> None:
        with self._lock:
            self._ctx_tokens = max(0, int(n))

    def stop(self) -> None:
        if not self._installed:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self in _active_status_bars:
            _active_status_bars.remove(self)
        # Reset scroll region and clear the status row, then return the
        # cursor to wherever streaming left it — NOT to row N-1.
        # Forcing it down would leave a blank gap between a short
        # response and the next prompt (the visible bug that motivated
        # this fix).
        size = shutil.get_terminal_size((80, 24))
        rows = max(2, size.lines)
        sys.stdout.write(
            "\033[s"                            # save cursor
            "\033[r"                            # reset DECSTBM
            f"\033[{rows};1H\033[2K"            # clear status row
            "\033[u"                            # restore cursor
        )
        sys.stdout.flush()
        self._installed = False

    def _run(self) -> None:
        # First paint right away so the bar shows up without a tick delay.
        self._draw()
        while not self._stop.wait(0.2):
            self._draw()

    def _draw(self) -> None:
        size = shutil.get_terminal_size((80, 24))
        rows = max(2, size.lines)
        cols = max(20, size.columns)
        # On resize, reset the scroll region so the bottom row stays
        # reserved against the actual new terminal bounds.
        resize_seq = ""
        if rows != self._last_rows:
            resize_seq = f"\033[1;{rows - 1}r"
            self._last_rows = rows
        with self._lock:
            elapsed = time.monotonic() - self._t0
            ctx = self._ctx_tokens
            out_tok = self._out_chars // 4   # ~3.7 chars/token for English
        timer = _fmt_elapsed(elapsed)
        parts = [f"⏱ {timer}", f"~{out_tok:,} tok"]
        if ctx:
            parts.append(f"ctx ~{ctx:,}")
        raw = "  " + "  ·  ".join(parts)
        if len(raw) > cols - 1:
            raw = raw[: cols - 2] + "…"
        styled = color(raw, MUTED) if _supports_color() else raw
        # One write so save→position→clear→write→restore can't be
        # interleaved by streaming text from the main thread.
        frame = (
            resize_seq
            + "\033[s"                          # save cursor
            + f"\033[{rows};1H\033[2K"          # to status row, clear
            + styled
            + "\033[u"                          # restore cursor
        )
        try:
            sys.stdout.write(frame)
            sys.stdout.flush()
        except (OSError, ValueError):
            # stdout closed — give up silently rather than crashing the
            # background thread and leaving the scroll region installed.
            self._stop.set()


def stop_all_status_bars() -> None:
    for b in list(_active_status_bars):
        try:
            b.stop()
        except Exception:
            pass


# ---------- log helpers ----------

def info(msg: str) -> None:
    print(color("  ℹ ", TEAL) + color(msg, SURFACE))


def warn(msg: str) -> None:
    print(color("  ⚠ ", WARN) + color(msg, WARN))


def error(msg: str) -> None:
    print(color("  ✖ ", ERR) + color(msg, ERR), file=sys.stderr)


def assistant(msg: str) -> None:
    """Print the assistant's answer as clean indented text — markdown-styled,
    no box. (Streaming answers are already shown live; this is the
    non-streaming render and the /resume replay.)"""
    rendered = render_markdown(msg)
    lines = _wrap_visible(rendered, max(20, width() - 4))
    for i, line in enumerate(lines or [""]):
        prefix = color("  ● ", TEAL_BRIGHT) if i == 0 else "    "
        print(prefix + line)


def thinking(msg: str) -> None:
    """Print <think>…</think> content as faint italic indented text — no box."""
    lines = _wrap_visible(msg, max(20, width() - 4))
    for i, line in enumerate(lines or [""]):
        prefix = color("  ◦ ", SOFT) if i == 0 else "    "
        print(prefix + color(line, SOFT + ITALIC))


def plan(items: list[str]) -> None:
    """Render a numbered plan in a panel.

    Each step is markdown-rendered (so **bold**, `code`, etc. show styled
    rather than leaking literal asterisks/backticks) and wrapped to the
    panel's inner width — long steps used to overflow the right border and
    leave the box ragged. Continuation lines hang-indent under the text.
    """
    if not items:
        return
    inner = max(20, width() - 4)  # mirrors panel()'s inner for inner_pad=1
    body: list[str] = []
    for i, step in enumerate(items, 1):
        num = f"  {i:>2}. "          # 6 visible columns
        indent = " " * _vlen(num)
        rendered = render_markdown(step.strip())
        wrapped = _wrap_visible(rendered, max(8, inner - _vlen(num))) or [""]
        for j, seg in enumerate(wrapped):
            prefix = color(num, TEAL) if j == 0 else indent
            body.append(prefix + seg)
    panel(body, title="plan", style="thick", color_c=TEAL, title_c=TEAL_BRIGHT)


def tilde(s: str) -> str:
    """Collapse the user's home directory prefix to '~' anywhere it appears
    in `s`. Keeps tool lines short and avoids leaking the full home path."""
    if not s:
        return s
    import os
    from pathlib import Path
    home = str(Path.home())
    out = s.replace(home, "~")
    # Also handle a differently-cased / trailing-slash home on Windows.
    if os.sep != "/":
        out = out.replace(home.replace("/", os.sep), "~")
    return out


def tool_call(name: str, summary: str) -> None:
    head = color("  ▸ ", TEAL_BRIGHT) + color(name, TEAL_BRIGHT)
    if summary:
        # Collapse newlines/tabs so a multi-line arg (e.g. a here-doc shell
        # command) doesn't produce a multi-row ▸ entry in scrollback.
        head += color(f"  {tilde(_sanitize_label(summary))}", MUTED)
    print(head)


def tool_result(summary: str, ok: bool = True) -> None:
    mark = color("    ✓", OK) if ok else color("    ✗", ERR)
    print(mark + color(f" {tilde(_sanitize_label(summary))}", MUTED))


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

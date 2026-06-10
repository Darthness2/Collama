"""check_syntax — parser-only syntax linting across many languages.

Lightweight, dependency-free where possible. Each checker is static and
side-effect-free — never executes the code under test. External-tool
checkers (node, tsc, gofmt, rustc, bash) gracefully skip when the binary
is missing so the tool stays useful on machines without every toolchain.
"""
from __future__ import annotations

import os
import subprocess

from .base import _resolve, _truncate, ToolContext


_SYNTAX_EXT_LANG = {
    ".py":   "python",
    ".pyi":  "python",
    ".json": "json",
    ".yaml": "yaml",
    ".yml":  "yaml",
    ".js":   "javascript",
    ".mjs":  "javascript",
    ".cjs":  "javascript",
    ".jsx":  "javascript",
    ".ts":   "typescript",
    ".tsx":  "typescript",
    ".sh":   "bash",
    ".bash": "bash",
    ".go":   "go",
    ".rs":   "rust",
    ".toml": "toml",
    ".html": "html",
    ".xml":  "xml",
    ".css":  "css",
}


def _check_python(src: str, label: str) -> tuple[bool, str]:
    try:
        compile(src, label, "exec")
    except SyntaxError as e:
        loc = f"{e.filename or label}:{e.lineno or '?'}:{e.offset or '?'}"
        return False, f"SyntaxError at {loc}: {e.msg}"
    return True, "ok"


def _check_json(src: str, label: str) -> tuple[bool, str]:
    import json as _json
    try:
        _json.loads(src)
    except _json.JSONDecodeError as e:
        return False, f"JSONDecodeError at {label}:{e.lineno}:{e.colno}: {e.msg}"
    return True, "ok"


def _check_yaml(src: str, label: str) -> tuple[bool, str]:
    try:
        import yaml  # type: ignore
    except ImportError:
        return True, "skipped (PyYAML not installed)"
    try:
        list(yaml.safe_load_all(src))
    except yaml.YAMLError as e:
        return False, f"YAMLError in {label}: {e}"
    return True, "ok"


def _check_toml(src: str, label: str) -> tuple[bool, str]:
    try:
        import tomllib  # py311+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore
        except ImportError:
            return True, "skipped (no tomllib/tomli)"
    try:
        tomllib.loads(src)
    except Exception as e:
        return False, f"TOMLDecodeError in {label}: {e}"
    return True, "ok"


def _check_via_cmd(cmd: list[str], src: str | None, label: str) -> tuple[bool, str]:
    """Run a parser-only external command. If the binary is missing return
    'skipped' so the tool stays useful on machines without every toolchain."""
    try:
        proc = subprocess.run(
            cmd,
            input=src if src is not None else None,
            capture_output=True,
            encoding="utf-8", errors="replace",
            timeout=15,
        )
    except FileNotFoundError:
        return True, f"skipped ({cmd[0]} not installed)"
    except subprocess.TimeoutExpired:
        return False, f"{cmd[0]} timed out checking {label}"
    if proc.returncode == 0:
        return True, "ok"
    err = (proc.stderr or proc.stdout).strip()
    return False, err or f"{cmd[0]} exit {proc.returncode}"


def _check_js(src: str, label: str, is_ts: bool) -> tuple[bool, str]:
    # `node --check` parses without running. For TS we fall back to a stripped
    # parse — there's no built-in TS parser. If `tsc` is around, prefer it.
    if is_ts:
        # tsc is heavy; only used when present.
        import shutil
        if shutil.which("tsc"):
            # Need a real file path for tsc; if src came in, write a temp.
            import tempfile
            suffix = ".tsx" if label.endswith(".tsx") else ".ts"
            with tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False) as f:
                f.write(src)
                tmp = f.name
            try:
                return _check_via_cmd(["tsc", "--noEmit", "--allowJs", tmp], None, label)
            finally:
                try: os.unlink(tmp)
                except OSError: pass
        return True, "skipped (tsc not installed; node --check can't parse TS)"
    return _check_via_cmd(["node", "--check", "-"], src, label)


def _check_bash(src: str, label: str) -> tuple[bool, str]:
    return _check_via_cmd(["bash", "-n"], src, label)


def _check_go(src: str, label: str) -> tuple[bool, str]:
    return _check_via_cmd(["gofmt", "-e"], src, label)


def _check_rust(src: str, label: str) -> tuple[bool, str]:
    # No stdin parse mode; write temp and use `rustc --edition=2021 --emit=metadata`.
    import shutil, tempfile
    if not shutil.which("rustc"):
        return True, "skipped (rustc not installed)"
    with tempfile.NamedTemporaryFile("w", suffix=".rs", delete=False) as f:
        f.write(src)
        tmp = f.name
    try:
        return _check_via_cmd(
            ["rustc", "--edition=2021", "--emit=metadata", "-o", os.devnull, tmp],
            None, label,
        )
    finally:
        try: os.unlink(tmp)
        except OSError: pass


def _check_xml_like(src: str, label: str) -> tuple[bool, str]:
    # HTML/XML — use stdlib parser; HTML uses html.parser which is lenient but
    # still catches gross issues. xml.etree is strict for XML.
    if label.lower().endswith((".html", ".htm")):
        from html.parser import HTMLParser
        class _P(HTMLParser):
            def error(self, msg): raise ValueError(msg)
        p = _P()
        try:
            p.feed(src); p.close()
        except Exception as e:
            return False, f"HTMLParseError in {label}: {e}"
        return True, "ok"
    import xml.etree.ElementTree as ET
    try:
        ET.fromstring(src)
    except ET.ParseError as e:
        return False, f"XMLParseError in {label}: {e}"
    return True, "ok"


def _check_css(src: str, label: str) -> tuple[bool, str]:
    # Stdlib has no CSS parser. Do a brace-balance sanity check; flag obvious
    # truncation. Not full validation, but catches the common "model cut the
    # file off mid-rule" case.
    depth = 0
    for i, ch in enumerate(src):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return False, f"CSS in {label}: stray '}}' at offset {i}"
    if depth != 0:
        return False, f"CSS in {label}: {depth} unclosed brace(s)"
    return True, "ok (brace-balance only; install a CSS linter for full check)"


_SYNTAX_CHECKERS = {
    "python":     lambda s, l: _check_python(s, l),
    "json":       lambda s, l: _check_json(s, l),
    "yaml":       lambda s, l: _check_yaml(s, l),
    "javascript": lambda s, l: _check_js(s, l, is_ts=False),
    "typescript": lambda s, l: _check_js(s, l, is_ts=True),
    "bash":       lambda s, l: _check_bash(s, l),
    "go":         lambda s, l: _check_go(s, l),
    "rust":       lambda s, l: _check_rust(s, l),
    "toml":       lambda s, l: _check_toml(s, l),
    "html":       lambda s, l: _check_xml_like(s, l),
    "xml":        lambda s, l: _check_xml_like(s, l),
    "css":        lambda s, l: _check_css(s, l),
}


def t_check_syntax(args: dict, ctx: ToolContext) -> str:
    """Parser-only syntax check across one or more files / inline snippets.
    Read-only — never executes the code under test. Returns a per-target
    pass/fail summary so the model can self-verify edits before declaring done.
    """
    paths = args.get("paths") or ([args["path"]] if args.get("path") else [])
    content = args.get("content")
    language = (args.get("language") or "").lower().strip() or None

    if content is not None and not paths:
        # Inline snippet mode.
        if not language:
            return "ERROR: 'language' is required when checking inline 'content'."
        checker = _SYNTAX_CHECKERS.get(language)
        if not checker:
            return f"ERROR: unsupported language '{language}'. Known: {', '.join(sorted(_SYNTAX_CHECKERS))}"
        ok, msg = checker(content, f"<inline:{language}>")
        return f"{'PASS' if ok else 'FAIL'}  <inline:{language}>  {msg}"

    if not paths:
        return "ERROR: provide 'paths' (list of files) or 'content' + 'language'."

    lines: list[str] = []
    any_fail = False
    for raw in paths:
        try:
            p = _resolve(raw, ctx.root)
        except Exception as e:
            lines.append(f"FAIL  {raw}  resolve error: {e}")
            any_fail = True
            continue
        if not p.exists():
            lines.append(f"FAIL  {raw}  no such file")
            any_fail = True
            continue
        try:
            src = p.read_text(encoding="utf-8")
        except Exception as e:
            lines.append(f"FAIL  {raw}  read error: {e}")
            any_fail = True
            continue
        lang = language or _SYNTAX_EXT_LANG.get(p.suffix.lower())
        if not lang:
            lines.append(f"SKIP  {raw}  unknown extension '{p.suffix}'")
            continue
        checker = _SYNTAX_CHECKERS.get(lang)
        if not checker:
            lines.append(f"SKIP  {raw}  no checker for '{lang}'")
            continue
        ok, msg = checker(src, str(p))
        if ok:
            lines.append(f"PASS  {raw}  [{lang}]  {msg}")
        else:
            any_fail = True
            lines.append(f"FAIL  {raw}  [{lang}]  {msg}")

    header = "syntax check: " + ("FAILED" if any_fail else "all passed")
    return _truncate(header + "\n" + "\n".join(lines))


TOOLS = {"check_syntax": t_check_syntax}

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "check_syntax",
        "description": (
            "Parser-only syntax check for one or more files (or an inline snippet). "
            "Read-only — never executes the code. Use this after edits to confirm "
            "the file still parses before declaring the task done. Languages auto-"
            "detected from extension: python, json, yaml, javascript, typescript, "
            "bash, go, rust, toml, html, xml, css. External-tool checkers (node, "
            "tsc, gofmt, rustc, bash) gracefully skip when the binary is missing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "paths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Files to check (absolute or workspace-relative).",
                },
                "path": {"type": "string", "description": "Single-file convenience alias for 'paths'."},
                "content": {"type": "string", "description": "Inline source to check instead of a file."},
                "language": {
                    "type": "string",
                    "description": "Force language (python|json|yaml|javascript|typescript|bash|go|rust|toml|html|xml|css). Required with 'content'.",
                },
            },
        },
    }},
]

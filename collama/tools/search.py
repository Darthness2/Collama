"""glob / tool_search / web_fetch / web_search — read-only discovery tools."""
from __future__ import annotations

from .base import ToolContext, _resolve, _truncate


def t_glob(args: dict, ctx: ToolContext) -> str:
    """GlobTool — file pattern matching (supports **)."""
    import fnmatch
    pattern = args["pattern"]
    base = _resolve(args.get("path", "."), ctx.root)
    if not base.exists() or not base.is_dir():
        return f"ERROR: not a directory: {base}"
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
    matches: list[str] = []
    if "**" in pattern or "/" in pattern:
        for p in base.rglob("*"):
            if any(part in skip_dirs for part in p.parts):
                continue
            rel = p.relative_to(base)
            if fnmatch.fnmatch(str(rel), pattern):
                matches.append(str(rel))
    else:
        for p in base.iterdir():
            if fnmatch.fnmatch(p.name, pattern):
                matches.append(p.name)
    matches.sort()
    return _truncate("\n".join(matches[:500]) if matches else "(no matches)")


def t_tool_search(args: dict, ctx: ToolContext) -> str:
    """ToolSearchTool — find tools by keyword in name or description."""
    # Lazy import to avoid circular dependency on registry → submodules → here.
    from .registry import TOOL_SCHEMAS
    q = (args.get("query") or "").lower().strip()
    out: list[str] = []
    for s in TOOL_SCHEMAS:
        fn = s.get("function") or {}
        name = fn.get("name", "")
        desc = fn.get("description", "")
        hay = f"{name} {desc}".lower()
        if not q or q in hay:
            out.append(f"{name}  —  {desc.splitlines()[0][:140] if desc else ''}")
    return _truncate("\n".join(out) if out else "(no matching tools)")


def t_web_fetch(args: dict, ctx: ToolContext) -> str:
    """WebFetchTool — fetch a URL and return text (HTML or JSON)."""
    import requests
    url = args["url"]
    if not url.startswith(("http://", "https://")):
        return "ERROR: url must be http:// or https://"
    timeout = int(args.get("timeout", 20))
    max_bytes = int(args.get("max_bytes", 200_000))
    headers = {"User-Agent": "collama/0.1 (+https://github.com/YOUR_USERNAME/Collama)"}
    try:
        r = requests.get(url, timeout=timeout, headers=headers,
                         verify=not ctx.insecure_ssl, stream=True)
    except requests.RequestException as e:
        return f"ERROR: fetch failed: {e}"
    chunks: list[bytes] = []
    seen = 0
    for chunk in r.iter_content(8192):
        chunks.append(chunk)
        seen += len(chunk)
        if seen >= max_bytes:
            break
    raw = b"".join(chunks)
    try:
        body = raw.decode(r.encoding or "utf-8", errors="replace")
    except Exception:
        body = raw.decode("utf-8", errors="replace")
    return _truncate(f"HTTP {r.status_code}  {url}\n{body}")


def t_web_search(args: dict, ctx: ToolContext) -> str:
    """WebSearchTool — DuckDuckGo HTML-frontend scrape (no API key needed)."""
    import re as _re
    import requests
    q = args["query"]
    n = int(args.get("limit", 10))
    try:
        r = requests.get(
            "https://duckduckgo.com/html/", params={"q": q},
            headers={"User-Agent": "Mozilla/5.0 collama/0.1"},
            timeout=15, verify=not ctx.insecure_ssl,
        )
    except requests.RequestException as e:
        return f"ERROR: search failed: {e}"
    if r.status_code != 200:
        return f"ERROR: HTTP {r.status_code}"
    # Lightweight parse — DuckDuckGo HTML wraps results in <a class="result__a">.
    rx = _re.compile(r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', _re.DOTALL)
    items = rx.findall(r.text)
    out: list[str] = []
    for href, title in items[:n]:
        title_text = _re.sub(r"<[^>]+>", "", title).strip()
        out.append(f"- {title_text}\n  {href}")
    return _truncate("\n".join(out) if out else "(no results)")


TOOLS = {
    "glob":        t_glob,
    "tool_search": t_tool_search,
    "web_fetch":   t_web_fetch,
    "web_search":  t_web_search,
}


TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "glob",
        "description": "File pattern matching. Supports ** for recursive globs (e.g. **/*.py, src/*.ts).",
        "parameters": {"type": "object", "properties": {
            "pattern": {"type": "string"}, "path": {"type": "string"},
        }, "required": ["pattern"]},
    }},
    {"type": "function", "function": {
        "name": "tool_search",
        "description": "Search the registered tools by keyword in their name or description.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
    }},
    {"type": "function", "function": {
        "name": "web_fetch",
        "description": "Fetch a URL (HTTP/HTTPS) and return the body text. Caps at ~200 KB.",
        "parameters": {"type": "object", "properties": {
            "url": {"type": "string"},
            "timeout": {"type": "integer"},
            "max_bytes": {"type": "integer"},
        }, "required": ["url"]},
    }},
    {"type": "function", "function": {
        "name": "web_search",
        "description": "Search the web (DuckDuckGo HTML frontend). Returns title + URL pairs.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"},
        }, "required": ["query"]},
    }},
]

"""glob / tool_search / web_fetch / web_search — read-only discovery tools."""
from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

from .base import PathEscapeError, ToolContext, _resolve_contained, _truncate

logger = logging.getLogger(__name__)

# Hard ceiling on bytes pulled from any single remote response, regardless of
# the caller-supplied max_bytes. Defends against decompression / streaming
# resource exhaustion.
_ABSOLUTE_MAX_BYTES = 5_000_000
# Cap on HTML fed to the DOTALL-heavy parsing regex to avoid catastrophic
# backtracking / ReDoS on adversarial pages.
_MAX_HTML_FOR_REGEX = 2_000_000
_MAX_REDIRECTS = 5


def _ip_is_blocked(ip_str: str) -> bool:
    """True if an IP literal is loopback / link-local / private / reserved —
    i.e. an address we must never let an LLM-driven fetch reach (SSRF guard)."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        # Unparseable — treat as unsafe rather than risk it.
        return True
    return (
        ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
        # IPv4-mapped IPv6 (e.g. ::ffff:127.0.0.1) — unwrap and re-check.
        or (getattr(ip, "ipv4_mapped", None) is not None and _ip_is_blocked(str(ip.ipv4_mapped)))
    )


def _assert_public_host(host: str) -> None:
    """Resolve `host` and refuse if ANY resolved address is non-public.

    Raises PermissionError with a user-facing message when the host resolves to
    a loopback/link-local/private/reserved address. Called BEFORE every connect
    (including each redirect hop) so we never open a socket to an internal IP.
    """
    if not host:
        raise PermissionError("ERROR: refusing to fetch: missing host")
    # A bare IP literal in the URL — check it directly.
    try:
        ipaddress.ip_address(host)
        if _ip_is_blocked(host):
            raise PermissionError(
                f"ERROR: refusing to fetch internal/non-public address: {host}"
            )
        return
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as e:
        raise PermissionError(f"ERROR: cannot resolve host {host}: {e}") from e
    for info in infos:
        ip_str = info[4][0]
        if _ip_is_blocked(ip_str):
            raise PermissionError(
                f"ERROR: refusing to fetch {host}: resolves to non-public "
                f"address {ip_str}"
            )


def _safe_get(url: str, *, headers: dict, timeout: int, verify: bool,
              max_bytes: int, params: dict | None = None):
    """SSRF-hardened HTTP GET.

    Manually follows redirects (max ``_MAX_REDIRECTS``), re-validating the host
    of every hop against :func:`_assert_public_host` BEFORE connecting, and
    reads at most ``max_bytes`` (further capped by ``_ABSOLUTE_MAX_BYTES``).

    Returns (status_code, final_url, raw_bytes, encoding). Raises
    PermissionError (with an ``ERROR: ...`` message) on a blocked host and
    requests.RequestException on transport errors.
    """
    import requests

    cap = min(int(max_bytes), _ABSOLUTE_MAX_BYTES)
    current = url
    for _hop in range(_MAX_REDIRECTS + 1):
        parsed = urlparse(current)
        if parsed.scheme not in ("http", "https"):
            raise PermissionError("ERROR: url must be http:// or https://")
        _assert_public_host(parsed.hostname or "")
        r = requests.get(
            current, headers=headers, params=params, timeout=timeout,
            verify=verify, stream=True, allow_redirects=False,
        )
        if r.is_redirect or r.status_code in (301, 302, 303, 307, 308):
            loc = r.headers.get("location")
            r.close()
            if not loc:
                raise requests.RequestException("redirect without Location header")
            # Resolve relative redirects against the current URL.
            current = requests.compat.urljoin(current, loc)
            params = None  # query already baked into the redirect target
            continue
        chunks: list[bytes] = []
        seen = 0
        for chunk in r.iter_content(8192):
            if not chunk:
                continue
            chunks.append(chunk)
            seen += len(chunk)
            if seen >= cap:
                break
        encoding = r.encoding
        status = r.status_code
        final_url = r.url
        r.close()
        return status, final_url, b"".join(chunks)[:cap], encoding
    raise requests.RequestException(f"too many redirects (>{_MAX_REDIRECTS})")


def t_glob(args: dict, ctx: ToolContext) -> str:
    """GlobTool — file pattern matching (supports **)."""
    import fnmatch
    pattern = args["pattern"]
    try:
        base = _resolve_contained(args.get("path", "."), ctx.root)
    except PathEscapeError as exc:
        return exc.message
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
        status, final_url, raw, encoding = _safe_get(
            url, headers=headers, timeout=timeout,
            verify=not ctx.insecure_ssl, max_bytes=max_bytes,
        )
    except PermissionError as e:
        return str(e)
    except requests.RequestException as e:
        logger.warning("web_fetch failed for %s", url, exc_info=True)
        return f"ERROR: fetch failed: {e}"
    try:
        body = raw.decode(encoding or "utf-8", errors="replace")
    except (LookupError, UnicodeDecodeError):
        body = raw.decode("utf-8", errors="replace")
    return _truncate(f"HTTP {status}  {final_url}\n{body}")


def t_web_search(args: dict, ctx: ToolContext) -> str:
    """WebSearchTool — DuckDuckGo HTML-frontend scrape (no API key needed)."""
    import re as _re
    import requests
    q = args["query"]
    n = int(args.get("limit", 10))
    headers = {"User-Agent": "Mozilla/5.0 collama/0.1"}
    try:
        status, _final_url, raw, encoding = _safe_get(
            "https://duckduckgo.com/html/", headers=headers, timeout=15,
            verify=not ctx.insecure_ssl, max_bytes=_MAX_HTML_FOR_REGEX,
            params={"q": q},
        )
    except PermissionError as e:
        return str(e)
    except requests.RequestException as e:
        logger.warning("web_search failed", exc_info=True)
        return f"ERROR: search failed: {e}"
    if status != 200:
        return f"ERROR: HTTP {status}"
    try:
        text = raw.decode(encoding or "utf-8", errors="replace")
    except (LookupError, UnicodeDecodeError):
        text = raw.decode("utf-8", errors="replace")
    # Cap input to the DOTALL regex below to avoid ReDoS / catastrophic
    # backtracking on a hostile/huge page.
    text = text[:_MAX_HTML_FOR_REGEX]
    # Lightweight parse — DuckDuckGo HTML wraps results in <a class="result__a">.
    rx = _re.compile(r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', _re.DOTALL)
    items = rx.findall(text)
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

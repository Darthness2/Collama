"""GitHub REST API helpers + tool implementations.

Uses a Personal Access Token from config (`github.token`) or the
GITHUB_TOKEN / GH_TOKEN env vars. All tools that mutate (create issue,
raw POST/PATCH/DELETE) go through ToolContext.confirm().
"""
from __future__ import annotations

import json
import logging
import os
from urllib.parse import quote
from typing import Any

import requests

from .tools import ToolContext, _truncate

_log = logging.getLogger(__name__)

API = "https://api.github.com"
UA = "collama/0.1"

_insecure_warnings_disabled = False


def _disable_insecure_warnings() -> None:
    """Suppress urllib3's InsecureRequestWarning — but ONLY once the user has
    actually opted into insecure SSL (see ``_request``). Doing this at import
    time would silence the warning process-wide for everyone."""
    global _insecure_warnings_disabled
    if _insecure_warnings_disabled:
        return
    try:
        from urllib3.exceptions import InsecureRequestWarning  # type: ignore
        requests.packages.urllib3.disable_warnings(InsecureRequestWarning)  # type: ignore
        _insecure_warnings_disabled = True
    except Exception as e:
        _log.warning("could not disable InsecureRequestWarning: %s", e, exc_info=True)


def _token(ctx: ToolContext) -> str | None:
    if ctx.github_token:
        return ctx.github_token
    return os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")


def _headers(ctx: ToolContext) -> dict[str, str]:
    h = {
        "Accept": "application/vnd.github+json",
        "User-Agent": UA,
        "X-GitHub-Api-Version": "2022-11-28",
    }
    tok = _token(ctx)
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _need_token(ctx: ToolContext) -> str | None:
    if not _token(ctx):
        return "ERROR: no GitHub token set. Run `/login github <token>` or set GITHUB_TOKEN."
    return None


def _request(method: str, path: str, ctx: ToolContext, **kw) -> tuple[int, Any]:
    url = path if path.startswith("http") else f"{API}{path}"
    if ctx.insecure_ssl:
        kw.setdefault("verify", False)
        _disable_insecure_warnings()
    try:
        r = requests.request(method, url, headers=_headers(ctx), timeout=30, **kw)
    except requests.RequestException as e:
        return 0, {"error": str(e)}
    try:
        body = r.json()
    except ValueError:
        body = r.text
    return r.status_code, body


def _fmt(status: int, body: Any, limit: int = 6000) -> str:
    if status == 0:
        return f"ERROR: network: {body.get('error') if isinstance(body, dict) else body}"
    text = json.dumps(body, indent=2) if not isinstance(body, str) else body
    prefix = f"HTTP {status}\n"
    if status >= 400:
        return f"ERROR: {prefix}{_truncate(text, limit)}"
    return _truncate(prefix + text, limit)


# ---------- tools ----------

def t_gh_whoami(args: dict, ctx: ToolContext) -> str:
    err = _need_token(ctx)
    if err:
        return err
    status, body = _request("GET", "/user", ctx)
    if status == 200 and isinstance(body, dict):
        return f"login: {body.get('login')}\nname:  {body.get('name')}\nid:    {body.get('id')}\nplan:  {body.get('plan', {}).get('name', '?')}"
    return _fmt(status, body)


def t_gh_list_repos(args: dict, ctx: ToolContext) -> str:
    err = _need_token(ctx)
    if err:
        return err
    visibility = args.get("visibility", "all")
    sort = args.get("sort", "updated")
    per_page = min(int(args.get("limit", 30)), 100)
    status, body = _request(
        "GET", "/user/repos", ctx,
        params={"visibility": visibility, "sort": sort, "per_page": per_page},
    )
    if status != 200 or not isinstance(body, list):
        return _fmt(status, body)
    rows = [f"{r['full_name']:<50} {('private' if r.get('private') else 'public '):<8} {r.get('description') or ''}"
            for r in body]
    return _truncate("\n".join(rows) if rows else "(no repos)")


def t_gh_get_repo(args: dict, ctx: ToolContext) -> str:
    repo = quote(args["repo"], safe="/")
    status, body = _request("GET", f"/repos/{repo}", ctx)
    if status == 200 and isinstance(body, dict):
        keep = ["full_name", "description", "default_branch", "private", "fork",
                "language", "stargazers_count", "open_issues_count", "html_url"]
        return "\n".join(f"{k}: {body.get(k)}" for k in keep)
    return _fmt(status, body)


def t_gh_get_file(args: dict, ctx: ToolContext) -> str:
    import base64
    repo = quote(args["repo"], safe="/")
    path = quote(args["path"], safe="/")
    ref = args.get("ref")
    params = {"ref": ref} if ref else None
    status, body = _request("GET", f"/repos/{repo}/contents/{path}", ctx, params=params)
    if status == 200 and isinstance(body, dict) and body.get("type") == "file":
        try:
            data = base64.b64decode(body.get("content", "")).decode("utf-8", errors="replace")
        except Exception as e:
            return f"ERROR: decode failed: {e}"
        return _truncate(f"{repo}:{path}@{ref or body.get('sha', '')[:7]}\n{data}")
    return _fmt(status, body)


def t_gh_list_issues(args: dict, ctx: ToolContext) -> str:
    repo = quote(args["repo"], safe="/")
    state = args.get("state", "open")
    per_page = min(int(args.get("limit", 30)), 100)
    status, body = _request("GET", f"/repos/{repo}/issues", ctx,
                            params={"state": state, "per_page": per_page})
    if status != 200 or not isinstance(body, list):
        return _fmt(status, body)
    rows = []
    for it in body:
        if "pull_request" in it:
            continue
        rows.append(f"#{it['number']:<5} [{it['state']:<6}] {it['title']}  ({it['user']['login']})")
    return _truncate("\n".join(rows) if rows else "(no issues)")


def t_gh_create_issue(args: dict, ctx: ToolContext) -> str:
    err = _need_token(ctx)
    if err:
        return err
    repo = args["repo"]
    title = args["title"]
    body = args.get("body", "")
    labels = args.get("labels") or []
    if not ctx.confirm("create GitHub issue", f"{repo}: {title!r}"):
        return "ERROR: user denied"
    repo_q = quote(repo, safe="/")
    status, resp = _request("POST", f"/repos/{repo_q}/issues", ctx,
                            json={"title": title, "body": body, "labels": labels})
    if status == 201 and isinstance(resp, dict):
        return f"OK: created {resp.get('html_url')}"
    return _fmt(status, resp)


def t_gh_list_pulls(args: dict, ctx: ToolContext) -> str:
    repo = quote(args["repo"], safe="/")
    state = args.get("state", "open")
    per_page = min(int(args.get("limit", 30)), 100)
    status, body = _request("GET", f"/repos/{repo}/pulls", ctx,
                            params={"state": state, "per_page": per_page})
    if status != 200 or not isinstance(body, list):
        return _fmt(status, body)
    rows = [f"#{p['number']:<5} [{p['state']:<6}] {p['title']}  ({p['user']['login']} → {p['base']['ref']})"
            for p in body]
    return _truncate("\n".join(rows) if rows else "(no PRs)")


def t_gh_get_pull(args: dict, ctx: ToolContext) -> str:
    repo = quote(args["repo"], safe="/")
    number = int(args["number"])
    status, body = _request("GET", f"/repos/{repo}/pulls/{number}", ctx)
    if status == 200 and isinstance(body, dict):
        keep = ["number", "title", "state", "merged", "mergeable", "base", "head",
                "additions", "deletions", "changed_files", "html_url"]
        out = []
        for k in keep:
            v = body.get(k)
            if isinstance(v, dict) and "ref" in v:
                v = v["ref"]
            out.append(f"{k}: {v}")
        out.append("---")
        out.append((body.get("body") or "")[:2000])
        return "\n".join(out)
    return _fmt(status, body)


def t_gh_search_code(args: dict, ctx: ToolContext) -> str:
    err = _need_token(ctx)
    if err:
        return err
    q = args["query"]
    per_page = min(int(args.get("limit", 20)), 100)
    status, body = _request("GET", "/search/code", ctx,
                            params={"q": q, "per_page": per_page})
    if status != 200 or not isinstance(body, dict):
        return _fmt(status, body)
    items = body.get("items", [])
    rows = [f"{it['repository']['full_name']}  {it['path']}" for it in items]
    return _truncate("\n".join(rows) if rows else "(no matches)")


def t_github_api(args: dict, ctx: ToolContext) -> str:
    """Raw GitHub API call — escape hatch."""
    method = args.get("method", "GET").upper()
    path = args["path"]
    body = args.get("body")
    if method not in ("GET", "POST", "PATCH", "PUT", "DELETE"):
        return f"ERROR: unsupported method: {method}"
    if method != "GET":
        if not ctx.confirm("GitHub API write", f"{method} {path}"):
            return "ERROR: user denied"
    kw: dict[str, Any] = {}
    if body is not None and method != "GET":
        kw["json"] = body
    elif method == "GET" and isinstance(body, dict):
        kw["params"] = body
    status, resp = _request(method, path, ctx, **kw)
    return _fmt(status, resp)


GITHUB_TOOLS = {
    "gh_whoami": t_gh_whoami,
    "gh_list_repos": t_gh_list_repos,
    "gh_get_repo": t_gh_get_repo,
    "gh_get_file": t_gh_get_file,
    "gh_list_issues": t_gh_list_issues,
    "gh_create_issue": t_gh_create_issue,
    "gh_list_pulls": t_gh_list_pulls,
    "gh_get_pull": t_gh_get_pull,
    "gh_search_code": t_gh_search_code,
    "github_api": t_github_api,
}


GITHUB_TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "gh_whoami",
        "description": "Show the authenticated GitHub user.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "gh_list_repos",
        "description": "List repositories the user has access to.",
        "parameters": {"type": "object", "properties": {
            "visibility": {"type": "string", "enum": ["all", "public", "private"]},
            "sort": {"type": "string", "enum": ["created", "updated", "pushed", "full_name"]},
            "limit": {"type": "integer"},
        }},
    }},
    {"type": "function", "function": {
        "name": "gh_get_repo",
        "description": "Get metadata for a repository (owner/name).",
        "parameters": {"type": "object", "properties": {
            "repo": {"type": "string", "description": "owner/name"},
        }, "required": ["repo"]},
    }},
    {"type": "function", "function": {
        "name": "gh_get_file",
        "description": "Read a file from a GitHub repo at an optional ref (branch/tag/sha).",
        "parameters": {"type": "object", "properties": {
            "repo": {"type": "string"}, "path": {"type": "string"}, "ref": {"type": "string"},
        }, "required": ["repo", "path"]},
    }},
    {"type": "function", "function": {
        "name": "gh_list_issues",
        "description": "List issues in a repo (excludes PRs).",
        "parameters": {"type": "object", "properties": {
            "repo": {"type": "string"},
            "state": {"type": "string", "enum": ["open", "closed", "all"]},
            "limit": {"type": "integer"},
        }, "required": ["repo"]},
    }},
    {"type": "function", "function": {
        "name": "gh_create_issue",
        "description": "Create an issue. Requires user approval.",
        "parameters": {"type": "object", "properties": {
            "repo": {"type": "string"}, "title": {"type": "string"},
            "body": {"type": "string"},
            "labels": {"type": "array", "items": {"type": "string"}},
        }, "required": ["repo", "title"]},
    }},
    {"type": "function", "function": {
        "name": "gh_list_pulls",
        "description": "List pull requests for a repo.",
        "parameters": {"type": "object", "properties": {
            "repo": {"type": "string"},
            "state": {"type": "string", "enum": ["open", "closed", "all"]},
            "limit": {"type": "integer"},
        }, "required": ["repo"]},
    }},
    {"type": "function", "function": {
        "name": "gh_get_pull",
        "description": "Get details on a pull request.",
        "parameters": {"type": "object", "properties": {
            "repo": {"type": "string"}, "number": {"type": "integer"},
        }, "required": ["repo", "number"]},
    }},
    {"type": "function", "function": {
        "name": "gh_search_code",
        "description": "Search code across GitHub. Use the GitHub search query syntax.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "github_api",
        "description": "Raw GitHub REST API call — escape hatch for endpoints not covered by other tools. Non-GET requires approval.",
        "parameters": {"type": "object", "properties": {
            "method": {"type": "string", "enum": ["GET", "POST", "PATCH", "PUT", "DELETE"]},
            "path": {"type": "string", "description": "API path like /repos/owner/name/labels"},
            "body": {"type": "object", "description": "JSON body for non-GET; query params for GET."},
        }, "required": ["path"]},
    }},
]

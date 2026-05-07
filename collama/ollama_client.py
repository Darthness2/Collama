"""Thin wrapper around the Ollama HTTP API."""
from __future__ import annotations

import json
from typing import Any, Iterator

import requests


def _looks_like_tools_error(body: str) -> bool:
    b = body.lower()
    return (
        "does not support tools" in b
        or "tool" in b and ("not support" in b or "unsupported" in b or "no tool" in b)
    )


class OllamaError(RuntimeError):
    pass


class ToolsUnsupportedError(OllamaError):
    """Raised when the chosen model rejects tool calls (e.g. deepseek-coder)."""


class OllamaClient:
    def __init__(self, host: str = "http://localhost:11434", timeout: int = 600):
        self.host = host.rstrip("/")
        self.timeout = timeout

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            raise OllamaError(f"could not reach Ollama at {self.host}: {e}") from e
        return [m["name"] for m in r.json().get("models", [])]

    def chat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        options: dict | None = None,
    ) -> dict:
        """Non-streaming chat call. Returns the raw 'message' object."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
        if options:
            payload["options"] = options
        try:
            r = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            raise OllamaError(f"chat request failed: {e}") from e
        if r.status_code != 200:
            body = r.text[:500]
            if r.status_code == 400 and tools and _looks_like_tools_error(body):
                raise ToolsUnsupportedError(body)
            raise OllamaError(f"chat HTTP {r.status_code}: {body}")
        try:
            data = r.json()
        except json.JSONDecodeError as e:
            raise OllamaError(f"invalid JSON from Ollama: {e}") from e
        if "message" not in data:
            raise OllamaError(f"unexpected response: {data}")
        return data["message"]

    def chat_stream(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        options: dict | None = None,
    ) -> Iterator[dict]:
        """Streaming chat. Yields raw chunks (each has a 'message' delta)."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
        if options:
            payload["options"] = options
        try:
            with requests.post(
                f"{self.host}/api/chat",
                json=payload,
                timeout=self.timeout,
                stream=True,
            ) as r:
                if r.status_code != 200:
                    body = r.text[:500]
                    if r.status_code == 400 and tools and _looks_like_tools_error(body):
                        raise ToolsUnsupportedError(body)
                    raise OllamaError(f"chat HTTP {r.status_code}: {body}")
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
        except requests.RequestException as e:
            raise OllamaError(f"streaming chat failed: {e}") from e

    def chat_stream_assembled(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        options: dict | None = None,
    ) -> Iterator[tuple[str, Any]]:
        """Higher-level streaming: yields ('delta', text) events as content
        arrives, then a single ('done', payload) with the fully assembled
        message and usage counters.

            payload = {
                'message': {'role','content','tool_calls'},
                'eval_count': int,
                'prompt_eval_count': int,
                'total_duration_ns': int,
            }
        """
        full = ""
        for chunk in self.chat_stream(model, messages, tools, options):
            msg = chunk.get("message") or {}
            delta = msg.get("content") or ""
            if delta:
                full += delta
                yield ("delta", delta)
            if chunk.get("done"):
                yield ("done", {
                    "message": {
                        "role": msg.get("role", "assistant"),
                        "content": full,
                        "tool_calls": msg.get("tool_calls") or [],
                    },
                    "eval_count": int(chunk.get("eval_count") or 0),
                    "prompt_eval_count": int(chunk.get("prompt_eval_count") or 0),
                    "total_duration_ns": int(chunk.get("total_duration") or 0),
                })
                return

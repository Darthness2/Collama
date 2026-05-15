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
    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: int = 600,
        connect_timeout: float = 15.0,
        read_timeout: float = 600.0,
        nonstream_read_timeout: float = 1800.0,
        keep_alive: str | int | None = "30m",
        num_ctx: int | None = 8192,
    ):
        """Timeouts split by transport because they mean different things:

        - connect_timeout: seconds to establish the TCP connection.
        - read_timeout (STREAMING only): max gap BETWEEN streamed chunks.
          A 30-minute generation is fine as long as tokens keep arriving.
        - nonstream_read_timeout (NON-STREAMING only): whole-response
          wall-clock budget. Defaults to 30 min because non-streaming
          can't observe progress — if a request really is slow, we don't
          want a fast 'read_timeout' tuned for streaming to kill it.
        - keep_alive: how long Ollama keeps the model resident in VRAM
          ('30m', '1h', 0 to unload immediately, -1 to keep forever).
        - num_ctx: caps the context window Ollama allocates; protects
          VRAM from KV-cache balloon on big prompts.

        `timeout` is kept for back-compat: if you pass it and leave both
        read timeouts at their defaults, it overrides them both.
        """
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.connect_timeout = connect_timeout
        legacy_override = timeout != 600
        self.read_timeout = (
            float(timeout) if legacy_override and read_timeout == 600.0 else read_timeout
        )
        self.nonstream_read_timeout = (
            float(timeout) if legacy_override and nonstream_read_timeout == 1800.0 else nonstream_read_timeout
        )
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx

    def _apply_keep_alive(self, payload: dict) -> None:
        if self.keep_alive is not None and "keep_alive" not in payload:
            payload["keep_alive"] = self.keep_alive
        if self.num_ctx:
            opts = payload.setdefault("options", {})
            opts.setdefault("num_ctx", self.num_ctx)

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.host}/api/tags", timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            raise OllamaError(f"could not reach Ollama at {self.host}: {e}") from e
        return [m["name"] for m in r.json().get("models", [])]

    def unload(self, model: str) -> bool:
        """Tell Ollama to evict `model` from VRAM immediately.

        Ollama unloads a model when a request specifies keep_alive=0. We POST
        a no-op generate request with that flag. Best-effort: returns True on
        success, False otherwise. Used by Collama on shutdown so closing the
        window doesn't leave a 14B model resident in your GPU.
        """
        try:
            r = requests.post(
                f"{self.host}/api/generate",
                json={"model": model, "keep_alive": 0},
                timeout=5,
            )
            return r.status_code == 200
        except requests.RequestException:
            return False

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
        self._apply_keep_alive(payload)
        try:
            r = requests.post(
                f"{self.host}/api/chat",
                json=payload,
                # Non-streaming: the read timeout is the whole-response
                # budget, so it has to be generous on slow models / big
                # contexts. Streaming has its own per-chunk timeout below.
                timeout=(self.connect_timeout, self.nonstream_read_timeout),
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
        self._apply_keep_alive(payload)
        try:
            with requests.post(
                f"{self.host}/api/chat",
                json=payload,
                # (connect, read): the read timeout is the max gap BETWEEN
                # streamed chunks — not the whole-response budget. As long as
                # tokens keep flowing, a long generation never times out.
                timeout=(self.connect_timeout, self.read_timeout),
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
        # Tool calls can arrive in ANY chunk — not necessarily the final
        # `done` one. Accumulate them across the whole stream; reading only
        # the done chunk silently drops calls and the turn renders nothing.
        tool_calls: list[dict] = []
        role = "assistant"
        for chunk in self.chat_stream(model, messages, tools, options):
            msg = chunk.get("message") or {}
            if msg.get("role"):
                role = msg["role"]
            delta = msg.get("content") or ""
            if delta:
                full += delta
                yield ("delta", delta)
            tc = msg.get("tool_calls")
            if tc:
                tool_calls.extend(tc)
            if chunk.get("done"):
                yield ("done", {
                    "message": {
                        "role": role,
                        "content": full,
                        "tool_calls": tool_calls,
                    },
                    "eval_count": int(chunk.get("eval_count") or 0),
                    "prompt_eval_count": int(chunk.get("prompt_eval_count") or 0),
                    "total_duration_ns": int(chunk.get("total_duration") or 0),
                })
                return

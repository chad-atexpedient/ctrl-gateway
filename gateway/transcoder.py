"""Per-endkind payload transcoding.

Adapters handle differences between provider APIs:
  - OpenAI-compatible (openai, llamacpp, groq, together, deepseek, etc.)
  - Anthropic Claude (native Messages API)
  - Google Gemini (native generateContent API)
  - Ollama (native /api/chat)

Each adapter encodes an OpenAI-format request into the provider's native
format. For non-OpenAI providers (anthropic, gemini), a response_decoder
callback is attached so the endpoint client can unwrap the native response
back into OpenAI format after receiving it.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from typing import Any

log = logging.getLogger("ctrl.transcoder")


class TranscodedRequest:
    __slots__ = ("url", "headers", "body", "endpoint_kind", "response_decoder", "stream_decoder")

    def __init__(
        self,
        url: str,
        headers: dict,
        body: dict,
        endpoint_kind: str,
        response_decoder: Callable[[dict], dict] | None = None,
        stream_decoder: Callable[[AsyncIterator[bytes]], AsyncIterator[bytes]] | None = None,
    ):
        self.url = url
        self.headers = headers
        self.body = body
        self.endpoint_kind = endpoint_kind
        self.response_decoder = response_decoder
        self.stream_decoder = stream_decoder


def _api_key(endpoint_cfg: dict) -> str:
    direct = endpoint_cfg.get("_api_key")
    if isinstance(direct, str) and direct:
        return direct
    key_env = endpoint_cfg.get("api_key_env")
    return os.environ.get(key_env, "") if key_env else ""


def _encode_openai_compat(
    *,
    kind: str,
    base_url: str,
    endpoint_cfg: dict,
    payload: dict,
    max_tokens_bump: int = 0,
    path: str,
    strip_model_alias: bool = False,
) -> TranscodedRequest:
    """Shared logic for OpenAI-compatible adapters (llama.cpp + openai kinds)."""
    body = dict(payload)
    if not strip_model_alias:
        body["model"] = endpoint_cfg.get("model_alias", "default")
    else:
        body.setdefault("model", endpoint_cfg.get("model_alias", "default"))
    body.setdefault("stream", False)
    if "max_completion_tokens" in body and "max_tokens" not in body:
        body["max_tokens"] = body.pop("max_completion_tokens")
    if max_tokens_bump > 0:
        cur = body.get("max_tokens") or 0
        body["max_tokens"] = max(cur, max_tokens_bump)
    headers = {"Content-Type": "application/json"}
    key = _api_key(endpoint_cfg)
    if key:
        headers["Authorization"] = f"Bearer {key}"
    return TranscodedRequest(
        url=f"{base_url.rstrip('/')}{path}",
        headers=headers,
        body=body,
        endpoint_kind=kind,
    )


class _LlamaCppAdapter:
    """llama.cpp with server-style OpenAI-compatible /v1/chat/completions."""

    kind = "llamacpp"

    def encode(
        self,
        base_url: str,
        endpoint_cfg: dict,
        payload: dict,
        max_tokens_bump: int = 0,
    ) -> TranscodedRequest:
        return _encode_openai_compat(
            kind=self.kind,
            base_url=base_url,
            endpoint_cfg=endpoint_cfg,
            payload=payload,
            max_tokens_bump=max_tokens_bump,
            path="/v1/chat/completions",
            strip_model_alias=True,
        )


class _OpenAIAdapter:
    """Generic OpenAI-compatible API."""

    kind = "openai"

    def encode(
        self,
        base_url: str,
        endpoint_cfg: dict,
        payload: dict,
        max_tokens_bump: int = 0,
    ) -> TranscodedRequest:
        return _encode_openai_compat(
            kind=self.kind,
            base_url=base_url,
            endpoint_cfg=endpoint_cfg,
            payload=payload,
            max_tokens_bump=max_tokens_bump,
            path="/chat/completions",
        )


class _OllamaAdapter:
    """Ollama needs explicit model name + custom path."""

    kind = "ollama"

    def encode(
        self,
        base_url: str,
        endpoint_cfg: dict,
        payload: dict,
        max_tokens_bump: int = 0,
    ) -> TranscodedRequest:
        body = dict(payload)
        # Ollama wants /api/chat with model field
        model = endpoint_cfg.get("model_alias", "llama3")
        body["model"] = model
        body.setdefault("stream", False)
        # max_tokens -> num_predict
        if "max_completion_tokens" in body and "max_tokens" not in body:
            body["max_tokens"] = body.pop("max_completion_tokens")
        if "max_tokens" in body:
            body["num_predict"] = body.pop("max_tokens")
        if max_tokens_bump > 0:
            cur = body.get("num_predict") or 0
            body["num_predict"] = max(cur, max_tokens_bump)
        response_format = body.pop("response_format", None)
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            body["format"] = "json"
        # Ollama supports OpenAI-style tools, but not these legacy controls.
        for k in ("tool_choice", "functions", "function_call"):
            body.pop(k, None)
        return TranscodedRequest(
            url=f"{base_url.rstrip('/')}/api/chat",
            headers={"Content-Type": "application/json"},
            body=body,
            endpoint_kind=self.kind,
            response_decoder=_decode_ollama_response,
            stream_decoder=_decode_ollama_stream,
        )


def _decode_ollama_response(raw: dict) -> dict:
    message = raw.get("message") or {}
    prompt_tokens = int(raw.get("prompt_eval_count") or 0)
    completion_tokens = int(raw.get("eval_count") or 0)
    out_message = {
        "role": message.get("role", "assistant"),
        "content": message.get("content", ""),
    }
    if message.get("tool_calls"):
        out_message["tool_calls"] = _normalize_tool_calls(message["tool_calls"])
    return {
        "id": raw.get("id") or f"chatcmpl-ollama-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": raw.get("model", ""),
        "choices": [{
            "index": 0,
            "message": out_message,
            "finish_reason": _openai_finish_reason(raw.get("done_reason")) if raw.get("done") else None,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _decode_ollama_stream(source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    async for raw in _iter_json_lines(source):
        message = raw.get("message") or {}
        delta = {}
        if message.get("content"):
            delta["content"] = message["content"]
        if message.get("tool_calls"):
            delta["tool_calls"] = message["tool_calls"]
        payload = _openai_stream_chunk(
            chunk_id=raw.get("id") or "chatcmpl-ollama",
            model=raw.get("model", ""),
            delta=delta,
            finish_reason=_openai_finish_reason(raw.get("done_reason")) if raw.get("done") else None,
        )
        yield _sse(payload)
    yield b"data: [DONE]\n\n"


class _AnthropicAdapter:
    """Anthropic Claude native Messages API.

    Translates OpenAI chat-completions format → Anthropic Messages format:
      - system messages become a top-level 'system' field
      - POST /v1/messages with x-api-key header
      - max_tokens is required
      - Response is unwrapped via _decode_anthropic_response
    """

    kind = "anthropic"

    def encode(
        self,
        base_url: str,
        endpoint_cfg: dict,
        payload: dict,
        max_tokens_bump: int = 0,
    ) -> TranscodedRequest:
        src = dict(payload)
        messages = src.get("messages", [])

        # Extract system messages into top-level 'system' field
        system_parts: list[str] = []
        chat_messages: list[dict] = []
        for m in messages:
            if m.get("role") in ("system", "developer"):
                system_parts.append(_text_content(m.get("content", "")))
            elif m.get("role") == "tool":
                chat_messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": m.get("tool_call_id", ""),
                        "content": _text_content(m.get("content", "")),
                    }],
                })
            else:
                content = _anthropic_content(m.get("content", ""))
                if m.get("role") == "assistant" and m.get("tool_calls"):
                    content = content if isinstance(content, list) else [{"type": "text", "text": str(content)}]
                    content.extend(_anthropic_tool_calls(m["tool_calls"]))
                chat_messages.append({"role": m.get("role", "user"), "content": content})

        body: dict[str, Any] = {
            "model": endpoint_cfg.get("model_alias", "claude-sonnet-4-20250514"),
            "messages": chat_messages,
            "max_tokens": src.get("max_tokens") or src.get("max_completion_tokens") or 4096,
        }
        if src.get("temperature") is not None:
            body["temperature"] = src["temperature"]
        if src.get("top_p") is not None:
            body["top_p"] = src["top_p"]
        if src.get("stop"):
            body["stop_sequences"] = src["stop"] if isinstance(src["stop"], list) else [src["stop"]]
        if src.get("tools"):
            body["tools"] = [
                {
                    "name": tool.get("function", {}).get("name", ""),
                    "description": tool.get("function", {}).get("description", ""),
                    "input_schema": tool.get("function", {}).get("parameters", {"type": "object"}),
                }
                for tool in src["tools"] if tool.get("type", "function") == "function"
            ]
        if src.get("tool_choice"):
            choice = src["tool_choice"]
            if choice == "auto":
                body["tool_choice"] = {"type": "auto"}
            elif choice == "required":
                body["tool_choice"] = {"type": "any"}
            elif isinstance(choice, dict) and choice.get("function", {}).get("name"):
                body["tool_choice"] = {"type": "tool", "name": choice["function"]["name"]}
        if src.get("stream"):
            body["stream"] = True
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        if max_tokens_bump > 0:
            body["max_tokens"] = max(body["max_tokens"], max_tokens_bump)

        api_key = ""
        api_key = _api_key(endpoint_cfg)

        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if api_key:
            headers["x-api-key"] = api_key

        base = base_url.rstrip("/")
        path = "/v1/messages"
        # Handle base_url that already includes /v1
        if base.endswith("/v1"):
            path = "/messages"

        return TranscodedRequest(
            url=f"{base}{path}",
            headers=headers,
            body=body,
            endpoint_kind=self.kind,
            response_decoder=_decode_anthropic_response,
            stream_decoder=_decode_anthropic_stream,
        )


def _decode_anthropic_response(raw: dict) -> dict:
    """Unwrap Anthropic Messages response → OpenAI chat.completion format."""
    content = ""
    tool_calls = []
    for block in raw.get("content", []):
        if isinstance(block, dict) and block.get("type") == "text":
            content += block.get("text", "")
        elif isinstance(block, dict) and block.get("type") == "tool_use":
            tool_calls.append({
                "id": block.get("id", ""),
                "type": "function",
                "function": {
                    "name": block.get("name", ""),
                    "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                },
            })
    usage = raw.get("usage", {})
    stop = raw.get("stop_reason", "end_turn")
    finish = _openai_finish_reason(stop)
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        "id": raw.get("id", "anthropic-msg"),
        "object": "chat.completion",
        "model": raw.get("model", ""),
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        },
    }


async def _decode_anthropic_stream(source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    message_id = "chatcmpl-anthropic"
    model = ""
    # Anthropic reports real token usage in-band: message_start carries the
    # (accurate) input_tokens up front, message_delta carries the cumulative
    # output_tokens near the end. Previously both were read only for
    # message_id/model/stop_reason and the usage sub-object was discarded, so
    # the gateway had no way to settle billing/quotas on real streamed token
    # counts and silently fell back to pre-request estimates for every
    # streamed Anthropic response. Track them and emit a trailing OpenAI-style
    # usage chunk (choices: [], usage: {...} — the same shape OpenAI itself
    # uses for stream_options.include_usage) so callers that look for it,
    # including this gateway's own settlement code, see real numbers.
    input_tokens = 0
    output_tokens = 0
    async for event in _iter_sse_data(source):
        event_type = event.get("type")
        if event_type == "message_start":
            message = event.get("message") or {}
            message_id = message.get("id") or message_id
            model = message.get("model") or model
            usage = message.get("usage") or {}
            input_tokens = int(usage.get("input_tokens", 0) or 0)
            output_tokens = int(usage.get("output_tokens", 0) or 0)
            yield _sse(_openai_stream_chunk(message_id, model, {"role": "assistant"}, None))
        elif event_type == "content_block_delta":
            delta = event.get("delta") or {}
            if delta.get("type") == "text_delta":
                yield _sse(_openai_stream_chunk(message_id, model, {"content": delta.get("text", "")}, None))
        elif event_type == "message_delta":
            usage = event.get("usage") or {}
            if "input_tokens" in usage:
                input_tokens = int(usage.get("input_tokens", 0) or 0)
            if "output_tokens" in usage:
                output_tokens = int(usage.get("output_tokens", 0) or 0)
            finish = _openai_finish_reason((event.get("delta") or {}).get("stop_reason"))
            yield _sse(_openai_stream_chunk(message_id, model, {}, finish))
    if input_tokens or output_tokens:
        yield _sse(_openai_usage_chunk(message_id, model, input_tokens, output_tokens))
    yield b"data: [DONE]\n\n"


# ---- Inbound Anthropic Messages API support (POST /v1/messages on this
# gateway itself) -----------------------------------------------------------
#
# Everything above this point translates OpenAI-pivot -> Anthropic for
# OUTBOUND calls to an Anthropic upstream. The functions below are the
# mirror image for the INBOUND side: a client calling this gateway's own
# /v1/messages endpoint in Anthropic's wire format, translated to/from the
# same OpenAI pivot the rest of the gateway (routing, policy, memory,
# billing) already speaks -- see app.py's chat_completions(wire_format=...)
# and anthropic_messages().


def decode_inbound_anthropic_request(body: dict) -> dict:
    """Convert an inbound Anthropic Messages API request body into the
    gateway's internal OpenAI chat-completions pivot format (the same shape
    chat_completions() already parses for native OpenAI-format callers).
    Reverse of _AnthropicAdapter.encode(). Pure dict->dict, like every
    other _decode_*/encode_* function in this module, so it's testable
    without the HTTP layer.
    """
    out: dict[str, Any] = {}
    if "model" in body:
        out["model"] = body["model"]

    messages: list[dict] = []
    system = body.get("system")
    if isinstance(system, str) and system:
        messages.append({"role": "system", "content": system})
    elif isinstance(system, list):
        text = "\n\n".join(
            part.get("text", "") for part in system
            if isinstance(part, dict) and part.get("type") == "text"
        )
        if text:
            messages.append({"role": "system", "content": text})

    for m in body.get("messages", []):
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue
        if not isinstance(content, list):
            messages.append({"role": role, "content": _text_content(content)})
            continue

        if role == "assistant":
            text_parts = []
            tool_calls = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif part.get("type") == "tool_use":
                    tool_calls.append({
                        "id": part.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": part.get("name", ""),
                            "arguments": json.dumps(part.get("input", {}), ensure_ascii=False),
                        },
                    })
            out_message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
            if tool_calls:
                out_message["tool_calls"] = tool_calls
            messages.append(out_message)
            continue

        # Non-assistant role (in practice always "user"): tool_result blocks
        # each become their own OpenAI `role: tool` message; remaining
        # text/image blocks become one regular message with OpenAI-style
        # content parts.
        openai_parts: list[dict] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type")
            if ptype == "tool_result":
                messages.append({
                    "role": "tool",
                    "tool_call_id": part.get("tool_use_id", ""),
                    "content": _text_content(part.get("content", "")),
                })
            elif ptype == "text":
                openai_parts.append({"type": "text", "text": part.get("text", "")})
            elif ptype == "image":
                source = part.get("source") or {}
                if source.get("type") == "base64":
                    media_type = source.get("media_type", "image/png")
                    data = source.get("data", "")
                    openai_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{media_type};base64,{data}"},
                    })
                elif source.get("type") == "url":
                    openai_parts.append({"type": "image_url", "image_url": {"url": source.get("url", "")}})
        if openai_parts:
            if len(openai_parts) == 1 and openai_parts[0]["type"] == "text":
                messages.append({"role": role, "content": openai_parts[0]["text"]})
            else:
                messages.append({"role": role, "content": openai_parts})

    out["messages"] = messages
    if "max_tokens" in body:
        out["max_tokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        out["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        out["top_p"] = body["top_p"]
    if body.get("stop_sequences"):
        out["stop"] = body["stop_sequences"]
    if "stream" in body:
        out["stream"] = bool(body["stream"])
    if body.get("tools"):
        out["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": tool.get("input_schema", {"type": "object"}),
                },
            }
            for tool in body["tools"]
        ]
    if body.get("tool_choice"):
        choice = body["tool_choice"]
        ctype = choice.get("type") if isinstance(choice, dict) else None
        if ctype == "auto":
            out["tool_choice"] = "auto"
        elif ctype == "any":
            out["tool_choice"] = "required"
        elif ctype == "tool" and choice.get("name"):
            out["tool_choice"] = {"type": "function", "function": {"name": choice["name"]}}
    return out


def _anthropic_stop_reason(reason: str | None) -> str:
    """Map an OpenAI-pivot finish_reason back to an Anthropic stop_reason."""
    if not reason:
        return "end_turn"
    normalized = str(reason).lower()
    if normalized in ("length", "max_tokens"):
        return "max_tokens"
    if normalized in ("tool_calls", "tool_use", "function_call"):
        return "tool_use"
    return "end_turn"


def encode_outbound_anthropic_response(resp: dict) -> dict:
    """Convert the gateway's internal OpenAI chat.completion response (the
    same shape chat_completions() returns to native OpenAI-format callers)
    into an Anthropic Messages API response, for the /v1/messages
    non-streaming path. Reverse of _decode_anthropic_response.
    """
    choice = (resp.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content: list[dict[str, Any]] = []
    text = message.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    elif isinstance(text, list):
        # Defensive: internal pivot content is normally a plain string for
        # assistant messages, but tolerate an OpenAI-style content-part list.
        joined = _text_content(text)
        if joined:
            content.append({"type": "text", "text": joined})
    for call in message.get("tool_calls") or []:
        function = call.get("function", {})
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        content.append({
            "type": "tool_use",
            "id": call.get("id", ""),
            "name": function.get("name", ""),
            "input": arguments,
        })
    usage = resp.get("usage") or {}
    return {
        "id": resp.get("id") or ("msg_" + uuid.uuid4().hex[:24]),
        "type": "message",
        "role": "assistant",
        "model": resp.get("model", ""),
        "content": content,
        "stop_reason": _anthropic_stop_reason(choice.get("finish_reason")),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _anthropic_sse(event_type: str, payload: dict) -> bytes:
    """Anthropic's streaming format names each event (unlike OpenAI's
    anonymous `data: {...}` frames): `event: <type>\\ndata: <json>\\n\\n`."""
    return f"event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


class AnthropicOutboundStreamEncoder:
    """Stateful translator: internal OpenAI-pivot SSE bytes -> Anthropic
    Messages API SSE bytes, for the /v1/messages streaming response path.

    By the time chat_completions() has bytes to write to the client, they
    are ALWAYS OpenAI-pivot-shaped SSE (chat.completion.chunk), regardless
    of which upstream actually served the request: EndpointClient.send()
    already runs the upstream's stream_decoder (_decode_anthropic_stream /
    _decode_ollama_stream / _decode_gemini_stream above) before app.py ever
    sees a byte, and openai/llamacpp-kind upstreams are already
    OpenAI-shaped. So this encoder only has to understand ONE input shape,
    no matter which upstream served the request.

    Raw passthrough chunks (openai/llamacpp kinds, which have no
    stream_decoder) are not guaranteed to land on SSE frame boundaries, so
    this buffers partial frames the same way _iter_sse_data does, rather
    than assuming one JSON object per feed() call.

    Text-delta streaming is fully translated. A tool_calls delta is
    accumulated and emitted as a single (non-incremental) tool_use content
    block at stream end instead of streamed token-by-token -- matching the
    scope of _decode_anthropic_stream above, which likewise only forwards
    text_delta content and not incremental tool-call streaming from a real
    Anthropic upstream. Extending both directions to stream tool-call
    arguments incrementally would need a shared design; not attempted here.
    """

    def __init__(self, message_id: str | None = None):
        self._buffer = b""
        self._message_id = message_id or ("msg_" + uuid.uuid4().hex[:24])
        self._model = ""
        self._started = False
        self._text_block_open = False
        self._output_tokens = 0
        self._finish_reason: str | None = None
        self._tool_calls: dict[int, dict[str, str]] = {}

    def feed(self, chunk: bytes) -> bytes:
        self._buffer += chunk
        out = bytearray()
        while b"\n\n" in self._buffer:
            frame, self._buffer = self._buffer.split(b"\n\n", 1)
            out += self._handle_frame(frame)
        return bytes(out)

    def finish(self) -> bytes:
        out = bytearray()
        if self._buffer.strip():
            out += self._handle_frame(self._buffer)
            self._buffer = b""
        out += self._close()
        return bytes(out)

    def _handle_frame(self, frame: bytes) -> bytes:
        out = bytearray()
        for line in frame.splitlines():
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if not data or data == b"[DONE]":
                continue
            try:
                payload = json.loads(data)
            except json.JSONDecodeError:
                continue
            out += self._handle_payload(payload)
        return bytes(out)

    def _handle_payload(self, payload: dict) -> bytes:
        out = bytearray()
        if not self._started:
            self._started = True
            self._model = payload.get("model") or self._model
            out += _anthropic_sse("message_start", {
                "type": "message_start",
                "message": {
                    "id": self._message_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": self._model,
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            })
        choices = payload.get("choices") or []
        if choices:
            choice = choices[0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                if not self._text_block_open:
                    self._text_block_open = True
                    out += _anthropic_sse("content_block_start", {
                        "type": "content_block_start",
                        "index": 0,
                        "content_block": {"type": "text", "text": ""},
                    })
                out += _anthropic_sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": delta["content"]},
                })
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = self._tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                function = tc.get("function") or {}
                if function.get("name"):
                    slot["name"] = function["name"]
                if function.get("arguments"):
                    slot["arguments"] += function["arguments"]
            if choice.get("finish_reason"):
                self._finish_reason = choice["finish_reason"]
        usage = payload.get("usage")
        if usage and "completion_tokens" in usage:
            self._output_tokens = int(usage.get("completion_tokens", 0) or 0)
        return bytes(out)

    def _close(self) -> bytes:
        out = bytearray()
        if not self._started:
            # Nothing was ever emitted (e.g. the stream failed before any
            # chunk arrived) -- don't emit a dangling message_delta/stop
            # with no matching message_start.
            return bytes(out)
        next_index = 0
        if self._text_block_open:
            out += _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": 0})
            next_index = 1
        for idx in sorted(self._tool_calls):
            slot = self._tool_calls[idx]
            block_index = next_index
            next_index += 1
            arguments = slot["arguments"]
            try:
                parsed_input = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                parsed_input = {"raw": arguments}
            out += _anthropic_sse("content_block_start", {
                "type": "content_block_start",
                "index": block_index,
                "content_block": {"type": "tool_use", "id": slot["id"], "name": slot["name"], "input": {}},
            })
            out += _anthropic_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": block_index,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(parsed_input, ensure_ascii=False)},
            })
            out += _anthropic_sse("content_block_stop", {"type": "content_block_stop", "index": block_index})
        stop_reason = self._finish_reason
        if stop_reason is None and self._tool_calls:
            stop_reason = "tool_calls"
        out += _anthropic_sse("message_delta", {
            "type": "message_delta",
            "delta": {"stop_reason": _anthropic_stop_reason(stop_reason), "stop_sequence": None},
            "usage": {"output_tokens": self._output_tokens},
        })
        out += _anthropic_sse("message_stop", {"type": "message_stop"})
        return bytes(out)


class _GeminiAdapter:
    """Google Gemini native generateContent API.

    Translates OpenAI format → Gemini format:
      - messages → contents[].parts[].text
      - system messages → systemInstruction.parts[].text
      - model is in the URL path, api_key in query string
      - Response unwrapped via _decode_gemini_response
    """

    kind = "gemini"

    def encode(
        self,
        base_url: str,
        endpoint_cfg: dict,
        payload: dict,
        max_tokens_bump: int = 0,
    ) -> TranscodedRequest:
        src = dict(payload)
        messages = src.get("messages", [])
        model = endpoint_cfg.get("model_alias", "gemini-2.0-flash")

        system_parts: list[str] = []
        contents: list[dict] = []
        for m in messages:
            if m.get("role") in ("system", "developer"):
                system_parts.append(_text_content(m.get("content", "")))
            else:
                role = "model" if m.get("role") == "assistant" else "user"
                parts = _gemini_parts(m)
                contents.append({"role": role, "parts": parts})

        body: dict[str, Any] = {"contents": contents}
        if src.get("temperature") is not None:
            body["generationConfig"] = {"temperature": src["temperature"]}
        requested_tokens = src.get("max_tokens") or src.get("max_completion_tokens")
        if max_tokens_bump > 0 or requested_tokens:
            gc = body.setdefault("generationConfig", {})
            gc["maxOutputTokens"] = requested_tokens or max_tokens_bump
        if system_parts:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        if src.get("tools"):
            body["tools"] = [{
                "functionDeclarations": [
                    {
                        "name": tool.get("function", {}).get("name", ""),
                        "description": tool.get("function", {}).get("description", ""),
                        "parameters": tool.get("function", {}).get("parameters", {"type": "object"}),
                    }
                    for tool in src["tools"] if tool.get("type", "function") == "function"
                ],
            }]

        api_key = ""
        api_key = _api_key(endpoint_cfg)

        base = base_url.rstrip("/")
        action = "streamGenerateContent?alt=sse" if src.get("stream") else "generateContent"
        url = f"{base}/v1beta/models/{model}:{action}"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["x-goog-api-key"] = api_key

        return TranscodedRequest(
            url=url,
            headers=headers,
            body=body,
            endpoint_kind=self.kind,
            response_decoder=_decode_gemini_response,
            stream_decoder=_decode_gemini_stream,
        )


def _decode_gemini_response(raw: dict) -> dict:
    """Unwrap Gemini generateContent response → OpenAI chat.completion format."""
    content = ""
    candidates = raw.get("candidates", [])
    tool_calls = []
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            if isinstance(part, dict):
                content += part.get("text", "")
                if part.get("functionCall"):
                    call = part["functionCall"]
                    tool_calls.append({
                        "id": "call_" + uuid.uuid4().hex[:12],
                        "type": "function",
                        "function": {
                            "name": call.get("name", ""),
                            "arguments": json.dumps(call.get("args", {}), ensure_ascii=False),
                        },
                    })
    usage = raw.get("usageMetadata", {})
    finish = "stop"
    if candidates and candidates[0].get("finishReason"):
        fr = candidates[0]["finishReason"]
        finish = "stop" if fr == "STOP" else fr.lower()
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls:
        message["tool_calls"] = tool_calls
        finish = "tool_calls"
    return {
        "id": "gemini-" + str(raw.get("responseId", "")),
        "object": "chat.completion",
        "model": "",
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish,
        }],
        "usage": {
            "prompt_tokens": usage.get("promptTokenCount", 0),
            "completion_tokens": usage.get("candidatesTokenCount", 0),
            "total_tokens": usage.get("totalTokenCount", 0),
        },
    }


async def _decode_gemini_stream(source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    chunk_id = "chatcmpl-gemini"
    async for raw in _iter_sse_data(source):
        candidates = raw.get("candidates") or []
        if not candidates:
            continue
        candidate = candidates[0]
        text = "".join(
            part.get("text", "") for part in candidate.get("content", {}).get("parts", [])
            if isinstance(part, dict)
        )
        finish = candidate.get("finishReason")
        yield _sse(_openai_stream_chunk(
            chunk_id,
            raw.get("modelVersion", ""),
            {"content": text} if text else {},
            _openai_finish_reason(finish) if finish else None,
        ))
    yield b"data: [DONE]\n\n"


def _text_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", "")) for part in content
            if isinstance(part, dict) and part.get("type") in ("text", "input_text")
        )
    return str(content or "")


def _anthropic_content(content):
    if not isinstance(content, list):
        return _text_content(content)
    blocks = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in ("text", "input_text"):
            blocks.append({"type": "text", "text": part.get("text", "")})
        elif part.get("type") in ("image_url", "input_image"):
            image = part.get("image_url") or part.get("image") or {}
            url = image.get("url", "") if isinstance(image, dict) else str(image)
            if url.startswith("data:") and ";base64," in url:
                header, data = url.split(",", 1)
                blocks.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": header[5:].split(";", 1)[0],
                        "data": data,
                    },
                })
            elif url:
                blocks.append({"type": "image", "source": {"type": "url", "url": url}})
    return blocks


def _anthropic_tool_calls(tool_calls: list[dict]) -> list[dict]:
    blocks = []
    for call in tool_calls:
        function = call.get("function", {})
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        blocks.append({
            "type": "tool_use",
            "id": call.get("id", ""),
            "name": function.get("name", ""),
            "input": arguments,
        })
    return blocks


def _normalize_tool_calls(tool_calls: list[dict]) -> list[dict]:
    normalized = []
    for index, call in enumerate(tool_calls):
        function = call.get("function", {})
        arguments = function.get("arguments", {})
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, ensure_ascii=False)
        normalized.append({
            "id": call.get("id") or f"call_{index}",
            "type": "function",
            "function": {"name": function.get("name", ""), "arguments": arguments},
        })
    return normalized


def _gemini_parts(message: dict) -> list[dict]:
    if message.get("role") == "tool":
        return [{
            "functionResponse": {
                "name": message.get("name", "tool"),
                "response": {"result": _text_content(message.get("content", ""))},
            },
        }]
    parts: list[dict[str, Any]] = []
    content = message.get("content", "")
    if isinstance(content, str):
        parts.append({"text": content})
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") in ("text", "input_text"):
                parts.append({"text": part.get("text", "")})
            elif part.get("type") in ("image_url", "input_image"):
                image = part.get("image_url") or part.get("image") or {}
                url = image.get("url", "") if isinstance(image, dict) else str(image)
                if url.startswith("data:") and ";base64," in url:
                    header, data = url.split(",", 1)
                    parts.append({"inlineData": {"mimeType": header[5:].split(";", 1)[0], "data": data}})
                elif url:
                    parts.append({"fileData": {"fileUri": url}})
    for call in message.get("tool_calls", []):
        function = call.get("function", {})
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}
        parts.append({"functionCall": {"name": function.get("name", ""), "args": arguments}})
    return parts or [{"text": ""}]


def _openai_finish_reason(reason: str | None) -> str:
    if not reason:
        return "stop"
    normalized = str(reason).lower()
    if normalized in ("stop", "end_turn", "stop_sequence"):
        return "stop"
    if normalized in ("length", "max_tokens"):
        return "length"
    if normalized in ("tool_use", "tool_calls"):
        return "tool_calls"
    if normalized in ("content_filter", "safety", "recitation"):
        return "content_filter"
    return normalized


def _openai_stream_chunk(
    chunk_id: str,
    model: str,
    delta: dict,
    finish_reason: str | None,
) -> dict:
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }


def _openai_usage_chunk(chunk_id: str, model: str, input_tokens: int, output_tokens: int) -> dict:
    """A trailing SSE chunk carrying real token usage, matching the shape
    OpenAI itself sends when a client requests stream_options.include_usage:
    an empty `choices` list plus a top-level `usage` object."""
    return {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


def _sse(payload: dict) -> bytes:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode()


async def _iter_json_lines(source: AsyncIterator[bytes]) -> AsyncIterator[dict]:
    buffer = b""
    async for chunk in source:
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            line = line.strip()
            if line:
                yield json.loads(line)
    if buffer.strip():
        yield json.loads(buffer)


async def _iter_sse_data(source: AsyncIterator[bytes]) -> AsyncIterator[dict]:
    buffer = b""
    async for chunk in source:
        buffer += chunk
        while b"\n\n" in buffer:
            frame, buffer = buffer.split(b"\n\n", 1)
            for line in frame.splitlines():
                if line.startswith(b"data:"):
                    data = line[5:].strip()
                    if data and data != b"[DONE]":
                        yield json.loads(data)
    for line in buffer.splitlines():
        if line.startswith(b"data:"):
            data = line[5:].strip()
            if data and data != b"[DONE]":
                yield json.loads(data)


_ADAPTERS: dict[str, Any] = {
    "llamacpp": _LlamaCppAdapter(),
    "openai": _OpenAIAdapter(),
    "ollama": _OllamaAdapter(),
    "anthropic": _AnthropicAdapter(),
    "gemini": _GeminiAdapter(),
}


def get_adapter(kind: str):
    if kind not in _ADAPTERS:
        raise ValueError(f"unknown endkind '{kind}'")
    return _ADAPTERS[kind]


def transcode(
    endpoint_cfg: dict,
    tier_cfg: dict,
    payload: dict,
) -> TranscodedRequest:
    adapter = get_adapter(endpoint_cfg["kind"])
    return adapter.encode(
        base_url=endpoint_cfg["base_url"],
        endpoint_cfg=endpoint_cfg,
        payload=payload,
        max_tokens_bump=tier_cfg.get("max_tokens_bump", 0),
    )

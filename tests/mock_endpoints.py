"""Mock LLM upstream endpoints for integration tests.

Each mock serves the OpenAI-compatible /v1/chat/completions (llamacpp kind)
and /chat/completions (openai kind) plus a health probe. Behavior is scripted
per endpoint name via the SCRIPT_* env vars:

  MOCK_FAIL_ENDPOINTS   comma-separated endpoint names that always 500
  MOCK_BREAK_ENDPOINTS  comma-separated names that fail first N requests then succeed
  MOCK_LATENCY_MS       base latency for responses (default 1)

Used by tests/test_integration.py. Run standalone:
  python tests/mock_endpoints.py --port 8091
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

from aiohttp import web


def _fail_endpoints() -> set:
    return set(filter(None, os.environ.get("MOCK_FAIL_ENDPOINTS", "").split(",")))


def _breaker_endpoints() -> dict:
    """name -> remaining failures before success."""
    out = {}
    for name in filter(None, os.environ.get("MOCK_BREAK_ENDPOINTS", "").split(",")):
        out[name] = int(os.environ.get(f"MOCK_BREAK_FAILS_{name}", "2"))
    return out


async def health(request: web.Request):
    # Honor MOCK_FAIL_ENDPOINTS so health probes can be scripted down too
    if request.app["name"] in _fail_endpoints():
        return web.json_response({"status": "down"}, status=503)
    return web.json_response({"status": "ok"})


async def chat_completions(request: web.Request):
    body = await request.json()
    name = request.app["name"]

    # Scripted failure
    if name in _fail_endpoints():
        return web.json_response({"error": "mock_internal"}, status=500)

    # Scripted breaker behavior: fail first N then succeed
    brk = request.app["breakers"]
    if name in brk and brk[name] > 0:
        brk[name] -= 1
        return web.json_response({"error": "mock_internal"}, status=503)

    # Scripted reviewer labels: MOCK_REVIEWER_LABELS -> return as model content
    review_labels = os.environ.get("MOCK_REVIEWER_LABELS")
    if review_labels:
        return web.json_response({
            "id": f"chatcmpl-mock-{name}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model", "mock"),
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": review_labels},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        })

    # Latency
    latency = request.app["latency_ms"]
    if latency:
        await asyncio.sleep(latency / 1000.0)

    # Streaming passthrough
    if body.get("stream"):
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream"},
        )
        await resp.prepare(request)
        for c in ["mock ", "streamed ", "from ", name]:
            payload = json.dumps({
                "id": f"chatcmpl-stream-{name}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": body.get("model", "mock"),
                "choices": [{"index": 0, "delta": {"content": c}, "finish_reason": None}],
            })
            await resp.write(f"data: {payload}\n\n".encode())
            await asyncio.sleep(0.005)
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    prompt_tokens = sum(len(m.get("content", "")) for m in body.get("messages", [])) // 4
    content = f"mock response from {name} for {body.get('model', '?')}"
    return web.json_response({
        "id": f"chatcmpl-mock-{name}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": body.get("model", "mock"),
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": max(prompt_tokens, 1),
            "completion_tokens": len(content) // 4,
            "total_tokens": max(prompt_tokens, 1) + len(content) // 4,
        },
    })


async def chat_completions_openai(request: web.Request):
    # openai kind uses /chat/completions with an api_key_env header check
    if request.app.get("require_api_key") and not request.headers.get("Authorization", "").startswith("Bearer "):
        return web.json_response({"error": "missing key"}, status=401)
    return await chat_completions(request)


async def chat_completions_ollama(request: web.Request):
    """Native Ollama /api/chat: JSON response or NDJSON stream (no SSE framing)."""
    body = await request.json()
    name = request.app["name"]
    if name in _fail_endpoints():
        return web.json_response({"error": "mock_internal"}, status=503)

    if body.get("stream"):
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "application/x-ndjson"},
        )
        await resp.prepare(request)
        await resp.write(json.dumps({
            "model": "ollama-m", "message": {"role": "assistant", "content": "ollama "},
            "done": False, "prompt_eval_count": 4, "eval_count": 0,
        }).encode() + b"\n")
        await resp.write(json.dumps({
            "model": "ollama-m", "message": {"role": "assistant", "content": "native"},
            "done": True, "done_reason": "stop", "prompt_eval_count": 4, "eval_count": 2,
        }).encode() + b"\n")
        await resp.write_eof()
        return resp

    return web.json_response({
        "model": "ollama-m",
        "message": {"role": "assistant", "content": f"ollama native from {name}"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 5,
        "eval_count": 3,
    })


async def make_app(name: str, latency_ms: int = 1) -> web.Application:
    app = web.Application()
    app["name"] = name
    app["latency_ms"] = latency_ms
    app["breakers"] = _breaker_endpoints()
    app.router.add_get("/health", health)
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_post("/chat/completions", chat_completions_openai)
    app.router.add_post("/api/chat", chat_completions_ollama)
    return app


async def _main(port: int, latency_ms: int):
    app = await make_app("mock", latency_ms)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", port)
    await site.start()
    print(f"mock endpoint listening on 127.0.0.1:{port}")
    try:
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8091)
    parser.add_argument("--latency-ms", type=int, default=1)
    args = parser.parse_args()
    asyncio.run(_main(args.port, args.latency_ms))

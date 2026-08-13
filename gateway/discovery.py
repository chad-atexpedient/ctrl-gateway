"""Local model discovery + endpoint connection testing.

probe_local(): scans localhost for running LLM servers (Ollama, LM Studio,
vLLM, llama.cpp) and returns discovered services with their model lists.

test_endpoint(): sends a tiny "hi" prompt to any endpoint (local or cloud)
and measures latency. Used by the dashboard's "Add Provider" flow to verify
a connection before adding it.
"""
from __future__ import annotations

import asyncio
import logging
import time

import aiohttp

from . import transcoder

log = logging.getLogger("ctrl.discovery")

PROBE_TIMEOUT = 3  # seconds per target
TEST_TIMEOUT = 20  # seconds for connection test

# Fixed targets (known default ports)
_TARGETS = [
    {
        "name": "ollama",
        "host": "localhost",
        "port": 11434,
        "kind": "ollama",
        "model_path": "/api/tags",
        "model_field": "models",
        "model_name_field": "name",
        "health_path": "/api/tags",
    },
    {
        "name": "lmstudio",
        "host": "localhost",
        "port": 1234,
        "kind": "openai",
        "model_path": "/v1/models",
        "model_field": "data",
        "model_name_field": "id",
        "health_path": "/v1/models",
    },
    {
        "name": "vllm",
        "host": "localhost",
        "port": 8000,
        "kind": "openai",
        "model_path": "/v1/models",
        "model_field": "data",
        "model_name_field": "id",
        "health_path": "/v1/models",
    },
]

# Additional llama.cpp ports to scan (server default ports vary)
_LLAMACPP_PORTS = [8070, 8071, 8072, 8078, 8079, 8080, 8088, 5000, 5001, 5005]


def _build_targets() -> list[dict]:
    """Build the full target list including llama.cpp port scan."""
    targets = list(_TARGETS)
    for port in _LLAMACPP_PORTS:
        targets.append({
            "name": f"llamacpp_{port}",
            "host": "localhost",
            "port": port,
            "kind": "llamacpp",
            "model_path": "/v1/models",
            "model_field": "data",
            "model_name_field": "id",
            "health_path": "/health",
        })
    return targets


async def _probe_one(session: aiohttp.ClientSession, target: dict) -> dict | None:
    """Probe a single target. Returns discovery dict or None if unreachable."""
    base = f"http://{target['host']}:{target['port']}"
    health_url = f"{base}{target['health_path']}"
    try:
        async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=PROBE_TIMEOUT)) as resp:
            if resp.status >= 400:
                return None
            # Try to parse models
            models: list[str] = []
            try:
                data = await resp.json()
                model_list = data.get(target["model_field"], [])
                models = [m.get(target["model_name_field"], "") for m in model_list if isinstance(m, dict)]
                models = [m for m in models if m]
            except Exception:
                pass
            return {
                "service": target["name"].split("_")[0],  # "ollama", "llamacpp", etc.
                "name": target["name"],
                "base_url": base,
                "kind": target["kind"],
                "models": models,
                "port": target["port"],
            }
    except (TimeoutError, aiohttp.ClientError, OSError):
        return None


async def probe_local() -> list[dict]:
    """Scan localhost for running LLM servers.

    Returns a list of discovered services, each with:
      service: ollama | lmstudio | vllm | llamacpp
      base_url: http://localhost:PORT
      kind: transcoder kind to use
      models: list of available model names
    """
    targets = _build_targets()
    timeout = aiohttp.ClientTimeout(total=PROBE_TIMEOUT + 1)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        results = await asyncio.gather(*[_probe_one(session, t) for t in targets], return_exceptions=True)
    discovered = []
    for r in results:
        if isinstance(r, dict) and r is not None:
            discovered.append(r)
    return discovered


async def test_endpoint(
    base_url: str,
    kind: str,
    model_alias: str,
    api_key_env: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Send a tiny 'hi' prompt to verify a provider connection.

    Returns {ok, latency_ms, response_preview, error?}.
    Works for any provider kind (openai, anthropic, gemini, ollama, llamacpp).
    """
    # Build a minimal endpoint + tier config for transcoding
    ep_cfg: dict = {
        "name": "_test",
        "kind": kind,
        "base_url": base_url,
        "model_alias": model_alias,
    }
    if api_key_env:
        ep_cfg["api_key_env"] = api_key_env
    if api_key:
        ep_cfg["_api_key"] = api_key

    tier_cfg = {"max_tokens_bump": 0}

    payload = {
        "model": model_alias,
        "messages": [{"role": "user", "content": "Say 'hello' in one word."}],
        "max_tokens": 10,
        "stream": False,
        "temperature": 0.0,
    }

    try:
        transcoded = transcoder.transcode(ep_cfg, tier_cfg, payload)
    except Exception as e:
        return {"ok": False, "error": f"transcode failed: {e}", "latency_ms": 0}

    t0 = time.time()
    try:
        from . import ssrf

        # base_url comes from the dashboard's "Add Provider" flow — an
        # admin-configured endpoint, not a hardcoded gateway-internal URL —
        # so it gets the same SSRF policy as every other admin-configured
        # integration in this codebase (allow_localhost=True,
        # allow_private=True is intentional here for local/self-hosted
        # providers — only the DNS-rebinding/redirect-bypass gaps around
        # that policy need closing, not the policy itself). Validate the
        # actual transcoded request URL, since that's what the request will
        # hit.
        try:
            ssrf.validate_url(transcoded.url, allow_localhost=True, allow_private=True)
        except ssrf.SSRFBlockedURL as e:
            return {"ok": False, "error": f"blocked by SSRF policy: {e.reason}", "latency_ms": 0}

        timeout = aiohttp.ClientTimeout(total=TEST_TIMEOUT)
        async with aiohttp.ClientSession(
            timeout=timeout,
            **ssrf.ssrf_client_kwargs(allow_localhost=True, allow_private=True),
        ) as session:
            async with session.post(
                transcoded.url, headers=transcoded.headers, json=transcoded.body,
            ) as resp:
                latency_ms = (time.time() - t0) * 1000
                if resp.status >= 400:
                    text = await resp.text()
                    return {
                        "ok": False,
                        "error": f"HTTP {resp.status}: {text[:200]}",
                        "latency_ms": latency_ms,
                    }
                data = await resp.json()
                # Unwrap native response if needed
                if transcoded.response_decoder:
                    data = transcoded.response_decoder(data)
                # Extract response text (OpenAI format)
                content = ""
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                return {
                    "ok": True,
                    "latency_ms": round(latency_ms),
                    "response_preview": content[:200],
                    "model": data.get("model", model_alias),
                    "usage": data.get("usage", {}),
                }
    except TimeoutError:
        return {"ok": False, "error": f"timeout after {TEST_TIMEOUT}s", "latency_ms": round((time.time() - t0) * 1000)}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "latency_ms": round((time.time() - t0) * 1000)}


async def probe_mcp_servers(
    hosts: list[str] | None = None,
    ports: list[int] | None = None,
) -> list[dict]:
    """Scan the local network for MCP tool servers (Model Context Protocol).

    Wraps `mcp_discovery.discover()` and returns the discovered servers in
    the discovery dict shape: {name, source_url, server_info, capabilities,
    transport, tools}. Errors are caught and never raised.
    """
    try:
        from . import mcp_discovery
        results = await mcp_discovery.discover(hosts=hosts, ports=ports, auto_register=False)
    except Exception as e:
        log.warning("probe_mcp_servers failed: %s", e)
        return []
    out = []
    for server in results:
        out.append(
            {
                "name": server.name,
                "source_url": server.source_url,
                "server_info": server.server_info,
                "capabilities": server.capabilities,
                "transport": server.transport,
                "tools": server.tools,
            }
        )
    return out

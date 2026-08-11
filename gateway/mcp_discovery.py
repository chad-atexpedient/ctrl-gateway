"""MCP server auto-discovery.

Extends `discovery.py` (which probes local LLM servers) to also discover
MCP tool servers (Model Context Protocol). Three probe methods:

  1. well-known: GET /.well-known/mcp.json or /.well-known/agent.json
  2. http: GET /mcp on candidate URLs (MCP HTTP transport)
  3. ports: scan a configurable port list for SSE endpoints

Discovered servers can be auto-registered as federated tools via
`memory.upsert_federated_tool(..., source="mcp_discovery")`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from . import memory

log = logging.getLogger("glint.mcp_discovery")


PROBE_TIMEOUT_SECONDS = 2.0
DEFAULT_PORTS = [8000, 8080, 8088, 9000, 9090, 3000, 3333]


@dataclass
class DiscoveredServer:
    name: str
    source_url: str
    server_info: dict[str, Any]
    capabilities: dict[str, Any]
    discovered_at: float = field(default_factory=time.time)
    transport: str = "http"  # http | sse | stdio
    tools: list[dict] = field(default_factory=list)


def _parse_capabilities(payload: Any) -> dict[str, Any]:
    """Extract MCP-style capabilities from a probe response."""
    if isinstance(payload, dict):
        caps = payload.get("capabilities")
        if isinstance(caps, dict):
            return caps
        info = payload.get("serverInfo") or payload.get("server_info")
        if isinstance(info, dict):
            return info
    return {}


async def _probe_well_known(
    session: aiohttp.ClientSession, base_url: str
) -> DiscoveredServer | None:
    for path in ("/.well-known/mcp.json", "/.well-known/agent.json"):
        url = f"{base_url.rstrip('/')}{path}"
        try:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=PROBE_TIMEOUT_SECONDS),
            ) as resp:
                if resp.status != 200:
                    continue
                payload = await resp.json(content_type=None)
                if not isinstance(payload, dict):
                    continue
                name = payload.get("name") or payload.get("server_name") or base_url
                caps = _parse_capabilities(payload)
                return DiscoveredServer(
                    name=str(name),
                    source_url=base_url,
                    server_info=payload,
                    capabilities=caps,
                    transport="http",
                )
        except (TimeoutError, aiohttp.ClientError, json.JSONDecodeError):
            continue
    return None


async def _probe_mcp_endpoint(
    session: aiohttp.ClientSession, base_url: str
) -> DiscoveredServer | None:
    """Probe a candidate as an MCP HTTP transport endpoint (POST initialize)."""
    url = f"{base_url.rstrip('/')}/mcp"
    init_payload = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "glint-discovery", "version": "1.0"},
        },
        "id": 1,
    }
    try:
        async with session.post(
            url,
            json=init_payload,
            timeout=aiohttp.ClientTimeout(total=PROBE_TIMEOUT_SECONDS),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        ) as resp:
            if resp.status >= 400:
                return None
            payload = await resp.json(content_type=None)
            if not isinstance(payload, dict):
                return None
            result = payload.get("result", {})
            if not isinstance(result, dict):
                return None
            info = result.get("serverInfo") or result.get("server_info") or {}
            caps = result.get("capabilities", {})
            name = (
                info.get("name") if isinstance(info, dict) else None
            ) or base_url
            return DiscoveredServer(
                name=str(name),
                source_url=base_url,
                server_info=info if isinstance(info, dict) else {},
                capabilities=caps if isinstance(caps, dict) else {},
                transport="http",
            )
    except (TimeoutError, aiohttp.ClientError, json.JSONDecodeError):
        return None


async def _probe_sse_endpoint(
    session: aiohttp.ClientSession, base_url: str
) -> DiscoveredServer | None:
    """Probe an SSE endpoint (GET /sse returns text/event-stream)."""
    url = f"{base_url.rstrip('/')}/sse"
    try:
        async with session.get(
            url,
            timeout=aiohttp.ClientTimeout(total=PROBE_TIMEOUT_SECONDS),
            headers={"Accept": "text/event-stream"},
        ) as resp:
            ct = resp.headers.get("Content-Type", "")
            if resp.status == 200 and "event-stream" in ct.lower():
                return DiscoveredServer(
                    name=f"sse-{base_url}",
                    source_url=base_url,
                    server_info={"transport": "sse"},
                    capabilities={},
                    transport="sse",
                )
    except (TimeoutError, aiohttp.ClientError):
        return None
    return None


async def _probe_one(
    session: aiohttp.ClientSession, base_url: str
) -> DiscoveredServer | None:
    """Try multiple probe methods for a single URL."""
    # SSRF protection: only scan localhost/private (discovery is local-only)
    from . import ssrf
    try:
        ssrf.validate_url(base_url, allow_localhost=True, allow_private=True)
    except ssrf.SSRFBlockedURL:
        return None
    discovered = await _probe_well_known(session, base_url)
    if discovered:
        return discovered
    discovered = await _probe_mcp_endpoint(session, base_url)
    if discovered:
        return discovered
    return await _probe_sse_endpoint(session, base_url)


async def discover(
    hosts: list[str] | None = None,
    ports: list[int] | None = None,
    auto_register: bool = False,
) -> list[DiscoveredServer]:
    """Discover MCP servers.

    Args:
        hosts: candidate hostnames/IPs (defaults to localhost + 127.0.0.1)
        ports: ports to scan (defaults to DEFAULT_PORTS)
        auto_register: if True, persist each discovery into federated_tools

    Returns:
        list of DiscoveredServer records
    """
    hosts = hosts or ["localhost", "127.0.0.1"]
    ports = ports or DEFAULT_PORTS
    candidates = [f"http://{h}:{p}" for h in hosts for p in ports]
    discovered: list[DiscoveredServer] = []
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(
            *(_probe_one(session, url) for url in candidates),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, DiscoveredServer):
                discovered.append(result)
                if auto_register:
                    try:
                        memory.upsert_federated_tool(
                            name=f"mcp_{result.name}",
                            source="mcp_discovery",
                            source_url=result.source_url,
                            tool={
                                "name": result.name,
                                "source_url": result.source_url,
                                "capabilities": result.capabilities,
                                "server_info": result.server_info,
                                "transport": result.transport,
                            },
                            enabled=True,
                        )
                    except Exception as e:
                        log.warning("auto-register failed: %s", e)
    log.info("MCP discovery found %d servers across %d candidates", len(discovered), len(candidates))
    return discovered


async def watch_loop(
    interval_seconds: int = 60,
    hosts: list[str] | None = None,
    ports: list[int] | None = None,
    auto_register: bool = True,
) -> None:
    """Background task: periodic re-discovery."""
    while True:
        try:
            await discover(hosts=hosts, ports=ports, auto_register=auto_register)
        except Exception as e:
            log.warning("mcp_discovery watch_loop error: %s", e)
        await asyncio.sleep(interval_seconds)

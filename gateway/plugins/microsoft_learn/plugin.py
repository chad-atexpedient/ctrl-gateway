"""Microsoft Learn plugin.

Demonstrates the plugin contract for aiohttp:
  - Exposes `build_router(context) -> web.RouteTableDef`
  - Receives a PluginContext (name, manifest, config, event_bus, emit_event, get_setting)
  - Routes registered at `manifest.prefix` (or unprefixed)
  - Events emitted via context.emit_event()
  - Errors never crash the gateway (caught at loader level)
"""
from __future__ import annotations

from typing import Any

import aiohttp
from aiohttp import web

# PluginContext type hints (kept loose to avoid hard import surface)
Context = Any


async def _search_docs(
    session: aiohttp.ClientSession,
    api_base: str,
    query: str,
    timeout_seconds: float,
) -> list[dict]:
    """Search Microsoft Learn via the public docs search API."""
    url = f"{api_base.rstrip('/')}/api/docs/search"
    params = {"q": query, "locale": "en-us"}
    try:
        async with session.get(
            url,
            params=params,
            timeout=aiohttp.ClientTimeout(total=timeout_seconds),
        ) as resp:
            if resp.status >= 400:
                return []
            payload = await resp.json(content_type=None)
    except (TimeoutError, aiohttp.ClientError):
        return []
    results = payload.get("results", []) if isinstance(payload, dict) else []
    out: list[dict] = []
    for entry in results[:10]:
        if not isinstance(entry, dict):
            continue
        out.append(
            {
                "title": entry.get("title", ""),
                "url": entry.get("url", ""),
                "summary": entry.get("summary", ""),
            }
        )
    return out


async def _list_modules(
    api_base: str, timeout_seconds: float
) -> list[dict]:
    """List available Microsoft Learn modules. Falls back to a static
    catalog when the API is unreachable so the plugin always responds."""
    url = f"{api_base.rstrip('/')}/api/modules"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                url,
                timeout=aiohttp.ClientTimeout(total=timeout_seconds),
            ) as resp:
                if resp.status >= 400:
                    raise aiohttp.ClientError(f"HTTP {resp.status}")
                payload = await resp.json(content_type=None)
        modules = payload.get("modules", []) if isinstance(payload, dict) else []
        return modules[:20] if isinstance(modules, list) else []
    except (TimeoutError, aiohttp.ClientError, ValueError):
        # Fallback to a known catalog snapshot
        return [
            {"id": "azure-fundamentals", "title": "Azure Fundamentals"},
            {"id": "ai-fundamentals", "title": "AI Fundamentals"},
            {"id": "powershell-basics", "title": "PowerShell Basics"},
        ]


def build_router(context: Context) -> web.RouteTableDef:
    """Build the plugin's aiohttp routes.

    The router prefix is added in the gateway app.add_routes call (the
    manifest's `prefix` is informational; aiohttp encodes the full path
    in each route). We follow the convention of prefixing the paths here
    to match the manifest.
    """
    api_base = context.get_setting("api_base", "https://learn.microsoft.com")
    timeout_seconds = float(context.get_setting("timeout_seconds", 10))
    routes = web.RouteTableDef()

    @routes.post("/integrations/microsoft-learn/search")
    async def search(request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "invalid JSON"}, status=400)
        query = body.get("query", "").strip()
        if not query:
            return web.json_response({"error": "query required"}, status=400)
        await context.emit_event(
            "learn.search.started", {"query": query}, severity="info"
        )
        try:
            async with aiohttp.ClientSession() as session:
                results = await _search_docs(session, api_base, query, timeout_seconds)
        except Exception as e:
            await context.emit_event(
                "learn.search.completed",
                {"query": query, "status": "failed", "detail": str(e)},
                severity="warn",
            )
            return web.json_response({"error": str(e)}, status=500)
        await context.emit_event(
            "learn.search.completed",
            {"query": query, "status": "success", "count": len(results)},
            severity="info",
        )
        return web.json_response({"results": results})

    @routes.get("/integrations/microsoft-learn/modules")
    async def modules(request: web.Request) -> web.Response:
        items = await _list_modules(api_base, timeout_seconds)
        return web.json_response({"modules": items})

    return routes

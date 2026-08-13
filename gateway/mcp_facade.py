"""Unified MCP (Model Context Protocol) facade.

Exposes the gateway's federated tools, A2A agents, and prompt templates
through a single JSON-RPC 2.0 endpoint compatible with MCP clients:

  POST /mcp
    {"jsonrpc":"2.0","method":"tools/list","id":1}
    {"jsonrpc":"2.0","method":"tools/call","params":{"name":"...","arguments":{}},"id":2}
    {"jsonrpc":"2.0","method":"prompts/list","id":3}
    {"jsonrpc":"2.0","method":"initialize","params":{...},"id":4}

Tool resolution order:
  1. Federated tools (from ContextForge sync / MCP discovery / embedded)
  2. A2A agents (invoked as tools with the a2a_ prefix)
  3. Gateway built-in tools (status, list_endpoints)

This is a read/call facade — it does not manage the tool registry. Tools
are registered via /admin/a2a/* and /admin/contextforge/*. This facade
makes them callable from any MCP-compatible client.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from aiohttp import web

from . import a2a_registry, memory

log = logging.getLogger("ctrl.mcp_facade")

PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "ctrl-gateway"
SERVER_VERSION = "2.0.0"


def _list_federated_tools_as_mcp(
    tenant_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return federated tools in MCP tools/list format.

    If tenant_id is provided, only tools scoped to "__all__" or that
    tenant are returned. If None, every enabled tool is returned (admin
    context — backward compatible).
    """
    rows = memory.list_federated_tools(enabled_only=True, tenant_id=tenant_id)
    tools: list[dict[str, Any]] = []
    for row in rows:
        try:
            tool_def = json.loads(row.get("tool_json") or "{}")
        except (json.JSONDecodeError, TypeError):
            tool_def = {}
        tools.append(
            {
                "name": row["name"],
                "description": tool_def.get("description", row.get("source", "")),
                "inputSchema": tool_def.get("inputSchema", {"type": "object", "properties": {}}),
                "annotations": {
                    "source": row.get("source", "unknown"),
                    "source_url": row.get("source_url", ""),
                },
            }
        )
    return tools


def _list_a2a_agents_as_mcp(
    tenant_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return A2A agents as MCP tools (prefixed with a2a_).

    If tenant_id is provided, only agents scoped to "__all__" or that
    tenant are returned.
    """
    agents = a2a_registry.list_agents(enabled_only=True, tenant_id=tenant_id)
    tools: list[dict[str, Any]] = []
    for agent in agents:
        tool_name = f"a2a_{agent.name}"
        tools.append(
            {
                "name": tool_name,
                "description": f"A2A agent: {agent.description or agent.name} ({agent.agent_type})",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "method": {"type": "string", "default": "message/send"},
                        "params": {"type": "object"},
                        "messages": {
                            "type": "array",
                            "items": {"type": "object"},
                            "description": "OpenAI-compatible messages array",
                        },
                    },
                },
                "annotations": {
                    "source": "a2a",
                    "agent_type": agent.agent_type,
                    "endpoint_url": agent.endpoint_url,
                },
            }
        )
    return tools


def _list_prompts_as_mcp(
    tenant_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return prompt templates in MCP prompts/list format.

    If tenant_id is provided, only templates scoped to "__all__" or that
    tenant are returned.
    """
    rows = memory.list_prompt_templates(enabled_only=True, tenant_id=tenant_id)
    prompts: list[dict[str, Any]] = []
    for row in rows:
        try:
            variables = json.loads(row.get("variables_json") or "[]")
        except (json.JSONDecodeError, TypeError):
            variables = []
        prompts.append(
            {
                "name": row["name"],
                "description": row.get("description", ""),
                "arguments": [
                    {"name": v, "required": False, "description": ""}
                    for v in variables
                ],
            }
        )
    return prompts


def _list_all_tools(
    tenant_id: str | None = None,
) -> list[dict[str, Any]]:
    """Combine federated tools + A2A agents into a single tools/list response."""
    return _list_federated_tools_as_mcp(tenant_id=tenant_id) + _list_a2a_agents_as_mcp(tenant_id=tenant_id)


async def handle_mcp_rpc(request: web.Request) -> web.Response:
    """Handle a single MCP JSON-RPC 2.0 request.

    Supports: initialize, tools/list, tools/call, prompts/list.
    """
    try:
        body = await request.json()
    except Exception:
        return web.json_response(
            {"jsonrpc": "2.0", "error": {"code": -32700, "message": "Parse error"}, "id": None},
            status=400,
        )
    method = body.get("method", "")
    req_id = body.get("id")
    params = body.get("params", {})
    # Tenant resolution: MUST come from the auth-verified identity, never from a
    # client-supplied header — auth_middleware sets request["tenant_id"] from the
    # resolved API key when auth is enabled. X-User-Id is only the fallback used
    # elsewhere in the gateway (chat_completions et al.) for trusted-reverse-proxy
    # deployments with auth disabled. Previously this read a raw `X-Tenant-Id`
    # header directly, which let any authenticated caller impersonate any other
    # tenant for tools/list, tools/call, and prompts/get — including invoking
    # another tenant's private A2A agents. Fixed to match every other tenant-scoped
    # handler in app.py.
    tenant_id = str(request.get("tenant_id") or request.headers.get("X-User-Id", "anonymous"))

    if method == "initialize":
        return web.json_response(
            {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {
                        "tools": {"listChanged": True},
                        "prompts": {"listChanged": True},
                    },
                    "serverInfo": {
                        "name": SERVER_NAME,
                        "version": SERVER_VERSION,
                    },
                },
                "id": req_id,
            }
        )

    if method == "tools/list":
        tools = _list_all_tools(tenant_id=tenant_id)
        return web.json_response(
            {"jsonrpc": "2.0", "result": {"tools": tools}, "id": req_id}
        )

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        # Check tool cache first
        from . import tool_cache
        cache = tool_cache.cache()
        if cache is not None:
            cached = cache.get(tool_name, arguments, tenant_id=tenant_id)
            if cached is not None:
                return web.json_response(
                    {
                        "jsonrpc": "2.0",
                        "result": {
                            "content": [{"type": "text", "text": json.dumps(cached, default=str)}],
                            "isError": False,
                            "_cached": True,
                        },
                        "id": req_id,
                    }
                )

        # Federated tool (no execution yet — just return definition)
        fed_tool = memory.get_federated_tool(tool_name, tenant_id=tenant_id)
        if fed_tool is not None:
            try:
                tool_def = json.loads(fed_tool.get("tool_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                tool_def = {}
            result = {
                "name": tool_name,
                "source": fed_tool.get("source", "unknown"),
                "definition": tool_def,
                "note": "Federated tool definition. Direct execution requires a registered transport.",
                "arguments_received": arguments,
            }
            if cache is not None:
                cache.set(tool_name, arguments, result, tenant_id=tenant_id)
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, default=str)}],
                        "isError": False,
                    },
                    "id": req_id,
                }
            )

        # A2A agent invocation
        if tool_name.startswith("a2a_"):
            agent_name = tool_name[4:]
            agent = a2a_registry.get_agent_by_name(agent_name, tenant_id=tenant_id)
            if agent is None:
                return _mcp_error(req_id, -32602, f"agent not found: {agent_name}")
            if not agent.enabled:
                return _mcp_error(req_id, -32603, f"agent disabled: {agent_name}")
            agent_result = await a2a_registry.invoke_agent(
                agent=agent,
                parameters=arguments,
                tenant_id=tenant_id,
                interaction_type="mcp_call",
            )
            a2a_data: dict[str, Any] = {
                "success": agent_result.success,
                "response": agent_result.response,
                "latency_ms": agent_result.latency_ms,
                "error": agent_result.error,
            }
            if cache is not None and agent_result.success:
                cache.set(tool_name, arguments, a2a_data, tenant_id=tenant_id)
            return web.json_response(
                {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(a2a_data, default=str),
                            }
                        ],
                        "isError": not agent_result.success,
                    },
                    "id": req_id,
                }
            )

        return _mcp_error(req_id, -32602, f"tool not found: {tool_name}")

    if method == "prompts/list":
        prompts = _list_prompts_as_mcp(tenant_id=tenant_id)
        return web.json_response(
            {"jsonrpc": "2.0", "result": {"prompts": prompts}, "id": req_id}
        )

    if method == "prompts/get":
        prompt_name = params.get("name", "")
        row = memory.get_prompt_template_by_name(prompt_name, tenant_id=tenant_id)
        if row is None:
            return _mcp_error(req_id, -32602, f"prompt not found: {prompt_name}")
        from . import prompt_registry
        rendered = prompt_registry.render_template(
            row["template_text"], params.get("arguments", {})
        )
        return web.json_response(
            {
                "jsonrpc": "2.0",
                "result": {
                    "description": row.get("description", ""),
                    "messages": [
                        {"role": "system", "content": {"type": "text", "text": rendered}}
                    ],
                },
                "id": req_id,
            }
        )

    # Unknown method
    return _mcp_error(req_id, -32601, f"method not found: {method}")


def _mcp_error(req_id: Any, code: int, message: str) -> web.Response:
    return web.json_response(
        {
            "jsonrpc": "2.0",
            "error": {"code": code, "message": message},
            "id": req_id,
        }
    )

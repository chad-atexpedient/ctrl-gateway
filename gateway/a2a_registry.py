"""A2A (Agent-to-Agent) registry.

Lifecycle + invocation of external AI agents. Inspired by IBM ContextForge's
A2A integration (`/a2a` endpoints + Agent-Tool State Cascade):

  - Register external agents (jsonrpc / openai / anthropic / custom)
  - Activate / deactivate / test / invoke
  - Virtual servers: named bundles of agents (ContextForge concept)
  - Per-agent metrics (success rate, latency, last interaction)
  - Auth propagation: api_key | bearer | oauth | none

The registry does NOT make HTTP calls directly — it stores the agent
configuration and exposes `invoke_agent(agent, parameters)` which formats
the request according to the agent's `agent_type`. The actual HTTP call is
made by an aiohttp ClientSession passed in by the caller (typically the
chat_completions handler or an admin /test endpoint).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import aiohttp

from . import memory

log = logging.getLogger("glint.a2a")


class A2AError(Exception):
    """Raised when A2A agent invocation fails."""


@dataclass
class A2AAgentRecord:
    id: int
    name: str
    endpoint_url: str
    agent_type: str
    description: str
    auth_type: str
    auth_value: str
    protocol_version: str
    capabilities: dict
    config: dict
    tags: list[str]
    enabled: bool


@dataclass
class A2AResult:
    success: bool
    response: Any
    latency_ms: float
    error: str | None = None
    status_code: int | None = None


def agent_from_row(row: dict) -> A2AAgentRecord:
    """Convert a DB row dict to an A2AAgentRecord."""
    try:
        caps = json.loads(row.get("capabilities_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        caps = {}
    try:
        cfg = json.loads(row.get("config_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        cfg = {}
    try:
        tags = json.loads(row.get("tags_json") or "[]")
    except (json.JSONDecodeError, TypeError):
        tags = []
    return A2AAgentRecord(
        id=int(row["id"]),
        name=row["name"],
        endpoint_url=row["endpoint_url"],
        agent_type=row["agent_type"],
        description=row.get("description", ""),
        auth_type=row.get("auth_type", "none"),
        auth_value=row.get("auth_value_encrypted", "") or "",
        protocol_version=row.get("protocol_version", "1.0"),
        capabilities=caps,
        config=cfg,
        tags=tags if isinstance(tags, list) else [],
        enabled=bool(row.get("enabled", True)),
    )


def list_agents(enabled_only: bool = False) -> list[A2AAgentRecord]:
    rows = memory.list_a2a_agents(enabled_only=enabled_only)
    return [agent_from_row(r) for r in rows]


def get_agent(agent_id: int) -> A2AAgentRecord | None:
    row = memory.get_a2a_agent(agent_id)
    return agent_from_row(row) if row else None


def get_agent_by_name(name: str) -> A2AAgentRecord | None:
    row = memory.get_a2a_agent_by_name(name)
    return agent_from_row(row) if row else None


def register_agent(
    name: str,
    endpoint_url: str,
    agent_type: str,
    description: str = "",
    auth_type: str = "none",
    auth_value: str | None = None,
    protocol_version: str | None = None,
    capabilities: dict | None = None,
    config: dict | None = None,
    tags: list[str] | None = None,
    enabled: bool = True,
) -> A2AAgentRecord | None:
    row = memory.upsert_a2a_agent(
        name=name,
        endpoint_url=endpoint_url,
        agent_type=agent_type,
        description=description,
        auth_type=auth_type,
        auth_value=auth_value,
        protocol_version=protocol_version,
        capabilities=capabilities,
        config=config,
        tags=tags,
        enabled=enabled,
    )
    if "error" in row:
        log.warning("register_agent failed: %s", row["error"])
        return None
    return get_agent_by_name(name)


def delete_agent(agent_id: int) -> bool:
    return memory.delete_a2a_agent(agent_id)


def set_agent_enabled(agent_id: int, enabled: bool) -> bool:
    return memory.set_a2a_agent_enabled(agent_id, enabled)


def build_auth_headers(agent: A2AAgentRecord) -> dict[str, str]:
    """Apply the agent's auth configuration to outbound headers."""
    if agent.auth_type == "api_key" or agent.auth_type == "bearer":
        if agent.auth_value:
            return {"Authorization": f"Bearer {agent.auth_value}"}
    return {}


def build_request_payload(
    agent: A2AAgentRecord, parameters: dict[str, Any]
) -> tuple[str, dict]:
    """Format the request body and method/path according to agent_type.

    Returns (full_url, json_payload).

    jsonrpc  → POST {endpoint}/ with body {"jsonrpc":"2.0","method":...,"params":...,"id":1}
    openai   → POST {endpoint} with chat.completions body
    anthropic→ POST {endpoint} with messages body
    custom   → POST {endpoint} with parameters as-is
    """
    if agent.agent_type == "jsonrpc":
        payload = {
            "jsonrpc": "2.0",
            "method": parameters.get("method", "message/send"),
            "params": parameters.get("params", {}),
            "id": parameters.get("id", 1),
        }
        url = agent.endpoint_url
        return url, payload
    if agent.agent_type == "openai":
        payload = {
            "model": parameters.get("model", agent.config.get("model", "gpt-4o-mini")),
            "messages": parameters.get("messages", []),
            "max_tokens": parameters.get("max_tokens", 1024),
            "temperature": parameters.get("temperature", agent.config.get("temperature", 0.7)),
        }
        return agent.endpoint_url, payload
    if agent.agent_type == "anthropic":
        payload = {
            "model": parameters.get("model", agent.config.get("model", "claude-3-5-sonnet-20240620")),
            "max_tokens": parameters.get("max_tokens", 1024),
            "messages": parameters.get("messages", []),
            "system": parameters.get("system", ""),
        }
        return agent.endpoint_url, payload
    # custom
    return agent.endpoint_url, parameters


async def invoke_agent(
    agent: A2AAgentRecord,
    parameters: dict[str, Any],
    session: aiohttp.ClientSession | None = None,
    timeout_seconds: float = 30.0,
    tenant_id: str = "anonymous",
    interaction_type: str = "invoke",
) -> A2AResult:
    """Invoke an A2A agent. Returns a typed result.

    Errors are caught and recorded into a2a_metrics; the result is never
    raised except for programming errors.
    """
    url, payload = build_request_payload(agent, parameters)
    headers = build_auth_headers(agent)
    headers.setdefault("Content-Type", "application/json")
    # SSRF protection: block private/loopback/link-local targets unless explicitly allowed
    from . import ssrf
    try:
        ssrf.validate_url(url, allow_localhost=True, allow_private=True)
    except ssrf.SSRFBlockedURL as e:
        return A2AResult(
            success=False, response=None, latency_ms=0.0,
            error=f"ssrf_blocked: {e.reason}", status_code=None,
        )
    started = time.monotonic()
    owns_session = False
    if session is None:
        session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout_seconds)
        )
        owns_session = True
    try:
        async with session.post(url, json=payload, headers=headers) as resp:
            latency_ms = (time.monotonic() - started) * 1000.0
            body_text = await resp.text()
            success = 200 <= resp.status < 300
            err: str | None = None if success else f"HTTP {resp.status}"
            response_payload: Any = body_text
            try:
                response_payload = json.loads(body_text)
                if agent.agent_type == "jsonrpc" and isinstance(
                    response_payload, dict
                ):
                    if "error" in response_payload and not success:
                        err = str(response_payload.get("error"))
            except json.JSONDecodeError:
                pass
            memory.record_a2a_metric(
                agent_id=agent.id,
                tenant_id=tenant_id,
                success=success,
                latency_ms=latency_ms,
                interaction_type=interaction_type,
                error=err,
            )
            return A2AResult(
                success=success,
                response=response_payload,
                latency_ms=latency_ms,
                error=err,
                status_code=resp.status,
            )
    except TimeoutError:
        latency_ms = (time.monotonic() - started) * 1000.0
        memory.record_a2a_metric(
            agent_id=agent.id,
            tenant_id=tenant_id,
            success=False,
            latency_ms=latency_ms,
            interaction_type=interaction_type,
            error="timeout",
        )
        return A2AResult(
            success=False,
            response=None,
            latency_ms=latency_ms,
            error="timeout",
            status_code=None,
        )
    except Exception as e:
        latency_ms = (time.monotonic() - started) * 1000.0
        memory.record_a2a_metric(
            agent_id=agent.id,
            tenant_id=tenant_id,
            success=False,
            latency_ms=latency_ms,
            interaction_type=interaction_type,
            error=str(e),
        )
        log.warning("a2a invoke_agent(%s) failed: %s", agent.name, e)
        return A2AResult(
            success=False, response=None, latency_ms=latency_ms, error=str(e)
        )
    finally:
        if owns_session:
            try:
                await session.close()
            except Exception:
                pass


# ----- Virtual servers -----


def list_virtual_servers() -> list[dict]:
    """Return all virtual servers with their associated agent IDs parsed."""
    rows = memory.list_a2a_virtual_servers()
    for row in rows:
        try:
            row["associated_agents"] = json.loads(
                row.get("associated_agents_json") or "[]"
            )
        except (json.JSONDecodeError, TypeError):
            row["associated_agents"] = []
    return rows


def get_virtual_server(server_id: int) -> dict | None:
    row = memory.get_a2a_virtual_server(server_id)
    if row is None:
        return None
    try:
        row["associated_agents"] = json.loads(
            row.get("associated_agents_json") or "[]"
        )
    except (json.JSONDecodeError, TypeError):
        row["associated_agents"] = []
    return row


def register_virtual_server(
    name: str,
    description: str = "",
    associated_agents: list[int] | None = None,
    enabled: bool = True,
) -> dict | None:
    row = memory.upsert_a2a_virtual_server(
        name=name,
        description=description,
        associated_agents=associated_agents,
        enabled=enabled,
    )
    if "error" in row:
        return None
    return get_virtual_server(int(row["id"])) if row else None


def delete_virtual_server(server_id: int) -> bool:
    return memory.delete_a2a_virtual_server(server_id)


def metrics_summary(agent_id: int) -> dict:
    return memory.a2a_agent_metrics_summary(agent_id)

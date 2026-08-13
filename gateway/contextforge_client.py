"""IBM ContextForge connector.

Pulls tools, A2A agents, virtual servers, and prompt templates FROM a
running ContextForge instance (HTTP mode) OR embeds a subset of
ContextForge's core logic directly (embedded mode). Both modes expose
the same interface.

External mode (HTTP client):
  - Connects to a ContextForge instance at `external_url` with optional
    `api_key` bearer auth
  - Pulls /a2a (agents), /servers (virtual servers), /rpc (tools/list),
    /prompts/list (templates) — exact routes vary by ContextForge version
  - Sync engine merges into local registries (memory tables)

Embedded mode:
  - Implements a subset of ContextForge's registry/admin API surface so
    tools can be added without an external ContextForge instance
  - Local federation registry, agent templates, prompt storage

Sync engine:
  - Periodic pull from external ContextForge (if configured)
  - Merges into gateway's plugin/a2a/prompt registries
  - Records each sync into contextforge_sync_log
  - Errors never crash the gateway

Reference: https://github.com/IBM/mcp-context-forge
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from . import a2a_registry, memory

log = logging.getLogger("ctrl.contextforge")


class ContextForgeError(Exception):
    """Raised when ContextForge connector operations fail."""


@dataclass
class SyncResult:
    sync_type: str
    source: str
    items_synced: int
    items_added: int
    items_updated: int
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


class ContextForgeClient:
    """Unified interface for external + embedded ContextForge."""

    def __init__(
        self,
        mode: str = "external",  # "external" | "embedded" | "both"
        external_url: str | None = None,
        api_key: str | None = None,
        sync_interval_seconds: int = 300,
        auto_sync: bool = False,
        timeout_seconds: float = 30.0,
    ):
        if mode not in ("external", "embedded", "both"):
            raise ContextForgeError(f"invalid mode: {mode}")
        if mode in ("external", "both") and not external_url:
            raise ContextForgeError("external_url required for external/both mode")
        self.mode = mode
        self.external_url = (external_url or "").rstrip("/")
        self.api_key = api_key
        self.sync_interval_seconds = sync_interval_seconds
        self.auto_sync = auto_sync
        self.timeout_seconds = timeout_seconds
        self._sync_task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"Accept": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            from . import ssrf
            # connector re-checks the SSRF policy at actual connect time,
            # closing the TOCTOU/DNS-rebinding gap the validate_url() call
            # in _fetch() below can't close on its own (see
            # ssrf.SSRFSafeResolver). This session is reused across many
            # requests to the same configured external_url, so one shared
            # connector is fine here.
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
                connector=ssrf.safe_connector(allow_localhost=True, allow_private=True),
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None
        self.stop_sync_loop()

    def _auth_headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    # ----- External mode: HTTP calls to ContextForge -----

    async def _fetch(self, path: str) -> Any:
        """GET path on external ContextForge. Raises on non-2xx."""
        session = await self._get_session()
        url = f"{self.external_url}{path}"
        # SSRF protection
        from . import ssrf
        ssrf.validate_url(url, allow_localhost=True, allow_private=True)
        try:
            async with session.get(url, headers=self._auth_headers()) as resp:
                if resp.status >= 400:
                    body = await resp.text()
                    raise ContextForgeError(
                        f"GET {path} -> HTTP {resp.status}: {body[:200]}"
                    )
                return await resp.json(content_type=None)
        except aiohttp.ClientError as e:
            raise ContextForgeError(f"GET {path} failed: {e}") from e

    async def fetch_a2a_agents(self) -> list[dict]:
        """Pull A2A agents from /a2a endpoint."""
        try:
            data = await self._fetch("/a2a")
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "agents" in data:
                agents = data["agents"]
                return agents if isinstance(agents, list) else []
            return []
        except ContextForgeError as e:
            log.warning("fetch_a2a_agents: %s", e)
            return []

    async def fetch_virtual_servers(self) -> list[dict]:
        """Pull virtual servers from /servers endpoint."""
        try:
            data = await self._fetch("/servers")
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "servers" in data:
                servers = data["servers"]
                return servers if isinstance(servers, list) else []
            return []
        except ContextForgeError as e:
            log.warning("fetch_virtual_servers: %s", e)
            return []

    async def fetch_tools(self) -> list[dict]:
        """Pull tools list from /rpc tools/list."""
        try:
            data = await self._fetch("/rpc")
        except ContextForgeError as e:
            log.warning("fetch_tools: %s", e)
            return []
        # tools/list returns {"result": {"tools": [...]}} in JSON-RPC 2.0
        try:
            if isinstance(data, dict):
                result = data.get("result", {})
                if isinstance(result, dict) and "tools" in result:
                    tools = result["tools"]
                    return tools if isinstance(tools, list) else []
            return []
        except Exception:
            return []

    async def fetch_prompts(self) -> list[dict]:
        """Pull prompt templates from /prompts/list."""
        try:
            data = await self._fetch("/prompts/list")
            if isinstance(data, list):
                return data
            if isinstance(data, dict) and "prompts" in data:
                prompts = data["prompts"]
                return prompts if isinstance(prompts, list) else []
            return []
        except ContextForgeError as e:
            log.warning("fetch_prompts: %s", e)
            return []

    # ----- Sync engine: merge into local registries -----

    async def sync_all(self) -> dict[str, SyncResult]:
        """Run a full sync across all asset types. Returns a dict keyed by
        sync_type (agents | servers | tools | prompts)."""
        results: dict[str, SyncResult] = {}
        if self.mode == "embedded":
            log.debug("embedded mode: no external sync needed")
            return results
        for sync_type, fetch_fn, merge_fn in [
            ("agents", self.fetch_a2a_agents, self._merge_agents),
            ("servers", self.fetch_virtual_servers, self._merge_servers),
            ("tools", self.fetch_tools, self._merge_tools),
            ("prompts", self.fetch_prompts, self._merge_prompts),
        ]:
            started = time.monotonic()
            try:
                items = await fetch_fn()
                merged = merge_fn(items)
                duration_ms = (time.monotonic() - started) * 1000.0
                result = SyncResult(
                    sync_type=sync_type,
                    source=self.external_url,
                    items_synced=merged["synced"],
                    items_added=merged["added"],
                    items_updated=merged["updated"],
                    errors=merged["errors"],
                    duration_ms=duration_ms,
                )
                results[sync_type] = result
                memory.record_contextforge_sync(
                    sync_type=sync_type,
                    source=self.external_url,
                    items_synced=merged["synced"],
                    items_added=merged["added"],
                    items_updated=merged["updated"],
                    errors=merged["errors"],
                    duration_ms=duration_ms,
                )
            except Exception as e:
                duration_ms = (time.monotonic() - started) * 1000.0
                result = SyncResult(
                    sync_type=sync_type,
                    source=self.external_url,
                    items_synced=0,
                    items_added=0,
                    items_updated=0,
                    errors=[str(e)],
                    duration_ms=duration_ms,
                )
                results[sync_type] = result
                memory.record_contextforge_sync(
                    sync_type=sync_type,
                    source=self.external_url,
                    items_synced=0,
                    items_added=0,
                    items_updated=0,
                    errors=[str(e)],
                    duration_ms=duration_ms,
                )
                log.warning("sync %s failed: %s", sync_type, e)
        return results

    def _merge_agents(self, items: list[dict]) -> dict[str, Any]:
        synced = added = updated = 0
        errors: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                continue
            try:
                existing = a2a_registry.get_agent_by_name(str(name))
                if existing:
                    a2a_registry.register_agent(
                        name=str(name),
                        endpoint_url=item.get(
                            "endpoint_url", existing.endpoint_url
                        ),
                        agent_type=item.get("agent_type", existing.agent_type),
                        description=item.get("description", existing.description),
                        auth_type=item.get("auth_type", existing.auth_type),
                        auth_value=item.get("auth_value"),
                        protocol_version=item.get(
                            "protocol_version", existing.protocol_version
                        ),
                        capabilities=item.get("capabilities", existing.capabilities),
                        config=item.get("config", existing.config),
                        tags=item.get("tags", existing.tags),
                        enabled=item.get("enabled", existing.enabled),
                    )
                    updated += 1
                else:
                    a2a_registry.register_agent(
                        name=str(name),
                        endpoint_url=item.get("endpoint_url", ""),
                        agent_type=item.get("agent_type", "jsonrpc"),
                        description=item.get("description", ""),
                        auth_type=item.get("auth_type", "none"),
                        auth_value=item.get("auth_value"),
                        protocol_version=item.get("protocol_version"),
                        capabilities=item.get("capabilities", {}),
                        config=item.get("config", {}),
                        tags=item.get("tags", []),
                        enabled=item.get("enabled", True),
                    )
                    added += 1
                synced += 1
            except Exception as e:
                errors.append(f"agent {name}: {e}")
        return {"synced": synced, "added": added, "updated": updated, "errors": errors}

    def _merge_servers(self, items: list[dict]) -> dict[str, Any]:
        synced = added = updated = 0
        errors: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                continue
            try:
                agents_raw = item.get("associated_a2a_agents", [])
                associated_ids: list[int] = []
                if isinstance(agents_raw, list):
                    for a in agents_raw:
                        if isinstance(a, dict) and "id" in a:
                            try:
                                associated_ids.append(int(a["id"]))
                            except (ValueError, TypeError):
                                continue
                        elif isinstance(a, int):
                            associated_ids.append(a)
                existing = memory.get_a2a_virtual_server_by_name(str(name))
                if existing:
                    memory.upsert_a2a_virtual_server(
                        name=str(name),
                        description=item.get("description", existing.get("description", "")),
                        associated_agents=associated_ids,
                        enabled=item.get("enabled", existing.get("enabled", True)),
                    )
                    updated += 1
                else:
                    a2a_registry.register_virtual_server(
                        name=str(name),
                        description=item.get("description", ""),
                        associated_agents=associated_ids,
                        enabled=item.get("enabled", True),
                    )
                    added += 1
                synced += 1
            except Exception as e:
                errors.append(f"server {name}: {e}")
        return {"synced": synced, "added": added, "updated": updated, "errors": errors}

    def _merge_tools(self, items: list[dict]) -> dict[str, Any]:
        synced = added = updated = 0
        errors: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                continue
            try:
                existing = memory.get_federated_tool(str(name))
                memory.upsert_federated_tool(
                    name=str(name),
                    source="contextforge",
                    source_url=self.external_url,
                    tool=item,
                    enabled=item.get("enabled", True),
                )
                if existing:
                    updated += 1
                else:
                    added += 1
                synced += 1
            except Exception as e:
                errors.append(f"tool {name}: {e}")
        return {"synced": synced, "added": added, "updated": updated, "errors": errors}

    def _merge_prompts(self, items: list[dict]) -> dict[str, Any]:
        synced = added = updated = 0
        errors: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            name = item.get("name")
            if not name:
                continue
            try:
                template_text = item.get("template") or item.get("template_text") or item.get("content")
                if not template_text:
                    continue
                variables = item.get("variables", [])
                if isinstance(variables, list) and variables and isinstance(variables[0], dict):
                    variables = [v.get("name", "") for v in variables if isinstance(v, dict)]
                existing = memory.get_prompt_template_by_name(str(name))
                memory.upsert_prompt_template(
                    name=str(name),
                    description=item.get("description", ""),
                    template_text=str(template_text),
                    variables=variables if isinstance(variables, list) else [],
                    category=item.get("category", "contextforge"),
                    enabled=item.get("enabled", True),
                    is_builtin=False,
                    source="contextforge",
                )
                if existing:
                    updated += 1
                else:
                    added += 1
                synced += 1
            except Exception as e:
                errors.append(f"prompt {name}: {e}")
        return {"synced": synced, "added": added, "updated": updated, "errors": errors}

    # ----- Sync loop -----

    async def sync_loop(self) -> None:
        """Background task: periodic sync."""
        while True:
            try:
                await self.sync_all()
            except Exception as e:
                log.warning("contextforge sync_loop error: %s", e)
            await asyncio.sleep(self.sync_interval_seconds)

    def start_sync_loop(self) -> None:
        if self._sync_task is not None and not self._sync_task.done():
            return
        if not self.auto_sync:
            return
        try:
            loop = asyncio.get_running_loop()
            self._sync_task = loop.create_task(self.sync_loop())
        except RuntimeError:
            log.debug("no running event loop; contextforge sync loop not started")

    def stop_sync_loop(self) -> None:
        if self._sync_task is not None and not self._sync_task.done():
            self._sync_task.cancel()
            self._sync_task = None

    # ----- Embedded mode helpers -----

    def embedded_register_tool(
        self, name: str, tool_definition: dict, source_url: str = "embedded"
    ) -> dict:
        """Register a tool in the local federated registry (embedded mode)."""
        return memory.upsert_federated_tool(
            name=name,
            source="embedded",
            source_url=source_url,
            tool=tool_definition,
            enabled=True,
        )

    def embedded_register_prompt(
        self,
        name: str,
        template_text: str,
        description: str = "",
        variables: list[str] | None = None,
        category: str = "embedded",
    ) -> dict:
        """Register a prompt template in the local registry (embedded mode)."""
        return memory.upsert_prompt_template(
            name=name,
            description=description,
            template_text=template_text,
            variables=variables or [],
            category=category,
            enabled=True,
            is_builtin=False,
            source="embedded",
        )


_default_client: ContextForgeClient | None = None


def init_client(
    mode: str = "external",
    external_url: str | None = None,
    api_key: str | None = None,
    sync_interval_seconds: int = 300,
    auto_sync: bool = False,
    timeout_seconds: float = 30.0,
) -> ContextForgeClient:
    global _default_client
    try:
        client = ContextForgeClient(
            mode=mode,
            external_url=external_url,
            api_key=api_key,
            sync_interval_seconds=sync_interval_seconds,
            auto_sync=auto_sync,
            timeout_seconds=timeout_seconds,
        )
    except ContextForgeError:
        client = ContextForgeClient(mode="embedded")
    _default_client = client
    return client


def client() -> ContextForgeClient | None:
    return _default_client

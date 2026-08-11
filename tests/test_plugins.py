"""Tests for the Gateway Extensions — plugins, A2A, ContextForge, prompts,
webhooks, tool cache.

Covers:
  - memory CRUD for all new tables (plugins, a2a_agents, a2a_virtual_servers,
    a2a_metrics, prompt_templates, webhooks, webhook_deliveries,
    contextforge_sync_log, federated_tools)
  - plugin.PluginLoader + PluginContext (manifest.yaml + build_router)
  - a2a_registry.build_request_payload for jsonrpc/openai/anthropic/custom
  - a2a_registry.invoke_agent with mocked HTTP
  - contextforge_client.ContextForgeClient (mode validation, embedded helpers)
  - mcp_discovery.discover (returns list, errors caught)
  - prompt_registry.render_template, extract_variables, inject_into_messages
  - webhook_dispatcher.WebhookDispatcher (HMAC signing, retry)
  - tool_cache.ToolCache (LRU eviction, TTL expiry, bypass keys)
  - /admin/plugins, /admin/a2a/*, /admin/prompts, /admin/webhooks,
    /admin/cache/*, /admin/contextforge/*, /admin/mcp/discover HTTP routes
"""
from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from aiohttp import web


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


async def _build_test_server(app):
    from aiohttp.test_utils import TestClient, TestServer
    server = TestServer(app)
    client = TestClient(server)
    await client.start_server()
    server._client = client  # type: ignore[attr-defined]
    return server


# ============================================================================
# Memory: new tables (plugins, a2a_agents, prompt_templates, webhooks,
# federated_tools, contextforge_sync_log, a2a_metrics)
# ============================================================================


class MemoryNewTablesTests(unittest.TestCase):

    def setUp(self):
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        tmpdir = tempfile.mkdtemp()
        self.tmpdir = tmpdir
        memory.init_engine(f"sqlite:///{tmpdir}/test.db")

    def tearDown(self):
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]

    # ----- plugins -----

    def test_plugin_upsert_and_list(self):
        from gateway import memory
        memory.upsert_plugin(
            name="slack",
            version="0.1.0",
            description="Slack connector",
            prefix="/integrations/slack",
            module_path="./plugins/slack/plugin.py",
            config={"channel_env": "SLACK_CHANNEL_ID"},
        )
        plugins = memory.list_plugins()
        self.assertEqual(len(plugins), 1)
        self.assertEqual(plugins[0]["name"], "slack")

    def test_plugin_update_via_upsert(self):
        from gateway import memory
        memory.upsert_plugin(name="foo", description="v1")
        memory.upsert_plugin(name="foo", description="v2")
        plugin = memory.get_plugin("foo")
        self.assertEqual(plugin["description"], "v2")

    def test_plugin_set_loaded_and_enabled(self):
        from gateway import memory
        memory.upsert_plugin(name="bar")
        ok = memory.set_plugin_loaded("bar", True, error=None)
        self.assertTrue(ok)
        plugin = memory.get_plugin("bar")
        self.assertTrue(plugin["loaded"])
        memory.set_plugin_enabled("bar", False)
        plugin = memory.get_plugin("bar")
        self.assertFalse(plugin["enabled"])

    def test_plugin_delete_refuses_builtin(self):
        from gateway import memory
        memory.upsert_plugin(name="core", is_builtin=True)
        self.assertFalse(memory.delete_plugin("core"))
        memory.upsert_plugin(name="user", is_builtin=False)
        self.assertTrue(memory.delete_plugin("user"))

    # ----- a2a_agents -----

    def test_a2a_agent_upsert_get_list(self):
        from gateway import memory
        row = memory.upsert_a2a_agent(
            name="hello",
            endpoint_url="http://localhost:9999/",
            agent_type="jsonrpc",
            description="hello world",
            auth_type="bearer",
            auth_value="secret123",
        )
        self.assertNotIn("error", row)
        agent = memory.get_a2a_agent_by_name("hello")
        self.assertIsNotNone(agent)
        self.assertEqual(agent["agent_type"], "jsonrpc")
        agents = memory.list_a2a_agents()
        self.assertEqual(len(agents), 1)

    def test_a2a_agent_invalid_type(self):
        from gateway import memory
        row = memory.upsert_a2a_agent(
            name="bad", endpoint_url="http://x", agent_type="unknown"
        )
        self.assertIn("error", row)

    def test_a2a_agent_set_enabled_and_delete(self):
        from gateway import memory
        memory.upsert_a2a_agent(name="ag", endpoint_url="http://x", agent_type="jsonrpc")
        agent = memory.get_a2a_agent_by_name("ag")
        memory.set_a2a_agent_enabled(agent["id"], False)
        self.assertFalse(memory.get_a2a_agent(agent["id"])["enabled"])
        self.assertTrue(memory.delete_a2a_agent(agent["id"]))

    def test_a2a_metrics_record_and_summary(self):
        from gateway import memory
        memory.upsert_a2a_agent(name="a", endpoint_url="http://x", agent_type="jsonrpc")
        agent = memory.get_a2a_agent_by_name("a")
        memory.record_a2a_metric(agent["id"], "alice", True, 12.5)
        memory.record_a2a_metric(agent["id"], "bob", False, 50.0, error="timeout")
        summary = memory.a2a_agent_metrics_summary(agent["id"])
        self.assertEqual(summary["total_invocations"], 2)
        self.assertEqual(summary["successful"], 1)
        self.assertGreater(summary["latency_avg_ms"], 0)

    # ----- a2a_virtual_servers -----

    def test_virtual_server_crud(self):
        from gateway import memory
        memory.upsert_a2a_agent(name="a1", endpoint_url="http://x", agent_type="jsonrpc")
        agent = memory.get_a2a_agent_by_name("a1")
        vs = memory.upsert_a2a_virtual_server(
            name="server1",
            description="test",
            associated_agents=[agent["id"]],
        )
        self.assertNotIn("error", vs)
        servers = memory.list_a2a_virtual_servers()
        self.assertEqual(len(servers), 1)
        self.assertEqual(servers[0]["name"], "server1")

    # ----- prompt_templates -----

    def test_prompt_upsert_render(self):
        from gateway import memory
        row = memory.upsert_prompt_template(
            name="t1",
            template_text="Hello {name}, you have {count} messages",
            description="greeting",
            variables=["name", "count"],
            category="general",
        )
        self.assertNotIn("error", row)
        tmpl = memory.get_prompt_template_by_name("t1")
        self.assertIsNotNone(tmpl)
        self.assertEqual(tmpl["template_text"], "Hello {name}, you have {count} messages")
        # Second upsert should increment version
        memory.upsert_prompt_template(name="t1", template_text="Hello {name}")
        tmpl2 = memory.get_prompt_template_by_name("t1")
        self.assertEqual(tmpl2["version"], 2)

    def test_prompt_set_enabled_and_delete_builtin(self):
        from gateway import memory
        memory.upsert_prompt_template(name="x", template_text="x", is_builtin=True)
        tmpl = memory.get_prompt_template_by_name("x")
        self.assertFalse(memory.delete_prompt_template(tmpl["id"]))
        memory.upsert_prompt_template(name="y", template_text="y", is_builtin=False)
        tmpl2 = memory.get_prompt_template_by_name("y")
        self.assertTrue(memory.delete_prompt_template(tmpl2["id"]))

    # ----- webhooks -----

    def test_webhook_upsert_delete(self):
        from gateway import memory
        row = memory.upsert_webhook(
            name="w1",
            url="http://example.com/webhook",
            events=["chat.completion", "security.alert"],
            secret="topsecret",
        )
        self.assertNotIn("error", row)
        wh = memory.get_webhook_by_name("w1")
        self.assertIsNotNone(wh)
        # record delivery
        delivery_id = memory.record_webhook_delivery(
            webhook_id=wh["id"],
            event_type="chat.completion",
            tenant_id="alice",
            status_code=200,
            payload_json="{}",
            response_body="OK",
            attempt=1,
            duration_ms=10.0,
        )
        self.assertGreater(delivery_id, 0)
        deliveries = memory.list_webhook_deliveries(webhook_id=wh["id"])
        self.assertEqual(len(deliveries), 1)
        self.assertTrue(memory.delete_webhook(wh["id"]))

    # ----- contextforge_sync_log + federated_tools -----

    def test_contextforge_sync_record_and_list(self):
        from gateway import memory
        memory.record_contextforge_sync(
            sync_type="agents",
            source="http://localhost:4444",
            items_synced=5,
            items_added=3,
            items_updated=2,
            errors=[],
            duration_ms=120.0,
        )
        log = memory.list_contextforge_sync_log()
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["sync_type"], "agents")

    def test_federated_tool_upsert_and_delete(self):
        from gateway import memory
        row = memory.upsert_federated_tool(
            name="mcp_test",
            source="mcp_discovery",
            source_url="http://localhost:8000",
            tool={"name": "mcp_test", "capabilities": {}},
        )
        self.assertNotIn("error", row)
        tools = memory.list_federated_tools()
        self.assertEqual(len(tools), 1)
        self.assertTrue(memory.delete_federated_tool("mcp_test"))


# ============================================================================
# Plugin loader — manifest.yaml + build_router
# ============================================================================


class PluginLoaderTests(unittest.TestCase):

    def _make_manifest(self, name, *, with_module=True, broken=False):
        tmpdir = Path(tempfile.mkdtemp())
        plugin_dir = tmpdir / name
        plugin_dir.mkdir()
        manifest = {
            "name": name,
            "version": "0.1.0",
            "description": f"Test plugin {name}",
            "module": "plugin.py",
            "prefix": f"/integrations/{name}",
        }
        (plugin_dir / "manifest.yaml").write_text(
            "\n".join(f"{k}: {v!r}" if isinstance(v, str) else f"{k}: {v}"
                     for k, v in manifest.items())
        )
        if with_module:
            if broken:
                (plugin_dir / "plugin.py").write_text("raise RuntimeError('broken')\n")
            else:
                code = (
                    "from aiohttp import web\n"
                    "def build_router(context):\n"
                    "    r = web.RouteTableDef()\n"
                    "    @r.get('/integrations/" + name + "/ping')\n"
                    "    async def ping(request):\n"
                    "        return web.json_response({'pong': True, 'name': context.name})\n"
                    "    return r\n"
                )
                (plugin_dir / "plugin.py").write_text(code)
        return tmpdir

    def test_load_all_success(self):
        from gateway import events
        from gateway import plugin as plugin_mod
        # Create a real plugin in tmpdir
        root = self._make_manifest("test_plugin")
        loader = plugin_mod.PluginLoader(
            app=web.Application(),
            config=MagicMock(),
            event_bus=events.EventBus(),
            plugin_root=root,
        )
        count = loader.load_all()
        self.assertEqual(count, 1)
        self.assertIn("test_plugin", loader.plugins)
        self.assertTrue(loader.plugins["test_plugin"].loaded)
        self.assertIsNone(loader.plugins["test_plugin"].error)

    def test_load_broken_plugin_records_error(self):
        from gateway import events
        from gateway import plugin as plugin_mod
        root = self._make_manifest("broken_plugin", broken=True)
        loader = plugin_mod.PluginLoader(
            app=web.Application(),
            config=MagicMock(),
            event_bus=events.EventBus(),
            plugin_root=root,
        )
        count = loader.load_all()
        self.assertEqual(count, 0)
        self.assertIn("broken_plugin", loader.plugins)
        self.assertFalse(loader.plugins["broken_plugin"].loaded)
        self.assertIsNotNone(loader.plugins["broken_plugin"].error)

    def test_load_plugin_without_module(self):
        from gateway import events
        from gateway import plugin as plugin_mod
        root = self._make_manifest("nomod", with_module=False)
        loader = plugin_mod.PluginLoader(
            app=web.Application(),
            config=MagicMock(),
            event_bus=events.EventBus(),
            plugin_root=root,
        )
        # No module -> error
        loader.load_all()
        self.assertIn("nomod", loader.plugins)
        self.assertFalse(loader.plugins["nomod"].loaded)

    def test_root_missing_is_noop(self):
        from gateway import events
        from gateway import plugin as plugin_mod
        loader = plugin_mod.PluginLoader(
            app=web.Application(),
            config=MagicMock(),
            event_bus=events.EventBus(),
            plugin_root="/no/such/directory/anywhere",
        )
        self.assertEqual(loader.load_all(), 0)

    def test_plugin_context_helpers(self):
        from gateway import events
        from gateway import plugin as plugin_mod
        ctx = plugin_mod.PluginContext(
            name="myctx",
            manifest={"settings": {"x": 1, "y": "hi"}},
            config=MagicMock(),
            event_bus=events.EventBus(),
        )
        self.assertEqual(ctx.get_setting("x"), 1)
        self.assertEqual(ctx.get_setting("missing", "default"), "default")
        # emit_event should not raise even with no event loop
        _run(ctx.emit_event("test.event", {"k": "v"}, tenant_id="alice"))


# ============================================================================
# A2A registry — request payload formats + auth headers
# ============================================================================


class A2ARegistryTests(unittest.TestCase):

    def _make_agent(self, **overrides):
        from gateway import a2a_registry
        defaults = {
            "id": 1,
            "name": "test_agent",
            "endpoint_url": "http://localhost:9999/",
            "agent_type": "jsonrpc",
            "description": "",
            "auth_type": "none",
            "auth_value": "",
            "protocol_version": "1.0",
            "capabilities": {},
            "config": {},
            "tags": [],
            "enabled": True,
        }
        defaults.update(overrides)
        return a2a_registry.A2AAgentRecord(**defaults)

    def test_jsonrpc_payload(self):
        from gateway import a2a_registry
        agent = self._make_agent(agent_type="jsonrpc")
        url, body = a2a_registry.build_request_payload(
            agent, {"method": "message/send", "params": {"x": 1}}
        )
        self.assertEqual(url, "http://localhost:9999/")
        self.assertEqual(body["jsonrpc"], "2.0")
        self.assertEqual(body["method"], "message/send")
        self.assertEqual(body["params"], {"x": 1})

    def test_openai_payload(self):
        from gateway import a2a_registry
        agent = self._make_agent(agent_type="openai")
        url, body = a2a_registry.build_request_payload(
            agent, {"messages": [{"role": "user", "content": "hi"}]}
        )
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])
        self.assertIn("max_tokens", body)

    def test_anthropic_payload(self):
        from gateway import a2a_registry
        agent = self._make_agent(agent_type="anthropic")
        url, body = a2a_registry.build_request_payload(
            agent, {"messages": [{"role": "user", "content": "hi"}], "system": "be nice"}
        )
        self.assertEqual(body["system"], "be nice")

    def test_custom_payload(self):
        from gateway import a2a_registry
        agent = self._make_agent(agent_type="custom")
        url, body = a2a_registry.build_request_payload(
            agent, {"anything": "goes"}
        )
        self.assertEqual(body, {"anything": "goes"})

    def test_auth_headers(self):
        from gateway import a2a_registry
        agent_none = self._make_agent(auth_type="none")
        self.assertEqual(a2a_registry.build_auth_headers(agent_none), {})
        agent_bearer = self._make_agent(auth_type="bearer", auth_value="abc")
        self.assertEqual(
            a2a_registry.build_auth_headers(agent_bearer),
            {"Authorization": "Bearer abc"},
        )
        agent_apikey = self._make_agent(auth_type="api_key", auth_value="xyz")
        self.assertEqual(
            a2a_registry.build_auth_headers(agent_apikey),
            {"Authorization": "Bearer xyz"},
        )

    def test_invoke_agent_records_metrics(self):
        from gateway import a2a_registry, memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        tmpdir = tempfile.mkdtemp()
        memory.init_engine(f"sqlite:///{tmpdir}/test.db")
        try:
            # Register agent
            memory.upsert_a2a_agent(
                name="inv", endpoint_url="http://localhost:65535/", agent_type="jsonrpc"
            )
            agent_id = memory.get_a2a_agent_by_name("inv")["id"]
            agent = a2a_registry.get_agent(agent_id)
            # Invoke with mocked session — connection refused -> error
            mock_session = MagicMock()
            resp = MagicMock()
            resp.status = 503
            resp.text = AsyncMock(return_value="service unavailable")
            resp.__aenter__ = AsyncMock(return_value=resp)
            resp.__aexit__ = AsyncMock(return_value=None)
            mock_session.post = MagicMock(return_value=resp)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=None)
            result = _run(a2a_registry.invoke_agent(agent, {"method": "ping"}, session=mock_session, tenant_id="alice"))
            self.assertFalse(result.success)
            self.assertEqual(result.status_code, 503)
            summary = memory.a2a_agent_metrics_summary(agent_id)
            self.assertEqual(summary["total_invocations"], 1)
            self.assertEqual(summary["successful"], 0)
        finally:
            memory.close_engine()
            memory._engine = None  # type: ignore[attr-defined]


# ============================================================================
# ContextForge connector — mode validation + embedded helpers
# ============================================================================


class ContextForgeClientTests(unittest.TestCase):

    def test_invalid_mode_raises(self):
        from gateway import contextforge_client
        with self.assertRaises(contextforge_client.ContextForgeError):
            contextforge_client.ContextForgeClient(mode="bad")

    def test_external_mode_requires_url(self):
        from gateway import contextforge_client
        with self.assertRaises(contextforge_client.ContextForgeError):
            contextforge_client.ContextForgeClient(mode="external", external_url=None)

    def test_embedded_mode_works(self):
        from gateway import contextforge_client
        c = contextforge_client.ContextForgeClient(mode="embedded")
        self.assertEqual(c.mode, "embedded")
        self.assertEqual(c.external_url, "")

    def test_both_mode_with_url(self):
        from gateway import contextforge_client
        c = contextforge_client.ContextForgeClient(
            mode="both", external_url="http://localhost:4444", api_key="secret"
        )
        self.assertEqual(c.mode, "both")
        self.assertEqual(c.api_key, "secret")

    def test_embedded_register_tool_and_prompt(self):
        from gateway import contextforge_client, memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        tmpdir = tempfile.mkdtemp()
        memory.init_engine(f"sqlite:///{tmpdir}/test.db")
        try:
            c = contextforge_client.ContextForgeClient(mode="embedded")
            tool_row = c.embedded_register_tool(
                "test_tool", {"name": "test_tool", "description": "demo"}
            )
            self.assertNotIn("error", tool_row)
            prompt_row = c.embedded_register_prompt(
                "test_prompt", "Hello {name}", description="greeting",
                variables=["name"], category="demo",
            )
            self.assertNotIn("error", prompt_row)
        finally:
            memory.close_engine()
            memory._engine = None  # type: ignore[attr-defined]


# ============================================================================
# MCP discovery
# ============================================================================


class MCPDiscoveryTests(unittest.TestCase):

    def test_discover_returns_list(self):
        from gateway import mcp_discovery
        # Use invalid port to ensure we get empty list fast
        results = _run(mcp_discovery.discover(hosts=["localhost"], ports=[1]))
        self.assertIsInstance(results, list)
        # Empty list since port 1 should not have any MCP servers
        self.assertEqual(len(results), 0)

    def test_discover_handles_errors(self):
        from gateway import mcp_discovery
        # Even with weird inputs, should not raise
        results = _run(mcp_discovery.discover(hosts=[], ports=[]))
        self.assertIsInstance(results, list)


# ============================================================================
# Prompt template registry — render + inject
# ============================================================================


class PromptRegistryTests(unittest.TestCase):

    def setUp(self):
        from gateway import memory, prompt_registry
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        tmpdir = tempfile.mkdtemp()
        self.tmpdir = tmpdir
        memory.init_engine(f"sqlite:///{tmpdir}/test.db")
        prompt_registry.seed_builtin_templates()

    def tearDown(self):
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]

    def test_extract_variables(self):
        from gateway import prompt_registry
        vars_ = prompt_registry.extract_variables("Hello {name}, count={count}")
        self.assertEqual(vars_, ["count", "name"])  # sorted

    def test_render_template_substitutes(self):
        from gateway import prompt_registry
        result = prompt_registry.render_template(
            "Hello {name}, you have {count} messages",
            {"name": "Alice", "count": 5},
        )
        self.assertEqual(result, "Hello Alice, you have 5 messages")

    def test_render_template_keeps_unknown_placeholders(self):
        from gateway import prompt_registry
        result = prompt_registry.render_template(
            "Hello {name}", {"other": "x"}
        )
        self.assertEqual(result, "Hello {name}")

    def test_inject_into_messages_prepends_system(self):
        from gateway import prompt_registry
        messages = [{"role": "user", "content": "hi"}]
        out = prompt_registry.inject_into_messages(
            messages, "router_coder", position="system"
        )
        self.assertEqual(out[0]["role"], "system")
        self.assertEqual(out[1]["role"], "user")

    def test_inject_into_messages_replaces_existing_system(self):
        from gateway import prompt_registry
        messages = [
            {"role": "system", "content": "old prompt"},
            {"role": "user", "content": "hi"},
        ]
        out = prompt_registry.inject_into_messages(
            messages, "router_coder", position="system", replace_existing_system=True
        )
        self.assertEqual(out[0]["role"], "system")
        self.assertNotEqual(out[0]["content"], "old prompt")
        self.assertEqual(len(out), 2)

    def test_inject_skips_disabled_template(self):
        from gateway import memory, prompt_registry
        # Disable the template
        tmpl = memory.get_prompt_template_by_name("router_coder")
        memory.set_prompt_template_enabled(tmpl["id"], False)
        messages = [{"role": "user", "content": "hi"}]
        out = prompt_registry.inject_into_messages(
            messages, "router_coder", position="system"
        )
        # No system message added
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["role"], "user")

    def test_category_for_vertical(self):
        from gateway import prompt_registry
        self.assertEqual(prompt_registry.category_for_vertical("code"), "code")
        self.assertEqual(prompt_registry.category_for_vertical("translation"), "translation")
        self.assertIsNone(prompt_registry.category_for_vertical(None))
        self.assertIsNone(prompt_registry.category_for_vertical("unknown_vertical"))


# ============================================================================
# Webhook dispatcher — HMAC signing
# ============================================================================


class WebhookDispatcherTests(unittest.TestCase):

    def test_sign(self):
        from gateway import webhook_dispatcher
        sig = webhook_dispatcher._sign("topsecret", '{"x":1}')
        self.assertTrue(sig.startswith("sha256="))
        self.assertEqual(len(sig), 7 + 64)

    def test_init_dispatcher_creates_default(self):
        from gateway import webhook_dispatcher
        d = webhook_dispatcher.init_dispatcher()
        self.assertIsNotNone(d)
        self.assertEqual(d.max_retries, 3)
        self.assertIs(webhook_dispatcher.dispatcher(), d)

    def test_matches_event(self):
        from gateway import memory, webhook_dispatcher
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        tmpdir = tempfile.mkdtemp()
        memory.init_engine(f"sqlite:///{tmpdir}/test.db")
        try:
            wh_star = webhook_dispatcher.WebhookRecord(
                id=1, name="all", url="http://x",
                events=["*"], secret="", enabled=True, description="",
            )
            wh_specific = webhook_dispatcher.WebhookRecord(
                id=2, name="specific", url="http://x",
                events=["chat.completion"], secret="", enabled=True, description="",
            )
            self.assertTrue(webhook_dispatcher._matches(wh_star, "anything"))
            self.assertTrue(webhook_dispatcher._matches(wh_specific, "chat.completion"))
            self.assertFalse(webhook_dispatcher._matches(wh_specific, "other.event"))
        finally:
            memory.close_engine()
            memory._engine = None  # type: ignore[attr-defined]


# ============================================================================
# Tool cache — LRU + TTL
# ============================================================================


class ToolCacheTests(unittest.TestCase):

    def test_set_get_hit_miss(self):
        from gateway import tool_cache
        c = tool_cache.ToolCache(max_entries=10, default_ttl_seconds=60)
        self.assertIsNone(c.get("tool", {"x": 1}))
        c.set("tool", {"x": 1}, {"result": "v"})
        self.assertEqual(c.get("tool", {"x": 1}), {"result": "v"})
        stats = c.stats()
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["sets"], 1)

    def test_ttl_expiration(self):

        from gateway import tool_cache
        c = tool_cache.ToolCache(max_entries=10, default_ttl_seconds=1)
        c.set("tool", {"x": 1}, {"v": 1}, ttl_seconds=1)
        self.assertEqual(c.get("tool", {"x": 1}), {"v": 1})
        # Manually mark expired by manipulating entry
        with c._lock:
            for entry in c._entries.values():
                entry.expires_at = 0.0  # epoch
        self.assertIsNone(c.get("tool", {"x": 1}))
        # Verify expiration counter incremented
        self.assertGreater(c.stats()["expirations"], 0)

    def test_lru_eviction(self):
        from gateway import tool_cache
        c = tool_cache.ToolCache(max_entries=2, default_ttl_seconds=60)
        c.set("t", {"i": 1}, 1)
        c.set("t", {"i": 2}, 2)
        c.set("t", {"i": 3}, 3)  # evicts i=1
        self.assertEqual(c.get("t", {"i": 2}), 2)
        self.assertIsNone(c.get("t", {"i": 1}))
        self.assertEqual(c.get("t", {"i": 3}), 3)
        self.assertGreater(c.stats()["evictions"], 0)

    def test_bypass_keys(self):
        from gateway import tool_cache
        c = tool_cache.ToolCache(
            max_entries=10, default_ttl_seconds=60, bypass_keys=["secret"]
        )
        c.set("t", {"secret": "abc"}, "stored")
        self.assertIsNone(c.get("t", {"secret": "abc"}))
        # Non-bypassed still works
        c.set("t", {"safe": "value"}, "stored")
        self.assertEqual(c.get("t", {"safe": "value"}), "stored")

    def test_invalidate_by_tool_name(self):
        from gateway import tool_cache
        c = tool_cache.ToolCache(max_entries=10, default_ttl_seconds=60)
        c.set("tool_a", {"x": 1}, "v1")
        c.set("tool_b", {"x": 2}, "v2")
        removed = c.invalidate(tool_name="tool_a")
        self.assertEqual(removed, 1)
        self.assertIsNone(c.get("tool_a", {"x": 1}))
        self.assertEqual(c.get("tool_b", {"x": 2}), "v2")

    def test_invalidate_all(self):
        from gateway import tool_cache
        c = tool_cache.ToolCache(max_entries=10, default_ttl_seconds=60)
        c.set("a", {}, 1)
        c.set("b", {}, 2)
        removed = c.invalidate()
        self.assertEqual(removed, 2)
        self.assertEqual(c.stats()["size"], 0)

    def test_tenant_isolation(self):
        from gateway import tool_cache
        c = tool_cache.ToolCache(max_entries=10, default_ttl_seconds=60)
        c.set("t", {"x": 1}, "alice", tenant_id="alice")
        c.set("t", {"x": 1}, "bob", tenant_id="bob")
        self.assertEqual(c.get("t", {"x": 1}, tenant_id="alice"), "alice")
        self.assertEqual(c.get("t", {"x": 1}, tenant_id="bob"), "bob")

    def test_snapshot(self):
        from gateway import tool_cache
        c = tool_cache.ToolCache(max_entries=10, default_ttl_seconds=60)
        c.set("t", {"x": 1}, "v", tenant_id="alice")
        snap = c.snapshot(limit=10)
        self.assertEqual(len(snap), 1)
        self.assertEqual(snap[0]["tool_name"], "t")
        self.assertEqual(snap[0]["tenant_id"], "alice")

    def test_init_cache(self):
        from gateway import tool_cache
        c = tool_cache.init_cache(max_entries=5, default_ttl_seconds=10)
        self.assertIsNotNone(c)
        self.assertEqual(tool_cache.cache(), c)


# ============================================================================
# HTTP integration tests for new admin routes
# ============================================================================


class _NewRoutesTestBase(unittest.TestCase):

    def setUp(self):
        from gateway import app as app_mod
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        tmpdir = tempfile.mkdtemp()
        self.tmpdir = tmpdir
        db_url = f"sqlite:///{tmpdir}/test.db"
        config = self._build_config(db_url)
        cfg_path = f"{tmpdir}/gateway-config.json"
        with open(cfg_path, "w") as f:
            json.dump(config, f)
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.app = self.loop.run_until_complete(app_mod.init_app(cfg_path))
        self.server = self.loop.run_until_complete(_build_test_server(self.app))

    def tearDown(self):
        try:
            self.loop.run_until_complete(self.server._client.close())
        except Exception:
            pass
        try:
            self.loop.run_until_complete(self.server.close())
        except Exception:
            pass
        self.loop.close()
        asyncio.set_event_loop(None)
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]

    def _build_config(self, db_url):
        return {
            "db_url": db_url,
            "mode": "single",
            "host": "127.0.0.1",
            "port": 0,
            "tenants": {"*": {"tier_access": ["tier0"], "budget_usd_per_day": 100.0}},
            "endpoints": [
                {
                    "name": "ep_test", "kind": "llamacpp", "base_url": "http://127.0.0.1:1",
                    "model_alias": "m",
                    "pricing": {"fixed_per_request": 0.0, "in_per_1k_tokens": 0.0, "out_per_1k_tokens": 0.0},
                    "concurrency": 1,
                    "breaker": {"failure_threshold": 1, "open_duration_seconds": 1, "half_open_max_probes": 1},
                },
            ],
            "tiers": [
                {"name": "tier0", "endpoints": ["ep_test"], "max_context": 32768,
                 "capability_per_vertical": {"_default": 0.95}, "max_tokens_bump": 0},
            ],
            "security": {"injection_regex": [], "provider_allowlist": {"enabled": False, "default_action": "block", "global_patterns": []}},
            "auth": {"enabled": False, "keys": {}, "admin_paths": ["/admin"]},
            "http": {"max_body_bytes": 4 * 1024 * 1024, "cors_origins": []},
            "memory": {"enabled": False},
            "embedding": {"onnx_path": "x", "model_id": "y", "checksum_sha256": ""},
            "routing": {"ood_threshold": 0.25, "cost_first": {"fallback_endpoint": "ep_test"}},
            "logging": {"flagged_retention_days": 7},
            "reviewer": {"endpoint": "http://127.0.0.1:1", "model": "m", "api_key_env": "x", "timeout_seconds": 30, "batch_size": 1, "caps": {"per_request_usd": 1.0, "per_hour_usd": 1.0, "per_day_usd": 1.0, "per_month_usd": 1.0}},
            "trainer": {"auto_retrain": False, "trigger_threshold_new_samples": 500, "trigger_accuracy_drop_below": 0.0, "min_trust_score_to_train": 0.0},
            "drift": {"enabled": False},
            "policy": {"_loaded_from": "tests"},
            "plugins": {"enabled": True, "root": self.tmpdir + "/plugins", "scan_interval_seconds": 30, "auto_load": False},
            "a2a": {"enabled": True, "max_agents": 100, "default_timeout": 30, "max_retries": 3, "metrics_enabled": True},
            "contextforge": {"enabled": False, "mode": "embedded", "external_url": "", "api_key": "", "sync_interval_seconds": 300, "auto_sync": False, "timeout_seconds": 30},
            "mcp_discovery": {"enabled": False, "probe_interval_seconds": 60, "auto_register": False, "hosts": ["localhost"], "ports": [8000]},
            "prompts": {"enabled": True, "auto_inject": False, "default_category": "general"},
            "webhooks": {"enabled": True, "max_retries": 3, "initial_backoff_seconds": 1.0, "backoff_multiplier": 2.0, "delivery_timeout_seconds": 10.0, "max_concurrent_deliveries": 16},
            "tool_cache": {"enabled": False, "max_entries": 100, "default_ttl_seconds": 60},
        }

    async def _async_get(self, path):
        client = self.server._client
        resp = await client.get(path)
        return resp, await resp.json()

    async def _async_post(self, path, body):
        client = self.server._client
        resp = await client.post(path, json=body)
        return resp, await resp.json()

    async def _async_put(self, path, body):
        client = self.server._client
        resp = await client.put(path, json=body)
        return resp, await resp.json()

    async def _async_delete(self, path):
        client = self.server._client
        resp = await client.delete(path)
        return resp, await resp.json()

    def _run_coro(self, coro):
        return self.loop.run_until_complete(coro)


class PluginRoutesTests(_NewRoutesTestBase):

    def test_list_plugins_empty(self):
        resp, data = self._run_coro(self._async_get("/admin/plugins"))
        self.assertIn("plugins", data)
        self.assertEqual(data["plugins"], [])

    def test_upsert_get_delete_plugin(self):
        resp, row = self._run_coro(self._async_post("/admin/plugins/test_plugin", {
            "name": "test_plugin",
            "version": "0.1.0",
            "description": "Test plugin",
            "prefix": "/test",
            "module_path": "./plugins/test/plugin.py",
        }))
        self.assertEqual(row["name"], "test_plugin")
        resp, fetched = self._run_coro(self._async_get("/admin/plugins/test_plugin"))
        self.assertEqual(fetched["name"], "test_plugin")
        resp, deleted = self._run_coro(self._async_delete("/admin/plugins/test_plugin"))
        self.assertIn("deleted", deleted)


class A2ARoutesTests(_NewRoutesTestBase):

    def test_list_agents_empty(self):
        resp, data = self._run_coro(self._async_get("/admin/a2a/agents"))
        self.assertEqual(data["agents"], [])

    def test_create_get_invoke_delete_agent(self):
        resp, row = self._run_coro(self._async_post("/admin/a2a/agents", {
            "name": "ag1",
            "endpoint_url": "http://localhost:65535/",
            "agent_type": "jsonrpc",
            "description": "test",
        }))
        self.assertEqual(row["name"], "ag1")
        agent_id = row["id"]
        resp, fetched = self._run_coro(self._async_get(f"/admin/a2a/agents/{agent_id}"))
        self.assertEqual(fetched["name"], "ag1")
        # Update
        resp, updated = self._run_coro(self._async_put(f"/admin/a2a/agents/{agent_id}", {
            "description": "updated",
        }))
        self.assertEqual(updated["description"], "updated")
        # Invoke (will fail because localhost:65535 isn't running, but should not 500)
        resp, inv = self._run_coro(self._async_post(f"/admin/a2a/agents/{agent_id}/invoke", {
            "parameters": {"method": "message/send", "params": {}},
        }))
        self.assertIn("success", inv)
        # Metrics
        resp, metrics = self._run_coro(self._async_get(f"/admin/a2a/agents/{agent_id}/metrics"))
        self.assertEqual(metrics["agent_id"], agent_id)
        # Delete
        resp, deleted = self._run_coro(self._async_delete(f"/admin/a2a/agents/{agent_id}"))
        self.assertIn("deleted", deleted)

    def test_list_servers_empty(self):
        resp, data = self._run_coro(self._async_get("/admin/a2a/servers"))
        self.assertEqual(data["servers"], [])

    def test_create_and_delete_virtual_server(self):
        # First create an agent
        resp, agent = self._run_coro(self._async_post("/admin/a2a/agents", {
            "name": "ag2",
            "endpoint_url": "http://localhost:65535/",
            "agent_type": "jsonrpc",
        }))
        resp, server = self._run_coro(self._async_post("/admin/a2a/servers", {
            "name": "server1",
            "description": "test server",
            "associated_agents": [agent["id"]],
        }))
        self.assertEqual(server["name"], "server1")
        resp, deleted = self._run_coro(self._async_delete(f"/admin/a2a/servers/{server['id']}"))
        self.assertIn("deleted", deleted)


class PromptRoutesTests(_NewRoutesTestBase):

    def test_seeded_templates_present(self):
        resp, data = self._run_coro(self._async_get("/admin/prompts"))
        names = [t["name"] for t in data["templates"]]
        self.assertIn("router_coder", names)
        self.assertIn("translator", names)
        self.assertIn("summarizer", names)
        self.assertIn("safety_refusal", names)

    def test_create_and_delete_prompt(self):
        resp, row = self._run_coro(self._async_post("/admin/prompts", {
            "name": "test_prompt",
            "template_text": "Hello {name}",
            "category": "general",
            "variables": ["name"],
        }))
        self.assertEqual(row["name"], "test_prompt")
        resp, deleted = self._run_coro(self._async_delete(f"/admin/prompts/{row['id']}"))
        self.assertIn("deleted", deleted)

    def test_create_prompt_missing_name(self):
        resp, body = self._run_coro(self._async_post("/admin/prompts", {
            "template_text": "no name",
        }))
        self.assertEqual(resp.status, 400)


class WebhookRoutesTests(_NewRoutesTestBase):

    def test_create_list_delete_webhook(self):
        resp, row = self._run_coro(self._async_post("/admin/webhooks", {
            "name": "w1",
            "url": "http://localhost:9999/webhook",
            "events": ["chat.completion", "security.alert"],
        }))
        self.assertEqual(row["name"], "w1")
        resp, listing = self._run_coro(self._async_get("/admin/webhooks"))
        self.assertGreaterEqual(len(listing["webhooks"]), 1)
        resp, deliveries = self._run_coro(self._async_get("/admin/webhooks/deliveries"))
        self.assertIn("deliveries", deliveries)
        resp, deleted = self._run_coro(self._async_delete(f"/admin/webhooks/{row['id']}"))
        self.assertIn("deleted", deleted)

    def test_create_webhook_missing_fields(self):
        resp, body = self._run_coro(self._async_post("/admin/webhooks", {
            "name": "incomplete",
        }))
        self.assertEqual(resp.status, 400)


class ContextForgeRoutesTests(_NewRoutesTestBase):

    def test_sync_disabled(self):
        # Default config has contextforge.enabled=False
        resp, body = self._run_coro(self._async_post("/admin/contextforge/sync", {}))
        self.assertEqual(resp.status, 400)

    def test_sync_log_listing(self):
        resp, body = self._run_coro(self._async_get("/admin/contextforge/sync-log"))
        self.assertIn("sync_log", body)

    def test_tools_listing(self):
        resp, body = self._run_coro(self._async_get("/admin/contextforge/tools"))
        self.assertIn("tools", body)


class MCPRoutesTests(_NewRoutesTestBase):

    def test_discover(self):
        resp, body = self._run_coro(self._async_post("/admin/mcp/discover", {
            "hosts": ["localhost"],
            "ports": [1],  # Invalid port — should return empty list fast
        }))
        self.assertEqual(resp.status, 200)
        self.assertIn("discovered", body)


class CacheRoutesTests(_NewRoutesTestBase):

    def test_cache_disabled(self):
        # Default config has tool_cache.enabled=False
        resp, body = self._run_coro(self._async_get("/admin/cache/stats"))
        self.assertEqual(resp.status, 400)

    def test_cache_enabled(self):
        # Enable cache via hot reload by re-initializing tool_cache directly
        from gateway import tool_cache
        old_cache = tool_cache.cache()
        try:
            tool_cache.init_cache(max_entries=10, default_ttl_seconds=60)
            resp, body = self._run_coro(self._async_get("/admin/cache/stats"))
            self.assertEqual(resp.status, 200)
            self.assertIn("stats", body)
            # Invalidate
            resp, inv = self._run_coro(self._async_post("/admin/cache/invalidate", {}))
            self.assertEqual(resp.status, 200)
            self.assertIn("invalidated", inv)
        finally:
            # Restore old cache state (None) for next tests
            if old_cache is None:
                tool_cache._default_cache = None  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()

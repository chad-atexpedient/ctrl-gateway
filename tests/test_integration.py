"""End-to-end integration tests.

Runs the REAL gateway (init_app) against in-process mock upstreams
(tests/mock_endpoints.py) and asserts full-pipeline contracts:

  - chat request -> routed -> forwarded -> response + logged decision
  - streaming request -> SSE passthrough assembled
  - fallback ladder: chosen endpoint down -> same-tier endpoint used -> fallback_used
  - data flywheel: review (mock reviewer) -> review_result -> curated_samples row
  - /trace returns valid JSON
  - auth enabled: admin 401 without key, chat maps key to tenant

Run from repo root:  python -m unittest tests.test_integration
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from tests import mock_endpoints

REPO = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class MockUpstream:
    """In-process mock LLM server on an ephemeral port.

    Behavior is scripted via env vars (MOCK_FAIL_ENDPOINTS,
    MOCK_BREAK_ENDPOINTS, MOCK_REVIEWER_LABELS) which the mock reads
    per-request — no restart needed.
    """

    def __init__(self, name: str, latency_ms: int = 1):
        self.name = name
        self.latency_ms = latency_ms
        self.port = _free_port()
        self.runner = None
        self.site = None
        self.app = None

    async def start(self):
        self.app = await mock_endpoints.make_app(self.name, self.latency_ms)
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", self.port)
        await self.site.start()

    async def stop(self):
        if self.runner:
            await self.runner.cleanup()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _base_config(tmpdir: str, *, ep_a: MockUpstream, ep_b: MockUpstream, ep_ollama: MockUpstream, reviewer: MockUpstream, auth: dict | None = None) -> dict:
    return {
        "mode": "single",
        "db_url": f"sqlite:///{tmpdir}/itest.db",
        "host": "127.0.0.1",
        "port": 0,
        "tenants": {
            "*": {
                "tier_access": ["tier0", "tier1", "tier2", "tier3", "tier4", "tier_medical"],
                "budget_usd_per_day": 100.0,
                "rps_limit": 1000,
                "concurrent_limit": 50,
                "tokens_per_min": 10000000,
            }
        },
        "endpoints": [
            {
                "name": "mock_a",
                "kind": "llamacpp",
                "base_url": ep_a.base_url,
                "model_alias": "mock-a",
                "pricing": {"fixed_per_request": 0.001, "in_per_1k_tokens": 0.002, "out_per_1k_tokens": 0.004},
                "concurrency": 4,
                "breaker": {"failure_threshold": 2, "open_duration_seconds": 2, "half_open_max_probes": 1},
                "health_probe": "/health",
            },
            {
                "name": "mock_b",
                "kind": "llamacpp",
                "base_url": ep_b.base_url,
                "model_alias": "mock-b",
                "pricing": {"fixed_per_request": 0.001, "in_per_1k_tokens": 0.003, "out_per_1k_tokens": 0.005},
                "concurrency": 4,
                "breaker": {"failure_threshold": 2, "open_duration_seconds": 2, "half_open_max_probes": 1},
                "health_probe": "/health",
            },
            {
                "name": "mock_ollama",
                "kind": "ollama",
                "base_url": ep_ollama.base_url,
                "model_alias": "ollama-model",
                # Deliberately expensive so cost-first never picks it over mock_a/mock_b.
                "pricing": {"fixed_per_request": 5.0, "in_per_1k_tokens": 1.0, "out_per_1k_tokens": 1.0},
                "concurrency": 4,
                "breaker": {"failure_threshold": 2, "open_duration_seconds": 2, "half_open_max_probes": 1},
                "health_probe": "/health",
            },
        ],
        "tiers": [
            {
                "name": "tier0",
                "endpoints": ["mock_a", "mock_b", "mock_ollama"],
                "max_context": 32768,
                "capability_per_vertical": {"_default": 0.95},
                "max_tokens_bump": 0,
            },
            {
                "name": "tier1",
                "endpoints": ["mock_b"],
                "max_context": 65536,
                "capability_per_vertical": {"_default": 0.98},
                "max_tokens_bump": 0,
            },
        ],
        "routing": {
            "cost_first": {"fit_threshold": 0.9, "capability_sigmoid_k": 20.0, "retry_penalty_multiplier": 5.0, "fallback_endpoint": "mock_b"},
            "escalation": {"ood_flag_to_tier": "tier1", "confidence_threshold": 0.5, "top2_epsilon": 0.10, "cost_margin_abstain_pct": 5.0},
            "efficiency_tiebreak": {"prefer_healthier": True, "prefer_lower_load": True},
            "working_memory": {"enabled": False},
        },
        "reviewer": {
            "endpoint": reviewer.base_url,
            "model": "mock-reviewer",
            "api_key_env": "TEST_REVIEWER_KEY",
            "timeout_seconds": 30,
            "batch_size": 10,
            "max_prompt_tokens": 100000,
            "caps": {"per_request_usd": 100.0, "per_hour_usd": 100.0, "per_day_usd": 100.0, "per_month_usd": 100.0},
            "estimated_in_per_1k": 0.0,
            "estimated_out_per_1k": 0.0,
        },
        "trainer": {
            "auto_retrain": False,
            "trigger_threshold_new_samples": 500,
            "trigger_accuracy_drop_below": 0.0,
            "min_trust_score_to_train": 0.0,
        },
        "security": {"injection_regex": []},
        "drift": {"enabled": False},
        "logging": {"trace_retention_days": None},
        "memory": {"enabled": True, "force_observe_on_close": False},
        "auth": auth or {"enabled": False, "keys": {}, "admin_paths": ["/admin", "/retrain", "/reload", "/config", "/export", "/registry", "/metrics"]},
        "http": {"max_body_bytes": 1048576},
    }


class IntegrationBase(unittest.TestCase):
    """Boots gateway + mocks. Subclasses set auth config via _auth()."""

    @classmethod
    def _auth(cls) -> dict | None:
        return None

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._async_setup())

    async def _async_setup(self):
        from gateway import memory
        memory.close_engine()  # fresh DB per test
        self.tmpdir = tempfile.mkdtemp()
        self.ep_a = MockUpstream("mock_a")
        self.ep_b = MockUpstream("mock_b")
        self.ep_ollama = MockUpstream("mock_ollama")
        self.rev = MockUpstream("mock_reviewer", latency_ms=0)
        await self.ep_a.start()
        await self.ep_b.start()
        await self.ep_ollama.start()
        await self.rev.start()

        self.conf_path = str(Path(self.tmpdir) / "gateway-config.json")
        cfg = _base_config(
            self.tmpdir, ep_a=self.ep_a, ep_b=self.ep_b,
            ep_ollama=self.ep_ollama, reviewer=self.rev, auth=self._auth(),
        )
        Path(self.conf_path).write_text(json.dumps(cfg), encoding="utf-8")

        from gateway import app as app_mod
        self.app = await app_mod.init_app(self.conf_path)
        self.client = TestClient(TestServer(self.app))
        await self.client.start_server()

    def tearDown(self):
        self.loop.run_until_complete(self._async_teardown())
        os.environ.pop("MOCK_FAIL_ENDPOINTS", None)
        os.environ.pop("MOCK_REVIEWER_LABELS", None)
        self.loop.close()
        asyncio.set_event_loop(None)

    async def _async_teardown(self):
        await self.client.close()
        # Client close triggers on_cleanup (workers stopped, pool closed)
        await self.ep_a.stop()
        await self.ep_b.stop()
        await self.ep_ollama.stop()
        await self.rev.stop()

    # --- helpers ---

    def _chat(self, text: str, stream: bool = False, headers: dict | None = None):
        body = {
            "model": "gateway",
            "messages": [{"role": "user", "content": text}],
            "stream": stream,
        }
        return self.client.post("/v1/chat/completions", json=body, headers=headers)

    def _trace(self):
        r = self.loop.run_until_complete(self.client.get("/trace?limit=5"))
        self.assertEqual(r.status, 200)
        return self.loop.run_until_complete(r.json())


class TestChatPipeline(IntegrationBase):
    def test_chat_routes_forwards_and_logs(self):
        r = self.loop.run_until_complete(self._chat("hello there"))
        self.assertEqual(r.status, 200, self.loop.run_until_complete(r.text()))
        data = self.loop.run_until_complete(r.json())
        content = data["choices"][0]["message"]["content"]
        self.assertIn("mock response from", content)
        trace = self._trace()
        self.assertEqual(trace["count"], 1)
        d = trace["decisions"][0]
        self.assertEqual(d["tier"], "tier0")
        self.assertIn("mock_", d["endpoint"])
        self.assertTrue(d["response_ok"])

    def test_streaming_passthrough(self):
        r = self.loop.run_until_complete(self._chat("stream this", stream=True))
        self.assertEqual(r.status, 200)
        text = self.loop.run_until_complete(r.text())
        self.assertIn("data:", text)
        self.assertIn("mock", text)
        trace = self._trace()
        self.assertTrue(trace["decisions"][0]["response_ok"])

    def test_trace_valid_json(self):
        self.loop.run_until_complete(self._chat("first"))
        self.loop.run_until_complete(self._chat("second"))
        trace = self._trace()
        self.assertEqual(trace["count"], 2)
        # JSON round-trip already proven by _trace(); assert structure
        self.assertIn("vertical", trace["decisions"][0])

    def test_fallback_ladder_uses_second_endpoint(self):
        # Make mock_a (cheapest, first candidate) fail hard (env read per-request)
        os.environ["MOCK_FAIL_ENDPOINTS"] = "mock_a"
        r = self.loop.run_until_complete(self._chat("should fall back"))
        self.assertEqual(r.status, 200)
        data = self.loop.run_until_complete(r.json())
        self.assertIn("mock response from mock_b", data["choices"][0]["message"]["content"])
        trace = self._trace()
        d = trace["decisions"][0]
        self.assertEqual(d["endpoint"], "mock_b")
        self.assertTrue(d["fallback_used"])

    def test_all_down_returns_502(self):
        os.environ["MOCK_FAIL_ENDPOINTS"] = "mock_a,mock_b,mock_ollama"
        r = self.loop.run_until_complete(self._chat("will fail"))
        self.assertEqual(r.status, 502)
        trace = self._trace()
        self.assertFalse(trace["decisions"][0]["response_ok"])

    def test_explicit_model_routes_directly_and_models_lists_it(self):
        r = self.loop.run_until_complete(self.client.post("/v1/chat/completions", json={
            "model": "mock-b",
            "messages": [{"role": "user", "content": "direct"}],
        }))
        self.assertEqual(r.status, 200)
        data = self.loop.run_until_complete(r.json())
        self.assertIn("mock_b", data["choices"][0]["message"]["content"])
        models = self.loop.run_until_complete(self.client.get("/v1/models"))
        model_data = self.loop.run_until_complete(models.json())
        self.assertIn("mock-b", {item["id"] for item in model_data["data"]})

    def test_unknown_model_returns_openai_error(self):
        r = self.loop.run_until_complete(self.client.post("/v1/chat/completions", json={
            "model": "does-not-exist",
            "messages": [{"role": "user", "content": "hello"}],
        }))
        self.assertEqual(r.status, 404)
        data = self.loop.run_until_complete(r.json())
        self.assertEqual(data["error"]["code"], "model_not_found")


class TestOllamaNative(IntegrationBase):
    """Ollama native /api/chat must be decoded into OpenAI chat.completions."""

    def test_ollama_native_response_decoded(self):
        r = self.loop.run_until_complete(self.client.post("/v1/chat/completions", json={
            "model": "mock_ollama",
            "messages": [{"role": "user", "content": "native please"}],
        }))
        self.assertEqual(r.status, 200, self.loop.run_until_complete(r.text()))
        data = self.loop.run_until_complete(r.json())
        self.assertEqual(data["choices"][0]["message"]["content"], "ollama native from mock_ollama")
        self.assertEqual(data["usage"]["total_tokens"], 8)
        trace = self._trace()
        self.assertEqual(trace["decisions"][0]["endpoint"], "mock_ollama")

    def test_ollama_native_stream_translated_to_openai_sse(self):
        r = self.loop.run_until_complete(self.client.post("/v1/chat/completions", json={
            "model": "mock_ollama",
            "messages": [{"role": "user", "content": "stream native"}],
            "stream": True,
        }))
        self.assertEqual(r.status, 200)
        body = self.loop.run_until_complete(r.text())
        # Raw Ollama NDJSON has no "data:" framing — translation is proven by it.
        self.assertIn("data: ", body)
        self.assertIn('"object": "chat.completion.chunk"', body)
        self.assertIn('"content": "ollama "', body)
        self.assertIn('"content": "native"', body)
        self.assertTrue(body.rstrip().endswith("data: [DONE]"))


class TestFlywheel(IntegrationBase):
    """Reviewer -> review_result -> curated_samples end-to-end."""

    def test_review_curates_high_agreement(self):
        from gateway import memory, reviewer
        # Drive the reviewer worker manually for determinism
        rw = reviewer.worker()
        rw._sleep_seconds = 0.05
        # Log a decision with known router labels + high confidence
        did = memory.log_decision(
            tenant_id="t1", session_id="s1", model_version="stub-v0", policy_version=1,
            query_hash="h1", query_preview="write a python function",
            vertical="programming", complexity=2,
            flags={"code": True, "math": False, "reasoning": False, "long_output": False},
            tier="tier0", endpoint="mock_a", source="arith",
            ms_classify=1.0, ms_total=10.0, est_cost_usd=0.001,
            escalated=False, fallback_used=False, has_image=False,
            has_injection_signal=False, vertical_top2_prob=0.9,
        )
        # Script the mock reviewer to return labels matching the router
        # (_parse_labels expects a JSON array, one object per prompt)
        labels = {
            "vertical": "programming", "complexity": 2, "code": True,
            "math": False, "reasoning": False, "long_output": False,
        }
        os.environ["MOCK_REVIEWER_LABELS"] = json.dumps([labels])

        reviewer.enqueue_for_review(did, "t1", cost_estimate=0.001)

        # Pump the worker until the review is done
        async def pump():
            for _ in range(50):
                await asyncio.sleep(0.1)
                with memory.engine().connect() as conn:
                    from sqlalchemy import func, select
                    n = conn.execute(select(func.count(memory.curated_samples.c.id))).scalar()
                if n and n >= 1:
                    return n
            return 0

        n = self.loop.run_until_complete(pump())
        self.assertGreaterEqual(n, 1, "flywheel did not curate any samples")
        with memory.engine().connect() as conn:
            from sqlalchemy import select
            row = conn.execute(select(memory.curated_samples).limit(1)).first()
        self.assertEqual(row.vertical, "programming")
        self.assertEqual(row.trust_score, 0.9)


class TestOpsSurface(IntegrationBase):
    """/ready, /metrics, /config redaction, budget propagation."""

    def test_ready_ok_with_live_endpoints(self):
        r = self.loop.run_until_complete(self.client.get("/ready"))
        self.assertEqual(r.status, 200)
        data = self.loop.run_until_complete(r.json())
        self.assertTrue(data["ready"])

    def test_ready_503_when_all_endpoints_down(self):
        # Health poller probes the mocks; make all fail
        os.environ["MOCK_FAIL_ENDPOINTS"] = "mock_a,mock_b,mock_ollama"
        from gateway import app as app_mod

        async def force_poll():
            await app_mod._poll_health(self.app)
        self.loop.run_until_complete(force_poll())
        r = self.loop.run_until_complete(self.client.get("/ready"))
        self.assertEqual(r.status, 503)

    def test_metrics_renders(self):
        r = self.loop.run_until_complete(self.client.get("/metrics"))
        self.assertEqual(r.status, 200)
        body = self.loop.run_until_complete(r.text())
        self.assertIn("glint_uptime_seconds", body)

    def test_config_redacts_auth_keys(self):
        r = self.loop.run_until_complete(self.client.get("/config"))
        self.assertEqual(r.status, 200)
        data = self.loop.run_until_complete(r.json())
        keys = data["config"]["auth"]["keys"]
        self.assertIn("**redacted**", keys)

    def test_budget_update_propagates(self):
        from gateway import memory, tenant
        # Create tenant then bump budget via admin API; state must refresh
        r = self.loop.run_until_complete(self.client.post(
            "/admin/users", json={"tenant_id": "bob", "budget_usd_per_day": 3.0}
        ))
        self.assertEqual(r.status, 200)
        r = self.loop.run_until_complete(self.client.post(
            "/admin/users/bob/budget", json={"budget_usd_per_day": 42.0}
        ))
        self.assertEqual(r.status, 200)
        st = tenant.manager().get_or_create("bob")
        self.assertEqual(st.budget_usd_per_day, 42.0)
        # And DB row reflects the new value
        row = memory.get_or_create_user("bob", {})
        self.assertEqual(row["budget_usd_per_day"], 42.0)


class TestAuth(IntegrationBase):
    @classmethod
    def _auth(cls) -> dict:
        return {
            "enabled": True,
            "keys": {
                "sk-admin-1": {"tenant_id": "admin", "scope": ["admin", "user"]},
                "sk-user-1": {"tenant_id": "alice", "scope": ["user"]},
                "sk-user-2": {"tenant_id": "bob", "scope": ["user"]},
            },
            "admin_paths": ["/admin", "/retrain", "/reload", "/config", "/export", "/registry", "/metrics"],
            "public_paths": ["/", "/health", "/ready", "/dashboard"],
        }

    def test_admin_requires_admin_scope(self):
        r = self.loop.run_until_complete(self.client.get("/admin/users"))
        self.assertEqual(r.status, 401)
        r = self.loop.run_until_complete(
            self.client.get("/admin/users", headers={"Authorization": "Bearer sk-user-1"})
        )
        self.assertEqual(r.status, 401)
        r = self.loop.run_until_complete(
            self.client.get("/admin/users", headers={"Authorization": "Bearer sk-admin-1"})
        )
        self.assertEqual(r.status, 200)

    def test_chat_key_maps_to_tenant(self):
        r = self.loop.run_until_complete(self._chat(
            "hello from alice", headers={"Authorization": "Bearer sk-user-1"},
        ))
        self.assertEqual(r.status, 200)
        trace = self.loop.run_until_complete(self.client.get(
            "/trace", headers={"Authorization": "Bearer sk-user-1"},
        ))
        data = self.loop.run_until_complete(trace.json())
        self.assertEqual(data["decisions"][0]["tenant_id"], "alice")

    def test_user_routes_require_valid_key(self):
        r = self.loop.run_until_complete(self._chat("anonymous is rejected"))
        self.assertEqual(r.status, 401)
        r = self.loop.run_until_complete(self._chat(
            "invalid is rejected", headers={"Authorization": "Bearer invalid"},
        ))
        self.assertEqual(r.status, 401)

    def test_trace_and_feedback_are_tenant_scoped(self):
        self.loop.run_until_complete(self._chat(
            "alice request", headers={"Authorization": "Bearer sk-user-1"},
        ))
        self.loop.run_until_complete(self._chat(
            "bob request", headers={"Authorization": "Bearer sk-user-2"},
        ))
        alice_trace = self.loop.run_until_complete(self.client.get(
            "/trace?tenant_id=bob", headers={"Authorization": "Bearer sk-user-1"},
        ))
        data = self.loop.run_until_complete(alice_trace.json())
        self.assertTrue(data["decisions"])
        self.assertTrue(all(d["tenant_id"] == "alice" for d in data["decisions"]))

        from gateway import memory
        bob_id = memory.get_decisions(tenant_id="bob")[0]["id"]
        forbidden = self.loop.run_until_complete(self.client.post(
            "/feedback",
            json={"decision_id": bob_id, "correct": True},
            headers={"Authorization": "Bearer sk-user-1"},
        ))
        self.assertEqual(forbidden.status, 403)

    def test_memory_resource_is_tenant_scoped(self):
        r = self.loop.run_until_complete(self.client.get(
            "/memory/working/bob", headers={"Authorization": "Bearer sk-user-1"},
        ))
        self.assertEqual(r.status, 403)


if __name__ == "__main__":
    unittest.main()

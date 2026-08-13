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


class TestAnthropicMessagesEndpoint(IntegrationBase):
    """POST /v1/messages -- Anthropic Messages API-compatible entry point.

    Exercises wire_format="anthropic" end-to-end: inbound request
    translation, routing through the exact same pipeline /v1/chat/completions
    uses (mock upstreams, memory, billing), and outbound response/stream
    translation back to Anthropic's shape. tests/test_unit.py's
    TestAnthropicWireFormat covers the translation functions in isolation.
    """

    def _messages(self, text: str, stream: bool = False, **extra):
        body = {
            "model": "gateway",
            "max_tokens": 256,
            "messages": [{"role": "user", "content": text}],
            "stream": stream,
            **extra,
        }
        return self.client.post("/v1/messages", json=body)

    def test_non_streaming_response_is_anthropic_shaped(self):
        r = self.loop.run_until_complete(self._messages("hello there"))
        self.assertEqual(r.status, 200, self.loop.run_until_complete(r.text()))
        data = self.loop.run_until_complete(r.json())
        self.assertEqual(data["type"], "message")
        self.assertEqual(data["role"], "assistant")
        self.assertEqual(data["content"][0]["type"], "text")
        self.assertIn("mock response from", data["content"][0]["text"])
        self.assertEqual(data["stop_reason"], "end_turn")
        self.assertIn("input_tokens", data["usage"])
        self.assertIn("output_tokens", data["usage"])
        self.assertNotIn("choices", data)
        self.assertNotIn("object", data)
        # Routing/memory/billing pipeline ran exactly like the OpenAI path
        trace = self._trace()
        self.assertEqual(trace["count"], 1)
        self.assertTrue(trace["decisions"][0]["response_ok"])

    def test_system_prompt_and_multi_turn_history_translated(self):
        r = self.loop.run_until_complete(self.client.post("/v1/messages", json={
            "model": "gateway",
            "max_tokens": 128,
            "system": "You are terse.",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ack"},
                {"role": "user", "content": "second"},
            ],
        }))
        self.assertEqual(r.status, 200, self.loop.run_until_complete(r.text()))
        data = self.loop.run_until_complete(r.json())
        self.assertEqual(data["type"], "message")

    def test_streaming_response_is_anthropic_sse(self):
        r = self.loop.run_until_complete(self._messages("stream this", stream=True))
        self.assertEqual(r.status, 200)
        body = self.loop.run_until_complete(r.text())
        self.assertIn("event: message_start", body)
        self.assertIn("event: content_block_delta", body)
        self.assertIn('"type": "text_delta"', body)
        self.assertIn("event: message_stop", body)
        self.assertIn('"stop_reason": "end_turn"', body)
        # Proof this is genuinely translated, not the OpenAI SSE passed through
        self.assertNotIn("chat.completion.chunk", body)
        self.assertNotIn("data: [DONE]", body)
        trace = self._trace()
        self.assertTrue(trace["decisions"][0]["response_ok"])

    def test_unknown_model_returns_anthropic_error_shape(self):
        r = self.loop.run_until_complete(self.client.post("/v1/messages", json={
            "model": "does-not-exist",
            "max_tokens": 16,
            "messages": [{"role": "user", "content": "hello"}],
        }))
        self.assertEqual(r.status, 404)
        data = self.loop.run_until_complete(r.json())
        self.assertEqual(data["type"], "error")
        self.assertEqual(data["error"]["type"], "not_found_error")
        self.assertNotIn("code", data["error"])  # OpenAI-only field shouldn't leak through

    def test_missing_messages_returns_anthropic_error_shape(self):
        r = self.loop.run_until_complete(self.client.post("/v1/messages", json={
            "model": "gateway",
            "max_tokens": 16,
            "messages": [],
        }))
        self.assertEqual(r.status, 400)
        data = self.loop.run_until_complete(r.json())
        self.assertEqual(data["type"], "error")
        self.assertEqual(data["error"]["type"], "invalid_request_error")


class TestAdminKeyKind(IntegrationBase):
    """OverlayManager.generate_key()'s `kind` field: informational metadata
    for which wire format/upstream family a key is meant for. Reuses the
    same recommended vocabulary as endpoint `kind` (openai/anthropic/...)
    but must NOT be a hardcoded enum -- any short identifier is valid, so
    tagging a key for a new wire format/provider never needs a gateway code
    change.
    """

    def test_generate_key_defaults_to_openai_kind(self):
        r = self.loop.run_until_complete(self.client.post("/admin/keys", json={"tenant_id": "acme"}))
        self.assertEqual(r.status, 201, self.loop.run_until_complete(r.text()))
        data = self.loop.run_until_complete(r.json())
        self.assertEqual(data["kind"], "openai")
        listed = self.loop.run_until_complete(self.client.get("/admin/keys"))
        keys = self.loop.run_until_complete(listed.json())["keys"]
        self.assertTrue(any(k["tenant_id"] == "acme" and k["kind"] == "openai" for k in keys))

    def test_generate_key_with_anthropic_kind(self):
        r = self.loop.run_until_complete(self.client.post("/admin/keys", json={
            "tenant_id": "acme", "kind": "anthropic",
        }))
        self.assertEqual(r.status, 201)
        data = self.loop.run_until_complete(r.json())
        self.assertEqual(data["kind"], "anthropic")
        self.assertTrue(data["key"].startswith("ctrl-"))

    def test_generate_key_kind_is_not_a_hardcoded_enum(self):
        """The user's own ask: kind must stay generally applicable, not
        restricted to a fixed openai/anthropic pair."""
        r = self.loop.run_until_complete(self.client.post("/admin/keys", json={
            "tenant_id": "acme", "kind": "my-custom-upstream",
        }))
        self.assertEqual(r.status, 201, self.loop.run_until_complete(r.text()))
        data = self.loop.run_until_complete(r.json())
        self.assertEqual(data["kind"], "my-custom-upstream")

    def test_generate_key_normalizes_kind_case(self):
        r = self.loop.run_until_complete(self.client.post("/admin/keys", json={
            "tenant_id": "acme", "kind": "Anthropic",
        }))
        self.assertEqual(r.status, 201)
        data = self.loop.run_until_complete(r.json())
        self.assertEqual(data["kind"], "anthropic")

    def test_generate_key_rejects_invalid_kind(self):
        r = self.loop.run_until_complete(self.client.post("/admin/keys", json={
            "tenant_id": "acme", "kind": "not a valid kind!",
        }))
        self.assertEqual(r.status, 400)


class TestQualityProfileTracking(IntegrationBase):
    """Regression tests for the budget-aware router's quality-profile feed.

    Previously chat_completions never called memory.record_quality_sample()
    after a request settled, so model_quality_profiles stayed empty forever
    and policy.estimate_success_probability() always fell back to the
    no-data prior -- the "budget-aware" routing feature quietly never
    activated on real traffic. These tests exercise every call site added in
    gateway/app.py's chat_completions handler (non-streaming success/failure,
    fallback attribution, and streaming success) and assert the DB is
    actually populated afterward.
    """

    def test_successful_chat_populates_quality_profile(self):
        from gateway import memory
        r = self.loop.run_until_complete(self._chat("populate quality profile"))
        self.assertEqual(r.status, 200, self.loop.run_until_complete(r.text()))
        d = self._trace()["decisions"][0]
        self.assertTrue(d["response_ok"])
        profile = memory.get_quality_profile(d["endpoint"], d["vertical"], d["complexity"])
        self.assertIsNotNone(profile, "successful chat completion should record a quality sample")
        self.assertEqual(profile["total_count"], 1)
        self.assertEqual(profile["success_count"], 1)

    def test_fallback_success_records_quality_against_serving_endpoint(self):
        # mock_a (the cheapest, first-choice endpoint) fails; mock_b serves the
        # request instead. The quality sample must land on mock_b (the endpoint
        # that actually responded), not on mock_a (the originally-chosen one).
        from gateway import memory
        os.environ["MOCK_FAIL_ENDPOINTS"] = "mock_a"
        r = self.loop.run_until_complete(self._chat("should fall back"))
        self.assertEqual(r.status, 200)
        d = self._trace()["decisions"][0]
        self.assertEqual(d["endpoint"], "mock_b")
        served_profile = memory.get_quality_profile("mock_b", d["vertical"], d["complexity"])
        self.assertIsNotNone(served_profile)
        self.assertEqual(served_profile["success_count"], 1)
        failed_profile = memory.get_quality_profile("mock_a", d["vertical"], d["complexity"])
        self.assertIsNone(failed_profile, "the endpoint that never responded should not get a success sample")

    def test_all_endpoints_down_records_failure_sample(self):
        from gateway import memory
        os.environ["MOCK_FAIL_ENDPOINTS"] = "mock_a,mock_b,mock_ollama"
        r = self.loop.run_until_complete(self._chat("will fail"))
        self.assertEqual(r.status, 502)
        d = self._trace()["decisions"][0]
        self.assertFalse(d["response_ok"])
        profile = memory.get_quality_profile(d["endpoint"], d["vertical"], d["complexity"])
        self.assertIsNotNone(profile, "a fully-failed request should still record a failure sample")
        self.assertEqual(profile["total_count"], 1)
        self.assertEqual(profile["success_count"], 0)

    def test_streaming_success_populates_quality_profile(self):
        from gateway import memory
        r = self.loop.run_until_complete(self._chat("stream this", stream=True))
        self.assertEqual(r.status, 200)
        self.loop.run_until_complete(r.text())  # drain the SSE body so the handler completes
        d = self._trace()["decisions"][0]
        self.assertTrue(d["response_ok"])
        profile = memory.get_quality_profile(d["endpoint"], d["vertical"], d["complexity"])
        self.assertIsNotNone(profile, "successful streaming completion should record a quality sample")
        self.assertEqual(profile["success_count"], 1)


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

    def test_persist_failure_marks_review_failed_not_stuck(self):
        """Regression: if persisting a review result raised (e.g. a DB error),
        the queue item was left in whatever state dequeue_review() set it to
        ("in_progress"). requeue_stale_reviews() would flip it back to
        "pending" once its staleness window elapsed, so it would be
        dequeued and retried forever -- repeatedly re-spending reviewer API
        budget on an item that fails the same deterministic way every time.
        It must instead be marked "failed", like every other error path in
        _process_batch, so it fails cleanly once and never returns to the
        queue.
        """
        from unittest.mock import patch

        from sqlalchemy import select

        from gateway import memory, reviewer

        did = memory.log_decision(
            tenant_id="t1", session_id="s1", model_version="stub-v0", policy_version=1,
            query_hash="h2", query_preview="write a python function",
            vertical="programming", complexity=2,
            flags={"code": True, "math": False, "reasoning": False, "long_output": False},
            tier="tier0", endpoint="mock_a", source="arith",
            ms_classify=1.0, ms_total=10.0, est_cost_usd=0.001,
            escalated=False, fallback_used=False, has_image=False,
            has_injection_signal=False, vertical_top2_prob=0.9,
        )
        labels = {
            "vertical": "programming", "complexity": 2, "code": True,
            "math": False, "reasoning": False, "long_output": False,
        }
        os.environ["MOCK_REVIEWER_LABELS"] = json.dumps([labels])
        reviewer.enqueue_for_review(did, "t1", cost_estimate=0.001)

        item = memory.dequeue_review()
        self.assertIsNotNone(item)
        # dequeue_review() returns the pre-update snapshot (still "pending");
        # confirm what it actually persisted was the "in_progress" transition.
        with memory.engine().connect() as conn:
            pre_row = conn.execute(
                select(memory.review_queue).where(memory.review_queue.c.id == item["id"])
            ).first()
        self.assertEqual(pre_row.status, "in_progress")

        rw = reviewer.worker()
        with patch.object(
            memory, "store_review_result",
            side_effect=RuntimeError("simulated persistence failure"),
        ):
            self.loop.run_until_complete(rw._process_batch([item]))

        with memory.engine().connect() as conn:
            row = conn.execute(
                select(memory.review_queue).where(memory.review_queue.c.id == item["id"])
            ).first()
        self.assertIsNotNone(row)
        self.assertEqual(row.status, "failed")
        self.assertNotEqual(row.status, "in_progress")
        self.assertIn("simulated persistence failure", row.last_error or "")


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
        self.assertIn("ctrl_gateway_uptime_seconds", body)

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

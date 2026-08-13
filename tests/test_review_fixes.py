"""Regression tests for code-review fixes (batch 2026-08)."""
import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DB_URL = "sqlite:///:memory:"


def _reset_memory():
    from gateway import memory
    memory.close_engine()
    memory.init_engine(DB_URL)


def _conf():
    from gateway import config as cfg
    return cfg.ConfigManager().current()


class TestRouterStubFixes(unittest.TestCase):
    def test_stub_empty_verticals(self):
        from gateway import router
        rt = router.Router()
        rt.init_stub([])
        out = rt.predict("hi there")
        self.assertEqual(out.vertical, "other")
        self.assertGreaterEqual(out.complexity, 1)

    def test_stub_keyword_vertical(self):
        from gateway import router
        rt = router.Router()
        rt.init_stub(["chat", "programming", "math_advanced"])
        out = rt.predict("write a python function to parse JSON")
        self.assertEqual(out.vertical, "programming")

    def test_stub_predict_empty_text(self):
        from gateway import router
        rt = router.Router()
        rt.init_stub(["chat"])
        out = rt.predict("")
        self.assertIsNotNone(out)
        self.assertEqual(out.complexity, 1)


class TestPolicyDslFixes(unittest.TestCase):
    def setUp(self):
        from gateway import policy
        self.conf = _conf()
        policy.set_current_config(self.conf)
        from gateway import ood
        self._ood = ood.OODResult(is_ood=False, score=0.1, max_prob=0.9,
                                  top_vertical="chat", threshold=0.25)

    def _ctx(self, complexity, flags):
        from gateway import policy
        return policy.RequestContext(
            text="example", has_image=False, flags=flags,
            complexity=complexity, vertical="chat",
            vertical_top2=[("chat", 0.9), ("other", 0.05)],
            ood=self._ood, model_version="v1", policy_version=1,
            session_id="s", tenant_id="t",
            estimated_input_tokens=10, estimated_output_tokens=20,
        )

    def test_override_or_parens(self):
        """router_complexity >= 4 AND (router_flag_code OR router_flag_math)"""
        from gateway import policy
        rule = {"condition": "router_complexity >= 4 AND (router_flag_code OR router_flag_math)"}
        self.assertTrue(policy._evaluate_override(
            rule, self._ctx(4, {"code": True, "math": False})))
        self.assertTrue(policy._evaluate_override(
            rule, self._ctx(5, {"code": False, "math": True})))
        self.assertFalse(policy._evaluate_override(
            rule, self._ctx(4, {"code": False, "math": False})))
        self.assertFalse(policy._evaluate_override(
            rule, self._ctx(2, {"code": True, "math": False})))

    def test_override_nested_not(self):
        from gateway import policy
        rule = {"condition": "router_complexity <= 2 AND NOT router_flag_code AND NOT router_flag_reasoning"}
        self.assertTrue(policy._evaluate_override(
            rule, self._ctx(1, {"code": False, "reasoning": False})))
        self.assertFalse(policy._evaluate_override(
            rule, self._ctx(1, {"code": True, "reasoning": False})))

    def test_override_prototype_match(self):
        from gateway import policy
        rule = {"condition": "prototype_match(name='system_design', threshold=0.85)"}
        ctx = policy.RequestContext(
            text="Design a URL shortener that handles 10K requests per second",
            has_image=False, flags={"code": False, "math": False, "reasoning": True, "long_output": False},
            complexity=4, vertical="system_design",
            vertical_top2=[("system_design", 0.8)],
            ood=self._ood, model_version="v1", policy_version=1,
            session_id="s", tenant_id="t",
            estimated_input_tokens=100, estimated_output_tokens=200,
        )
        self.assertTrue(policy._evaluate_override(rule, ctx))

    def test_override_prototype_or(self):
        from gateway import policy
        rule = {"condition": "prototype_match(kind='structural', name='hard_programming', threshold=0.85) OR prototype_match(kind='structural', name='system_design', threshold=0.85)"}
        ctx = policy.RequestContext(
            text="Solve the traveling salesman problem using dynamic programming",
            has_image=False, flags={"code": True, "math": False, "reasoning": True, "long_output": False},
            complexity=5, vertical="hard_programming",
            vertical_top2=[("hard_programming", 0.7)],
            ood=self._ood, model_version="v1", policy_version=1,
            session_id="s", tenant_id="t",
            estimated_input_tokens=100, estimated_output_tokens=200,
        )
        self.assertTrue(policy._evaluate_override(rule, ctx))


class TestConfigValidationFixes(unittest.TestCase):
    def _write(self, tmp, config, policy="{}", tax="verticals:\n  - name: foo", proto="{\"version\":1,\"prototypes\":[]}"):
        pc = Path(tmp) / "config.json"
        pp = Path(tmp) / "policy.json"
        ty = Path(tmp) / "tax.yaml"
        pr = Path(tmp) / "proto.json"
        pc.write_text(json.dumps(config))
        pp.write_text(policy)
        ty.write_text(tax)
        pr.write_text(proto)
        return pc, pp, ty, pr

    def _base_config(self, endpoints=None, tiers=None, reviewer=None):
        return {
            "mode": "single", "db_url": "sqlite:///:memory:",
            "endpoints": endpoints or [
                {"name": "e1", "kind": "llamacpp", "base_url": "http://localhost:1",
                 "model_alias": "x", "pricing": {}, "concurrency": 1, "breaker": {}}
            ],
            "tiers": tiers or [
                {"name": "tier0", "endpoints": ["e1"], "capability_per_vertical": {"_default": 0.5}}
            ],
            "reviewer": reviewer or {"model": "x", "api_key_env": "TEACHER_API_KEY"},
        }

    def test_unknown_tier_endpoint_rejected(self):
        from gateway import config as cfg
        with tempfile.TemporaryDirectory() as tmp:
            c = self._base_config(tiers=[
                {"name": "tier0", "endpoints": ["does_not_exist"], "capability_per_vertical": {"_default": 0.5}}
            ])
            pc, pp, ty, pr = self._write(tmp, c)
            with self.assertRaises(Exception) as ctx:
                cfg.ConfigManager(config_path=pc, policy_path=pp, taxonomy_path=ty, prototypes_path=pr)
            self.assertIn("unknown endpoint", str(ctx.exception))

    def test_zero_per_request_cap_rejected(self):
        from gateway import config as cfg
        with tempfile.TemporaryDirectory() as tmp:
            c = self._base_config(reviewer={
                "model": "x", "api_key_env": "TEACHER_API_KEY",
                "caps": {"per_request_usd": 0}
            })
            pc, pp, ty, pr = self._write(tmp, c)
            with self.assertRaises(Exception) as ctx:
                cfg.ConfigManager(config_path=pc, policy_path=pp, taxonomy_path=ty, prototypes_path=pr)
            self.assertIn("per_request_usd", str(ctx.exception))

    def test_valid_config_passes(self):
        from gateway import config as cfg
        with tempfile.TemporaryDirectory() as tmp:
            c = self._base_config(reviewer={
                "model": "x", "api_key_env": "TEACHER_API_KEY",
                "caps": {"per_request_usd": 0.10, "per_hour_usd": 2.0}
            })
            pc, pp, ty, pr = self._write(tmp, c)
            mgr = cfg.ConfigManager(config_path=pc, policy_path=pp, taxonomy_path=ty, prototypes_path=pr)
            self.assertIsNotNone(mgr.current())

    def test_generate_data_prompt_formats(self):
        from router_model.generate_data import GEN_SYSTEM_PROMPT
        rendered = GEN_SYSTEM_PROMPT.format(n_per_vertical=3)
        self.assertIn("{verticals:", rendered)
        self.assertIn("{text, complexity", rendered)


class TestAnthropicGeminiTranscoders(unittest.TestCase):
    """Native Anthropic + Gemini transcoder adapters + response decoding."""

    def test_anthropic_encode(self):
        from gateway import transcoder
        ep = {
            "name": "claude", "kind": "anthropic",
            "base_url": "https://api.anthropic.com",
            "model_alias": "claude-sonnet-4-20250514",
            "api_key_env": "ANTHROPIC_API_KEY",
        }
        os.environ["ANTHROPIC_API_KEY"] = "sk-ant-test"
        req = transcoder.transcode(ep, {"max_tokens_bump": 0}, {
            "model": "x",
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ],
            "max_tokens": 100,
        })
        self.assertIn("/v1/messages", req.url)
        self.assertEqual(req.headers["x-api-key"], "sk-ant-test")
        self.assertEqual(req.headers["anthropic-version"], "2023-06-01")
        # system extracted to top-level
        self.assertEqual(req.body["system"], "You are helpful.")
        self.assertEqual(len(req.body["messages"]), 1)
        self.assertEqual(req.body["messages"][0]["role"], "user")
        self.assertEqual(req.body["max_tokens"], 100)
        self.assertIsNotNone(req.response_decoder)

    def test_anthropic_decode(self):
        from gateway import transcoder
        raw = {
            "id": "msg_01",
            "model": "claude-sonnet-4-20250514",
            "content": [{"type": "text", "text": "Hello!"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        decoded = transcoder._decode_anthropic_response(raw)
        self.assertEqual(decoded["object"], "chat.completion")
        self.assertEqual(decoded["choices"][0]["message"]["content"], "Hello!")
        self.assertEqual(decoded["choices"][0]["finish_reason"], "stop")
        self.assertEqual(decoded["usage"]["prompt_tokens"], 10)
        self.assertEqual(decoded["usage"]["completion_tokens"], 5)

    def test_anthropic_tools_round_trip(self):
        from gateway import transcoder
        req = transcoder.transcode({
            "name": "claude", "kind": "anthropic", "base_url": "https://api.anthropic.com",
            "model_alias": "claude", "api_key_env": "ANTHROPIC_API_KEY",
        }, {}, {
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [{"type": "function", "function": {
                "name": "weather", "description": "Weather lookup",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            }}],
        })
        self.assertEqual(req.body["tools"][0]["name"], "weather")
        decoded = transcoder._decode_anthropic_response({
            "content": [{"type": "tool_use", "id": "tool_1", "name": "weather", "input": {"city": "NYC"}}],
            "stop_reason": "tool_use", "usage": {},
        })
        call = decoded["choices"][0]["message"]["tool_calls"][0]
        self.assertEqual(call["function"]["name"], "weather")
        self.assertEqual(decoded["choices"][0]["finish_reason"], "tool_calls")

    def test_gemini_encode(self):
        from gateway import transcoder
        ep = {
            "name": "gemini", "kind": "gemini",
            "base_url": "https://generativelanguage.googleapis.com",
            "model_alias": "gemini-2.0-flash",
            "api_key_env": "GOOGLE_API_KEY",
        }
        os.environ["GOOGLE_API_KEY"] = "AIza-test"
        req = transcoder.transcode(ep, {"max_tokens_bump": 0}, {
            "model": "x",
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "what is 2+2"},
            ],
        })
        self.assertIn("generateContent", req.url)
        self.assertNotIn("key=", req.url)
        self.assertEqual(req.headers["x-goog-api-key"], "AIza-test")
        self.assertIn("models/gemini-2.0-flash", req.url)
        # system instruction extracted
        self.assertEqual(req.body["systemInstruction"]["parts"][0]["text"], "Be brief.")
        self.assertEqual(len(req.body["contents"]), 1)
        self.assertEqual(req.body["contents"][0]["role"], "user")
        self.assertIsNotNone(req.response_decoder)

    def test_gemini_decode(self):
        from gateway import transcoder
        raw = {
            "candidates": [{
                "content": {"parts": [{"text": "4"}], "role": "model"},
                "finishReason": "STOP",
            }],
            "usageMetadata": {"promptTokenCount": 8, "candidatesTokenCount": 1, "totalTokenCount": 9},
        }
        decoded = transcoder._decode_gemini_response(raw)
        self.assertEqual(decoded["object"], "chat.completion")
        self.assertEqual(decoded["choices"][0]["message"]["content"], "4")
        self.assertEqual(decoded["choices"][0]["finish_reason"], "stop")
        self.assertEqual(decoded["usage"]["prompt_tokens"], 8)
        self.assertEqual(decoded["usage"]["completion_tokens"], 1)

    def test_gemini_function_call_decode(self):
        from gateway import transcoder
        decoded = transcoder._decode_gemini_response({
            "candidates": [{
                "content": {"parts": [{"functionCall": {"name": "weather", "args": {"city": "NYC"}}}]},
                "finishReason": "STOP",
            }],
        })
        self.assertEqual(decoded["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "weather")
        self.assertEqual(decoded["choices"][0]["finish_reason"], "tool_calls")

    def test_ollama_decode(self):
        from gateway import transcoder
        decoded = transcoder._decode_ollama_response({
            "model": "llama3", "message": {"role": "assistant", "content": "hello"},
            "done": True, "done_reason": "stop", "prompt_eval_count": 3, "eval_count": 2,
        })
        self.assertEqual(decoded["choices"][0]["message"]["content"], "hello")
        self.assertEqual(decoded["usage"]["total_tokens"], 5)

    def test_native_stream_decoders_emit_openai_sse(self):
        import asyncio

        from gateway import transcoder

        async def collect(decoder, chunks):
            async def source():
                for chunk in chunks:
                    yield chunk
            return b"".join([part async for part in decoder(source())])

        ollama = asyncio.run(collect(transcoder._decode_ollama_stream, [
            b'{"model":"llama3","message":{"content":"hi"},"done":false}\n',
            b'{"model":"llama3","message":{"content":""},"done":true,"done_reason":"stop"}\n',
        ]))
        self.assertIn(b'"object": "chat.completion.chunk"', ollama)
        self.assertTrue(ollama.endswith(b"data: [DONE]\n\n"))

        anthropic = asyncio.run(collect(transcoder._decode_anthropic_stream, [
            b'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","model":"claude"}}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta","delta":{"type":"text_delta","text":"hi"}}\n\n',
        ]))
        self.assertIn(b'"content": "hi"', anthropic)

        gemini = asyncio.run(collect(transcoder._decode_gemini_stream, [
            b'data: {"candidates":[{"content":{"parts":[{"text":"hi"}]},"finishReason":"STOP"}]}\n\n',
        ]))
        self.assertIn(b'"content": "hi"', gemini)

    def test_anthropic_stream_surfaces_real_usage(self):
        """Regression: _decode_anthropic_stream used to read message_start
        and message_delta only for id/model/stop_reason and silently drop
        their `usage` sub-object, so a streamed Anthropic response never
        reported real token counts — callers (including the gateway's own
        settlement code) had no choice but to fall back to pre-request
        estimates. It should now emit a trailing OpenAI-style usage chunk
        (empty `choices`, top-level `usage`) built from the real
        input_tokens (message_start) and output_tokens (message_delta).
        """
        import asyncio
        import json as json_mod

        from gateway import transcoder

        async def collect(decoder, chunks):
            async def source():
                for chunk in chunks:
                    yield chunk
            return b"".join([part async for part in decoder(source())])

        raw = asyncio.run(collect(transcoder._decode_anthropic_stream, [
            b'event: message_start\ndata: {"type":"message_start","message":'
            b'{"id":"msg_1","model":"claude","usage":{"input_tokens":12,"output_tokens":1}}}\n\n',
            b'event: content_block_delta\ndata: {"type":"content_block_delta",'
            b'"delta":{"type":"text_delta","text":"hi"}}\n\n',
            b'event: message_delta\ndata: {"type":"message_delta",'
            b'"delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":7}}\n\n',
        ]))

        self.assertTrue(raw.rstrip(b"\n").endswith(b"data: [DONE]"))
        events = []
        for line in raw.split(b"\n\n"):
            if line.startswith(b"data:"):
                payload = line[len(b"data:"):].strip()
                if payload and payload != b"[DONE]":
                    events.append(json_mod.loads(payload))
        self.assertEqual(len(events), 4)  # role, content, finish, usage

        usage_events = [e for e in events if "usage" in e]
        self.assertEqual(len(usage_events), 1, "exactly one trailing usage chunk expected")
        usage_chunk = usage_events[0]
        self.assertEqual(usage_chunk["choices"], [])
        self.assertEqual(
            usage_chunk["usage"],
            {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19},
        )
        # The usage chunk must be appended after the content delta, not before it.
        content_index = next(
            i for i, e in enumerate(events)
            if (e.get("choices") or [{}])[0].get("delta", {}).get("content") == "hi"
        )
        self.assertLess(content_index, events.index(usage_chunk))

    def test_provider_presets_loaded(self):
        from gateway import config as cfg
        conf = cfg.ConfigManager().current()
        presets = conf.provider_presets.get("presets", {})
        self.assertIn("openai", presets)
        self.assertIn("anthropic", presets)
        self.assertIn("google", presets)
        self.assertIn("deepseek", presets)
        self.assertEqual(presets["anthropic"]["kind"], "anthropic")
        self.assertEqual(presets["google"]["kind"], "gemini")


class TestCompactionPctFix(unittest.TestCase):
    def setUp(self):
        from gateway import config as cfg
        from gateway import memory
        self.conf = cfg.ConfigManager().current()
        _reset_memory()
        from gateway import memory_observational as om
        self.om = om
        # OM tables live in a separate MetaData; create them for the test DB
        om.memory_metadata.create_all(memory.engine())

    def test_compaction_pct_config_respected(self):
        """With compaction_token_threshold_pct=75 and tier max_context=10000,
        threshold = 7500."""
        om_cfg = self.conf.policy.get("memory", {})
        self.assertEqual(om_cfg.get("compaction_token_threshold_pct"), 75)
        ctx = self.om.MemoryContext(
            resource_id="r", thread_id="t",
            recency_messages=[], working_memory_content=None,
            recalled_messages=[], observations=None, reflection=None,
            total_tokens_estimate=8000,
        )
        self.assertTrue(self.om.compaction_required(ctx, self.conf, 10000))
        ctx2 = self.om.MemoryContext(
            resource_id="r", thread_id="t",
            recency_messages=[], working_memory_content=None,
            recalled_messages=[], observations=None, reflection=None,
            total_tokens_estimate=7000,
        )
        self.assertFalse(self.om.compaction_required(ctx2, self.conf, 10000))

    def test_assemble_messages_does_not_mutate(self):
        request_messages = [
            {"role": "user", "content": "hello"},
            {"role": "system", "content": "original system prompt"},
        ]
        original = json.dumps(request_messages)
        ctx = self.om.MemoryContext(
            resource_id="r", thread_id="t",
            recency_messages=[], working_memory_content="## Profile\nName: Bob",
            recalled_messages=[], observations=None, reflection=None,
        )
        self.om.assemble_messages(request_messages, ctx, self.conf)
        self.assertEqual(json.dumps(request_messages), original)

    def test_ensure_working_memory_persists(self):
        self.om.ensure_working_memory("test_resource_xyz")
        content = self.om._load_working_memory("test_resource_xyz")
        self.assertIsNotNone(content)
        self.assertIn("test_resource_xyz", content)


class TestMemorySchemaFixes(unittest.TestCase):
    def setUp(self):
        _reset_memory()

    def test_vertical_top2_prob_column_and_persist(self):
        from gateway import memory
        memory.log_decision(
            tenant_id="t1", session_id="s1", model_version="v1", policy_version=1,
            query_hash="abc", query_preview="hi", vertical="chat", complexity=1,
            flags={}, tier="tier0", endpoint="e", source="arith",
            ms_classify=0.0, ms_total=10.0, est_cost_usd=0.0,
            escalated=False, fallback_used=False, has_image=False,
            has_injection_signal=False, vertical_top2_prob=0.85,
        )
        decisions = memory.get_decisions(limit=10)
        self.assertEqual(decisions[0]["vertical_top2_prob"], 0.85)

    def test_session_stats_portable(self):
        from gateway import memory
        stats = memory.session_stats()
        self.assertIn("total_sessions", stats)
        self.assertIn("active_last_hour", stats)

    def test_get_or_create_session_idempotent(self):
        from gateway import memory
        s1 = memory.get_or_create_session("sess_x", "tenant_x")
        s2 = memory.get_or_create_session("sess_x", "tenant_x")
        self.assertEqual(s1["session_id"], s2["session_id"])

    def test_get_decisions_json_safe(self):
        """Datetime fields must be ISO-serializable (trace endpoint regression)."""
        import json as _json

        from gateway import memory
        memory.log_decision(
            tenant_id="t1", session_id="s1", model_version="v1", policy_version=1,
            query_hash="abc", query_preview="hi", vertical="chat", complexity=1,
            flags={}, tier="tier0", endpoint="e", source="arith",
            ms_classify=0.0, ms_total=10.0, est_cost_usd=0.0,
            escalated=False, fallback_used=False, has_image=False,
            has_injection_signal=False,
        )
        decisions = memory.get_decisions(limit=10)
        # Should serialize without error
        _json.dumps(decisions)

    def test_purge_old_traces(self):
        from gateway import memory
        # Fresh rows should survive a 1-day purge
        memory.log_decision(
            tenant_id="t1", session_id="s1", model_version="v1", policy_version=1,
            query_hash="abc", query_preview="hi", vertical="chat", complexity=1,
            flags={}, tier="tier0", endpoint="e", source="arith",
            ms_classify=0.0, ms_total=10.0, est_cost_usd=0.0,
            escalated=False, fallback_used=False, has_image=False,
            has_injection_signal=False,
        )
        purged = memory.purge_old_traces(days=1)
        self.assertEqual(purged, 0)
        self.assertEqual(len(memory.get_decisions(limit=10)), 1)


class TestCircuitBreaker(unittest.TestCase):
    def setUp(self):
        _reset_memory()

    def test_transitions(self):
        from gateway import circuit
        c = circuit.BreakerConfig(failure_threshold=3, open_duration_seconds=0.05, half_open_max_probes=1)
        b = circuit.CircuitBreaker("ep_test", c)
        self.assertEqual(b.state(), "CLOSED")
        for _ in range(3):
            b.record_failure()
        self.assertEqual(b.state(), "OPEN")
        self.assertFalse(b.allow())
        time.sleep(0.06)
        self.assertTrue(b.allow())  # half-open probe
        b.record_success()
        self.assertEqual(b.state(), "CLOSED")


class TestEventsBus(unittest.TestCase):
    def test_sync_publish_no_crash(self):
        from gateway import events
        bus = events.EventBus()
        received = []
        unsub = bus.subscribe_sync(lambda e: received.append(e))
        bus.publish(events.Event(source=events.EventSource.SYSTEM, type="t1"))
        self.assertEqual(len(received), 1)
        unsub()
        bus.publish(events.Event(source=events.EventSource.SYSTEM, type="t2"))
        self.assertEqual(len(received), 1)

    def test_ring_buffer_recent(self):
        from gateway import events
        bus = events.EventBus(ring_buffer_size=10)
        for i in range(15):
            bus.publish(events.Event(source=events.EventSource.ROUTING, type=f"e{i}"))
        recent = bus.recent(limit=5)
        self.assertEqual(len(recent), 5)
        self.assertEqual(recent[-1].type, "e14")


class TestSwarm(unittest.TestCase):
    def test_should_swarm_off(self):
        from gateway import swarm
        conf = _conf()
        ctx = type("Ctx", (), {"complexity": 5, "flags": {"code": True, "reasoning": True, "long_output": False}})()
        ok, reason = swarm.should_swarm(ctx, conf, "tier4", 0.1)
        self.assertFalse(ok)

    def test_decompose_chunks(self):
        from gateway import swarm
        conf = _conf()
        ctx = type("Ctx", (), {"text": "para one.\n\npara two.\n\npara three."})()
        # decompose() is async (llm_plan mode needs to await a planner call);
        # with llm_plan unset/False and no pool, it resolves straight to the
        # synchronous chunking path — asyncio.run() is enough here, no need
        # for an async test case.
        subtasks = asyncio.run(
            swarm.decompose(ctx, conf, {"chunk_max_chars": 100, "tier_pyramid": ["tier0", "tier2", "tier3"]})
        )
        self.assertGreaterEqual(len(subtasks), 1)

    def test_decompose_llm_plan_falls_back_without_pool(self):
        """llm_plan:true but pool=None must fall back to chunking, not crash."""
        from gateway import swarm
        conf = _conf()
        ctx = type("Ctx", (), {"text": "para one.\n\npara two.\n\npara three."})()
        subtasks = asyncio.run(
            swarm.decompose(
                ctx, conf,
                {"llm_plan": True, "chunk_max_chars": 100, "tier_pyramid": ["tier0", "tier2", "tier3"]},
                None,
            )
        )
        self.assertGreaterEqual(len(subtasks), 1)

    def test_topological_layers_no_deps(self):
        """Subtasks with no depends_on all land in a single layer (matches
        the old fully-parallel behavior for the default chunking path)."""
        from gateway.swarm import SubTask, _topological_layers
        subtasks = [SubTask(id=f"s{i}", prompt="x", target_tier="tier0") for i in range(3)]
        layers = _topological_layers(subtasks)
        self.assertEqual(len(layers), 1)
        self.assertEqual(len(layers[0]), 3)

    def test_topological_layers_respects_dependency(self):
        from gateway.swarm import SubTask, _topological_layers
        subtasks = [
            SubTask(id="a", prompt="x", target_tier="tier0"),
            SubTask(id="b", prompt="x", target_tier="tier0", depends_on=["a"]),
        ]
        layers = _topological_layers(subtasks)
        self.assertEqual([st.id for st in layers[0]], ["a"])
        self.assertEqual([st.id for st in layers[1]], ["b"])

    def test_topological_layers_cycle_falls_back_to_single_layer(self):
        from gateway.swarm import SubTask, _topological_layers
        subtasks = [
            SubTask(id="a", prompt="x", target_tier="tier0", depends_on=["b"]),
            SubTask(id="b", prompt="x", target_tier="tier0", depends_on=["a"]),
        ]
        layers = _topological_layers(subtasks)
        self.assertEqual(len(layers), 1)
        self.assertEqual(len(layers[0]), 2)

    def test_has_cycle_detects_self_referencing_graph(self):
        from gateway.swarm import SubTask, _has_cycle
        subtasks = [
            SubTask(id="a", prompt="x", target_tier="tier0", depends_on=["b"]),
            SubTask(id="b", prompt="x", target_tier="tier0", depends_on=["a"]),
        ]
        self.assertTrue(_has_cycle(subtasks))

    def test_has_cycle_false_for_dag(self):
        from gateway.swarm import SubTask, _has_cycle
        subtasks = [
            SubTask(id="a", prompt="x", target_tier="tier0"),
            SubTask(id="b", prompt="x", target_tier="tier0", depends_on=["a"]),
        ]
        self.assertFalse(_has_cycle(subtasks))


class TestReviewerCaps(unittest.TestCase):
    def test_caps_ok_no_caps(self):
        from gateway import reviewer
        self.assertTrue(reviewer._caps_ok({}))

    def test_per_request_cap_validation_in_config(self):
        from gateway import config as cfg
        with tempfile.TemporaryDirectory() as tmp:
            pc = Path(tmp) / "config.json"
            pp = Path(tmp) / "policy.json"
            ty = Path(tmp) / "tax.yaml"
            pr = Path(tmp) / "proto.json"
            pc.write_text(json.dumps({
                "mode": "single", "db_url": "sqlite:///:memory:",
                "endpoints": [{"name": "e1", "kind": "llamacpp", "base_url": "http://localhost:1",
                               "model_alias": "x", "pricing": {}, "concurrency": 1, "breaker": {}}],
                "tiers": [{"name": "tier0", "endpoints": ["e1"], "capability_per_vertical": {"_default": 0.5}}],
                "reviewer": {"model": "x", "api_key_env": "TEACHER_API_KEY", "caps": {"per_request_usd": -1}},
            }))
            pp.write_text("{}")
            ty.write_text("verticals:\n  - name: foo")
            pr.write_text("{\"version\":1,\"prototypes\":[]}")
            with self.assertRaises(Exception) as ctx:
                cfg.ConfigManager(config_path=pc, policy_path=pp, taxonomy_path=ty, prototypes_path=pr)
            self.assertIn("per_request_usd", str(ctx.exception))


class TestTranslatorIntegration(unittest.TestCase):
    def test_apply_to_payload_inserts_system(self):
        from gateway import translation
        payload = {"messages": [{"role": "user", "content": "hola"}]}
        override = translation.build_translation_rewrite(
            translation.TranslationIntent(is_translation=True, target_language="es"))
        new_payload = translation.apply_to_payload(payload, override)
        self.assertEqual(new_payload["messages"][0]["role"], "system")
        self.assertIn("translate", new_payload["messages"][0]["content"].lower())
        self.assertEqual(len(payload["messages"]), 1)  # original untouched


class TestTenantOpsFixes(unittest.TestCase):
    """Budget/tier edits must take effect (upsert + refresh)."""

    def setUp(self):
        _reset_memory()
        from gateway import tenant
        tenant._manager = None

    def test_get_or_create_user_overwrite_updates_existing(self):
        from gateway import memory
        u = memory.get_or_create_user("t1", {"budget_usd_per_day": 1.0, "tier_access": ["tier0"]})
        self.assertEqual(u["budget_usd_per_day"], 1.0)
        u2 = memory.get_or_create_user("t1", {"budget_usd_per_day": 5.0}, overwrite=True)
        self.assertEqual(u2["budget_usd_per_day"], 5.0)
        # Without overwrite, existing row is untouched
        u3 = memory.get_or_create_user("t1", {"budget_usd_per_day": 9.0})
        self.assertEqual(u3["budget_usd_per_day"], 5.0)

    def test_tenant_manager_refresh_applies_budget(self):
        from gateway import memory, tenant
        mgr = tenant.TenantManager(
            {"tier_access": ["tier0"], "budget_usd_per_day": 1.0,
             "rps_limit": 100, "concurrent_limit": 20, "tokens_per_min": 1000}
        )
        st = mgr.get_or_create("alice")
        self.assertEqual(st.budget_usd_per_day, 1.0)
        memory.get_or_create_user("alice", {"budget_usd_per_day": 7.5}, overwrite=True)
        st2 = mgr.refresh("alice")
        self.assertEqual(st2.budget_usd_per_day, 7.5)

    def test_preconfigured_tenant_override(self):
        from gateway import tenant
        mgr = tenant.TenantManager(
            {"tier_access": ["tier0"], "budget_usd_per_day": 1.0,
             "rps_limit": 100, "concurrent_limit": 20, "tokens_per_min": 1000},
            preconfigured={"bob": {"tier_access": ["tier0", "tier4"], "budget_usd_per_day": 50.0}},
        )
        st = mgr.get_or_create("bob")
        self.assertIn("tier4", st.tier_access)
        self.assertEqual(st.budget_usd_per_day, 50.0)

    def test_usage_reservation_enforces_and_releases_budget(self):
        from gateway import memory
        ok, _ = memory.reserve_usage(
            "alice", budget_limit_usd=0.015, rps_limit=10,
            token_limit_per_minute=1000, estimated_tokens_in=10,
            estimated_tokens_out=10, estimated_cost_usd=0.01,
        )
        self.assertTrue(ok)
        ok, reason = memory.reserve_usage(
            "alice", budget_limit_usd=0.015, rps_limit=10,
            token_limit_per_minute=1000, estimated_tokens_in=10,
            estimated_tokens_out=10, estimated_cost_usd=0.01,
        )
        self.assertFalse(ok)
        self.assertIn("daily budget", reason)
        memory.settle_reserved_usage(
            "alice", reserved_tokens_in=10, reserved_tokens_out=10,
            reserved_cost_usd=0.01, actual_tokens_in=0, actual_tokens_out=0,
            actual_cost_usd=0.0, completed=False,
        )
        ok, _ = memory.reserve_usage(
            "alice", budget_limit_usd=0.015, rps_limit=10,
            token_limit_per_minute=1000, estimated_tokens_in=10,
            estimated_tokens_out=10, estimated_cost_usd=0.01,
        )
        self.assertTrue(ok)


class TestLiveEvalFixes(unittest.TestCase):
    """Live-eval sampling + checkpoint registry."""

    def setUp(self):
        _reset_memory()

    def _seed_reviewed_decision(self, agree: bool = True):
        from gateway import memory
        did = memory.log_decision(
            tenant_id="t", session_id="s", model_version="v1", policy_version=1,
            query_hash="h", query_preview="write a python function",
            vertical="programming", complexity=2,
            flags={"code": True, "math": False, "reasoning": False, "long_output": False},
            tier="tier0", endpoint="e", source="arith", ms_classify=1.0, ms_total=10.0,
            est_cost_usd=0.0, escalated=False, fallback_used=False,
            has_image=False, has_injection_signal=False, vertical_top2_prob=0.9,
        )
        labels = {"vertical": "programming", "complexity": 2, "code": True,
                  "math": False, "reasoning": False, "long_output": False}
        router_labels = labels if agree else {"vertical": "chat", "complexity": 1,
                                              "code": False, "math": False,
                                              "reasoning": False, "long_output": False}
        memory.store_review_result(
            decision_id=did, reviewer_model="rm", reviewer_endpoint="re",
            labels=labels, truncated=False, router_labels=router_labels,
            router_confidence=0.9, cost_usd=0.0, raw_response="{}",
            min_trust_to_curate=0.0,
        )
        return did

    def test_record_live_eval_idempotent(self):
        from gateway import memory
        did = self._seed_reviewed_decision()
        ok1 = memory.record_live_eval(
            decision_id=did, query_hash="h", text="t",
            ground_truth_vertical="programming", ground_truth_complexity=2,
            ground_truth_flags={"code": True, "math": False, "reasoning": False, "long_output": False},
        )
        self.assertTrue(ok1)
        ok2 = memory.record_live_eval(
            decision_id=did, query_hash="h", text="t",
            ground_truth_vertical="programming", ground_truth_complexity=2,
            ground_truth_flags={},
        )
        self.assertFalse(ok2)
        samples = memory.live_eval_samples()
        self.assertEqual(len(samples), 1)

    def test_negative_feedback_curates_reviewer_correction(self):
        from sqlalchemy import select

        from gateway import memory
        did = self._seed_reviewed_decision(agree=False)
        with memory.engine().connect() as conn:
            self.assertIsNone(conn.execute(
                select(memory.curated_samples).where(memory.curated_samples.c.decision_id == did)
            ).first())
        memory.record_feedback(did, correct=False, comment="router was wrong")
        with memory.engine().connect() as conn:
            sample = conn.execute(
                select(memory.curated_samples).where(memory.curated_samples.c.decision_id == did)
            ).first()
        self.assertIsNotNone(sample)
        self.assertEqual(sample.vertical, "programming")
        self.assertEqual(sample.source, "human_reviewed")

    def test_checkpoint_record_and_history(self):
        from gateway import memory
        memory.register_model_version("v-1", None, "emb", "hash1")
        memory.record_checkpoint("v-1", "v-1")
        memory.mark_checkpoint_promoted("v-1", {"base_accuracy": 0.95})
        hist = memory.checkpoint_history()
        self.assertEqual(len(hist), 1)
        self.assertTrue(hist[0]["promoted"])
        self.assertEqual(memory.active_model_version(), "v-1")

    def test_register_existing_version_reactivates_it(self):
        from gateway import memory
        memory.register_model_version("v-1", None, "emb", "hash1")
        memory.register_model_version("v-2", "v-1", "emb", "hash2")
        self.assertEqual(memory.active_model_version(), "v-2")
        memory.register_model_version("v-1", None, "emb", "hash1")
        self.assertEqual(memory.active_model_version(), "v-1")

    def test_trainer_watermark_state(self):
        from gateway import memory
        memory.set_trainer_state("last_curated_id", "42")
        self.assertEqual(memory.get_trainer_state("last_curated_id"), "42")

    def test_sampler_end_to_end(self):
        """TrainerWorker._sample_live_eval must seed live_eval_set + jsonl."""
        import asyncio

        from gateway import trainer_worker as tw
        self._seed_reviewed_decision()
        worker = tw.TrainerWorker(_conf())

        async def run():
            await worker._sample_live_eval()
        asyncio.new_event_loop().run_until_complete(run())

        from gateway import memory
        samples = memory.live_eval_samples()
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["ground_truth_vertical"], "programming")
        out = Path("router_model/data/live-eval/live_eval.jsonl")
        self.assertTrue(out.exists())
        self.assertGreaterEqual(len(out.read_text(encoding="utf-8").strip().splitlines()), 1)

    def test_purge_old_flags(self):
        from gateway import memory
        memory.flag_input("t", None, "injection_signal", "regex", "preview", "logged")
        self.assertEqual(len(memory.list_flagged()), 1)
        self.assertEqual(memory.purge_old_flags(days=1), 0)  # fresh rows survive


class TestChecksumVerification(unittest.TestCase):
    def test_mismatch_refuses_load(self):
        from gateway import router
        rt = router.Router()
        rt.init_stub(["chat"])
        # No real onnx on disk — exercise the path where checksum would be verified
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            onnx_fake = Path(tmp) / "model.onnx"
            onnx_fake.write_bytes(b"fake-onnx-bytes")
            heads_fake = Path(tmp) / "heads.npz"
            heads_fake.write_bytes(b"fake-heads")
            ok = rt.try_load_real(
                onnx_path=str(onnx_fake), heads_path=str(heads_fake),
                vertical_names=["chat"], checksum_sha256="deadbeef" * 16,
            )
            self.assertFalse(ok)  # checksum mismatch -> refuse
            self.assertTrue(rt.is_stub())


class TestComplexityDecode(unittest.TestCase):
    """Gateway complexity decode must match eval.py (threshold count, not argmax)."""

    def _model(self):
        from gateway import router
        return router._RealModel(
            embedding_session=None,
            heads_weights={},
            heads_metadata={"version": "test"},
            vertical_names=["chat", "math"],
        )

    def test_threshold_count_matches_eval(self):
        import numpy as np
        heads_out = {
            "vertical_probs": np.array([0.7, 0.3]),
            "complexity_probs": np.array([0.99, 0.9, 0.4, 0.1, 0.0]),
            "code_prob": 0.1, "math_prob": 0.2,
            "reasoning_prob": 0.3, "long_output_prob": 0.4,
            "projection": None,
        }
        out = self._model()._build_output("test text", "hash1", heads_out)
        # heads > 0.5: [0.99, 0.9] -> 2. (argmax would have returned 1.)
        self.assertEqual(out.complexity, 2)

    def test_complexity_clamped_high(self):
        import numpy as np
        heads_out = {
            "vertical_probs": np.array([0.7, 0.3]),
            "complexity_probs": np.array([0.99, 0.99, 0.99, 0.99, 0.99]),
            "code_prob": 0.1, "math_prob": 0.2,
            "reasoning_prob": 0.3, "long_output_prob": 0.4,
            "projection": None,
        }
        out = self._model()._build_output("t", "h2", heads_out)
        self.assertEqual(out.complexity, 5)

    def test_complexity_clamped_low(self):
        import numpy as np
        heads_out = {
            "vertical_probs": np.array([0.7, 0.3]),
            "complexity_probs": np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
            "code_prob": 0.1, "math_prob": 0.2,
            "reasoning_prob": 0.3, "long_output_prob": 0.4,
            "projection": None,
        }
        out = self._model()._build_output("t", "h3", heads_out)
        self.assertEqual(out.complexity, 1)


class TestReviewerCapsTenantFilter(unittest.TestCase):
    """Reviewer caps must count ONLY __reviewer__ spend, not user traffic."""

    def setUp(self):
        _reset_memory()

    def test_user_spend_does_not_count_toward_caps(self):
        from gateway import memory, reviewer
        memory.record_usage("user_a", tokens_in=1000, tokens_out=100, cost_usd=5.0)
        memory.record_usage("__reviewer__", tokens_in=100, tokens_out=50, cost_usd=1.0)
        caps = {"per_hour_usd": 2.0}
        # Reviewer spent 1.0 < 2.0 -> caps OK (True). Without the tenant filter
        # the total would be 6.0 >= 2.0 -> False.
        self.assertTrue(reviewer._caps_ok(caps))

    def test_reviewer_spend_over_cap_blocks(self):
        from gateway import memory, reviewer
        memory.record_usage("__reviewer__", tokens_in=100, tokens_out=50, cost_usd=3.0)
        self.assertFalse(reviewer._caps_ok({"per_hour_usd": 2.0}))


class TestRequeueStaleReviews(unittest.TestCase):
    def test_requeue_stale_in_progress(self):
        from datetime import UTC, datetime, timedelta

        from sqlalchemy import update

        from gateway import memory
        _reset_memory()
        did = memory.log_decision(
            tenant_id="t", session_id="s", model_version="v1", policy_version=1,
            query_hash="h", query_preview="x", vertical="chat", complexity=1,
            flags={}, tier="tier0", endpoint="e", source="arith",
            ms_classify=0.0, ms_total=1.0, est_cost_usd=0.0,
            escalated=False, fallback_used=False, has_image=False,
            has_injection_signal=False,
        )
        memory.enqueue_review(did, "t")
        item = memory.dequeue_review()
        self.assertIsNotNone(item)
        # Age the row so it looks stale
        with memory.engine().begin() as conn:
            conn.execute(
                update(memory.review_queue)
                .where(memory.review_queue.c.id == item["id"])
                .values(started_at=datetime.now(UTC) - timedelta(hours=1))
            )
        requeued = memory.requeue_stale_reviews(stale_after_seconds=300)
        self.assertEqual(requeued, 1)
        # And it's dequeuable again
        item2 = memory.dequeue_review()
        self.assertIsNotNone(item2)
        self.assertEqual(item2["id"], item["id"])


class TestEventsTenantFilter(unittest.TestCase):
    def test_recent_filters_by_tenant(self):
        from gateway import events
        bus = events.EventBus()
        bus.publish(events.Event(source=events.EventSource.ROUTING, type="a", tenant_id="alice"))
        bus.publish(events.Event(source=events.EventSource.ROUTING, type="b", tenant_id="bob"))
        bus.publish(events.Event(source=events.EventSource.ROUTING, type="c", tenant_id=None))
        alice = bus.recent(limit=10, tenant_id="alice")
        self.assertEqual([e.type for e in alice], ["a", "c"])  # own + tenantless
        bob = bus.recent(limit=10, tenant_id="bob")
        self.assertEqual([e.type for e in bob], ["b", "c"])
        all_ev = bus.recent(limit=10)
        self.assertEqual(len(all_ev), 3)


class TestHashedAuthKeys(unittest.TestCase):
    def test_hashed_key_resolves_without_storing_plaintext(self):
        import hashlib

        from gateway.auth import AuthManager
        raw = "ctrl-secret"
        digest = "sha256:" + hashlib.sha256(raw.encode()).hexdigest()
        manager = AuthManager({
            "enabled": True,
            "keys": {digest: {"tenant_id": "alice", "scope": ["user"]}},
        })
        self.assertEqual(manager.resolve(raw)["tenant_id"], "alice")
        self.assertIsNone(manager.resolve("wrong"))


if __name__ == "__main__":
    unittest.main()

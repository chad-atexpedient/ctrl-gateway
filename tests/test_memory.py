"""Tests for observational memory + event bus + memory tiers."""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_async(coro):
    """Helper: run a coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestEventBus(unittest.TestCase):
    def test_emit_and_recent(self):
        # Reset the bus singleton for clean state
        import gateway.events as ev_mod
        from gateway import events
        ev_mod._bus = events.EventBus()
        b = events.bus()

        events.emit(
            events.EventSource.ROUTING,
            "test_event",
            {"foo": "bar"},
            tenant_id="t1",
        )
        recent = b.recent(limit=10)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].type, "test_event")
        self.assertEqual(recent[0].data, {"foo": "bar"})
        self.assertEqual(recent[0].tenant_id, "t1")

    def test_emit_status(self):
        import gateway.events as ev_mod
        from gateway import events
        ev_mod._bus = events.EventBus()
        b = events.bus()
        events.emit_status(
            events.EventSource.OBSERVER,
            "Compressing...",
            done=False,
            tenant_id="t1",
        )
        recent = b.recent()
        self.assertEqual(recent[0].type, "status")
        self.assertEqual(recent[0].severity, events.EventSeverity.STATUS)
        self.assertEqual(recent[0].data["description"], "Compressing...")
        self.assertFalse(recent[0].data["done"])

    def test_filter_by_source(self):
        import gateway.events as ev_mod
        from gateway import events
        ev_mod._bus = events.EventBus()
        b = events.bus()
        events.emit(events.EventSource.ROUTING, "r1")
        events.emit(events.EventSource.OBSERVER, "o1")
        events.emit(events.EventSource.OBSERVER, "o2")
        routing = b.recent(source=events.EventSource.ROUTING)
        self.assertEqual(len(routing), 1)
        observer = b.recent(source=events.EventSource.OBSERVER)
        self.assertEqual(len(observer), 2)

    def test_subscribe_async(self):
        import gateway.events as ev_mod
        from gateway import events
        ev_mod._bus = events.EventBus()

        async def runner():
            bus = events.bus()
            q, unsub = await bus.subscribe(max_queue=10)
            events.emit(events.EventSource.ROUTING, "sub_test")
            # Wait briefly for event to land
            await asyncio.sleep(0.01)
            self.assertFalse(q.empty())
            ev = q.get_nowait()
            self.assertEqual(ev.type, "sub_test")
            unsub()

        run_async(runner())

    def test_subscribe_sync_callback(self):
        import gateway.events as ev_mod
        from gateway import events
        ev_mod._bus = events.EventBus()

        captured = []
        def cb(ev):
            captured.append(ev)

        unsub = events.bus().subscribe_sync(cb)
        events.emit(events.EventSource.TRAINER, "sync_test", {"x": 1})
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0].data, {"x": 1})
        unsub()


class TestWorkingMemory(unittest.TestCase):
    def setUp(self):
        from pathlib import Path

        from gateway import memory
        from gateway import memory_observational as om
        tmpdir = tempfile.mkdtemp()
        self._db_path = str(Path(tmpdir) / "test_om.db")
        memory.close_engine()
        memory.init_engine(f"sqlite:///{self._db_path}")
        om.memory_metadata.create_all(memory.engine())

    def test_ensure_working_memory(self):
        from gateway import memory_observational as om
        om.ensure_working_memory("resource-1")
        wm = om._load_working_memory("resource-1")
        self.assertIsNotNone(wm)
        self.assertIn("Customer Profile", wm)
        self.assertIn("resource-1", wm)

    def test_update_working_memory(self):
        from gateway import memory_observational as om
        om.ensure_working_memory("resource-2")
        new_content = "# Customer Profile (resource-2)\n\nName: Alice\nTier: enterprise\n"
        om.update_working_memory("resource-2", new_content, source="test")
        wm = om._load_working_memory("resource-2")
        self.assertEqual(wm, new_content)

    def test_record_message_and_recency(self):
        from gateway import memory_observational as om
        om.record_message(
            resource_id="r3", thread_id="t1",
            role="user", content="hello there",
            token_estimate=2,
        )
        om.record_message(
            resource_id="r3", thread_id="t1",
            role="assistant", content="hi!",
            token_estimate=1,
        )
        recency = om._load_recency("r3", "t1", limit=10)
        self.assertEqual(len(recency), 2)
        self.assertEqual(recency[0]["role"], "user")
        self.assertEqual(recency[1]["role"], "assistant")

    def test_recency_is_resource_scoped(self):
        from gateway import memory_observational as om
        om.record_message(resource_id="alice", thread_id="shared", role="user", content="alice secret")
        om.record_message(resource_id="bob", thread_id="shared", role="user", content="bob secret")
        alice = om._load_recency("alice", "shared", limit=10)
        self.assertEqual([m["content"] for m in alice], ["alice secret"])


class TestMemoryContext(unittest.TestCase):
    def setUp(self):
        from gateway import config as cfg
        self.conf = cfg.ConfigManager().current()
        from pathlib import Path

        from gateway import memory
        tmpdir = tempfile.mkdtemp()
        self._db_path = str(Path(tmpdir) / "test_ctx.db")
        memory.close_engine()
        memory.init_engine(f"sqlite:///{self._db_path}")
        from gateway import memory_observational as om
        om.memory_metadata.create_all(memory.engine())

    def test_load_empty(self):
        from gateway import memory_observational as om
        ctx = om.load_memory_context(
            conf=self.conf,
            resource_id="nonexistent",
            thread_id="t1",
        )
        self.assertEqual(ctx.resource_id, "nonexistent")
        self.assertEqual(len(ctx.recency_messages), 0)
        self.assertIsNone(ctx.working_memory_content)
        self.assertEqual(ctx.total_tokens_estimate, 0)

    def test_load_with_history(self):
        from gateway import memory_observational as om
        # Seed some messages
        for i in range(5):
            om.record_message(
                resource_id="r1", thread_id="t1",
                role="user" if i % 2 == 0 else "assistant",
                content=f"message {i}" * (i + 1),  # growing content
                token_estimate=(i + 1) * 5,
            )
        ctx = om.load_memory_context(
            conf=self.conf,
            resource_id="r1",
            thread_id="t1",
            last_messages_count=3,
        )
        self.assertEqual(len(ctx.recency_messages), 3)
        self.assertGreater(ctx.total_tokens_estimate, 0)


class TestCompaction(unittest.TestCase):
    def setUp(self):
        from gateway import config as cfg
        self.conf = cfg.ConfigManager().current()

    def test_no_compaction_within_budget(self):
        from gateway import memory_observational as om
        ctx = om.MemoryContext(
            resource_id="r", thread_id="t",
            recency_messages=[{"content": "hi"}] * 5,
            working_memory_content=None,
            recalled_messages=[],
            observations=None,
            reflection=None,
            total_tokens_estimate=100,
        )
        self.assertFalse(om.compaction_required(ctx, self.conf, 32768))

    def test_compaction_required_when_over_budget(self):
        from gateway import memory_observational as om
        ctx = om.MemoryContext(
            resource_id="r", thread_id="t",
            recency_messages=[],
            working_memory_content=None,
            recalled_messages=[],
            observations="x" * 30000,
            reflection=None,
            total_tokens_estimate=30000,
        )
        self.assertTrue(om.compaction_required(ctx, self.conf, 32768))

    def test_redirect_to_higher_tier(self):
        from gateway import memory_observational as om
        # Build a context that exceeds tier2 (64K) but fits tier3 (131K)
        ctx = om.MemoryContext(
            resource_id="r", thread_id="t",
            recency_messages=[],
            working_memory_content=None,
            recalled_messages=[],
            observations="x" * 80000,  # 80K tokens
            reflection=None,
            total_tokens_estimate=80000,
        )
        redirect, target, reason = om.should_redirect_for_compaction(ctx, self.conf, "tier2")
        self.assertTrue(redirect)
        # target should be tier3 or tier4 (with larger max_context)
        self.assertIn(target, ["tier3", "tier4"])


class TestAssembleMessages(unittest.TestCase):
    def setUp(self):
        from gateway import config as cfg
        self.conf = cfg.ConfigManager().current()

    def test_assemble_with_recency(self):
        from gateway import memory_observational as om
        ctx = om.MemoryContext(
            resource_id="r", thread_id="t",
            recency_messages=[
                {"role": "user", "content": "earlier question"},
                {"role": "assistant", "content": "earlier answer"},
            ],
            working_memory_content=None,
            recalled_messages=[],
            observations=None,
            reflection=None,
            total_tokens_estimate=10,
        )
        req_messages = [{"role": "user", "content": "new question"}]
        out = om.assemble_messages(req_messages, ctx, self.conf)
        # recency + user request
        self.assertGreaterEqual(len(out), 3)
        # user's content should be present
        self.assertTrue(any(m["content"] == "new question" for m in out))

    def test_assemble_with_working_memory(self):
        from gateway import memory_observational as om
        ctx = om.MemoryContext(
            resource_id="r", thread_id="t",
            recency_messages=[],
            working_memory_content="Name: Bob\nTier: pro",
            recalled_messages=[],
            observations=None,
            reflection=None,
            total_tokens_estimate=5,
        )
        req_messages = [{"role": "user", "content": "hi"}]
        out = om.assemble_messages(req_messages, ctx, self.conf)
        # Find the system message and check it contains working memory
        sys_msgs = [m for m in out if m["role"] == "system"]
        self.assertEqual(len(sys_msgs), 1)
        self.assertIn("Working Memory", sys_msgs[0]["content"])
        self.assertIn("Bob", sys_msgs[0]["content"])

    def test_dedup_recent_with_request(self):
        """Messages already in the request shouldn't be duplicated from recency."""
        from gateway import memory_observational as om
        ctx = om.MemoryContext(
            resource_id="r", thread_id="t",
            recency_messages=[
                {"role": "user", "content": "what is 2+2"},
                {"role": "assistant", "content": "4"},
            ],
            working_memory_content=None,
            recalled_messages=[],
            observations=None,
            reflection=None,
            total_tokens_estimate=10,
        )
        # Request repeats the assistant message
        req_messages = [
            {"role": "user", "content": "what is 2+2"},
            {"role": "assistant", "content": "4"},
        ]
        out = om.assemble_messages(req_messages, ctx, self.conf)
        # Count occurrences of the duplicated text
        text_blob = json.dumps([m["content"] for m in out])
        # The user message should appear only once
        self.assertEqual(text_blob.count("what is 2+2"), 1)


if __name__ == "__main__":
    unittest.main()

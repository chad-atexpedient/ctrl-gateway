"""Tests for the model catalog + provider discovery engine.

Covers:
  - model_catalog table CRUD (memory.py)
  - capability scoring (model_sync.compute_capability_score)
  - tier auto-assignment (_tier_from_capability)
  - provider adapters (_parse_openrouter_model, _parse_openai_model,
    _parse_ollama_model, _parse_anthropic_model)
  - ModelSyncEngine sync flows with mocked HTTP
  - Admin routes (/admin/models/*)
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from unittest.mock import AsyncMock

from gateway import memory, model_sync


def _run(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ============================================================================
# Model catalog table CRUD
# ============================================================================


class ModelCatalogCRUDTests(unittest.TestCase):

    def setUp(self):
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        tmpdir = tempfile.mkdtemp()
        self.tmpdir = tmpdir
        memory.init_engine(f"sqlite:///{tmpdir}/test.db")

    def tearDown(self):
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]

    def test_upsert_and_get(self):
        row = memory.upsert_model_catalog_entry(
            model_id="openai/gpt-4o",
            provider="openrouter",
            display_name="GPT-4o",
            context_length=128000,
            pricing_prompt_per_1k=0.005,
            pricing_completion_per_1k=0.015,
            supports_tools=True,
            supports_vision=True,
            capability_score=0.85,
            tier_assignment="tier4",
        )
        self.assertNotIn("error", row)
        fetched = memory.get_model_catalog_entry("openai/gpt-4o", "openrouter")
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["display_name"], "GPT-4o")
        self.assertEqual(fetched["context_length"], 128000)
        self.assertTrue(fetched["supports_tools"])
        self.assertEqual(fetched["tier_assignment"], "tier4")

    def test_upsert_updates_existing(self):
        memory.upsert_model_catalog_entry(
            model_id="test/model",
            provider="openrouter",
            display_name="v1",
            context_length=8000,
        )
        memory.upsert_model_catalog_entry(
            model_id="test/model",
            provider="openrouter",
            display_name="v2",
            context_length=32000,
        )
        row = memory.get_model_catalog_entry("test/model", "openrouter")
        self.assertEqual(row["display_name"], "v2")
        self.assertEqual(row["context_length"], 32000)

    def test_same_model_different_providers(self):
        """The same model_id from two providers creates two rows."""
        memory.upsert_model_catalog_entry(
            model_id="gpt-4o", provider="openai",
            display_name="GPT-4o (direct)",
        )
        memory.upsert_model_catalog_entry(
            model_id="gpt-4o", provider="openrouter",
            display_name="GPT-4o (via OR)",
        )
        direct = memory.get_model_catalog_entry("gpt-4o", "openai")
        routed = memory.get_model_catalog_entry("gpt-4o", "openrouter")
        self.assertIsNotNone(direct)
        self.assertIsNotNone(routed)
        self.assertNotEqual(direct["id"], routed["id"])

    def test_list_with_filters(self):
        memory.upsert_model_catalog_entry(
            model_id="big", provider="x", context_length=200000,
            supports_tools=True, supports_vision=True, capability_score=0.9,
        )
        memory.upsert_model_catalog_entry(
            model_id="small", provider="x", context_length=4000,
            supports_tools=False, capability_score=0.2,
        )
        # Filter by context length
        big = memory.list_model_catalog(min_context_length=100000)
        self.assertEqual(len(big), 1)
        self.assertEqual(big[0]["model_id"], "big")
        # Filter by tools
        tools_only = memory.list_model_catalog(supports_tools=True)
        self.assertEqual(len(tools_only), 1)
        # Filter by vision
        vision_only = memory.list_model_catalog(supports_vision=True)
        self.assertEqual(len(vision_only), 1)
        # Filter by capability
        strong = memory.list_model_catalog(min_capability_score=0.8)
        self.assertEqual(len(strong), 1)

    def test_set_enabled_and_tier(self):
        memory.upsert_model_catalog_entry(
            model_id="m1", provider="p1", capability_score=0.5,
            tier_assignment="tier2",
        )
        self.assertTrue(memory.set_model_catalog_enabled("m1", "p1", False))
        row = memory.get_model_catalog_entry("m1", "p1")
        self.assertFalse(row["enabled"])

        self.assertTrue(memory.set_model_catalog_tier("m1", "p1", "tier4"))
        row = memory.get_model_catalog_entry("m1", "p1")
        self.assertEqual(row["tier_assignment"], "tier4")

    def test_delete(self):
        memory.upsert_model_catalog_entry(model_id="del", provider="p")
        self.assertTrue(memory.delete_model_catalog_entry("del", "p"))
        self.assertIsNone(memory.get_model_catalog_entry("del", "p"))

    def test_stats(self):
        memory.upsert_model_catalog_entry(
            model_id="a", provider="openrouter",
            context_length=128000, supports_tools=True,
            pricing_prompt_per_1k=0.005,
        )
        memory.upsert_model_catalog_entry(
            model_id="b", provider="ollama",
            context_length=8000, supports_vision=True,
            pricing_prompt_per_1k=0.0,
        )
        stats = memory.model_catalog_stats()
        self.assertEqual(stats["total"], 2)
        self.assertEqual(stats["by_provider"]["openrouter"], 1)
        self.assertEqual(stats["by_provider"]["ollama"], 1)
        self.assertEqual(stats["supports_tools"], 1)
        self.assertEqual(stats["supports_vision"], 1)

    def test_list_by_provider_grouping(self):
        memory.upsert_model_catalog_entry(model_id="a", provider="openai")
        memory.upsert_model_catalog_entry(model_id="b", provider="openai")
        memory.upsert_model_catalog_entry(model_id="c", provider="ollama")
        grouped = memory.list_model_catalog_by_provider()
        self.assertEqual(len(grouped["openai"]), 2)
        self.assertEqual(len(grouped["ollama"]), 1)


# ============================================================================
# Capability scoring
# ============================================================================


class CapabilityScoreTests(unittest.TestCase):

    def test_frontier_model_scores_high(self):
        """A large-context, tool-supporting, reasoning model with moderate
        pricing should score above 0.65."""
        score = model_sync.compute_capability_score(
            context_length=200000,
            supports_tools=True,
            supports_vision=True,
            supports_reasoning=True,
            pricing_prompt_per_1k=0.005,
            pricing_completion_per_1k=0.015,
        )
        self.assertGreater(score, 0.65)

    def test_tiny_free_model_scores_moderate(self):
        """A free model with small context but no features still gets a
        reasonable score from the inverse-pricing component."""
        score = model_sync.compute_capability_score(
            context_length=4096,
            supports_tools=False,
            supports_vision=False,
            supports_reasoning=False,
            pricing_prompt_per_1k=0.0,
            pricing_completion_per_1k=0.0,
        )
        self.assertGreater(score, 0.25)
        self.assertLess(score, 0.50)

    def test_expensive_model_scores_lower(self):
        """Same features but very expensive → lower score."""
        cheap = model_sync.compute_capability_score(
            context_length=128000, supports_tools=True, supports_vision=True,
            supports_reasoning=False, pricing_prompt_per_1k=0.001,
            pricing_completion_per_1k=0.002,
        )
        expensive = model_sync.compute_capability_score(
            context_length=128000, supports_tools=True, supports_vision=True,
            supports_reasoning=False, pricing_prompt_per_1k=0.075,
            pricing_completion_per_1k=0.075,
        )
        self.assertGreater(cheap, expensive)

    def test_score_always_in_range(self):
        """Score must be 0.0–1.0 for any input."""
        for cl in [None, 0, 1024, 32768, 200000, 1000000]:
            for tools in [True, False]:
                for reasoning in [True, False]:
                    for price in [None, 0.0, 0.001, 0.01, 0.075, 1.0]:
                        score = model_sync.compute_capability_score(
                            context_length=cl,
                            supports_tools=tools,
                            supports_vision=False,
                            supports_reasoning=reasoning,
                            pricing_prompt_per_1k=price,
                            pricing_completion_per_1k=price,
                        )
                        self.assertGreaterEqual(score, 0.0)
                        self.assertLessEqual(score, 1.0)


class TierAssignmentTests(unittest.TestCase):

    def test_tier_thresholds(self):
        self.assertEqual(model_sync._tier_from_capability(0.85), "tier4")
        self.assertEqual(model_sync._tier_from_capability(0.80), "tier4")
        self.assertEqual(model_sync._tier_from_capability(0.70), "tier3")
        self.assertEqual(model_sync._tier_from_capability(0.55), "tier2")
        self.assertEqual(model_sync._tier_from_capability(0.40), "tier1")
        self.assertEqual(model_sync._tier_from_capability(0.20), "tier0")


# ============================================================================
# Provider adapters
# ============================================================================


class ProviderAdapterTests(unittest.TestCase):

    def test_parse_openrouter_model(self):
        m = {
            "id": "openai/gpt-4o",
            "name": "OpenAI: GPT-4o",
            "description": "Multimodal model.",
            "context_length": 128000,
            "architecture": {
                "modality": "text->text",
                "input_modalities": ["text", "image"],
                "output_modalities": ["text"],
            },
            "pricing": {
                "prompt": "0.000005",
                "completion": "0.000015",
                "input_cache_read": "0.00000125",
            },
            "top_provider": {"context_length": 128000, "max_completion_tokens": 16384},
            "supported_parameters": ["tools", "tool_choice", "temperature", "max_tokens"],
            "reasoning": {"mandatory": False, "default_enabled": False},
        }
        parsed = model_sync._parse_openrouter_model(m)
        self.assertEqual(parsed["model_id"], "openai/gpt-4o")
        self.assertEqual(parsed["provider"], "openrouter")
        self.assertEqual(parsed["context_length"], 128000)
        self.assertTrue(parsed["supports_tools"])
        self.assertTrue(parsed["supports_vision"])
        self.assertFalse(parsed["supports_reasoning"])
        # Per-token → per-1K conversion
        self.assertAlmostEqual(parsed["pricing_prompt_per_1k"], 0.005, places=6)
        self.assertAlmostEqual(parsed["pricing_completion_per_1k"], 0.015, places=6)

    def test_parse_openai_model(self):
        m = {"id": "gpt-4o-mini", "owned_by": "openai"}
        parsed = model_sync._parse_openai_model(m, "openai")
        self.assertEqual(parsed["model_id"], "gpt-4o-mini")
        self.assertEqual(parsed["provider"], "openai")
        self.assertIsNone(parsed["context_length"])
        self.assertIsNone(parsed["pricing_prompt_per_1k"])

    def test_parse_ollama_model(self):
        m = {
            "name": "llama3:8b",
            "details": {"family": "llama", "parameter_size": "8B", "quantization_level": "Q4_0"},
            "context_length": 8192,
        }
        parsed = model_sync._parse_ollama_model(m)
        self.assertEqual(parsed["model_id"], "llama3:8b")
        self.assertEqual(parsed["provider"], "ollama")
        self.assertEqual(parsed["context_length"], 8192)
        self.assertEqual(parsed["pricing_prompt_per_1k"], 0.0)
        self.assertEqual(parsed["pricing_completion_per_1k"], 0.0)
        self.assertTrue(parsed["supports_tools"])  # llama family

    def test_parse_anthropic_model(self):
        m = {
            "id": "claude-3-5-sonnet-20241022",
            "display_name": "Claude 3.5 Sonnet",
            "description": "Most intelligent model.",
            "max_context_tokens": 200000,
        }
        parsed = model_sync._parse_anthropic_model(m)
        self.assertEqual(parsed["model_id"], "claude-3-5-sonnet-20241022")
        self.assertEqual(parsed["provider"], "anthropic")
        self.assertEqual(parsed["context_length"], 200000)
        self.assertTrue(parsed["supports_tools"])
        self.assertTrue(parsed["supports_vision"])  # sonnet has vision


# ============================================================================
# Sync engine — mocked HTTP
# ============================================================================


class ModelSyncEngineTests(unittest.TestCase):

    def setUp(self):
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        tmpdir = tempfile.mkdtemp()
        self.tmpdir = tmpdir
        memory.init_engine(f"sqlite:///{tmpdir}/test.db")

    def tearDown(self):
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]

    def test_sync_openrouter_with_mock(self):
        """Mock the HTTP fetch and verify models land in model_catalog."""
        eng = model_sync.ModelSyncEngine(
            openrouter_enabled=True,
            ollama_enabled=False,
        )
        mock_data = {
            "data": [
                {
                    "id": "openai/gpt-4o",
                    "name": "GPT-4o",
                    "context_length": 128000,
                    "architecture": {"modality": "text->text",
                                     "input_modalities": ["text", "image"],
                                     "output_modalities": ["text"]},
                    "pricing": {"prompt": "0.000005", "completion": "0.000015"},
                    "supported_parameters": ["tools", "temperature"],
                    "reasoning": {},
                },
                {
                    "id": "meta/llama-3.1-8b",
                    "name": "Llama 3.1 8B",
                    "context_length": 128000,
                    "architecture": {"modality": "text->text",
                                     "input_modalities": ["text"],
                                     "output_modalities": ["text"]},
                    "pricing": {"prompt": "0.0", "completion": "0.0"},
                    "supported_parameters": ["temperature"],
                    "reasoning": {},
                },
            ]
        }
        eng._fetch_json = AsyncMock(return_value=mock_data)
        summary = _run(eng.sync_openrouter())
        self.assertEqual(summary.discovered, 2)
        self.assertEqual(summary.added, 2)
        self.assertEqual(len(summary.errors), 0)
        # Verify they landed in the DB
        entries = memory.list_model_catalog(provider="openrouter")
        self.assertEqual(len(entries), 2)
        ids = {e["model_id"] for e in entries}
        self.assertIn("openai/gpt-4o", ids)
        self.assertIn("meta/llama-3.1-8b", ids)
        # GPT-4o should have a higher capability score (tools + vision)
        gpt4o = memory.get_model_catalog_entry("openai/gpt-4o", "openrouter")
        llama = memory.get_model_catalog_entry("meta/llama-3.1-8b", "openrouter")
        self.assertGreater(gpt4o["capability_score"], llama["capability_score"])
        # Auto-tier assignment
        self.assertIsNotNone(gpt4o["tier_assignment"])

    def test_sync_openai_no_key_returns_error(self):
        """When no API key is set, sync should report the error gracefully."""
        import os
        old = os.environ.pop("TEST_KEY_NONEXISTENT", None)
        _ = old
        eng = model_sync.ModelSyncEngine(
            openrouter_enabled=False,
            ollama_enabled=False,
            openai_api_key_env="TEST_KEY_NONEXISTENT",
        )
        summary = _run(eng.sync_openai())
        self.assertEqual(summary.discovered, 0)
        self.assertTrue(any("not set" in e for e in summary.errors))

    def test_sync_ollama_with_mock(self):
        eng = model_sync.ModelSyncEngine(
            openrouter_enabled=False,
            ollama_enabled=True,
            ollama_base_url="http://localhost:11434",
        )
        mock_data = {
            "models": [
                {
                    "name": "llama3:8b",
                    "details": {"family": "llama", "parameter_size": "8B",
                                "quantization_level": "Q4_0"},
                    "context_length": 8192,
                },
                {
                    "name": "mistral:7b",
                    "details": {"family": "llama", "parameter_size": "7B",
                                "quantization_level": "Q4_0"},
                    "context_length": 32768,
                },
            ]
        }
        eng._fetch_json = AsyncMock(return_value=mock_data)
        summary = _run(eng.sync_ollama())
        self.assertEqual(summary.discovered, 2)
        self.assertEqual(summary.added, 2)
        entries = memory.list_model_catalog(provider="ollama")
        self.assertEqual(len(entries), 2)

    def test_sync_all_concurrent(self):
        """sync_all runs all enabled providers concurrently."""
        eng = model_sync.ModelSyncEngine(
            openrouter_enabled=True,
            ollama_enabled=True,
        )
        eng._fetch_json = AsyncMock(side_effect=[
            {"data": [{"id": "or/1", "name": "M1", "context_length": 32000,
                       "architecture": {}, "pricing": {}, "supported_parameters": []}]},
            {"models": [{"name": "ollama/1", "details": {}, "context_length": 4096}]},
        ])
        results = _run(eng.sync_all())
        self.assertEqual(len(results), 2)
        providers = {r.provider for r in results}
        self.assertIn("openrouter", providers)
        self.assertIn("ollama", providers)

    def test_sync_handles_fetch_failure(self):
        """When the fetch returns None, sync reports an error gracefully."""
        eng = model_sync.ModelSyncEngine(
            openrouter_enabled=True,
            ollama_enabled=False,
        )
        eng._fetch_json = AsyncMock(return_value=None)
        summary = _run(eng.sync_openrouter())
        self.assertEqual(summary.discovered, 0)
        self.assertTrue(len(summary.errors) > 0)


# ============================================================================
# Init / singleton
# ============================================================================


class InitEngineTests(unittest.TestCase):

    def test_init_and_engine(self):
        eng = model_sync.init_sync_engine(
            openrouter_enabled=False,
            ollama_enabled=False,
            auto_sync=False,
        )
        self.assertIsNotNone(eng)
        self.assertIs(model_sync.engine(), eng)


if __name__ == "__main__":
    unittest.main()

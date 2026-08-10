"""Tests for the per-admin/user token budgets + per-model limits + budget-aware routing.

Covers:
  - Subscription plan CRUD (memory.upsert_plan, list_plans, get_plan)
  - Tenant plan assignment (assign_tenant_plan, get_tenant_plan)
  - Per-model token limits (upsert_model_token_limit, get_model_token_limit)
  - Daily token budget enforcement (reserve_usage)
  - Per-model USD/token enforcement (reserve_usage)
  - max_request_tokens enforcement
  - Model quality profiles (Wilson lower bound, conservative prior)
  - 99th-percentile cost-to-complete estimation
  - Budget-aware routing (cascade chain, target success probability)
  - Cascade P(success) math
  - Admin API endpoints for plans, model limits, token budgets
  - User-facing API endpoints
"""
from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path


class PlanQuotaTests(unittest.TestCase):
    """Subscription plan CRUD and tenant binding."""

    def setUp(self):
        from gateway import memory
        self.tmpdir = tempfile.mkdtemp()
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        memory.init_engine(f"sqlite:///{self.tmpdir}/test.db")

    def tearDown(self):
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]

    def test_create_and_get_plan(self):
        from gateway import memory
        plan = memory.upsert_plan(
            "pro",
            name="Pro",
            daily_token_limit=1_000_000,
            daily_usd_limit=25.0,
            required_success_probability=0.99,
            allowed_models=["tier1_model", "tier2_model"],
        )
        self.assertEqual(plan["plan_id"], "pro")
        self.assertEqual(plan["daily_token_limit"], 1_000_000)
        self.assertEqual(plan["daily_usd_limit"], 25.0)
        self.assertEqual(plan["required_success_probability"], 0.99)
        self.assertEqual(plan["allowed_models"], ["tier1_model", "tier2_model"])

        loaded = memory.get_plan("pro")
        self.assertEqual(loaded["name"], "Pro")

    def test_update_plan(self):
        from gateway import memory
        memory.upsert_plan("free", daily_token_limit=100_000)
        plan = memory.upsert_plan("free", daily_token_limit=200_000, name="Free V2")
        self.assertEqual(plan["daily_token_limit"], 200_000)
        self.assertEqual(plan["name"], "Free V2")

    def test_list_plans(self):
        from gateway import memory
        memory.upsert_plan("free")
        memory.upsert_plan("pro")
        plans = memory.list_plans()
        self.assertEqual({p["plan_id"] for p in plans}, {"free", "pro"})

    def test_assign_tenant_plan(self):
        from gateway import memory
        # Create both plans first (FK requires plan to exist)
        memory.upsert_plan("pro", daily_token_limit=2_000_000)
        memory.upsert_plan("free", daily_token_limit=100_000)
        binding = memory.assign_tenant_plan("acme", "pro", notes="VIP")
        self.assertEqual(binding["tenant_id"], "acme")
        self.assertEqual(binding["plan_id"], "pro")
        self.assertEqual(binding["notes"], "VIP")

        # Calling again updates the binding (not the user)
        binding = memory.assign_tenant_plan("acme", "free")
        self.assertEqual(binding["plan_id"], "free")
        # The user row's plan_id is also updated
        u = memory.get_or_create_user("acme", {})
        self.assertEqual(u["plan_id"], "free")

    def test_get_tenant_plan_quota(self):
        from gateway import memory
        memory.upsert_plan("pro", daily_token_limit=500_000, daily_usd_limit=10.0)
        memory.assign_tenant_plan("acme", "pro")
        quota = memory.get_tenant_plan_quota("acme")
        self.assertIsNotNone(quota)
        self.assertEqual(quota["plan_id"], "pro")
        self.assertEqual(quota["daily_token_limit"], 500_000)
        self.assertEqual(quota["daily_usd_limit"], 10.0)

    def test_get_tenant_plan_quota_unassigned(self):
        from gateway import memory
        quota = memory.get_tenant_plan_quota("nobody")
        self.assertIsNone(quota)


class ModelTokenLimitTests(unittest.TestCase):

    def setUp(self):
        from gateway import memory
        self.tmpdir = tempfile.mkdtemp()
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        memory.init_engine(f"sqlite:///{self.tmpdir}/test.db")

    def tearDown(self):
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]

    def test_upsert_and_get_model_limit(self):
        from gateway import memory
        ml = memory.upsert_model_token_limit(
            "acme", "tier1_model",
            daily_token_limit=100_000, max_request_tokens=8192,
        )
        self.assertEqual(ml["daily_token_limit"], 100_000)
        self.assertEqual(ml["max_request_tokens"], 8192)

        loaded = memory.get_model_token_limit("acme", "tier1_model")
        self.assertEqual(loaded["daily_token_limit"], 100_000)
        self.assertEqual(loaded["max_request_tokens"], 8192)

    def test_update_model_limit(self):
        from gateway import memory
        memory.upsert_model_token_limit("acme", "tier1_model", daily_token_limit=100_000)
        ml = memory.upsert_model_token_limit("acme", "tier1_model", daily_token_limit=200_000)
        self.assertEqual(ml["daily_token_limit"], 200_000)

    def test_list_model_token_limits(self):
        from gateway import memory
        memory.upsert_model_token_limit("acme", "tier1_model", daily_token_limit=100_000)
        memory.upsert_model_token_limit("acme", "tier2_model", daily_token_limit=200_000)
        limits = memory.list_model_token_limits("acme")
        self.assertEqual({entry["endpoint_name"] for entry in limits}, {"tier1_model", "tier2_model"})


class TokenBudgetEnforcementTests(unittest.TestCase):
    """Reserve_usage must enforce daily_token_limit, per-model limits, and max_request_tokens."""

    def setUp(self):
        from gateway import memory
        self.tmpdir = tempfile.mkdtemp()
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        memory.init_engine(f"sqlite:///{self.tmpdir}/test.db")
        memory.get_or_create_user("acme", {
            "daily_token_limit": 1000,
            "budget_usd_per_day": 100.0,
            "rps_limit": 100,
            "tokens_per_min": 1_000_000,
        })

    def tearDown(self):
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]

    def test_daily_token_limit_enforced(self):
        from gateway import memory
        # First request: 800 tokens — fits
        ok, reason = memory.reserve_usage(
            "acme",
            budget_limit_usd=100.0,
            rps_limit=100,
            token_limit_per_minute=1_000_000,
            estimated_tokens_in=400,
            estimated_tokens_out=400,
            estimated_cost_usd=0.01,
            daily_token_limit=1000,
            endpoint_name="tier1_model",
        )
        self.assertTrue(ok, reason)

        # Second request: 300 tokens — would exceed 1000
        ok, reason = memory.reserve_usage(
            "acme",
            budget_limit_usd=100.0,
            rps_limit=100,
            token_limit_per_minute=1_000_000,
            estimated_tokens_in=200,
            estimated_tokens_out=100,
            estimated_cost_usd=0.01,
            daily_token_limit=1000,
            endpoint_name="tier1_model",
        )
        self.assertFalse(ok)
        self.assertIn("daily token limit", reason)

    def test_per_model_token_limit_enforced(self):
        from gateway import memory
        # Tenant has unlimited daily tokens, but this model has 500 token limit
        memory.get_or_create_user("acme", {
            "daily_token_limit": 100_000,
            "budget_usd_per_day": 100.0,
            "rps_limit": 100,
            "tokens_per_min": 1_000_000,
        })
        ok, _ = memory.reserve_usage(
            "acme",
            budget_limit_usd=100.0,
            rps_limit=100,
            token_limit_per_minute=1_000_000,
            estimated_tokens_in=400,
            estimated_tokens_out=100,
            estimated_cost_usd=0.01,
            daily_token_limit=100_000,
            endpoint_name="tier1_model",
            model_token_limit=500,
        )
        self.assertTrue(ok)

        # Now model limit is exhausted
        ok, reason = memory.reserve_usage(
            "acme",
            budget_limit_usd=100.0,
            rps_limit=100,
            token_limit_per_minute=1_000_000,
            estimated_tokens_in=100,
            estimated_tokens_out=100,
            estimated_cost_usd=0.01,
            daily_token_limit=100_000,
            endpoint_name="tier1_model",
            model_token_limit=500,
        )
        self.assertFalse(ok)
        self.assertIn("model", reason)
        self.assertIn("tier1_model", reason)

    def test_per_model_usd_limit_enforced(self):
        from gateway import memory
        memory.get_or_create_user("acme", {
            "daily_token_limit": 0,  # unlimited
            "budget_usd_per_day": 100.0,
            "rps_limit": 100,
            "tokens_per_min": 1_000_000,
        })
        ok, _ = memory.reserve_usage(
            "acme",
            budget_limit_usd=100.0,
            rps_limit=100,
            token_limit_per_minute=1_000_000,
            estimated_tokens_in=100,
            estimated_tokens_out=100,
            estimated_cost_usd=5.0,
            daily_token_limit=0,
            endpoint_name="tier1_model",
            model_usd_limit=5.0,
        )
        self.assertTrue(ok)

        # Now model USD limit is exhausted
        ok, reason = memory.reserve_usage(
            "acme",
            budget_limit_usd=100.0,
            rps_limit=100,
            token_limit_per_minute=1_000_000,
            estimated_tokens_in=100,
            estimated_tokens_out=100,
            estimated_cost_usd=1.0,
            daily_token_limit=0,
            endpoint_name="tier1_model",
            model_usd_limit=5.0,
        )
        self.assertFalse(ok)
        self.assertIn("tier1_model", reason)
        self.assertIn("USD limit", reason)

    def test_max_request_tokens_enforced(self):
        from gateway import memory
        memory.get_or_create_user("acme", {
            "daily_token_limit": 100_000,
            "budget_usd_per_day": 100.0,
            "rps_limit": 100,
            "tokens_per_min": 1_000_000,
        })
        ok, reason = memory.reserve_usage(
            "acme",
            budget_limit_usd=100.0,
            rps_limit=100,
            token_limit_per_minute=1_000_000,
            estimated_tokens_in=5000,
            estimated_tokens_out=5000,
            estimated_cost_usd=0.01,
            daily_token_limit=100_000,
            endpoint_name="tier1_model",
            max_request_tokens=4096,
        )
        self.assertFalse(ok)
        self.assertIn("max_request_tokens", reason)

    def test_endpoint_name_recorded_in_usage(self):
        from gateway import memory
        memory.get_or_create_user("acme", {
            "daily_token_limit": 100_000,
            "budget_usd_per_day": 100.0,
            "rps_limit": 100,
            "tokens_per_min": 1_000_000,
        })
        memory.reserve_usage(
            "acme",
            budget_limit_usd=100.0,
            rps_limit=100,
            token_limit_per_minute=1_000_000,
            estimated_tokens_in=100,
            estimated_tokens_out=200,
            estimated_cost_usd=0.01,
            daily_token_limit=100_000,
            endpoint_name="tier1_model",
        )
        # Per-model spend should reflect the usage
        spent = memory.get_today_token_spend("acme", "tier1_model")
        self.assertEqual(spent, 300)
        # Tenant-wide spend should also be 300
        spent_all = memory.get_today_token_spend("acme")
        self.assertEqual(spent_all, 300)

    def test_settlement_releases_tokens_on_failure(self):
        from gateway import memory
        memory.get_or_create_user("acme", {
            "daily_token_limit": 1000,
            "budget_usd_per_day": 100.0,
            "rps_limit": 100,
            "tokens_per_min": 1_000_000,
        })
        memory.reserve_usage(
            "acme",
            budget_limit_usd=100.0,
            rps_limit=100,
            token_limit_per_minute=1_000_000,
            estimated_tokens_in=400,
            estimated_tokens_out=400,
            estimated_cost_usd=0.01,
            daily_token_limit=1000,
            endpoint_name="tier1_model",
        )
        # Failed request: release all reserved tokens
        memory.settle_reserved_usage(
            "acme",
            reserved_tokens_in=400,
            reserved_tokens_out=400,
            reserved_cost_usd=0.01,
            actual_tokens_in=0,
            actual_tokens_out=0,
            actual_cost_usd=0.0,
            completed=False,
            endpoint_name="tier1_model",
        )
        # Spent should be 0 after the failed release
        spent = memory.get_today_token_spend("acme", "tier1_model")
        self.assertEqual(spent, 0)


class QualityProfileTests(unittest.TestCase):

    def setUp(self):
        from gateway import memory
        self.tmpdir = tempfile.mkdtemp()
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        memory.init_engine(f"sqlite:///{self.tmpdir}/test.db")

    def tearDown(self):
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]

    def test_record_quality_sample(self):
        from gateway import memory
        for _ in range(9):
            memory.record_quality_sample("tier1_model", "programming", 3, success=True)
        memory.record_quality_sample("tier1_model", "programming", 3, success=False)
        profile = memory.get_quality_profile("tier1_model", "programming", 3)
        self.assertIsNotNone(profile)
        self.assertEqual(profile["total_count"], 10)
        self.assertEqual(profile["success_count"], 9)

    def test_conservative_prior_with_no_samples(self):
        from gateway import policy
        p, n = policy.estimate_success_probability(
            "tier1_model", "programming", 3,
            conservative_prior=0.5, min_samples=10,
        )
        self.assertEqual(p, 0.5)
        self.assertEqual(n, 0)

    def test_conservative_prior_with_few_samples(self):
        from gateway import memory, policy
        for _ in range(5):
            memory.record_quality_sample("tier1_model", "math", 3, success=True)
        p, n = policy.estimate_success_probability(
            "tier1_model", "math", 3, min_samples=10,
        )
        # Under min_samples, still returns the conservative prior
        self.assertEqual(p, 0.5)
        self.assertEqual(n, 5)

    def test_wilson_lower_bound_with_many_samples(self):
        from gateway import memory, policy
        # 95/100 success rate → lower bound should be below 0.95 but above 0.84
        for _ in range(95):
            memory.record_quality_sample("tier1_model", "programming", 3, success=True)
        for _ in range(5):
            memory.record_quality_sample("tier1_model", "programming", 3, success=False)
        p, n = policy.estimate_success_probability(
            "tier1_model", "programming", 3, min_samples=10,
        )
        self.assertEqual(n, 100)
        self.assertGreater(p, 0.84)
        self.assertLess(p, 0.95)

    def test_imperfect_model_returns_lower_bound(self):
        from gateway import memory, policy
        # 50/100 success rate → lower bound should be well below 0.5
        for _ in range(50):
            memory.record_quality_sample("tier1_model", "failing", 3, success=True)
        for _ in range(50):
            memory.record_quality_sample("tier1_model", "failing", 3, success=False)
        p, n = policy.estimate_success_probability(
            "tier1_model", "failing", 3, min_samples=10,
        )
        self.assertEqual(n, 100)
        self.assertLess(p, 0.5)


class CostToCompleteTests(unittest.TestCase):

    def test_p99_cost_with_high_success(self):
        from gateway import policy
        # p_success=1.0 → p99 cost = base cost (no retries needed)
        cost = policy.cost_to_complete_p99(base_cost=1.0, p_success=1.0)
        self.assertEqual(cost, 1.0)

    def test_p99_cost_with_low_success(self):
        from gateway import policy
        # p_success=0.5 → likely need retries
        cost = policy.cost_to_complete_p99(base_cost=1.0, p_success=0.5, max_retries=3)
        self.assertGreater(cost, 1.0)

    def test_p99_cost_zero_success(self):
        from gateway import policy
        cost = policy.cost_to_complete_p99(base_cost=1.0, p_success=0.0, max_retries=2)
        # With 0 success and 2 retries, max attempts = 3 → cost = 3.0
        self.assertEqual(cost, 3.0)

    def test_cascade_success_probability(self):
        from gateway import policy
        c1 = policy.BudgetAwareCandidate(
            tier_name="tier0", endpoint_name="cheep",
            cost=0.01, fit=0.95, success_probability=0.95,
            p_completed=0.95, cost_to_complete_p99=0.011,
            estimated_tokens=100, estimated_cost_usd=0.01,
            fits_remaining_tokens=True, allowed_by_plan=True,
        )
        c2 = policy.BudgetAwareCandidate(
            tier_name="tier1", endpoint_name="big",
            cost=0.05, fit=0.99, success_probability=0.99,
            p_completed=0.99, cost_to_complete_p99=0.051,
            estimated_tokens=100, estimated_cost_usd=0.05,
            fits_remaining_tokens=True, allowed_by_plan=True,
        )
        # 1 - (1 - 0.95) * (1 - 0.99) = 1 - 0.05 * 0.01 = 0.9995
        p = policy.cascade_success_probability([c1, c2])
        self.assertAlmostEqual(p, 0.9995, places=4)

    def test_plan_allowed_endpoint(self):
        from gateway import policy
        # No plan = unrestricted
        self.assertTrue(policy.plan_allowed_endpoint(None, "anything"))
        # Empty allowed_models = unrestricted
        self.assertTrue(policy.plan_allowed_endpoint({"allowed_models": []}, "anything"))
        # Non-empty allowed_models = whitelist
        self.assertTrue(policy.plan_allowed_endpoint({"allowed_models": ["a", "b"]}, "a"))
        self.assertFalse(policy.plan_allowed_endpoint({"allowed_models": ["a", "b"]}, "c"))


class BudgetAwareRoutingTests(unittest.TestCase):
    """End-to-end budget-aware routing with cascade + 99% probability."""

    def _build_request_context(self, vertical="programming", complexity=2):
        from gateway import ood, policy
        return policy.RequestContext(
            text="write a python function",
            has_image=False,
            flags={"code": True, "math": False, "reasoning": False, "long_output": False},
            complexity=complexity,
            vertical=vertical,
            vertical_top2=[(vertical, 0.9)],
            ood=ood.OODResult(is_ood=False, score=0.0, max_prob=0.9, top_vertical=vertical, threshold=0.5),
            model_version="stub-v0",
            policy_version=1,
            session_id="s1",
            tenant_id="acme",
            estimated_input_tokens=200,
            estimated_output_tokens=200,
        )

    def _build_config(self):
        from gateway import config
        return config.Config(
            config={
                "tiers": [
                    {
                        "name": "tier0",
                        "endpoints": ["ollama_local"],
                        "max_context": 32768,
                        "capability_per_vertical": {"_default": 0.95, "programming": 0.7},
                        "max_tokens_bump": 0,
                    },
                    {
                        "name": "tier1",
                        "endpoints": ["frontier"],
                        "max_context": 65536,
                        "capability_per_vertical": {"_default": 0.99, "programming": 0.95},
                        "max_tokens_bump": 0,
                    },
                ],
                "endpoints": [
                    {"name": "ollama_local", "pricing": {"fixed_per_request": 0.0, "in_per_1k_tokens": 0.0, "out_per_1k_tokens": 0.0}},
                    {"name": "frontier", "pricing": {"fixed_per_request": 0.0, "in_per_1k_tokens": 0.01, "out_per_1k_tokens": 0.03}},
                ],
                "routing": {
                    "cost_first": {"fit_threshold": 0.9, "capability_sigmoid_k": 20.0, "retry_penalty_multiplier": 5.0},
                    "escalation": {"ood_flag_to_tier": "tier1", "confidence_threshold": 0.5, "top2_epsilon": 0.1, "cost_margin_abstain_pct": 5.0},
                },
                "ladder": {"context_reserve_pct": 25},
            },
            policy={"ladder": {"context_reserve_pct": 25}},
            taxonomy={"verticals": [{"name": "programming", "min_capability": 0.5}], "complexity_weights": [0.1, 0.2, 0.3, 0.2, 0.2]},
            prototypes={"verticals": {}},
            version=1,
        )

    def test_route_picks_single_when_target_met(self):
        from gateway import policy, tenant
        tmpdir = tempfile.mkdtemp()
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        memory.init_engine(f"sqlite:///{tmpdir}/test.db")
        memory.get_or_create_user("acme", {"daily_token_limit": 1_000_000, "target_success_probability": 0.95})
        # Record enough high-quality samples on ollama so it meets the 0.95 target
        for _ in range(200):
            memory.record_quality_sample("ollama_local", "programming", 2, success=True)
        tm = tenant.TenantManager({"daily_token_limit": 1_000_000, "target_success_probability": 0.95})
        ctx = self._build_request_context()
        conf = self._build_config()
        decision = policy.budget_aware_route(
            ctx, conf, {}, {},
            tenant_mgr=tm,
            tenant_id="acme",
        )
        # The cascade length should be 1 (single model meets target)
        self.assertEqual(len(decision.cascade), 1)
        self.assertEqual(decision.decision.endpoint, "ollama_local")
        self.assertGreaterEqual(decision.achieved_success_probability, 0.95)

    def test_route_uses_cascade_when_target_not_met(self):
        from gateway import policy, tenant
        tmpdir = tempfile.mkdtemp()
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        memory.init_engine(f"sqlite:///{tmpdir}/test.db")
        memory.get_or_create_user("acme", {"daily_token_limit": 1_000_000, "target_success_probability": 0.95})
        # Record perfect samples for both models
        for _ in range(100):
            memory.record_quality_sample("ollama_local", "programming", 2, success=True)
            memory.record_quality_sample("frontier", "programming", 2, success=True)
        tm = tenant.TenantManager({"daily_token_limit": 1_000_000, "target_success_probability": 0.95})
        ctx = self._build_request_context()
        conf = self._build_config()
        decision = policy.budget_aware_route(
            ctx, conf, {}, {},
            tenant_mgr=tm,
            tenant_id="acme",
        )
        # Cascade should chain tier0 + tier1 to meet 0.95
        self.assertGreaterEqual(len(decision.cascade), 1)
        # Cheapest first
        self.assertEqual(decision.cascade[0].endpoint_name, "ollama_local")
        # Both models are 100% successful, so the cascade achieves ~99%
        self.assertGreaterEqual(decision.achieved_success_probability, 0.95)

    def test_route_rejects_when_no_model_eligible(self):
        from gateway import policy, tenant
        tmpdir = tempfile.mkdtemp()
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        memory.init_engine(f"sqlite:///{tmpdir}/test.db")
        # Plan that restricts to models that don't exist in config
        memory.upsert_plan(
            "restricted", allowed_models=["nonexistent_model"],
            required_success_probability=0.99,
        )
        memory.assign_tenant_plan("acme", "restricted")
        tm = tenant.TenantManager({})
        ctx = self._build_request_context()
        conf = self._build_config()
        with self.assertRaises(policy.BudgetError) as cx:
            policy.budget_aware_route(
                ctx, conf, {}, {},
                tenant_mgr=tm,
                tenant_id="acme",
            )
        self.assertEqual(cx.exception.code, "quality_target_unmet")

    def test_route_rejects_when_no_tokens_remaining(self):
        from gateway import policy, tenant
        tmpdir = tempfile.mkdtemp()
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        memory.init_engine(f"sqlite:///{tmpdir}/test.db")
        # Burn the entire daily token budget before the request
        for _ in range(10):
            memory.reserve_usage(
                "acme",
                budget_limit_usd=100.0,
                rps_limit=100,
                token_limit_per_minute=1_000_000,
                estimated_tokens_in=100,
                estimated_tokens_out=100,
                estimated_cost_usd=0.01,
                daily_token_limit=1000,
                endpoint_name="ollama_local",
            )
        tm = tenant.TenantManager({"daily_token_limit": 1000})
        ctx = self._build_request_context()
        conf = self._build_config()
        with self.assertRaises(policy.BudgetError) as cx:
            policy.budget_aware_route(
                ctx, conf, {}, {},
                tenant_mgr=tm,
                tenant_id="acme",
            )
        self.assertEqual(cx.exception.code, "quality_target_unmet")


class TenantStateTests(unittest.TestCase):

    def setUp(self):
        from gateway import memory
        self.tmpdir = tempfile.mkdtemp()
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        memory.init_engine(f"sqlite:///{self.tmpdir}/test.db")

    def tearDown(self):
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]

    def test_tenant_state_has_new_fields(self):
        from gateway import tenant
        tm = tenant.TenantManager({
            "tier_access": ["tier0"],
            "budget_usd_per_day": 5.0,
            "rps_limit": 50,
            "concurrent_limit": 10,
            "tokens_per_min": 100_000,
            "daily_token_limit": 50_000,
            "target_success_probability": 0.97,
        })
        st = tm.get_or_create("acme")
        self.assertEqual(st.daily_token_limit, 50_000)
        self.assertEqual(st.target_success_probability, 0.97)
        self.assertEqual(st.budget_usd_per_day, 5.0)

    def test_tenant_state_default_target_is_99(self):
        from gateway import tenant
        tm = tenant.TenantManager({})
        st = tm.get_or_create("acme")
        self.assertEqual(st.target_success_probability, 0.99)

    def test_remaining_tokens_today(self):
        from gateway import memory, tenant
        self.tmpdir = tempfile.mkdtemp()
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        memory.init_engine(f"sqlite:///{self.tmpdir}/test.db")
        memory.get_or_create_user(
            "acme", {"daily_token_limit": 1000, "budget_usd_per_day": 100.0,
                     "rps_limit": 100, "tokens_per_min": 1_000_000},
        )
        tm = tenant.TenantManager({"daily_token_limit": 1000})
        # Initially remaining = 1000
        self.assertEqual(tm.remaining_tokens_today("acme"), 1000)
        # Burn 400 tokens
        memory.reserve_usage(
            "acme",
            budget_limit_usd=100.0,
            rps_limit=100,
            token_limit_per_minute=1_000_000,
            estimated_tokens_in=200,
            estimated_tokens_out=200,
            estimated_cost_usd=0.01,
            daily_token_limit=1000,
            endpoint_name="ollama_local",
        )
        self.assertEqual(tm.remaining_tokens_today("acme"), 600)

    def test_remaining_tokens_unlimited_when_zero_limit(self):
        from gateway import tenant
        tm = tenant.TenantManager({})
        self.assertEqual(tm.remaining_tokens_today("acme"), -1)

    def test_tenant_state_picks_up_plan_quota(self):
        from gateway import memory, tenant
        self.tmpdir = tempfile.mkdtemp()
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        memory.init_engine(f"sqlite:///{self.tmpdir}/test.db")
        memory.upsert_plan(
            "pro", daily_token_limit=2_000_000, daily_usd_limit=20.0,
            required_success_probability=0.98,
        )
        memory.assign_tenant_plan("acme", "pro")
        tm = tenant.TenantManager({})
        st = tm.get_or_create("acme")
        self.assertEqual(st.daily_token_limit, 2_000_000)
        self.assertEqual(st.budget_usd_per_day, 20.0)
        self.assertEqual(st.target_success_probability, 0.98)


class AdminAPIIntegrationTests(unittest.TestCase):
    """End-to-end via the running app: plan CRUD, model limits, budget read."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        memory.init_engine(f"sqlite:///{self.tmpdir}/test.db")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        try:
            self.loop.close()
        except Exception:
            pass

    def _bootstrap_app(self, auth: dict | None = None):
        from aiohttp.test_utils import TestClient, TestServer

        from gateway import app as app_mod
        cfg_path = str(Path(self.tmpdir) / "gateway-config.json")
        cfg = {
            "mode": "single",
            "db_url": f"sqlite:///{self.tmpdir}/itest.db",
            "host": "127.0.0.1",
            "port": 0,
            "tenants": {"*": {
                "tier_access": ["tier0", "tier1"],
                "budget_usd_per_day": 100.0,
                "rps_limit": 1000,
                "concurrent_limit": 50,
                "tokens_per_min": 10_000_000,
            }},
            "endpoints": [
                {"name": "ep_a", "kind": "llamacpp", "base_url": "http://127.0.0.1:1",
                 "model_alias": "a", "pricing": {"fixed_per_request": 0.0, "in_per_1k_tokens": 0.0, "out_per_1k_tokens": 0.0},
                 "concurrency": 1, "breaker": {"failure_threshold": 1, "open_duration_seconds": 1, "half_open_max_probes": 1},
                 "health_probe": "/health"},
            ],
            "tiers": [
                {"name": "tier0", "endpoints": ["ep_a"], "max_context": 32768,
                 "capability_per_vertical": {"_default": 0.95}, "max_tokens_bump": 0},
            ],
            "routing": {"cost_first": {"fit_threshold": 0.9, "capability_sigmoid_k": 20.0, "retry_penalty_multiplier": 5.0, "fallback_endpoint": "ep_a"}, "escalation": {"ood_flag_to_tier": "tier0", "confidence_threshold": 0.5, "top2_epsilon": 0.1, "cost_margin_abstain_pct": 5.0}},
            "reviewer": {"endpoint": "http://127.0.0.1:1", "model": "m", "api_key_env": "x", "timeout_seconds": 30, "batch_size": 1, "caps": {"per_request_usd": 1.0, "per_hour_usd": 1.0, "per_day_usd": 1.0, "per_month_usd": 1.0}},
            "trainer": {"auto_retrain": False, "trigger_threshold_new_samples": 500, "trigger_accuracy_drop_below": 0.0, "min_trust_score_to_train": 0.0},
            "security": {"injection_regex": []},
            "drift": {"enabled": False},
            "logging": {"trace_retention_days": None},
            "memory": {"enabled": True, "force_observe_on_close": False},
            "auth": auth or {"enabled": False, "keys": {}, "admin_paths": ["/admin"]},
            "http": {"max_body_bytes": 1048576},
        }
        Path(cfg_path).write_text(__import__("json").dumps(cfg), encoding="utf-8")

        async def _build():
            the_app = await app_mod.init_app(cfg_path)
            client = TestClient(TestServer(the_app))
            await client.start_server()
            return client

        return self.loop.run_until_complete(_build())

    def test_admin_plan_crud(self):
        client = self._bootstrap_app()
        try:
            # Create plan
            r = self.loop.run_until_complete(client.post("/admin/plans", json={
                "plan_id": "pro",
                "name": "Pro",
                "daily_token_limit": 1_000_000,
                "daily_usd_limit": 25.0,
                "required_success_probability": 0.99,
                "allowed_models": ["ep_a"],
            }))
            self.assertEqual(r.status, 201)
            j = self.loop.run_until_complete(r.json())
            self.assertEqual(j["plan_id"], "pro")

            # List plans
            r = self.loop.run_until_complete(client.get("/admin/plans"))
            self.assertEqual(r.status, 200)
            plans = self.loop.run_until_complete(r.json())
            self.assertEqual(len(plans["plans"]), 1)

            # Update plan
            r = self.loop.run_until_complete(client.put("/admin/plans/pro", json={
                "daily_token_limit": 2_000_000,
            }))
            self.assertEqual(r.status, 200)
            j = self.loop.run_until_complete(r.json())
            self.assertEqual(j["daily_token_limit"], 2_000_000)

            # Assign plan to tenant
            r = self.loop.run_until_complete(client.post("/admin/users/acme/subscription", json={
                "plan_id": "pro",
            }))
            self.assertEqual(r.status, 200)
            j = self.loop.run_until_complete(r.json())
            self.assertEqual(j["plan_id"], "pro")

            # Get tenant's plan
            r = self.loop.run_until_complete(client.get("/admin/users/acme/subscription"))
            self.assertEqual(r.status, 200)
            j = self.loop.run_until_complete(r.json())
            self.assertEqual(j["binding"]["plan_id"], "pro")
            self.assertEqual(j["quota"]["daily_token_limit"], 2_000_000)
        finally:
            self.loop.run_until_complete(client.close())

    def test_admin_model_limits(self):
        client = self._bootstrap_app()
        try:
            # Set per-model limit
            r = self.loop.run_until_complete(client.put("/admin/users/acme/models/ep_a/limits", json={
                "daily_token_limit": 50_000,
                "max_request_tokens": 4096,
                "daily_usd_limit": 5.0,
            }))
            self.assertEqual(r.status, 200)
            j = self.loop.run_until_complete(r.json())
            self.assertEqual(j["daily_token_limit"], 50_000)
            self.assertEqual(j["max_request_tokens"], 4096)

            # Get per-model limit
            r = self.loop.run_until_complete(client.get("/admin/users/acme/models/ep_a/limits"))
            self.assertEqual(r.status, 200)
            j = self.loop.run_until_complete(r.json())
            self.assertEqual(j["daily_token_limit"], 50_000)

            # Get all limits
            r = self.loop.run_until_complete(client.get("/admin/users/acme/limits"))
            self.assertEqual(r.status, 200)
            j = self.loop.run_until_complete(r.json())
            self.assertEqual(len(j["model_limits"]), 1)
            self.assertEqual(j["model_limits"][0]["endpoint_name"], "ep_a")
        finally:
            self.loop.run_until_complete(client.close())

    def test_admin_daily_token_budget(self):
        client = self._bootstrap_app()
        try:
            # Set daily token budget
            r = self.loop.run_until_complete(client.put("/admin/users/acme/budget/tokens", json={
                "daily_token_limit": 100_000,
                "target_success_probability": 0.97,
            }))
            self.assertEqual(r.status, 200)
            j = self.loop.run_until_complete(r.json())
            self.assertEqual(j["daily_token_limit"], 100_000)
            self.assertEqual(j["target_success_probability"], 0.97)

            # Get user's budget
            r = self.loop.run_until_complete(client.get("/admin/users/acme/budget"))
            self.assertEqual(r.status, 200)
            j = self.loop.run_until_complete(r.json())
            self.assertEqual(j["daily_token_limit"], 100_000)
            self.assertEqual(j["target_success_probability"], 0.97)
        finally:
            self.loop.run_until_complete(client.close())


class UserFacingAPIIntegrationTests(unittest.TestCase):
    """User-facing /usage and /usage/limits endpoints."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        memory.init_engine(f"sqlite:///{self.tmpdir}/test.db")
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        from gateway import memory
        memory.close_engine()
        memory._engine = None  # type: ignore[attr-defined]
        try:
            self.loop.close()
        except Exception:
            pass

    def _bootstrap_authenticated_app(self):
        from aiohttp.test_utils import TestClient, TestServer

        from gateway import app as app_mod
        cfg_path = str(Path(self.tmpdir) / "gateway-config.json")
        cfg = {
            "mode": "single",
            "db_url": f"sqlite:///{self.tmpdir}/itest.db",
            "host": "127.0.0.1",
            "port": 0,
            "tenants": {"*": {"tier_access": ["tier0"], "budget_usd_per_day": 100.0, "rps_limit": 1000, "concurrent_limit": 50, "tokens_per_min": 10_000_000}},
            "endpoints": [{"name": "ep_a", "kind": "llamacpp", "base_url": "http://127.0.0.1:1", "model_alias": "a", "pricing": {"fixed_per_request": 0.0, "in_per_1k_tokens": 0.0, "out_per_1k_tokens": 0.0}, "concurrency": 1, "breaker": {"failure_threshold": 1, "open_duration_seconds": 1, "half_open_max_probes": 1}, "health_probe": "/health"}],
            "tiers": [{"name": "tier0", "endpoints": ["ep_a"], "max_context": 32768, "capability_per_vertical": {"_default": 0.95}, "max_tokens_bump": 0}],
            "routing": {"cost_first": {"fit_threshold": 0.9, "capability_sigmoid_k": 20.0, "retry_penalty_multiplier": 5.0, "fallback_endpoint": "ep_a"}, "escalation": {"ood_flag_to_tier": "tier0", "confidence_threshold": 0.5, "top2_epsilon": 0.1, "cost_margin_abstain_pct": 5.0}},
            "reviewer": {"endpoint": "http://127.0.0.1:1", "model": "m", "api_key_env": "x", "timeout_seconds": 30, "batch_size": 1, "caps": {"per_request_usd": 1.0, "per_hour_usd": 1.0, "per_day_usd": 1.0, "per_month_usd": 1.0}},
            "trainer": {"auto_retrain": False, "trigger_threshold_new_samples": 500, "trigger_accuracy_drop_below": 0.0, "min_trust_score_to_train": 0.0},
            "security": {"injection_regex": []},
            "drift": {"enabled": False},
            "logging": {"trace_retention_days": None},
            "memory": {"enabled": True, "force_observe_on_close": False},
            "auth": {
                "enabled": True,
                "keys": {"sk-user-1": {"tenant_id": "acme", "scope": ["user"]}},
                "admin_paths": ["/admin"],
                "public_paths": ["/", "/health", "/ready", "/dashboard", "/usage", "/usage/limits"],
            },
            "http": {"max_body_bytes": 1048576},
        }
        Path(cfg_path).write_text(__import__("json").dumps(cfg), encoding="utf-8")

        async def _build():
            the_app = await app_mod.init_app(cfg_path)
            client = TestClient(TestServer(the_app))
            await client.start_server()
            return client

        return self.loop.run_until_complete(_build())

    def test_user_can_read_own_usage(self):
        client = self._bootstrap_authenticated_app()
        try:
            r = self.loop.run_until_complete(client.get("/usage", headers={"Authorization": "Bearer sk-user-1"}))
            self.assertEqual(r.status, 200)
            j = self.loop.run_until_complete(r.json())
            self.assertEqual(j["tenant_id"], "acme")
        finally:
            self.loop.run_until_complete(client.close())

    def test_user_can_read_own_limits(self):
        from gateway import memory
        memory.get_or_create_user("acme", {"daily_token_limit": 50_000, "target_success_probability": 0.95})
        client = self._bootstrap_authenticated_app()
        try:
            r = self.loop.run_until_complete(client.get("/usage/limits", headers={"Authorization": "Bearer sk-user-1"}))
            self.assertEqual(r.status, 200)
            j = self.loop.run_until_complete(r.json())
            self.assertEqual(j["daily_token_limit"], 50_000)
            self.assertEqual(j["target_success_probability"], 0.95)
            self.assertEqual(j["remaining_tokens_today"], 50_000)
        finally:
            self.loop.run_until_complete(client.close())


if __name__ == "__main__":
    unittest.main()

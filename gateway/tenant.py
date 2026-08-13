"""Multi-tenant rate and budget enforcement.

Per-user:
  - Token bucket for RPS (in-memory + persistent counter for persistence across restarts)
  - Concurrent request semaphore
  - Daily USD budget (persisted in usage_counters)
  - Daily token budget (per subscription, persisted in usage_counters)
  - Per-model token and USD limits (persisted in model_token_limits)
  - Tokens-per-minute cap
  - Target success probability (used by budget-aware routing)

Tenant config is loaded from gateway-config.json -> tenants; users are
created lazily via memory.get_or_create_user on first request. Tenant plans
(plan_quotas) further override budgets and constrain allowed_models.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass, field

from . import memory

log = logging.getLogger("ctrl.tenant")


class BudgetExceeded(Exception):
    pass


class RateLimited(Exception):
    def __init__(self, retry_after_seconds: float):
        self.retry_after_seconds = retry_after_seconds
        super().__init__(f"rate limited; retry after {retry_after_seconds:.1f}s")


class _TokenBucket:
    """In-memory token bucket. Thread-safe via lock."""

    def __init__(self, rate_per_sec: float, capacity: float):
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.last_refill = time.time()
        self._lock = threading.Lock()

    def consume(self, tokens: float = 1.0) -> bool:
        with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            self.last_refill = now
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

    def time_to_available(self, tokens: float = 1.0) -> float:
        with self._lock:
            now = time.time()
            elapsed = now - self.last_refill
            current = min(self.capacity, self.tokens + elapsed * self.rate)
            if current >= tokens:
                return 0.0
            deficit = tokens - current
            return deficit / max(self.rate, 1e-9)


@dataclass
class TenantState:
    tenant_id: str
    tier_access: list[str]
    budget_usd_per_day: float
    rps_limit: int
    concurrent_limit: int
    tokens_per_min: int
    daily_token_limit: int = 0
    target_success_probability: float = 0.99
    plan_id: str | None = None
    bucket: _TokenBucket = field(default_factory=lambda: _TokenBucket(rate_per_sec=100, capacity=200))
    semaphore: asyncio.Semaphore = field(default_factory=lambda: asyncio.Semaphore(20))
    user_lock: threading.Lock = field(default_factory=threading.Lock)


class TenantManager:
    def __init__(self, default_config: dict, preconfigured: dict | None = None):
        self._default = default_config
        # Per-user overrides from gateway-config.json -> tenants (excl "*" / docs)
        self._preconfigured: dict[str, dict] = preconfigured or {}
        self._states: dict[str, TenantState] = {}
        self._lock = threading.Lock()

    def _cfg_or_default(self, cfg: dict, key: str, default_value):
        """Return cfg[key] if the key is explicitly present (not None),
        else fall back to self._default[key], else default_value.

        Deliberately uses `is None` rather than truthiness: several of
        these fields have a legitimate, intentional value of 0 or [] that
        is semantically distinct from "unset" — e.g. tier_access=[]
        (explicitly restrict a tenant to zero tiers), concurrent_limit=0
        (hard-block: asyncio.Semaphore(0) never admits a request),
        rps_limit=0 / tokens_per_min=0 / budget_usd_per_day=0 /
        daily_token_limit=0 (explicitly "unlimited", overriding a
        non-zero default). A truthiness check (`cfg.get(key) or default`)
        can't distinguish any of those explicit values from "key absent"
        and silently replaces them with the default instead.
        """
        value = cfg.get(key)
        if value is not None:
            return value
        return self._default.get(key, default_value)

    def _build_state(self, tenant_id: str, cfg: dict) -> TenantState:
        tier_access = cfg.get("tier_access")
        if isinstance(tier_access, str):
            import json
            tier_access = json.loads(tier_access) if tier_access else []
        elif tier_access is None:
            tier_access = self._default.get("tier_access", [])
        rps = int(self._cfg_or_default(cfg, "rps_limit", 100))
        conc = int(self._cfg_or_default(cfg, "concurrent_limit", 20))
        return TenantState(
            tenant_id=tenant_id,
            tier_access=tier_access,
            budget_usd_per_day=float(self._cfg_or_default(cfg, "budget_usd_per_day", 1.0)),
            rps_limit=rps,
            concurrent_limit=conc,
            tokens_per_min=int(self._cfg_or_default(cfg, "tokens_per_min", 200000)),
            daily_token_limit=int(self._cfg_or_default(cfg, "daily_token_limit", 0)),
            target_success_probability=float(
                self._cfg_or_default(cfg, "target_success_probability", 0.99)
            ),
            plan_id=cfg.get("plan_id"),
            bucket=_TokenBucket(rate_per_sec=rps, capacity=rps * 2),
            semaphore=asyncio.Semaphore(conc),
            user_lock=threading.Lock(),
        )

    def reconfigure(self, default_config: dict, preconfigured: dict | None = None) -> None:
        """Replace tenant defaults/overrides; existing requests keep old state."""
        with self._lock:
            self._default = default_config
            self._preconfigured = preconfigured or {}
            self._states = {}

    def get_or_create(self, tenant_id: str) -> TenantState:
        with self._lock:
            st = self._states.get(tenant_id)
            if st:
                return st
            # Precedence: config-file override > DB row > plan quota > defaults
            if tenant_id in self._preconfigured:
                merged = {**self._default, **self._preconfigured[tenant_id]}
                st = self._build_state(tenant_id, merged)
            else:
                db_cfg = memory.get_or_create_user(tenant_id, defaults=self._default)
                db_keys = (
                    "tier_access",
                    "budget_usd_per_day",
                    "rps_limit",
                    "concurrent_limit",
                    "tokens_per_min",
                    "daily_token_limit",
                    "target_success_probability",
                    "plan_id",
                )
                merged = {**self._default, **{k: v for k, v in db_cfg.items() if k in db_keys and v is not None}}
                # Plan-level quota overrides (if tenant assigned to a plan)
                plan_quota = memory.get_tenant_plan_quota(tenant_id)
                if plan_quota:
                    if plan_quota.get("daily_token_limit", 0):
                        merged["daily_token_limit"] = plan_quota["daily_token_limit"]
                    if plan_quota.get("daily_usd_limit", 0):
                        merged["budget_usd_per_day"] = plan_quota["daily_usd_limit"]
                    if plan_quota.get("required_success_probability"):
                        merged["target_success_probability"] = plan_quota["required_success_probability"]
                st = self._build_state(tenant_id, merged)
            self._states[tenant_id] = st
            return st

    def refresh(self, tenant_id: str) -> TenantState:
        """Rebuild the cached state from DB/config (admin edits take effect)."""
        with self._lock:
            self._states.pop(tenant_id, None)
        return self.get_or_create(tenant_id)

    def remaining_tokens_today(self, tenant_id: str) -> int:
        """Return remaining tenant-wide daily token budget. -1 = unlimited."""
        st = self.get_or_create(tenant_id)
        if st.daily_token_limit <= 0:
            return -1
        spent = memory.get_today_token_spend(tenant_id)
        return max(0, st.daily_token_limit - spent)

    def remaining_model_tokens_today(self, tenant_id: str, endpoint_name: str) -> int:
        """Return remaining per-model daily token budget. -1 = unlimited."""
        mtl = memory.get_model_token_limit(tenant_id, endpoint_name)
        if not mtl or int(mtl.get("daily_token_limit", 0)) <= 0:
            return -1
        spent = memory.get_today_token_spend(tenant_id, endpoint_name)
        return max(0, int(mtl["daily_token_limit"]) - spent)

    def allowed_models(self, tenant_id: str) -> list[str] | None:
        """Return list of allowed model names for the tenant's plan. None = unrestricted."""
        plan_quota = memory.get_tenant_plan_quota(tenant_id)
        if not plan_quota:
            return None
        return plan_quota.get("allowed_models") or None

    def check_rate_limit(self, tenant_id: str) -> None:
        st = self.get_or_create(tenant_id)
        if not st.bucket.consume(1.0):
            retry = st.bucket.time_to_available(1.0)
            raise RateLimited(retry_after_seconds=max(retry, 0.05))

    def check_budget(self, tenant_id: str, additional_cost_usd: float) -> None:
        st = self.get_or_create(tenant_id)
        if st.budget_usd_per_day <= 0:
            return  # 0 = unlimited
        spent = memory.get_today_spend(tenant_id)
        if spent + additional_cost_usd > st.budget_usd_per_day:
            raise BudgetExceeded(
                f"daily budget ${st.budget_usd_per_day:.2f} exceeded (spent ${spent:.2f})"
            )

    def reserve_usage(
        self,
        tenant_id: str,
        *,
        estimated_tokens_in: int,
        estimated_tokens_out: int,
        estimated_cost_usd: float,
        endpoint_name: str = "unknown",
        model_token_limit: int = 0,
        model_usd_limit: float = 0.0,
        max_request_tokens: int = 0,
    ) -> None:
        st = self.get_or_create(tenant_id)
        ok, reason = memory.reserve_usage(
            tenant_id,
            budget_limit_usd=st.budget_usd_per_day,
            rps_limit=st.rps_limit,
            token_limit_per_minute=st.tokens_per_min,
            estimated_tokens_in=estimated_tokens_in,
            estimated_tokens_out=estimated_tokens_out,
            estimated_cost_usd=estimated_cost_usd,
            daily_token_limit=st.daily_token_limit,
            endpoint_name=endpoint_name,
            model_token_limit=model_token_limit,
            model_usd_limit=model_usd_limit,
            max_request_tokens=max_request_tokens,
        )
        if not ok:
            if reason and (
                reason.startswith("daily budget")
                or reason.startswith("daily token limit")
                or reason.startswith("model")
                or reason.startswith("max_request_tokens")
            ):
                raise BudgetExceeded(reason)
            retry_after = 1.0 if reason and reason.startswith("requests-per-second") else 60.0
            raise RateLimited(retry_after_seconds=retry_after)

    @staticmethod
    def settle_usage(
        tenant_id: str,
        *,
        reserved_tokens_in: int,
        reserved_tokens_out: int,
        reserved_cost_usd: float,
        actual_tokens_in: int,
        actual_tokens_out: int,
        actual_cost_usd: float,
        completed: bool,
        endpoint_name: str = "unknown",
    ) -> None:
        memory.settle_reserved_usage(
            tenant_id,
            reserved_tokens_in=reserved_tokens_in,
            reserved_tokens_out=reserved_tokens_out,
            reserved_cost_usd=reserved_cost_usd,
            actual_tokens_in=actual_tokens_in,
            actual_tokens_out=actual_tokens_out,
            actual_cost_usd=actual_cost_usd,
            completed=completed,
            endpoint_name=endpoint_name,
        )

    def can_access_tier(self, tenant_id: str, tier_name: str) -> bool:
        st = self.get_or_create(tenant_id)
        return tier_name in st.tier_access

    def highest_accessible_tier(self, tenant_id: str, candidates: list[str]) -> str | None:
        st = self.get_or_create(tenant_id)
        for t in candidates:
            if t in st.tier_access:
                return t
        return None

    def all_states(self) -> dict[str, dict]:
        with self._lock:
            return {
                tid: {
                    "tenant_id": s.tenant_id,
                    "tier_access": s.tier_access,
                    "budget_usd_per_day": s.budget_usd_per_day,
                    "rps_limit": s.rps_limit,
                    "concurrent_limit": s.concurrent_limit,
                    "tokens_per_min": s.tokens_per_min,
                    "budget_spent_today_usd": memory.get_today_spend(tid),
                }
                for tid, s in self._states.items()
            }


_manager: TenantManager | None = None


def init_manager(default_config: dict, preconfigured: dict | None = None) -> TenantManager:
    global _manager
    _manager = TenantManager(default_config, preconfigured=preconfigured)
    return _manager


def manager() -> TenantManager:
    if _manager is None:
        raise RuntimeError("tenant manager not initialized")
    return _manager

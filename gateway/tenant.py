"""Multi-tenant rate and budget enforcement.

Per-user:
  - Token bucket for RPS (in-memory + persistent counter for persistence across restarts)
  - Concurrent request semaphore
  - Daily USD budget (persisted in usage_counters)
  - Tokens-per-minute cap

Tenant config is loaded from gateway-config.json -> tenants; users are
created lazily via memory.get_or_create_user on first request.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass

from . import memory

log = logging.getLogger("glint.tenant")


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
    bucket: _TokenBucket
    semaphore: asyncio.Semaphore
    user_lock: threading.Lock


class TenantManager:
    def __init__(self, default_config: dict, preconfigured: dict | None = None):
        self._default = default_config
        # Per-user overrides from gateway-config.json -> tenants (excl "*" / docs)
        self._preconfigured: dict[str, dict] = preconfigured or {}
        self._states: dict[str, TenantState] = {}
        self._lock = threading.Lock()

    def _build_state(self, tenant_id: str, cfg: dict) -> TenantState:
        tier_access = cfg.get("tier_access")
        if isinstance(tier_access, str):
            import json
            tier_access = json.loads(tier_access) if tier_access else []
        elif not tier_access:
            tier_access = self._default.get("tier_access", [])
        return TenantState(
            tenant_id=tenant_id,
            tier_access=tier_access,
            budget_usd_per_day=float(cfg.get("budget_usd_per_day") or self._default.get("budget_usd_per_day", 1.0)),
            rps_limit=int(cfg.get("rps_limit") or self._default.get("rps_limit", 100)),
            concurrent_limit=int(cfg.get("concurrent_limit") or self._default.get("concurrent_limit", 20)),
            tokens_per_min=int(cfg.get("tokens_per_min") or self._default.get("tokens_per_min", 200000)),
            bucket=_TokenBucket(
                rate_per_sec=int(cfg.get("rps_limit") or self._default.get("rps_limit", 100)),
                capacity=int(cfg.get("rps_limit") or self._default.get("rps_limit", 100)) * 2,
            ),
            semaphore=asyncio.Semaphore(int(cfg.get("concurrent_limit") or self._default.get("concurrent_limit", 20))),
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
            # Precedence: config-file override > DB row > defaults
            if tenant_id in self._preconfigured:
                merged = {**self._default, **self._preconfigured[tenant_id]}
                st = self._build_state(tenant_id, merged)
            else:
                db_cfg = memory.get_or_create_user(tenant_id, defaults=self._default)
                keys = ("tier_access", "budget_usd_per_day", "rps_limit", "concurrent_limit", "tokens_per_min")
                merged = {**self._default, **{k: v for k, v in db_cfg.items() if k in keys and v is not None}}
                st = self._build_state(tenant_id, merged)
            self._states[tenant_id] = st
            return st

    def refresh(self, tenant_id: str) -> TenantState:
        """Rebuild the cached state from DB/config (admin edits take effect)."""
        with self._lock:
            self._states.pop(tenant_id, None)
        return self.get_or_create(tenant_id)

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
        )
        if not ok:
            if reason and reason.startswith("daily budget"):
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

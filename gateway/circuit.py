"""Per-endpoint circuit breaker.

States: CLOSED (normal) -> OPEN (skip endpoint) -> HALF_OPEN (probe) -> CLOSED.

Persists state in DB so restarts don't reset. Thread-safe.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import UTC, datetime

from . import memory

log = logging.getLogger("ctrl.circuit")


@dataclass
class BreakerConfig:
    failure_threshold: int = 3
    open_duration_seconds: float = 60.0
    half_open_max_probes: int = 1


class CircuitBreaker:
    """One breaker per endpoint. State persisted in DB."""

    def __init__(self, endpoint_name: str, config: BreakerConfig):
        self.endpoint_name = endpoint_name
        self.config = config
        self._lock = threading.Lock()
        self._state = self._load_state()

    def _load_state(self) -> dict:
        row = memory.get_breaker_state(self.endpoint_name)
        # Auto-transition OPEN -> HALF_OPEN if duration elapsed
        if row.get("state") == "OPEN" and row.get("opened_at"):
            opened_at = row["opened_at"]
            if opened_at.tzinfo is None:
                opened_at = opened_at.replace(tzinfo=UTC)
            elapsed = (datetime.now(UTC) - opened_at).total_seconds()
            if elapsed >= self.config.open_duration_seconds:
                row["state"] = "HALF_OPEN"
                row["half_open_probes_remaining"] = self.config.half_open_max_probes
                memory.set_breaker_state(
                    self.endpoint_name,
                    state="HALF_OPEN",
                    consecutive_failures=row.get("consecutive_failures", 0),
                    opened_at=row.get("opened_at"),
                    last_failure_at=row.get("last_failure_at"),
                    last_success_at=row.get("last_success_at"),
                    half_open_probes_remaining=self.config.half_open_max_probes,
                )
        return row

    def allow(self) -> bool:
        """Should we allow a request through this endpoint right now?"""
        with self._lock:
            state = self._state.get("state", "CLOSED")
            if state == "CLOSED":
                return True
            if state == "OPEN":
                # Auto-transition to HALF_OPEN once the open duration has elapsed
                opened_at = self._state.get("opened_at")
                if opened_at is not None:
                    if opened_at.tzinfo is None:
                        opened_at = opened_at.replace(tzinfo=UTC)
                    elapsed = (datetime.now(UTC) - opened_at).total_seconds()
                    if elapsed >= self.config.open_duration_seconds:
                        remaining = self.config.half_open_max_probes
                        self._state["state"] = "HALF_OPEN"
                        self._state["half_open_probes_remaining"] = remaining - 1
                        memory.set_breaker_state(
                            self.endpoint_name,
                            state="HALF_OPEN",
                            consecutive_failures=self._state.get("consecutive_failures", 0),
                            opened_at=opened_at,
                            last_failure_at=self._state.get("last_failure_at"),
                            last_success_at=self._state.get("last_success_at"),
                            half_open_probes_remaining=remaining - 1,
                        )
                        return remaining > 0
                return False
            if state == "HALF_OPEN":
                remaining = self._state.get("half_open_probes_remaining", 0)
                if remaining > 0:
                    self._state["half_open_probes_remaining"] = remaining - 1
                    memory.set_breaker_state(
                        self.endpoint_name,
                        state="HALF_OPEN",
                        consecutive_failures=self._state.get("consecutive_failures", 0),
                        opened_at=self._state.get("opened_at"),
                        last_failure_at=self._state.get("last_failure_at"),
                        last_success_at=self._state.get("last_success_at"),
                        half_open_probes_remaining=remaining - 1,
                    )
                    return True
                # No probes left; treat as OPEN until duration elapses
                return False
            return True

    def record_success(self):
        with self._lock:
            if self._state.get("state") != "CLOSED":
                log.info("breaker %s -> CLOSED (success)", self.endpoint_name)
            self._state = {
                "endpoint_name": self.endpoint_name,
                "state": "CLOSED",
                "consecutive_failures": 0,
                "opened_at": None,
                "last_failure_at": self._state.get("last_failure_at"),
                "last_success_at": datetime.now(UTC),
                "half_open_probes_remaining": 0,
            }
            memory.set_breaker_state(
                self.endpoint_name,
                state="CLOSED",
                consecutive_failures=0,
                last_success_at=datetime.now(UTC),
                half_open_probes_remaining=0,
            )

    def record_failure(self):
        with self._lock:
            cf = self._state.get("consecutive_failures", 0) + 1
            now = datetime.now(UTC)
            if cf >= self.config.failure_threshold:
                if self._state.get("state") != "OPEN":
                    log.warning(
                        "breaker %s -> OPEN (failures=%d, threshold=%d)",
                        self.endpoint_name, cf, self.config.failure_threshold,
                    )
                self._state = {
                    "endpoint_name": self.endpoint_name,
                    "state": "OPEN",
                    "consecutive_failures": cf,
                    "opened_at": now,
                    "last_failure_at": now,
                    "last_success_at": self._state.get("last_success_at"),
                    "half_open_probes_remaining": 0,
                }
                memory.set_breaker_state(
                    self.endpoint_name,
                    state="OPEN",
                    consecutive_failures=cf,
                    opened_at=now,
                    last_failure_at=now,
                    half_open_probes_remaining=0,
                )
            else:
                self._state["consecutive_failures"] = cf
                self._state["last_failure_at"] = now
                memory.set_breaker_state(
                    self.endpoint_name,
                    state=self._state.get("state", "CLOSED"),
                    consecutive_failures=cf,
                    last_failure_at=now,
                )

    def state(self) -> str:
        with self._lock:
            return self._state.get("state", "CLOSED")


class BreakerRegistry:
    """Manages all per-endpoint breakers. Lazily creates on first use."""

    def __init__(self):
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get(self, endpoint_name: str, config: BreakerConfig) -> CircuitBreaker:
        with self._lock:
            b = self._breakers.get(endpoint_name)
            if b is None:
                b = CircuitBreaker(endpoint_name, config)
                self._breakers[endpoint_name] = b
            elif b.config != config:
                b.config = config
            return b

    def all_states(self) -> dict[str, str]:
        with self._lock:
            return {name: b.state() for name, b in self._breakers.items()}


_registry: BreakerRegistry | None = None


def init_registry() -> BreakerRegistry:
    global _registry
    _registry = BreakerRegistry()
    return _registry


def registry() -> BreakerRegistry:
    if _registry is None:
        raise RuntimeError("breaker registry not initialized")
    return _registry

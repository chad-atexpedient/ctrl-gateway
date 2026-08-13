"""Tool response cache.

LRU + TTL cache for tool and plugin responses. Keyed by a deterministic
hash of (tool_name, arguments, tenant_id). Honors:

  - default_ttl_seconds (applied when no per-tool override)
  - per_tool_ttl_seconds (overrides default)
  - max_entries (LRU eviction)
  - bypass_keys (string list — if any matches the args, skip caching)

Cache hit/miss counters are exposed via `stats()` for the dashboard and
Prometheus metrics. The cache is in-memory only (no DB persistence) — a
gateway restart drops it; this is intentional (responses may include
freshness-critical data and we don't want stale entries surviving a
restart).
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("ctrl.tool_cache")


@dataclass
class CacheEntry:
    key: str
    value: Any
    expires_at: float
    created_at: float = field(default_factory=time.time)
    tool_name: str = ""
    tenant_id: str | None = None
    hits: int = 0


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    sets: int = 0
    evictions: int = 0
    expirations: int = 0
    size: int = 0
    max_size: int = 0


def _stable_hash(*parts: Any) -> str:
    payload = json.dumps(parts, default=str, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class ToolCache:
    """Thread-safe LRU + TTL cache."""

    def __init__(
        self,
        max_entries: int = 1024,
        default_ttl_seconds: int = 300,
        per_tool_ttl_seconds: dict[str, int] | None = None,
        bypass_keys: list[str] | None = None,
    ):
        self.max_entries = max(1, max_entries)
        self.default_ttl_seconds = max(1, default_ttl_seconds)
        self.per_tool_ttl_seconds: dict[str, int] = per_tool_ttl_seconds or {}
        self.bypass_keys = [k.lower() for k in (bypass_keys or [])]
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, CacheEntry] = OrderedDict()
        self._stats = CacheStats(max_size=self.max_entries)

    def _now(self) -> float:
        return time.time()

    def _ttl_for(self, tool_name: str) -> int:
        return self.per_tool_ttl_seconds.get(tool_name, self.default_ttl_seconds)

    def _bypass_match(self, arguments: dict[str, Any]) -> bool:
        if not self.bypass_keys:
            return False
        for k, v in arguments.items():
            if any(b in str(k).lower() for b in self.bypass_keys):
                return True
            if any(b in str(v).lower() for b in self.bypass_keys):
                return True
        return False

    def _evict_expired(self) -> None:
        now = self._now()
        expired_keys = [
            k for k, entry in self._entries.items() if entry.expires_at <= now
        ]
        for k in expired_keys:
            self._entries.pop(k, None)
            self._stats.expirations += 1

    def _evict_lru_if_needed(self) -> None:
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self._stats.evictions += 1

    def get(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tenant_id: str | None = None,
    ) -> Any | None:
        """Return cached value or None on miss/expired/bypass."""
        if self._bypass_match(arguments):
            with self._lock:
                self._stats.misses += 1
            return None
        key = self._make_key(tool_name, arguments, tenant_id)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._stats.misses += 1
                return None
            if entry.expires_at <= self._now():
                self._entries.pop(key, None)
                self._stats.expirations += 1
                self._stats.misses += 1
                return None
            entry.hits += 1
            self._stats.hits += 1
            # Mark as recently used
            self._entries.move_to_end(key)
            return entry.value

    def set(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        value: Any,
        tenant_id: str | None = None,
        ttl_seconds: int | None = None,
    ) -> str | None:
        """Store a value. Returns the cache key, or None if bypassed."""
        if self._bypass_match(arguments):
            return None
        key = self._make_key(tool_name, arguments, tenant_id)
        effective_ttl = ttl_seconds if ttl_seconds is not None else self._ttl_for(tool_name)
        now = self._now()
        with self._lock:
            self._evict_expired()
            self._entries[key] = CacheEntry(
                key=key,
                value=value,
                expires_at=now + max(1, effective_ttl),
                tool_name=tool_name,
                tenant_id=tenant_id,
            )
            self._entries.move_to_end(key)
            self._evict_lru_if_needed()
            self._stats.sets += 1
            self._stats.size = len(self._entries)
        return key

    def _make_key(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        tenant_id: str | None,
    ) -> str:
        return _stable_hash(tool_name, arguments, tenant_id)

    def invalidate(self, tool_name: str | None = None) -> int:
        """Invalidate all entries, or all entries for a specific tool.
        Returns number of entries removed."""
        with self._lock:
            if tool_name is None:
                count = len(self._entries)
                self._entries.clear()
                self._stats.size = 0
                return count
            keys_to_remove = [
                k for k, entry in self._entries.items() if entry.tool_name == tool_name
            ]
            for k in keys_to_remove:
                self._entries.pop(k, None)
            self._stats.size = len(self._entries)
            return len(keys_to_remove)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            self._stats.size = len(self._entries)
            return {
                "hits": self._stats.hits,
                "misses": self._stats.misses,
                "sets": self._stats.sets,
                "evictions": self._stats.evictions,
                "expirations": self._stats.expirations,
                "size": self._stats.size,
                "max_size": self._stats.max_size,
                "hit_rate": (
                    self._stats.hits / (self._stats.hits + self._stats.misses)
                    if (self._stats.hits + self._stats.misses) > 0
                    else 0.0
                ),
            }

    def snapshot(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return a small summary of cache entries (tool_name, tenant_id,
        age, hits, expires_in). Useful for dashboards."""
        now = self._now()
        out: list[dict[str, Any]] = []
        with self._lock:
            for entry in list(self._entries.values())[:limit]:
                out.append(
                    {
                        "key": entry.key,
                        "tool_name": entry.tool_name,
                        "tenant_id": entry.tenant_id,
                        "age_seconds": now - entry.created_at,
                        "expires_in_seconds": max(0.0, entry.expires_at - now),
                        "hits": entry.hits,
                    }
                )
        return out


_default_cache: ToolCache | None = None


def init_cache(
    max_entries: int = 1024,
    default_ttl_seconds: int = 300,
    per_tool_ttl_seconds: dict[str, int] | None = None,
    bypass_keys: list[str] | None = None,
) -> ToolCache:
    global _default_cache
    _default_cache = ToolCache(
        max_entries=max_entries,
        default_ttl_seconds=default_ttl_seconds,
        per_tool_ttl_seconds=per_tool_ttl_seconds,
        bypass_keys=bypass_keys,
    )
    return _default_cache


def cache() -> ToolCache | None:
    return _default_cache

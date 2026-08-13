"""Model discovery and sync engine.

Discovers available models from all configured providers and populates the
`model_catalog` table. Supports:

  - OpenRouter (`/api/v1/models`) — aggregates 300+ models from all major
    providers with per-token pricing, context length, modality, and
    capability metadata.
  - OpenAI-compatible (`/v1/models`) — OpenAI direct, Groq, Together,
    Fireworks, Mistral, xAI, DeepSeek, local vLLM/LMStudio, etc.
  - Anthropic (`/v1/models`) — Claude model family.
  - Ollama (`/api/tags`) — locally installed models.

The sync engine normalizes all provider responses into a common shape and
computes a `capability_score` (0.0–1.0) used for:
  - Tier auto-assignment (Phase 2)
  - Spidergraph visualization (Phase 3)
  - Cost-first routing enrichment

Usage:
    from gateway.model_sync import sync_all, init_sync_engine

    engine = init_sync_engine(conf)
    summary = await sync_all()           # one-shot sync
    await engine.start_sync_loop()       # background periodic sync

The capability score is a weighted blend of:
  - context_length (log-scaled, capped at 200K)
  - supports_tools (+0.15)
  - supports_vision (+0.10)
  - supports_reasoning (+0.20)
  - inverse pricing (cheaper = higher score, log-scaled)

This is deliberately transparent and tunable — not a black box.
"""
from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

import aiohttp

from . import memory, ssrf

log = logging.getLogger("ctrl.model_sync")

# Default sync interval: 6 hours. New models don't appear every minute.
DEFAULT_SYNC_INTERVAL_SECONDS = 6 * 60 * 60
DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass
class SyncSummary:
    """Result of a single sync run."""
    provider: str
    discovered: int = 0
    added: int = 0
    updated: int = 0
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "discovered": self.discovered,
            "added": self.added,
            "updated": self.updated,
            "errors": self.errors,
            "duration_ms": round(self.duration_ms, 1),
        }


# =====================================================================
# Capability scoring
# =====================================================================


def compute_capability_score(
    context_length: int | None,
    supports_tools: bool,
    supports_vision: bool,
    supports_reasoning: bool,
    pricing_prompt_per_1k: float | None,
    pricing_completion_per_1k: float | None,
) -> float:
    """Compute a 0.0–1.0 capability score for a model.

    Weighted blend of context window, features, and inverse cost.
    Transparent and tunable — not a black box.
    """
    score = 0.0
    # Context length: log-scaled, capped at 200K. A 200K context model
    # gets the full 0.30; a 4K model gets ~0.17; a 1M model also gets 0.30
    # (diminishing returns past 200K for most workloads).
    if context_length and context_length > 0:
        cl = min(context_length, 200_000)
        score += 0.10 + 0.20 * (math.log2(max(cl, 1024)) / math.log2(200_000))
    else:
        score += 0.05  # unknown context — conservative
    # Feature bonuses
    if supports_tools:
        score += 0.15
    if supports_vision:
        score += 0.10
    if supports_reasoning:
        score += 0.20
    # Inverse pricing: cheaper models score higher. Uses the average of
    # input + output per-1K-token pricing. A free model gets the full 0.25;
    # a $10/M-token model gets ~0.10; a $75/M-token model gets ~0.05.
    if pricing_prompt_per_1k is not None and pricing_completion_per_1k is not None:
        avg_cost = (pricing_prompt_per_1k + pricing_completion_per_1k) / 2.0
        if avg_cost <= 0:
            score += 0.25
        else:
            # log-scaled inverse: 0.001/1K (cheap) → ~0.22, 0.01/1K → ~0.15, 0.075/1K → ~0.05
            score += max(0.0, 0.25 * (1.0 - math.log10(avg_cost * 1000 + 1) / 3.0))
    else:
        score += 0.10  # unknown pricing — neutral
    return round(min(max(score, 0.0), 1.0), 4)


# =====================================================================
# Provider adapters — each normalizes a provider's API response into
# the common model_catalog shape.
# =====================================================================


def _parse_openrouter_model(m: dict) -> dict:
    """Parse a single OpenRouter model entry into catalog shape."""
    pricing = m.get("pricing") or {}
    arch = m.get("architecture") or {}
    top = m.get("top_provider") or {}
    # OpenRouter pricing is per-token; convert to per-1K.
    prompt_per_token = float(pricing.get("prompt") or 0)
    completion_per_token = float(pricing.get("completion") or 0)
    cached_per_token = float(pricing.get("input_cache_read") or 0)
    input_mods = arch.get("input_modalities") or ["text"]
    output_mods = arch.get("output_modalities") or ["text"]
    supported_params = m.get("supported_parameters") or []
    reasoning = m.get("reasoning") or {}
    return {
        "model_id": m.get("id", ""),
        "provider": "openrouter",
        "display_name": m.get("name", m.get("id", "")),
        "description": (m.get("description") or "")[:2000],
        "context_length": m.get("context_length") or top.get("context_length"),
        "max_completion_tokens": top.get("max_completion_tokens"),
        "modality": arch.get("modality", "text->text"),
        "input_modalities": input_mods,
        "output_modalities": output_mods,
        "pricing_prompt_per_1k": round(prompt_per_token * 1000, 8),
        "pricing_completion_per_1k": round(completion_per_token * 1000, 8),
        "pricing_cached_per_1k": round(cached_per_token * 1000, 8) if cached_per_token else None,
        "supports_tools": "tools" in supported_params or "tool_choice" in supported_params,
        "supports_reasoning": bool(reasoning.get("mandatory") or reasoning.get("default_enabled")),
        "supports_vision": "image" in input_mods,
        "supported_parameters": supported_params,
        "raw_metadata": m,
    }


def _parse_openai_model(m: dict, provider_name: str) -> dict:
    """Parse an OpenAI-compatible /v1/models entry."""
    model_id = m.get("id", "")
    # OpenAI's /v1/models doesn't include pricing/context — those come from
    # the preset defaults or are left null for the admin to fill in.
    return {
        "model_id": model_id,
        "provider": provider_name,
        "display_name": model_id,
        "description": "",
        "context_length": m.get("context_length") or m.get("context_window"),
        "max_completion_tokens": m.get("max_completion_tokens"),
        "modality": m.get("modality", "text->text"),
        "input_modalities": m.get("input_modalities", ["text"]),
        "output_modalities": m.get("output_modalities", ["text"]),
        "pricing_prompt_per_1k": m.get("pricing", {}).get("prompt_per_1k") if isinstance(m.get("pricing"), dict) else None,
        "pricing_completion_per_1k": m.get("pricing", {}).get("completion_per_1k") if isinstance(m.get("pricing"), dict) else None,
        "pricing_cached_per_1k": None,
        "supports_tools": m.get("supports_tool_calls", m.get("supports_function_calling", False)),
        "supports_reasoning": "reasoning" in model_id.lower() or "o1" in model_id.lower() or "o3" in model_id.lower() or "o4" in model_id.lower(),
        "supports_vision": m.get("supports_vision", False),
        "supported_parameters": m.get("supported_parameters", []),
        "raw_metadata": m,
    }


def _parse_ollama_model(m: dict) -> dict:
    """Parse an Ollama /api/tags entry."""
    name = m.get("name", "")
    details = m.get("details") or {}
    return {
        "model_id": name,
        "provider": "ollama",
        "display_name": details.get("name", name),
        "description": f"{details.get('family', 'unknown')} · {details.get('parameter_size', '?')} · {details.get('quantization_level', '?')}",
        "context_length": m.get("context_length") or 32768,
        "max_completion_tokens": None,
        "modality": "text->text" if "vision" not in name.lower() and "llava" not in name.lower() else "text->image",
        "input_modalities": ["text"],
        "output_modalities": ["text"],
        "pricing_prompt_per_1k": 0.0,  # local = free
        "pricing_completion_per_1k": 0.0,
        "pricing_cached_per_1k": 0.0,
        "supports_tools": details.get("family") in ("llama", "qwen2"),
        "supports_reasoning": False,
        "supports_vision": "vision" in name.lower() or "llava" in name.lower(),
        "supported_parameters": ["temperature", "top_p", "stop"],
        "raw_metadata": m,
    }


def _parse_anthropic_model(m: dict) -> dict:
    """Parse an Anthropic /v1/models entry."""
    model_id = m.get("id", "")
    return {
        "model_id": model_id,
        "provider": "anthropic",
        "display_name": m.get("display_name", model_id),
        "description": (m.get("description") or "")[:2000],
        "context_length": m.get("max_context_tokens") or 200000,
        "max_completion_tokens": m.get("max_output_tokens"),
        "modality": "text->text",
        "input_modalities": m.get("input_modalities", ["text"]),
        "output_modalities": ["text"],
        "pricing_prompt_per_1k": None,  # Anthropic /v1/models doesn't include pricing
        "pricing_completion_per_1k": None,
        "pricing_cached_per_1k": None,
        "supports_tools": True,  # all Claude 3+ models support tools
        "supports_reasoning": "thinking" in model_id.lower() or "reasoning" in model_id.lower(),
        "supports_vision": "vision" in model_id.lower() or "sonnet" in model_id.lower() or "opus" in model_id.lower(),
        "supported_parameters": ["temperature", "top_p", "top_k", "stop_sequences", "max_tokens"],
        "raw_metadata": m,
    }


# =====================================================================
# Sync engine
# =====================================================================


class ModelSyncEngine:
    """Discovers available models from all configured providers.

    Created by init_sync_engine() with the gateway config. The config's
    `model_sync` section controls which providers are probed, the sync
    interval, and whether the background loop is enabled.

    One-shot sync: `await engine.sync_all()`.
    Background loop: `engine.start_sync_loop()`.
    """

    def __init__(
        self,
        openrouter_enabled: bool = True,
        openai_base_url: str | None = None,
        openai_api_key_env: str = "OPENAI_API_KEY",
        anthropic_api_key_env: str = "ANTHROPIC_API_KEY",
        anthropic_enabled: bool = False,
        ollama_base_url: str = "http://localhost:11434",
        ollama_enabled: bool = True,
        openai_compatible_providers: list[dict] | None = None,
        sync_interval_seconds: int = DEFAULT_SYNC_INTERVAL_SECONDS,
        auto_sync: bool = True,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ):
        self.openrouter_enabled = openrouter_enabled
        self.openai_base_url = openai_base_url
        self.openai_api_key_env = openai_api_key_env
        self.anthropic_api_key_env = anthropic_api_key_env
        self.anthropic_enabled = anthropic_enabled
        self.ollama_base_url = ollama_base_url
        self.ollama_enabled = ollama_enabled
        # Each entry: {"name": "groq", "base_url": "https://api.groq.com/openai/v1", "api_key_env": "GROQ_API_KEY"}
        self.openai_compatible_providers = openai_compatible_providers or []
        self.sync_interval_seconds = sync_interval_seconds
        self.auto_sync = auto_sync
        self.timeout_seconds = timeout_seconds
        self._sync_task: asyncio.Task | None = None

    async def _fetch_json(
        self, url: str, headers: dict | None = None
    ) -> dict | None:
        """Fetch JSON from a URL with SSRF protection."""
        try:
            ssrf.validate_url(url, allow_localhost=True, allow_private=True)
        except ssrf.SSRFBlockedURL as e:
            log.warning("model_sync SSRF blocked %s: %s", url, e.reason)
            return None
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_seconds)
            # connector re-checks the SSRF policy at actual connect time,
            # closing the TOCTOU/DNS-rebinding gap the validate_url() call
            # above can't close on its own (see ssrf.SSRFSafeResolver).
            connector = ssrf.safe_connector(allow_localhost=True, allow_private=True)
            async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
                async with session.get(url, headers=headers or {}) as resp:
                    if resp.status != 200:
                        log.warning(
                            "model_sync %s returned HTTP %d", url, resp.status
                        )
                        return None
                    return await resp.json()
        except Exception as e:
            log.warning("model_sync fetch %s failed: %s", url, e)
            return None

    async def sync_openrouter(self) -> SyncSummary:
        """Pull all models from OpenRouter /api/v1/models."""
        summary = SyncSummary(provider="openrouter")
        if not self.openrouter_enabled:
            return summary
        started = time.monotonic()
        data = await self._fetch_json("https://openrouter.ai/api/v1/models")
        if not data or "data" not in data:
            summary.errors.append("no data field in OpenRouter response")
            summary.duration_ms = (time.monotonic() - started) * 1000
            return summary
        models = data["data"]
        summary.discovered = len(models)
        for m in models:
            try:
                parsed = _parse_openrouter_model(m)
                self._upsert_parsed(parsed)
                summary.added += 1
            except Exception as e:
                summary.errors.append(f"{m.get('id', '?')}: {e}")
        summary.duration_ms = (time.monotonic() - started) * 1000
        log.info(
            "OpenRouter sync: %d models discovered, %d added, %d errors, %.0fms",
            summary.discovered, summary.added, len(summary.errors), summary.duration_ms,
        )
        return summary

    async def sync_openai(self) -> SyncSummary:
        """Pull models from an OpenAI-compatible /v1/models endpoint."""
        provider_name = "openai"
        base_url = self.openai_base_url or "https://api.openai.com/v1"
        summary = SyncSummary(provider=provider_name)
        api_key = os.environ.get(self.openai_api_key_env, "")
        if not api_key:
            summary.errors.append(f"env var {self.openai_api_key_env} not set")
            return summary
        started = time.monotonic()
        url = f"{base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = await self._fetch_json(url, headers=headers)
        if not data or "data" not in data:
            summary.errors.append("no data field in response")
            summary.duration_ms = (time.monotonic() - started) * 1000
            return summary
        models = data["data"]
        summary.discovered = len(models)
        for m in models:
            try:
                parsed = _parse_openai_model(m, provider_name)
                self._upsert_parsed(parsed)
                summary.added += 1
            except Exception as e:
                summary.errors.append(f"{m.get('id', '?')}: {e}")
        summary.duration_ms = (time.monotonic() - started) * 1000
        log.info("%s sync: %d models, %d added", provider_name, summary.discovered, summary.added)
        return summary

    async def sync_openai_compatible(self, name: str, base_url: str, api_key_env: str) -> SyncSummary:
        """Sync from any OpenAI-compatible provider (Groq, Together, etc.)."""
        summary = SyncSummary(provider=name)
        api_key = os.environ.get(api_key_env, "")
        if not api_key:
            summary.errors.append(f"env var {api_key_env} not set")
            return summary
        started = time.monotonic()
        url = f"{base_url.rstrip('/')}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = await self._fetch_json(url, headers=headers)
        if not data or "data" not in data:
            summary.errors.append("no data field in response")
            summary.duration_ms = (time.monotonic() - started) * 1000
            return summary
        models = data["data"]
        summary.discovered = len(models)
        for m in models:
            try:
                parsed = _parse_openai_model(m, name)
                self._upsert_parsed(parsed)
                summary.added += 1
            except Exception as e:
                summary.errors.append(f"{m.get('id', '?')}: {e}")
        summary.duration_ms = (time.monotonic() - started) * 1000
        log.info("%s sync: %d models, %d added", name, summary.discovered, summary.added)
        return summary

    async def sync_anthropic(self) -> SyncSummary:
        """Pull models from Anthropic /v1/models."""
        summary = SyncSummary(provider="anthropic")
        if not self.anthropic_enabled:
            return summary
        api_key = os.environ.get(self.anthropic_api_key_env, "")
        if not api_key:
            summary.errors.append(f"env var {self.anthropic_api_key_env} not set")
            return summary
        started = time.monotonic()
        url = "https://api.anthropic.com/v1/models"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        data = await self._fetch_json(url, headers=headers)
        if not data or "data" not in data:
            summary.errors.append("no data field in response")
            summary.duration_ms = (time.monotonic() - started) * 1000
            return summary
        models = data["data"]
        summary.discovered = len(models)
        for m in models:
            try:
                parsed = _parse_anthropic_model(m)
                self._upsert_parsed(parsed)
                summary.added += 1
            except Exception as e:
                summary.errors.append(f"{m.get('id', '?')}: {e}")
        summary.duration_ms = (time.monotonic() - started) * 1000
        log.info("Anthropic sync: %d models, %d added", summary.discovered, summary.added)
        return summary

    async def sync_ollama(self) -> SyncSummary:
        """Pull installed models from local Ollama /api/tags."""
        summary = SyncSummary(provider="ollama")
        if not self.ollama_enabled:
            return summary
        started = time.monotonic()
        url = f"{self.ollama_base_url.rstrip('/')}/api/tags"
        data = await self._fetch_json(url)
        if not data or "models" not in data:
            summary.errors.append("no models field in Ollama response")
            summary.duration_ms = (time.monotonic() - started) * 1000
            return summary
        models = data["models"]
        summary.discovered = len(models)
        for m in models:
            try:
                parsed = _parse_ollama_model(m)
                self._upsert_parsed(parsed)
                summary.added += 1
            except Exception as e:
                summary.errors.append(f"{m.get('name', '?')}: {e}")
        summary.duration_ms = (time.monotonic() - started) * 1000
        log.info("Ollama sync: %d models, %d added", summary.discovered, summary.added)
        return summary

    def _upsert_parsed(self, parsed: dict) -> None:
        """Compute capability score and upsert into model_catalog."""
        cap_score = compute_capability_score(
            context_length=parsed.get("context_length"),
            supports_tools=parsed.get("supports_tools", False),
            supports_vision=parsed.get("supports_vision", False),
            supports_reasoning=parsed.get("supports_reasoning", False),
            pricing_prompt_per_1k=parsed.get("pricing_prompt_per_1k"),
            pricing_completion_per_1k=parsed.get("pricing_completion_per_1k"),
        )
        parsed["capability_score"] = cap_score
        # Auto-assign tier based on capability score.
        parsed["tier_assignment"] = _tier_from_capability(cap_score)
        memory.upsert_model_catalog_entry(
            model_id=parsed["model_id"],
            provider=parsed["provider"],
            display_name=parsed.get("display_name", parsed["model_id"]),
            description=parsed.get("description", ""),
            context_length=parsed.get("context_length"),
            max_completion_tokens=parsed.get("max_completion_tokens"),
            modality=parsed.get("modality", "text->text"),
            input_modalities=parsed.get("input_modalities"),
            output_modalities=parsed.get("output_modalities"),
            pricing_prompt_per_1k=parsed.get("pricing_prompt_per_1k"),
            pricing_completion_per_1k=parsed.get("pricing_completion_per_1k"),
            pricing_cached_per_1k=parsed.get("pricing_cached_per_1k"),
            supports_tools=parsed.get("supports_tools", False),
            supports_reasoning=parsed.get("supports_reasoning", False),
            supports_vision=parsed.get("supports_vision", False),
            supported_parameters=parsed.get("supported_parameters"),
            capability_score=cap_score,
            tier_assignment=parsed["tier_assignment"],
            raw_metadata=parsed.get("raw_metadata"),
            enabled=True,
        )

    async def sync_all(self) -> list[SyncSummary]:
        """Run all enabled provider syncs concurrently."""
        tasks: list[Any] = []
        if self.openrouter_enabled:
            tasks.append(self.sync_openrouter())
        if self.openai_base_url or self.openai_api_key_env != "OPENAI_API_KEY" or os.environ.get("OPENAI_API_KEY"):
            tasks.append(self.sync_openai())
        if self.anthropic_enabled:
            tasks.append(self.sync_anthropic())
        if self.ollama_enabled:
            tasks.append(self.sync_ollama())
        for p in self.openai_compatible_providers:
            tasks.append(self.sync_openai_compatible(
                name=p.get("name", "unknown"),
                base_url=p.get("base_url", ""),
                api_key_env=p.get("api_key_env", ""),
            ))
        if not tasks:
            log.info("model_sync: no providers enabled")
            return []
        results = await asyncio.gather(*tasks, return_exceptions=True)
        summaries: list[SyncSummary] = []
        for r in results:
            if isinstance(r, SyncSummary):
                summaries.append(r)
            elif isinstance(r, Exception):
                log.warning("model_sync provider error: %s", r)
                s = SyncSummary(provider="unknown")
                s.errors.append(str(r))
                summaries.append(s)
        # Record sync log
        total_discovered = sum(s.discovered for s in summaries)
        total_added = sum(s.added for s in summaries)
        memory.record_contextforge_sync(
            sync_type="model_catalog",
            source=",".join(s.provider for s in summaries),
            items_synced=total_discovered,
            items_added=total_added,
            items_updated=0,
            errors=[e for s in summaries for e in s.errors],
            duration_ms=sum(s.duration_ms for s in summaries),
        )
        return summaries

    async def sync_loop(self) -> None:
        """Background task: periodic re-sync."""
        while True:
            try:
                await self.sync_all()
            except Exception as e:
                log.warning("model_sync loop error: %s", e)
            await asyncio.sleep(self.sync_interval_seconds)

    def start_sync_loop(self) -> None:
        if self._sync_task is not None and not self._sync_task.done():
            return
        if not self.auto_sync:
            return
        try:
            loop = asyncio.get_running_loop()
            self._sync_task = loop.create_task(self.sync_loop())
        except RuntimeError:
            log.debug("no running event loop; model_sync loop not started")

    def stop_sync_loop(self) -> None:
        if self._sync_task is not None and not self._sync_task.done():
            self._sync_task.cancel()
            self._sync_task = None


def _tier_from_capability(score: float) -> str:
    """Auto-assign a tier name based on capability score.

    tier0 = weakest/cheapest (score < 0.35)
    tier1 = budget (0.35–0.50)
    tier2 = mid-range (0.50–0.65)
    tier3 = strong (0.65–0.80)
    tier4 = frontier (>= 0.80)
    """
    if score >= 0.80:
        return "tier4"
    if score >= 0.65:
        return "tier3"
    if score >= 0.50:
        return "tier2"
    if score >= 0.35:
        return "tier1"
    return "tier0"


# =====================================================================
# Module-level singleton
# =====================================================================


_default_engine: ModelSyncEngine | None = None


def init_sync_engine(
    openrouter_enabled: bool = True,
    openai_base_url: str | None = None,
    openai_api_key_env: str = "OPENAI_API_KEY",
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY",
    anthropic_enabled: bool = False,
    ollama_base_url: str = "http://localhost:11434",
    ollama_enabled: bool = True,
    openai_compatible_providers: list[dict] | None = None,
    sync_interval_seconds: int = DEFAULT_SYNC_INTERVAL_SECONDS,
    auto_sync: bool = True,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> ModelSyncEngine:
    global _default_engine
    _default_engine = ModelSyncEngine(
        openrouter_enabled=openrouter_enabled,
        openai_base_url=openai_base_url,
        openai_api_key_env=openai_api_key_env,
        anthropic_api_key_env=anthropic_api_key_env,
        anthropic_enabled=anthropic_enabled,
        ollama_base_url=ollama_base_url,
        ollama_enabled=ollama_enabled,
        openai_compatible_providers=openai_compatible_providers,
        sync_interval_seconds=sync_interval_seconds,
        auto_sync=auto_sync,
        timeout_seconds=timeout_seconds,
    )
    return _default_engine


def engine() -> ModelSyncEngine | None:
    return _default_engine


async def sync_all() -> list[SyncSummary]:
    """Convenience: run sync on the default engine."""
    eng = engine()
    if eng is None:
        return []
    return await eng.sync_all()


# =====================================================================
# Phase 2: Catalog → Endpoint auto-registration
# =====================================================================

# Provider templates define how a catalog entry maps to an endpoint config.
# The catalog stores model metadata; the template provides the transport
# details (kind, base_url, auth). This is what breaks the 1:1
# endpoint:model coupling: one provider template + N catalog entries =
# N auto-created endpoints sharing the same transport.
PROVIDER_TEMPLATES: dict[str, dict] = {
    "openrouter": {
        "kind": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
    },
    "openai": {
        "kind": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
    },
    "anthropic": {
        "kind": "anthropic",
        "base_url": "https://api.anthropic.com",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "google": {
        "kind": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "api_key_env": "GOOGLE_API_KEY",
    },
    "groq": {
        "kind": "openai",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
    },
    "together": {
        "kind": "openai",
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
    },
    "deepseek": {
        "kind": "openai",
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "mistral": {
        "kind": "openai",
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
    },
    "ollama": {
        "kind": "ollama",
        "base_url": "http://localhost:11434",
        "api_key_env": "",
    },
    "xai": {
        "kind": "openai",
        "base_url": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
    },
}


def endpoint_name_for(model_id: str, provider: str) -> str:
    """Generate a unique, deterministic endpoint name from a catalog entry.

    Convention: sanitize the model_id to lowercase alphanumerics + hyphens.
    e.g. "openai/gpt-4o" from openrouter → "openai_gpt-4o"
         "llama3:8b" from ollama → "llama3-8b"
    """
    safe = model_id.replace("/", "_").replace(":", "-").replace(".", "-")
    return safe.lower()[:64]


def catalog_to_endpoint_config(entry: dict) -> dict | None:
    """Convert a model_catalog row to a gateway endpoint config dict.

    Returns None if the provider is not in PROVIDER_TEMPLATES (unknown
    transport). The returned dict is suitable for OverlayManager.add_endpoint.
    """
    provider = entry.get("provider", "")
    template = PROVIDER_TEMPLATES.get(provider)
    if template is None:
        return None
    prompt_price = entry.get("pricing_prompt_per_1k")
    completion_price = entry.get("pricing_completion_per_1k")
    return {
        "name": endpoint_name_for(entry["model_id"], provider),
        "kind": template["kind"],
        "base_url": template["base_url"],
        "api_key_env": template["api_key_env"],
        "model_alias": entry["model_id"],
        "max_context": entry.get("context_length") or 32768,
        "concurrency": 4,
        "pricing": {
            "in_per_1k_tokens": float(prompt_price) if prompt_price is not None else 0.0,
            "out_per_1k_tokens": float(completion_price) if completion_price is not None else 0.0,
            "fixed_per_request": 0.0,
        },
        "breaker": {
            "failure_threshold": 3,
            "open_duration_seconds": 60,
            "half_open_max_probes": 1,
        },
        "health_probe": "/health",
        # Custom fields for traceability
        "_catalog_provider": provider,
        "_catalog_model_id": entry["model_id"],
        "_capability_score": entry.get("capability_score"),
    }


def filter_catalog_for_registration(
    min_score: float = 0.0,
    provider: str | None = None,
    supports_tools: bool | None = None,
    supports_vision: bool | None = None,
    supports_reasoning: bool | None = None,
    min_context_length: int | None = None,
    limit: int = 100,
) -> list[dict]:
    """Read the catalog and return entries suitable for endpoint registration.

    Filters out entries whose provider is not in PROVIDER_TEMPLATES (can't
    create an endpoint without a transport template).
    """
    entries = memory.list_model_catalog(
        provider=provider,
        enabled_only=True,
        min_capability_score=min_score if min_score > 0 else None,
        supports_tools=supports_tools,
        supports_vision=supports_vision,
        supports_reasoning=supports_reasoning,
        min_context_length=min_context_length,
        limit=limit,
    )
    return [
        e for e in entries
        if e.get("provider") in PROVIDER_TEMPLATES
    ]


def build_registration_plan(
    entries: list[dict],
    existing_endpoint_names: set[str],
) -> dict:
    """Given catalog entries + existing endpoint names, return a registration
    plan with what would be created vs. skipped.

    Returns:
        {
            "to_create": [{endpoint_config, catalog_entry, tier}],
            "already_exists": [{name, model_id, provider}],
            "no_template": [{model_id, provider}],
        }
    """
    to_create: list[dict] = []
    already: list[dict] = []
    no_template: list[dict] = []
    for entry in entries:
        cfg = catalog_to_endpoint_config(entry)
        if cfg is None:
            no_template.append({
                "model_id": entry["model_id"],
                "provider": entry.get("provider", ""),
            })
            continue
        ep_name = cfg["name"]
        if ep_name in existing_endpoint_names:
            already.append({
                "name": ep_name,
                "model_id": entry["model_id"],
                "provider": entry.get("provider", ""),
            })
            continue
        to_create.append({
            "endpoint_config": cfg,
            "catalog_entry": {
                "model_id": entry["model_id"],
                "provider": entry.get("provider", ""),
                "display_name": entry.get("display_name", entry["model_id"]),
                "capability_score": entry.get("capability_score"),
            },
            "tier": entry.get("tier_assignment") or _tier_from_capability(
                entry.get("capability_score") or 0.0
            ),
        })
    return {
        "to_create": to_create,
        "already_exists": already,
        "no_template": no_template,
    }

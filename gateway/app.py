"""Glint-V2 gateway aiohttp application.

Wires together:
  - config (hot reload)
  - router (embedding + heads)
  - policy (cost-first + escalation)
  - endpoints (HTTP clients + breakers)
  - transcoder (per-endkind payload)
  - circuit (per-endpoint breakers)
  - tenant (rate/budget)
  - security (injection detection)
  - reviewer (async queue)
  - trainer (auto-retrain)

Endpoints: /v1/chat/completions (OpenAI-compatible), /v1/models, /stats,
/config, /reload, /trace, /feedback, /accuracy, /export, /memory, /verticals,
/cost, /review-stats, /retrain, /registry, /docs/model-card/<version>,
/admin/users, /admin/users/<id>/budget, /admin/users/<id>/stats, /admin/flags,
/admin/security/* (events, stats, injection-profiles, provider-allowlist,
  host firewall sync, status, test),
/health, /dashboard.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from aiohttp import web

from . import a2a_registry as a2a_mod
from . import admin as admin_mod
from . import auth as auth_mod
from . import (
    circuit,
    discovery,
    endpoints,
    events,
    memory,
    observer_worker,
    reviewer,
    security,
    tenant,
    trainer_worker,
    transcoder,
    translation,
)
from . import config as cfg_mod
from . import contextforge_client as cf_mod
from . import firewall as firewall_mod
from . import mcp_discovery as mcp_disc_mod
from . import mcp_facade as mcp_facade_mod
from . import memory_observational as om
from . import metrics as metrics_mod
from . import model_sync as model_sync_mod
from . import ood as ood_mod
from . import plugin as plugin_mod
from . import policy as policy_mod
from . import prompt_registry as prompt_mod
from . import router as router_mod
from . import swarm as swarm_mod
from . import tool_cache as tool_cache_mod
from . import webhook_dispatcher as webhook_mod

log = logging.getLogger("glint.app")

INJECTION_PATTERNS: list = []
INJECTION_PROFILES: list = []  # list[security.InjectionProfile]
FIREWALL: firewall_mod.DomainAllowlistEnforcer | None = None
HOST_FIREWALL: firewall_mod.HostFirewallManager | None = None
PLUGIN_LOADER: plugin_mod.PluginLoader | None = None
TOOL_CACHE: tool_cache_mod.ToolCache | None = None
WEBHOOK_DISPATCHER: webhook_mod.WebhookDispatcher | None = None
CONTEXTFORGE_CLIENT: cf_mod.ContextForgeClient | None = None
MCP_DISCOVERY_TASK: asyncio.Task | None = None
MODEL_SYNC_ENGINE: model_sync_mod.ModelSyncEngine | None = None


async def init_app(conf_path: str = "./gateway-config.json") -> web.Application:
    """Build the aiohttp app with all subsystems initialized."""
    conf_mgr = cfg_mod.init(conf_path)
    conf = conf_mgr.current()

    # Memory (DB)
    memory.init_engine(conf.config["db_url"])

    # Router (stub at start; auto-loaded if ONNX exists)
    vertical_names = [v["name"] for v in conf.verticals()]
    rt = router_mod.init_router(vertical_names)
    onnx_path = conf.config.get("embedding", {}).get("onnx_path")
    heads_path = onnx_path.replace("model.onnx", "heads.npz") if onnx_path else None
    if onnx_path and Path(onnx_path).exists() and heads_path and Path(heads_path).exists():
        loaded = rt.try_load_real(
            onnx_path=onnx_path,
            heads_path=heads_path,
            vertical_names=vertical_names,
            calibration_temperature=conf.policy.get("calibration", {}).get("temperature", 1.0),
            checksum_sha256=conf.config.get("embedding", {}).get("checksum_sha256") or None,
            tokenizer_source=conf.config.get("embedding", {}).get("model_id"),
        )
        if loaded:
            memory.register_model_version(
                version_id=rt.model_version(),
                parent_id=None,
                embedding_model=conf.config.get("embedding", {}).get("model_id", "unknown"),
                heads_hash=hashlib.sha256(Path(heads_path).read_bytes()).hexdigest()[:16],
            )

    # Breaker registry
    circuit.init_registry()

    # Endpoints pool
    pool = endpoints.init_pool()
    await pool.rebuild(conf)

    # Security hub: in-process firewall + host firewall manager.
    # Both default to disabled. The in-process firewall is built from
    # gateway-config.json security.provider_allowlist (and DB rows can be
    # layered in via /admin/security/*).
    fw_config = conf.config.get("security", {}).get("provider_allowlist", {}) or {}
    enforcer = firewall_mod.DomainAllowlistEnforcer(
        enabled=bool(fw_config.get("enabled", False)),
        default_action=fw_config.get("default_action", "block"),
    )
    enforcer.load_from_config(fw_config)
    # Layer DB-backed rules on top
    try:
        db_rules = await asyncio.to_thread(memory.list_provider_allowlist)
    except Exception:
        db_rules = []
    enforcer.load_from_db(db_rules)
    pool.set_firewall(enforcer)
    global FIREWALL
    FIREWALL = enforcer

    # Host firewall (off by default; needs admin/root)
    host_fw_config = fw_config.get("host_firewall", {}) or {}
    host_fw = firewall_mod.HostFirewallManager(
        enabled=bool(host_fw_config.get("enabled", False)),
        platform=host_fw_config.get("platform", "auto"),
        persist_on_shutdown=bool(host_fw_config.get("persist_on_shutdown", False)),
    )
    global HOST_FIREWALL
    HOST_FIREWALL = host_fw
    if host_fw.enabled:
        # Sync once at startup. Failures are logged but never block startup.
        try:
            # Collect patterns from the enforcer's rules
            patterns = [r.pattern for r in enforcer.list_rules() if r.tenant_id == "*"]
            host_fw.sync(patterns)
        except Exception as e:
            log.warning("host firewall sync on startup failed: %s", e)

    # Tenant manager — default from tenants["*"], per-user overrides from the rest
    tenants_cfg = conf.config.get("tenants", {}) or {}
    default_tenant_cfg = tenants_cfg.get("*") or {
        "tier_access": ["tier0", "tier1", "tier2"],
        "budget_usd_per_day": 1.0,
        "rps_limit": 100,
        "concurrent_limit": 20,
        "tokens_per_min": 200000,
    }
    preconfigured = {k: v for k, v in tenants_cfg.items() if k != "*" and not str(k).startswith("_")}
    tenant.init_manager(default_tenant_cfg, preconfigured=preconfigured)

    # Security patterns + injection profiles
    global INJECTION_PATTERNS, INJECTION_PROFILES
    INJECTION_PATTERNS = security.compile_patterns(
        conf.config.get("security", {}).get("injection_regex", [])
    )
    # Seed built-in profiles (idempotent: skip if name already exists).
    # We only seed + load when profiles are enabled (default = enabled).
    profiles_enabled = bool(conf.config.get("security", {}).get("injection_profiles_enabled", True))
    if profiles_enabled:
        try:
            await asyncio.to_thread(memory.seed_default_injection_profiles, security.DEFAULT_INJECTION_PROFILES)
        except Exception as e:
            log.warning("seed injection profiles failed: %s", e)
        # Load all enabled profiles
        try:
            db_profiles = await asyncio.to_thread(memory.list_injection_profiles, True)
        except Exception:
            db_profiles = []
        INJECTION_PROFILES = []
        for row in db_profiles:
            try:
                INJECTION_PROFILES.append(security.InjectionProfile.from_config(
                    name=row["name"],
                    regexes=row.get("regexes", []),
                    severity=row.get("severity", "medium"),
                    action=row.get("action", "alert"),
                    enabled=row.get("enabled", True),
                    is_builtin=row.get("is_builtin", False),
                ))
            except Exception as e:
                log.warning("skip invalid injection profile %r: %s", row.get("name"), e)
    else:
        INJECTION_PROFILES = []

    # Tool cache (LRU + TTL)
    cache_config = conf.config.get("tool_cache", {}) or {}
    if cache_config.get("enabled", False):
        TOOL_CACHE = tool_cache_mod.init_cache(
            max_entries=int(cache_config.get("max_entries", 1024)),
            default_ttl_seconds=int(cache_config.get("default_ttl_seconds", 300)),
            per_tool_ttl_seconds=cache_config.get("per_tool_ttl_seconds") or {},
            bypass_keys=cache_config.get("bypass_keys") or [],
        )
    else:
        # Explicitly clear the module-level singleton so that a prior
        # init_cache() call (e.g. from a different config or a test) does
        # not leave a stale cache lying around when the current config
        # disables it.
        TOOL_CACHE = None
        tool_cache_mod._default_cache = None  # type: ignore[attr-defined]

    # Webhook dispatcher
    webhook_config = conf.config.get("webhooks", {}) or {}
    if webhook_config.get("enabled", True):
        WEBHOOK_DISPATCHER = webhook_mod.init_dispatcher(
            max_retries=int(webhook_config.get("max_retries", 3)),
            initial_backoff_seconds=float(webhook_config.get("initial_backoff_seconds", 1.0)),
            backoff_multiplier=float(webhook_config.get("backoff_multiplier", 2.0)),
            delivery_timeout_seconds=float(webhook_config.get("delivery_timeout_seconds", 10.0)),
            max_concurrent_deliveries=int(webhook_config.get("max_concurrent_deliveries", 16)),
        )

    # Built-in prompt templates
    prompt_config = conf.config.get("prompts", {}) or {}
    if prompt_config.get("enabled", True):
        try:
            await asyncio.to_thread(prompt_mod.seed_builtin_templates)
        except Exception as e:
            log.warning("seed builtin prompt templates failed: %s", e)

    # IBM ContextForge connector
    cf_config = conf.config.get("contextforge", {}) or {}
    if cf_config.get("enabled", False):
        try:
            CONTEXTFORGE_CLIENT = cf_mod.init_client(
                mode=cf_config.get("mode", "embedded"),
                external_url=cf_config.get("external_url") or None,
                api_key=cf_config.get("api_key") or None,
                sync_interval_seconds=int(cf_config.get("sync_interval_seconds", 300)),
                auto_sync=bool(cf_config.get("auto_sync", False)),
                timeout_seconds=float(cf_config.get("timeout_seconds", 30.0)),
            )
            # Best-effort one-shot sync at startup
            try:
                sync_results = await CONTEXTFORGE_CLIENT.sync_all()
                for sync_type, result in sync_results.items():
                    events.emit(
                        events.EventSource.CONTEXTFORGE,
                        f"sync.{sync_type}",
                        {
                            "items_synced": result.items_synced,
                            "items_added": result.items_added,
                            "items_updated": result.items_updated,
                            "errors": result.errors,
                            "duration_ms": result.duration_ms,
                        },
                        severity=events.EventSeverity.INFO,
                    )
            except Exception as e:
                log.warning("ContextForge initial sync failed: %s", e)
            if CONTEXTFORGE_CLIENT.auto_sync:
                CONTEXTFORGE_CLIENT.start_sync_loop()
        except Exception as e:
            log.warning("ContextForge client init failed: %s", e)
    else:
        CONTEXTFORGE_CLIENT = None

    # Model sync engine (auto-discover models from OpenRouter, OpenAI, etc.)
    ms_config = conf.config.get("model_sync", {}) or {}
    if ms_config.get("enabled", True):
        try:
            # Build the list of OpenAI-compatible providers from presets + config.
            compat_providers = ms_config.get("providers", []) or []
            MODEL_SYNC_ENGINE = model_sync_mod.init_sync_engine(
                openrouter_enabled=bool(ms_config.get("openrouter_enabled", True)),
                openai_base_url=ms_config.get("openai_base_url"),
                openai_api_key_env=ms_config.get("openai_api_key_env", "OPENAI_API_KEY"),
                anthropic_api_key_env=ms_config.get("anthropic_api_key_env", "ANTHROPIC_API_KEY"),
                anthropic_enabled=bool(ms_config.get("anthropic_enabled", False)),
                ollama_base_url=ms_config.get("ollama_base_url", "http://localhost:11434"),
                ollama_enabled=bool(ms_config.get("ollama_enabled", True)),
                openai_compatible_providers=compat_providers,
                sync_interval_seconds=int(ms_config.get("sync_interval_seconds", model_sync_mod.DEFAULT_SYNC_INTERVAL_SECONDS)),
                auto_sync=bool(ms_config.get("auto_sync", False)),
                timeout_seconds=float(ms_config.get("timeout_seconds", 30.0)),
            )
            if MODEL_SYNC_ENGINE.auto_sync:
                MODEL_SYNC_ENGINE.start_sync_loop()
        except Exception as e:
            log.warning("Model sync engine init failed: %s", e)
    else:
        MODEL_SYNC_ENGINE = None

    # Set current config for policy atom-eval
    policy_mod.set_current_config(conf)

    # Reviewer worker
    rw = reviewer.init_worker(conf)
    await rw.start()

    # Trainer worker
    tw = trainer_worker.init_worker(conf)
    await tw.start()

    # Observer/Reflector worker (observational memory)
    ow = observer_worker.init_worker(conf, pool)
    await ow.start()

    # Ensure OM tables exist
    om.memory_metadata.create_all(memory.engine())

    # Auth manager (per-tenant API keys). Disabled unless config enables it.
    auth_mgr = auth_mod.init_manager(conf)
    auth_mod.set_manager(auth_mgr)

    # Overlay manager (live admin CRUD for providers/keys/tiers)
    overlay_mgr = admin_mod.OverlayManager(conf_mgr)
    max_body_bytes = int(conf.config.get("http", {}).get("max_body_bytes", 4 * 1024 * 1024))
    app = web.Application(client_max_size=max_body_bytes)
    app["conf_mgr"] = conf_mgr
    app["router"] = rt
    app["endpoint_pool"] = pool
    app["tenant_mgr"] = tenant.manager()
    app["reviewer_worker"] = rw
    app["trainer_worker"] = tw
    app["observer_worker"] = ow
    app["auth_manager"] = auth_mgr
    app["overlay_manager"] = overlay_mgr
    app["router_config_signature"] = _router_config_signature(conf)
    app["max_body_bytes"] = max_body_bytes
    app["cors_origins"] = list(conf.config.get("http", {}).get("cors_origins", []) or [])
    app["tool_cache"] = TOOL_CACHE
    app["webhook_dispatcher"] = WEBHOOK_DISPATCHER
    app["contextforge_client"] = CONTEXTFORGE_CLIENT
    app.middlewares.append(auth_mod.body_size_middleware)
    app.middlewares.append(auth_mod.cors_middleware)
    app.middlewares.append(auth_mod.security_headers_middleware)
    app.middlewares.append(auth_mod.auth_middleware)

    # Plugin loader (after middlewares so plugin routes see auth middleware)
    plugins_config = conf.config.get("plugins", {}) or {}
    if plugins_config.get("enabled", True):
        global PLUGIN_LOADER
        PLUGIN_LOADER = plugin_mod.init_loader(
            app=app,
            config=conf,
            event_bus=events.bus(),
            plugin_root=plugins_config.get("root", "./plugins"),
            scan_interval_seconds=int(plugins_config.get("scan_interval_seconds", 30)),
            auto_load=bool(plugins_config.get("auto_load", True)),
        )
        if plugins_config.get("auto_load", True):
            PLUGIN_LOADER.start_watch()

    # Routes
    app.router.add_post("/v1/chat/completions", chat_completions)
    app.router.add_get("/v1/models", list_models)
    app.router.add_get("/stats", get_stats)
    app.router.add_get("/config", get_config)
    app.router.add_post("/reload", post_reload)
    app.router.add_get("/trace", get_trace)
    app.router.add_post("/feedback", post_feedback)
    app.router.add_get("/accuracy", get_accuracy)
    app.router.add_get("/export", get_export)
    app.router.add_get("/memory", get_memory)
    app.router.add_get("/verticals", get_verticals)
    app.router.add_get("/cost", get_cost)
    app.router.add_get("/review-stats", get_review_stats)
    app.router.add_post("/retrain", post_retrain)
    app.router.add_get("/registry", get_registry)
    app.router.add_get("/docs/model-card/{version}", get_model_card)
    app.router.add_get("/admin/users", list_users)
    app.router.add_post("/admin/users", create_user)
    app.router.add_get("/admin/users/{tenant_id}/budget", get_user_budget)
    app.router.add_post("/admin/users/{tenant_id}/budget", set_user_budget)
    app.router.add_get("/admin/users/{tenant_id}/stats", get_user_stats)
    # Subscription plans + per-model token limits
    app.router.add_get("/admin/plans", admin_list_plans)
    app.router.add_post("/admin/plans", admin_create_plan)
    app.router.add_put("/admin/plans/{plan_id}", admin_update_plan)
    app.router.add_post("/admin/users/{tenant_id}/subscription", admin_assign_plan)
    app.router.add_delete("/admin/users/{tenant_id}/subscription", admin_unassign_plan)
    app.router.add_get("/admin/users/{tenant_id}/subscription", admin_get_plan)
    app.router.add_put("/admin/users/{tenant_id}/budget/tokens", admin_set_daily_token_budget)
    app.router.add_get("/admin/users/{tenant_id}/models/{endpoint}/limits", admin_get_model_limits)
    app.router.add_put("/admin/users/{tenant_id}/models/{endpoint}/limits", admin_set_model_limits)
    app.router.add_get("/admin/users/{tenant_id}/limits", admin_get_all_limits)
    app.router.add_get("/admin/models/quality", admin_list_quality_profiles)
    app.router.add_post("/admin/models/quality", admin_record_quality_sample)
    # User-facing (read-only) limits
    app.router.add_get("/usage", get_my_usage)
    app.router.add_get("/usage/limits", get_my_limits)
    app.router.add_get("/admin/flags", list_flags)
    app.router.add_get("/admin/security/events", admin_security_events)
    app.router.add_get("/admin/security/events/stats", admin_security_stats)
    app.router.add_get("/admin/security/injection-profiles", admin_list_injection_profiles)
    app.router.add_post("/admin/security/injection-profiles", admin_create_injection_profile)
    app.router.add_put("/admin/security/injection-profiles/{profile_id}", admin_update_injection_profile)
    app.router.add_delete("/admin/security/injection-profiles/{profile_id}", admin_delete_injection_profile)
    app.router.add_get("/admin/security/provider-allowlist", admin_list_provider_allowlist)
    app.router.add_post("/admin/security/provider-allowlist", admin_upsert_provider_allowlist)
    app.router.add_delete("/admin/security/provider-allowlist/{tenant_id}/{pattern}", admin_delete_provider_allowlist)
    app.router.add_post("/admin/security/sync-firewall", admin_sync_host_firewall)
    app.router.add_get("/admin/security/status", admin_security_status)
    app.router.add_post("/admin/security/test", admin_security_test)
    # Plugins
    app.router.add_get("/admin/plugins", admin_list_plugins)
    app.router.add_get("/admin/plugins/{name}", admin_get_plugin)
    app.router.add_post("/admin/plugins/{name}", admin_upsert_plugin)
    app.router.add_delete("/admin/plugins/{name}", admin_delete_plugin)
    app.router.add_post("/admin/plugins/{name}/reload", admin_reload_plugin)
    app.router.add_post("/admin/plugins/{name}/enable", admin_enable_plugin)
    app.router.add_post("/admin/plugins/{name}/disable", admin_disable_plugin)
    # A2A agents
    app.router.add_get("/admin/a2a/agents", admin_list_a2a_agents)
    app.router.add_post("/admin/a2a/agents", admin_create_a2a_agent)
    app.router.add_get("/admin/a2a/agents/{agent_id}", admin_get_a2a_agent)
    app.router.add_put("/admin/a2a/agents/{agent_id}", admin_update_a2a_agent)
    app.router.add_delete("/admin/a2a/agents/{agent_id}", admin_delete_a2a_agent)
    app.router.add_post("/admin/a2a/agents/{agent_id}/invoke", admin_invoke_a2a_agent)
    app.router.add_get("/admin/a2a/agents/{agent_id}/metrics", admin_a2a_agent_metrics)
    # A2A virtual servers
    app.router.add_get("/admin/a2a/servers", admin_list_a2a_servers)
    app.router.add_post("/admin/a2a/servers", admin_create_a2a_server)
    app.router.add_delete("/admin/a2a/servers/{server_id}", admin_delete_a2a_server)
    # ContextForge
    app.router.add_post("/admin/contextforge/sync", admin_contextforge_sync)
    app.router.add_get("/admin/contextforge/sync-log", admin_contextforge_sync_log)
    app.router.add_get("/admin/contextforge/tools", admin_contextforge_tools)
    # MCP discovery
    app.router.add_post("/admin/mcp/discover", admin_mcp_discover)
    # Prompts
    app.router.add_get("/admin/prompts", admin_list_prompts)
    app.router.add_post("/admin/prompts", admin_create_prompt)
    app.router.add_get("/admin/prompts/{template_id}", admin_get_prompt)
    app.router.add_put("/admin/prompts/{template_id}", admin_update_prompt)
    app.router.add_delete("/admin/prompts/{template_id}", admin_delete_prompt)
    # Webhooks
    app.router.add_get("/admin/webhooks", admin_list_webhooks)
    app.router.add_post("/admin/webhooks", admin_create_webhook)
    app.router.add_delete("/admin/webhooks/{webhook_id}", admin_delete_webhook)
    app.router.add_get("/admin/webhooks/deliveries", admin_webhook_deliveries)
    # Tool cache
    app.router.add_get("/admin/cache/stats", admin_cache_stats)
    app.router.add_post("/admin/cache/invalidate", admin_cache_invalidate)
    # Model catalog + sync
    app.router.add_post("/admin/models/sync", admin_models_sync)
    app.router.add_get("/admin/models/catalog", admin_models_catalog)
    app.router.add_get("/admin/models/stats", admin_models_stats)
    app.router.add_put("/admin/models/{model_id}/tier", admin_models_set_tier)
    app.router.add_put("/admin/models/{model_id}/enabled", admin_models_set_enabled)
    # MCP facade — unified JSON-RPC endpoint for tools/list, tools/call, prompts/list
    app.router.add_post("/mcp", mcp_facade_mod.handle_mcp_rpc)
    app.router.add_get("/admin/provider-presets", get_provider_presets)
    # Provider/Key/Tier CRUD
    app.router.add_post("/admin/endpoints", admin_add_endpoint)
    app.router.add_put("/admin/endpoints/{name}", admin_update_endpoint)
    app.router.add_delete("/admin/endpoints/{name}", admin_delete_endpoint)
    app.router.add_post("/admin/keys", admin_generate_key)
    app.router.add_get("/admin/keys", admin_list_keys)
    app.router.add_delete("/admin/keys/{key}", admin_revoke_key)
    app.router.add_put("/admin/tiers/{name}", admin_update_tier)
    app.router.add_post("/admin/tiers/{name}/endpoints/{endpoint}", admin_assign_endpoint)
    app.router.add_delete("/admin/tiers/{name}/endpoints/{endpoint}", admin_unassign_endpoint)
    app.router.add_get("/admin/probe-local", admin_probe_local)
    app.router.add_post("/admin/test-endpoint", admin_test_endpoint)
    app.router.add_post("/admin/revert/{version}", admin_revert)
    app.router.add_get("/admin/flywheel-graph", admin_flywheel_graph)
    app.router.add_get("/admin/training-status", admin_training_status)
    app.router.add_get("/health", health)
    app.router.add_get("/ready", ready)
    app.router.add_get("/metrics", get_metrics)
    app.router.add_get("/", root_index)
    app.router.add_get("/dashboard", dashboard_spa)
    app.router.add_static("/dashboard/", path=str(Path(__file__).parent / "dashboard"), show_index=True)

    # Observational memory + event stream endpoints
    app.router.add_get("/memory/context", get_memory_context)
    app.router.add_get("/memory/working/{resource_id}", get_working_memory)
    app.router.add_post("/memory/working/{resource_id}", update_working_memory_ep)
    app.router.add_get("/memory/observations/{resource_id}/{thread_id}", get_observations_ep)
    app.router.add_get("/memory/reflection/{resource_id}/{thread_id}", get_reflection_ep)
    app.router.add_post("/memory/observe", force_observe_ep)
    app.router.add_post("/memory/reflect", force_reflect_ep)
    app.router.add_get("/events", events_stream)
    app.router.add_get("/events/recent", events_recent)

    # Background config mtime poller
    app.on_startup.append(_start_config_poller)
    app.on_startup.append(_start_health_poller)
    app.on_cleanup.append(_cleanup_resources)

    return app


async def _start_health_poller(app: web.Application):
    """Periodically probe endpoint /health so /ready + /stats see live state,
    not just breaker state (which only changes after request failures)."""
    app["endpoint_health"] = {}

    async def poll():
        while True:
            try:
                # Interval is read per iteration so config reloads take effect
                interval = float(app["conf_mgr"].current().config.get("http", {}).get("health_poll_seconds", 30))
                app["health_poll_seconds"] = interval
                await _poll_health(app)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("health poll error: %s", e)
                interval = 30.0
            await asyncio.sleep(interval)

    app["health_poller"] = asyncio.create_task(poll())


async def _poll_health(app: web.Application):
    pool = app["endpoint_pool"]
    clients = pool.clients()

    async def probe(name: str, client):
        try:
            return name, await client.health_check()
        except Exception:
            return name, False

    results = await asyncio.gather(*[probe(n, c) for n, c in clients.items()], return_exceptions=True)
    health = {}
    for r in results:
        if isinstance(r, tuple):
            health[r[0]] = bool(r[1])
    app["endpoint_health"] = health


async def _start_config_poller(app: web.Application):
    last_purge_at = [0.0]

    async def poller():
        conf_mgr: cfg_mod.ConfigManager = app["conf_mgr"]
        while True:
            try:
                changed = conf_mgr.check_mtime_and_reload(poll_interval_seconds=5.0)
                if changed:
                    conf = conf_mgr.current()
                    await _apply_runtime_config(app, conf)
                    log.info("config reloaded (version=%d)", conf.version)
                # Hourly trace retention purge (config.logging.trace_retention_days)
                now = time.time()
                if now - last_purge_at[0] >= 3600:
                    last_purge_at[0] = now
                    log_cfg = conf_mgr.current().config.get("logging", {})
                    retention = log_cfg.get("trace_retention_days")
                    if retention:
                        purged = memory.purge_old_traces(int(retention))
                        if purged:
                            log.info("trace retention: purged %d old decisions (>%sd)", purged, retention)
                    flag_retention = log_cfg.get("flagged_retention_days")
                    if flag_retention:
                        purged = memory.purge_old_flags(int(flag_retention))
                        if purged:
                            log.info("flagged retention: purged %d old flags (>%sd)", purged, flag_retention)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.warning("config poller error: %s", e)
            await asyncio.sleep(5)

    app["config_poller"] = asyncio.create_task(poller())


async def _apply_runtime_config(app: web.Application, conf: cfg_mod.Config):
    """Propagate a new immutable config snapshot to every live subsystem."""
    policy_mod.set_current_config(conf)
    await app["endpoint_pool"].rebuild(conf)
    global INJECTION_PATTERNS, INJECTION_PROFILES, FIREWALL, HOST_FIREWALL
    INJECTION_PATTERNS = security.compile_patterns(
        conf.config.get("security", {}).get("injection_regex", [])
    )
    # Refresh injection profiles from the DB
    profiles_enabled = bool(conf.config.get("security", {}).get("injection_profiles_enabled", True))
    if profiles_enabled:
        try:
            db_profiles = await asyncio.to_thread(memory.list_injection_profiles, True)
        except Exception:
            db_profiles = []
        new_profiles = []
        for row in db_profiles:
            try:
                new_profiles.append(security.InjectionProfile.from_config(
                    name=row["name"],
                    regexes=row.get("regexes", []),
                    severity=row.get("severity", "medium"),
                    action=row.get("action", "alert"),
                    enabled=row.get("enabled", True),
                    is_builtin=row.get("is_builtin", False),
                ))
            except Exception:
                pass
        INJECTION_PROFILES = new_profiles
    else:
        INJECTION_PROFILES = []
    # Refresh in-process firewall
    fw_config = conf.config.get("security", {}).get("provider_allowlist", {}) or {}
    enforcer = firewall_mod.DomainAllowlistEnforcer(
        enabled=bool(fw_config.get("enabled", False)),
        default_action=fw_config.get("default_action", "block"),
    )
    enforcer.load_from_config(fw_config)
    try:
        db_rules = await asyncio.to_thread(memory.list_provider_allowlist)
    except Exception:
        db_rules = []
    enforcer.load_from_db(db_rules)
    app["endpoint_pool"].set_firewall(enforcer)
    FIREWALL = enforcer
    # Re-sync host firewall if enabled
    if HOST_FIREWALL and HOST_FIREWALL.enabled:
        try:
            patterns = [r.pattern for r in enforcer.list_rules() if r.tenant_id == "*"]
            HOST_FIREWALL.sync(patterns)
        except Exception as e:
            log.warning("host firewall sync on hot-reload failed: %s", e)

    # Refresh tool cache settings
    cache_config = conf.config.get("tool_cache", {}) or {}
    if cache_config.get("enabled", False):
        app["tool_cache"] = tool_cache_mod.init_cache(
            max_entries=int(cache_config.get("max_entries", 1024)),
            default_ttl_seconds=int(cache_config.get("default_ttl_seconds", 300)),
            per_tool_ttl_seconds=cache_config.get("per_tool_ttl_seconds") or {},
            bypass_keys=cache_config.get("bypass_keys") or [],
        )
    else:
        app["tool_cache"] = None

    app["auth_manager"].update(conf.config.get("auth", {}))
    app["reviewer_worker"].update_config(conf)
    app["trainer_worker"].update_config(conf)
    app["observer_worker"].update_config(conf)
    tenants_cfg = conf.config.get("tenants", {}) or {}
    default_cfg = tenants_cfg.get("*") or {}
    preconfigured = {k: v for k, v in tenants_cfg.items() if k != "*" and not str(k).startswith("_")}
    app["tenant_mgr"].reconfigure(default_cfg, preconfigured)
    app["max_body_bytes"] = int(conf.config.get("http", {}).get("max_body_bytes", app["max_body_bytes"]))
    app["cors_origins"] = list(conf.config.get("http", {}).get("cors_origins", []) or [])
    signature = _router_config_signature(conf)
    if signature != app.get("router_config_signature"):
        app["router_config_signature"] = signature
        vertical_names = [v["name"] for v in conf.verticals()]
        onnx_path = conf.config.get("embedding", {}).get("onnx_path")
        heads_path = onnx_path.replace("model.onnx", "heads.npz") if onnx_path else None
        loaded = False
        if onnx_path and heads_path and Path(onnx_path).exists() and heads_path and Path(heads_path).exists():
            loaded = await asyncio.to_thread(
                app["router"].try_load_real,
                onnx_path=onnx_path,
                heads_path=heads_path,
                vertical_names=vertical_names,
                calibration_temperature=conf.policy.get("calibration", {}).get("temperature", 1.0),
                checksum_sha256=conf.config.get("embedding", {}).get("checksum_sha256") or None,
                tokenizer_source=conf.config.get("embedding", {}).get("model_id"),
            )
        if not loaded:
            app["router"].init_stub(vertical_names)


def _router_config_signature(conf: cfg_mod.Config) -> tuple:
    embedding = conf.config.get("embedding", {})
    return (
        embedding.get("onnx_path"),
        embedding.get("checksum_sha256"),
        embedding.get("model_id"),
        conf.policy.get("calibration", {}).get("temperature", 1.0),
        tuple(v["name"] for v in conf.verticals()),
    )


async def _cleanup_resources(app: web.Application):
    if app.get("config_poller"):
        app["config_poller"].cancel()
    if app.get("health_poller"):
        app["health_poller"].cancel()
    if app.get("reviewer_worker"):
        await app["reviewer_worker"].stop()
    if app.get("trainer_worker"):
        await app["trainer_worker"].stop()
    if app.get("observer_worker"):
        await app["observer_worker"].stop()
    pool = app.get("endpoint_pool")
    if pool:
        await pool.close_all()
    # Optionally clear host-level firewall rules (only when persist_on_shutdown=False)
    global HOST_FIREWALL
    if HOST_FIREWALL and HOST_FIREWALL.enabled and not HOST_FIREWALL.persist_on_shutdown:
        try:
            HOST_FIREWALL.clear()
        except Exception as e:
            log.warning("host firewall clear on shutdown failed: %s", e)
    # Stop plugin loader watch loop
    if PLUGIN_LOADER is not None:
        PLUGIN_LOADER.stop_watch()
    # Close ContextForge client
    cf_client = app.get("contextforge_client")
    if cf_client is not None:
        try:
            await cf_client.close()
        except Exception as e:
            log.warning("contextforge client close failed: %s", e)
    # Close webhook dispatcher session
    webhook_dispatcher = webhook_mod.dispatcher()
    if webhook_dispatcher is not None:
        try:
            await webhook_dispatcher.close()
        except Exception as e:
            log.warning("webhook dispatcher close failed: %s", e)
    # Stop model sync loop
    ms_engine = model_sync_mod.engine()
    if ms_engine is not None:
        ms_engine.stop_sync_loop()
    await asyncio.to_thread(memory.close_engine)


# ============================================================
# Request handlers
# ============================================================


def _openai_error(
    code: str,
    message: str,
    status: int,
    *,
    error_type: str = "invalid_request_error",
    param: str | None = None,
    headers: dict[str, str] | None = None,
) -> web.Response:
    return web.json_response({
        "error": {"message": message, "type": error_type, "param": param, "code": code},
    }, status=status, headers=headers)


async def chat_completions(request: web.Request) -> web.StreamResponse:
    try:
        body = await request.json()
    except Exception:
        return _openai_error("invalid_json", "Body must be valid JSON.", 400)
    if not isinstance(body, dict):
        return _openai_error("invalid_body", "Body must be a JSON object.", 400)
    conf = request.app["conf_mgr"].current()
    rt: router_mod.Router = request.app["router"]
    tenant_mgr: tenant.TenantManager = request.app["tenant_mgr"]

    # Tenant resolution: auth middleware sets request["tenant_id"] when a valid
    # API key is present; X-User-Id remains the fallback identity.
    tenant_id: str = str(
        request.get("tenant_id") or request.headers.get("X-User-Id", "anonymous")
    )

    # Rate limit check
    try:
        await asyncio.to_thread(tenant_mgr.check_rate_limit, tenant_id)
    except tenant.RateLimited as e:
        return _openai_error(
            "rate_limit_exceeded", "Request rate limit exceeded.", 429,
            error_type="rate_limit_error", headers={"Retry-After": str(max(1, int(e.retry_after_seconds)))},
        )

    # Extract prompt
    messages = body.get("messages", [])
    if not messages:
        return _openai_error("missing_messages", "messages is required.", 400, param="messages")
    if not isinstance(messages, list) or not all(isinstance(m, dict) for m in messages):
        return _openai_error("invalid_messages", "messages must be an array of objects.", 400, param="messages")
    last_user = next((m for m in reversed(messages) if m.get("role") == "user"), None)
    if not last_user:
        return _openai_error("missing_user_message", "At least one user message is required.", 400, param="messages")
    text = _extract_text(last_user.get("content", ""))
    has_image = _has_image(last_user.get("content"))

    # Security check — profile-based. Each profile has a severity + action
    # (block / alert / log). High-severity matches with action=block reject
    # the request with HTTP 400. Routing is NEVER altered by injection
    # signals (we still classify + route as if the user typed normally).
    if INJECTION_PROFILES:
        inj = security.check_injection_with_action(
            text,
            INJECTION_PROFILES,
            strip_control_tokens=bool(conf.config.get("security", {}).get("strip_control_tokens", True)),
        )
    else:
        # Fallback to legacy single-list check
        legacy = security.check_injection(
            text,
            INJECTION_PATTERNS,
            strip_control_tokens=bool(conf.config.get("security", {}).get("strip_control_tokens", True)),
        )
        inj = security.InjectionResult(
            has_injection=legacy.has_injection_signal,
            severity="medium",
            matched_profiles=[
                {"name": "legacy", "pattern": p, "severity": "medium"}
                for p in legacy.matched_patterns
            ],
            action="alert",
            sanitized_text=legacy.sanitized_text,
        )
    has_injection = inj.has_injection

    # Block-or-alert handling for injection matches.
    if has_injection and inj.action == "block":
        # Security event audit + log
        top = inj.matched_profiles[0] if inj.matched_profiles else {
            "name": "unknown", "pattern": "", "severity": inj.severity,
        }
        event_id = 0
        try:
            event_id = await asyncio.to_thread(
                memory.record_security_event,
                tenant_id,
                "injection_blocked",
                inj.severity,
                f"blocked injection: {top['name']}",
                matched_pattern=top.get("pattern", "")[:256],
                query_preview=text[:1000],
                action_taken="block",
                request_metadata={"endpoint": "chat_completions"},
            )
        except Exception as e:
            log.warning("record security event failed: %s", e)
        # Also write to flagged_inputs for backwards compat
        try:
            await asyncio.to_thread(
                memory.flag_input,
                tenant_id,
                None,
                "injection_blocked",
                top.get("pattern", "")[:256],
                text,
                "blocked",
                inj.severity,
                top.get("name"),
                event_id,
            )
        except Exception as e:
            log.warning("flag_input failed: %s", e)
        return _openai_error(
            "injection_blocked",
            "Request blocked by security policy.",
            400,
            error_type="invalid_request_error",
        )
    elif has_injection and inj.action == "alert":
        try:
            top = inj.matched_profiles[0]
            await asyncio.to_thread(
                memory.record_security_event,
                tenant_id,
                "injection_alerted",
                inj.severity,
                f"alerted injection: {top['name']}",
                matched_pattern=top.get("pattern", "")[:256],
                query_preview=text[:1000],
                action_taken="alert",
            )
        except Exception as e:
            log.warning("record security event failed: %s", e)
    elif has_injection and inj.action == "log":
        try:
            top = inj.matched_profiles[0]
            await asyncio.to_thread(
                memory.record_security_event,
                tenant_id,
                "injection_alerted",
                "low",
                f"logged injection: {top['name']}",
                matched_pattern=top.get("pattern", "")[:256],
                query_preview=text[:1000],
                action_taken="log",
            )
        except Exception as e:
            log.warning("record security event failed: %s", e)

    # Session
    session_id = request.headers.get("X-Session-Id") or str(uuid.uuid4())

    # Concurrency acquire (per-tenant semaphore)
    tstate = tenant_mgr.get_or_create(tenant_id)
    async with tstate.semaphore:
            # Emit "thinking" status event for UI
            events.emit_status(
                events.EventSource.ROUTING,
                "Classifying request...",
                done=False,
                tenant_id=tenant_id,
                session_id=session_id,
            )

            memory_enabled = bool(conf.config.get("memory", {}).get("enabled", True))
            if memory_enabled:
                await asyncio.to_thread(om.ensure_working_memory, tenant_id)
                memory_ctx = await asyncio.to_thread(
                    om.load_memory_context,
                    conf=conf,
                    resource_id=tenant_id,
                    thread_id=session_id,
                    query_text=text,
                )
            else:
                memory_ctx = om.MemoryContext(
                    resource_id=tenant_id,
                    thread_id=session_id,
                    recency_messages=[],
                    working_memory_content=None,
                    recalled_messages=[],
                    observations=None,
                    reflection=None,
                )

            # Record the user's message in memory domain (for OM to compress later)
            recall_enabled = (
                memory_enabled
                and int(conf.policy.get("memory", {}).get("semantic_recall_top_k", 0)) > 0
                and not rt.is_stub()
            )
            if memory_enabled:
                try:
                    await asyncio.to_thread(
                        om.record_message,
                        resource_id=tenant_id,
                        thread_id=session_id,
                        role="user",
                        content=text,
                        token_estimate=len(text) // 4,
                        embed=recall_enabled,
                    )
                except Exception as e:
                    log.warning("record user message failed: %s", e)

            # Router classification
            try:
                r = await asyncio.to_thread(rt.predict, text)
            except Exception as e:
                log.exception("router predict failed: %s", e)
                return _openai_error("router_failed", "The routing model failed.", 500, error_type="server_error")

            # OOD check
            ood_threshold = conf.config.get("routing", {}).get("ood_threshold", 0.25)
            ood = ood_mod.detect(r.vertical_top2, ood_threshold)

            # Build request context
            ctx = policy_mod.RequestContext(
                text=text,
                has_image=has_image,
                flags=r.flags,
                complexity=r.complexity,
                vertical=r.vertical,
                vertical_top2=r.vertical_top2,
                projection=r.projection,
                ood=ood,
                model_version=r.model_version,
                policy_version=conf.version,
                session_id=session_id,
                tenant_id=tenant_id,
                estimated_input_tokens=_estimate_input_tokens(messages) + memory_ctx.total_tokens_estimate,
                estimated_output_tokens=body.get("max_tokens") or body.get("max_completion_tokens") or _estimate_output_tokens(r),
            )

            # Working memory inheritance: short + low-confidence query inherits
            # the session's last tier when the previous response was OK + fast.
            inherit_tier = None
            wm = conf.config.get("routing", {}).get("working_memory", {})
            if wm.get("enabled") and not has_image and not has_injection:
                if len(text) <= wm.get("max_query_chars_for_inherit", 80):
                    if ctx.ood.is_ood or (ctx.vertical_top2 and ctx.vertical_top2[0][1] < 0.55):
                        sess = await asyncio.to_thread(memory.get_or_create_session, session_id, tenant_id)
                        if sess.get("last_response_ok") and (
                            sess.get("last_response_ms") or 0
                        ) < wm.get("previous_response_time_threshold_ms", 30000):
                            inherit_tier = sess.get("last_tier")

            # Pre-route
            breaker_states = {b.endpoint_name: b.state() for b in [
                circuit.registry().get(ep["name"], _breaker_config(ep)) for ep in conf.config.get("endpoints", [])
            ]}
            requested_model = body.get("model", "gateway")
            direct_decision = _direct_model_route(
                conf, requested_model, tenant_mgr, tenant_id, breaker_states,
            )
            if requested_model not in (None, "", "gateway") and direct_decision is None:
                return web.json_response({
                    "error": {
                        "message": f"model '{requested_model}' not found or not accessible",
                        "type": "invalid_request_error",
                        "param": "model",
                        "code": "model_not_found",
                    },
                }, status=404)
            pre = policy_mod.pre_route(ctx, conf, breaker_states, memory_ctx=memory_ctx)

            # Emit routing decision event
            events.emit(
                events.EventSource.ROUTING,
                "decision_made",
                {
                    "vertical": r.vertical,
                    "complexity": r.complexity,
                    "source": direct_decision.source if direct_decision else (pre.source if pre.matched else "cost_first"),
                    "tier": direct_decision.tier if direct_decision else (pre.tier if pre.matched else None),
                    "memory_tokens": memory_ctx.total_tokens_estimate,
                    "has_observations": memory_ctx.observations is not None,
                },
                tenant_id=tenant_id,
                session_id=session_id,
            )

            if direct_decision is not None:
                decision = direct_decision
            elif pre.matched:
                decision = policy_mod.RoutingDecision(
                    tier=pre.tier,
                    endpoint=_first_endpoint_for_tier(conf, pre.tier, breaker_states),
                    source=pre.source,
                )
            else:
                # Cost-first gate
                endpoint_loads = request.app["endpoint_pool"].all_inflight()
                # Use budget-aware routing when the tenant has a plan, a daily
                # token budget, or any per-model token limits — otherwise the
                # simpler cost-first picker is sufficient.
                routing_cfg = conf.config.get("routing", {})
                use_budget_aware = bool(
                    routing_cfg.get("budget_aware", {}).get("enabled", True)
                )
                plan_quota = None
                if use_budget_aware:
                    plan_quota = (
                        await asyncio.to_thread(memory.get_tenant_plan_quota, tenant_id)
                    )
                    tenant_st = tenant_mgr.get_or_create(tenant_id)
                    has_token_limit = tenant_st.daily_token_limit > 0
                    has_model_limits = bool(
                        await asyncio.to_thread(
                            memory.list_model_token_limits, tenant_id,
                        )
                    )
                    if not (plan_quota or has_token_limit or has_model_limits):
                        # Plain cost-first when there's no budget to enforce
                        use_budget_aware = False
                if use_budget_aware:
                    try:
                        budget_routed = policy_mod.budget_aware_route(
                            ctx,
                            conf,
                            breaker_states,
                            endpoint_loads,
                            tenant_mgr=tenant_mgr,
                            tenant_id=tenant_id,
                            plan_quota=plan_quota,
                        )
                        decision = budget_routed.decision
                    except policy_mod.BudgetError as e:
                        # decision_id is not yet logged; return the error directly
                        return _openai_error(
                            e.code,
                            e.message,
                            429,
                            error_type="insufficient_quota",
                        )
                else:
                    decision = policy_mod.cost_first_route(ctx, conf, breaker_states, endpoint_loads)

                # Observational memory: redirect to a higher tier if compacted context overflows
                redirect, target_tier, reason = om.should_redirect_for_compaction(
                    memory_ctx, conf, decision.tier
                )
                if redirect and target_tier:
                    events.emit(
                        events.EventSource.MEMORY,
                        "compaction_redirect",
                        {
                            "from_tier": decision.tier,
                            "to_tier": target_tier,
                            "reason": reason,
                            "memory_tokens": memory_ctx.total_tokens_estimate,
                        },
                        tenant_id=tenant_id,
                        session_id=session_id,
                    )
                    decision.tier = target_tier
                    decision.endpoint = _first_endpoint_for_tier(conf, target_tier, breaker_states)
                    decision.rationale += f"; compacted_redirect:{reason}"
                    decision.source = "memory_compaction_redirect"

            # Tenant access check (downgrade if user can't access tier)
            if not tenant_mgr.can_access_tier(tenant_id, decision.tier):
                accessible = tenant_mgr.highest_accessible_tier(tenant_id, [t["name"] for t in conf.config.get("tiers", [])])
                if accessible:
                    decision.tier = accessible
                    decision.endpoint = _first_endpoint_for_tier(conf, accessible, breaker_states)
                    decision.rationale += "; downgraded for tenant access"
                    decision.source = decision.source + "_tenant_access"

            # Working-memory tier inheritance (only when cost-first chose the tier,
            # and never when a compaction redirect already escalated for context).
            if inherit_tier and not pre.matched and "compaction" not in (decision.source or ""):
                if conf.tier(inherit_tier) and tenant_mgr.can_access_tier(tenant_id, inherit_tier):
                    decision.tier = inherit_tier
                    decision.endpoint = _first_endpoint_for_tier(conf, inherit_tier, breaker_states)
                    decision.rationale += "; working-memory tier inheritance"
                    decision.source = decision.source + "_wm_inherit"

            if not _ensure_context_capacity(conf, decision, ctx, tenant_mgr, tenant_id, breaker_states):
                return _openai_error(
                    "context_length_exceeded",
                    "Request plus reserved output exceeds every accessible context window.",
                    400,
                    param="messages",
                )

            # Any redirect/downgrade/inheritance above may have changed the
            # endpoint after the original arithmetic decision.
            decision.cost_usd = _estimate_endpoint_cost(conf, decision.endpoint, ctx, decision.fit)

            # Swarm mode: parallel decomposition for complex requests (cost-aware)
            swarm_result = None
            swarm_reserved_cost = 0.0
            use_swarm, swarm_reason = swarm_mod.should_swarm(ctx, conf, decision.tier, decision.cost_usd)
            if use_swarm:
                swarm_cfg = conf.policy.get("swarm", {})
                required_tiers = set(swarm_cfg.get("tier_pyramid", []))
                required_tiers.add(swarm_cfg.get("synthesis_tier", "tier3"))
                inaccessible = [t for t in required_tiers if t and not tenant_mgr.can_access_tier(tenant_id, t)]
                if inaccessible:
                    use_swarm = False
                    swarm_reason = f"tenant cannot access swarm tiers: {', '.join(sorted(inaccessible))}"
            if use_swarm:
                events.emit(
                    events.EventSource.SWARM,
                    "swarm_triggered",
                    {
                        "reason": swarm_reason,
                        "from_tier": decision.tier,
                        "from_cost_usd": decision.cost_usd,
                    },
                    tenant_id=tenant_id,
                    session_id=session_id,
                )
                try:
                    swarm_reserved_cost = decision.cost_usd
                    await asyncio.to_thread(
                        tenant_mgr.reserve_usage,
                        tenant_id,
                        estimated_tokens_in=ctx.estimated_input_tokens,
                        estimated_tokens_out=ctx.estimated_output_tokens,
                        estimated_cost_usd=decision.cost_usd,
                        endpoint_name="swarm",
                    )
                except tenant.BudgetExceeded as e:
                    return _openai_error("insufficient_quota", str(e), 429, error_type="insufficient_quota")
                except tenant.RateLimited as e:
                    return _openai_error(
                        "tokens_per_minute_exceeded", "Token rate limit exceeded.", 429,
                        error_type="rate_limit_error", headers={"Retry-After": str(max(1, int(e.retry_after_seconds)))},
                    )
                subtasks = await swarm_mod.decompose(
                    ctx, conf, conf.policy.get("swarm", {}), request.app["endpoint_pool"],
                )
                try:
                    swarm_result = await swarm_mod.execute_swarm(
                        ctx,
                        conf,
                        subtasks,
                        request.app["endpoint_pool"],
                        synthesis_tier=conf.policy.get("swarm", {}).get("synthesis_tier", "tier3"),
                        request_id=session_id,
                    )
                except Exception:
                    await asyncio.to_thread(
                        tenant_mgr.settle_usage,
                        tenant_id,
                        reserved_tokens_in=ctx.estimated_input_tokens,
                        reserved_tokens_out=ctx.estimated_output_tokens,
                        reserved_cost_usd=decision.cost_usd,
                        actual_tokens_in=0,
                        actual_tokens_out=0,
                        actual_cost_usd=0.0,
                        completed=False,
                        endpoint_name="swarm",
                    )
                    raise
                decision.tier = "swarm"
                decision.endpoint = "swarm"
                decision.source = "swarm"
                decision.rationale = f"swarm({len(subtasks)} subtasks): {swarm_reason}"
                decision.cost_usd = swarm_result.total_cost_usd

            # Translation intent: detect + tag + optional dedicated-tier routing
            trans_cfg = conf.policy.get("translation", {})
            trans_mode = trans_cfg.get("mode", "off")
            trans_intent = None
            if trans_mode != "off" and not has_image:
                trans_intent = translation.detect_intent(text)
                if trans_intent.is_translation:
                    decision.extra["translation"] = True
                    decision.extra["translation_mode"] = trans_mode
                    if trans_intent.target_language:
                        decision.extra["target_language"] = trans_intent.target_language
                    if trans_mode == "dedicated_endpoint" and swarm_result is None:
                        dedicated = trans_cfg.get("dedicated_tier")
                        if dedicated and conf.tier(dedicated):
                            decision.tier = dedicated
                            decision.endpoint = _first_endpoint_for_tier(conf, dedicated, breaker_states)
                            decision.source = decision.source + "_translation"
                            decision.rationale += f"; translation->{dedicated}"
                    events.emit(
                        events.EventSource.TRANSLATION,
                        "translation_detected",
                        {
                            "mode": trans_mode,
                            "target_language": trans_intent.target_language,
                        },
                        tenant_id=tenant_id,
                        session_id=session_id,
                    )

            if swarm_result is None:
                decision.cost_usd = _estimate_endpoint_cost(conf, decision.endpoint, ctx, decision.fit)

            # Log decision
            query_hash = hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
            decision_id = await asyncio.to_thread(
                memory.log_decision,
                tenant_id=tenant_id,
                session_id=session_id,
                model_version=r.model_version,
                policy_version=conf.version,
                query_hash=query_hash,
                query_preview=text[:500],
                vertical=r.vertical,
                complexity=r.complexity,
                flags=r.flags,
                vertical_top2_prob=r.vertical_top2[0][1] if r.vertical_top2 else None,
                tier=decision.tier,
                endpoint=decision.endpoint,
                source=decision.source,
                ms_classify=r.ms_classify,
                ms_total=0.0,
                est_cost_usd=decision.cost_usd,
                escalated=decision.escalated,
                fallback_used=False,
                has_image=has_image,
                has_injection_signal=has_injection,
                extra=decision.extra or None,
            )

            # Flag injection if detected (backwards-compat audit trail)
            if has_injection:
                if isinstance(inj, security.InjectionResult):
                    matched_for_log = "; ".join(
                        m.get("pattern", "") for m in inj.matched_profiles
                    )
                else:
                    matched_for_log = "; ".join(inj.matched_patterns)
                await asyncio.to_thread(
                    memory.flag_input,
                    tenant_id=tenant_id,
                    decision_id=decision_id,
                    reason="injection_signal",
                    matched_regex=matched_for_log[:200],
                    query_preview=text[:200],
                    action_taken="logged_routing_unaffected",
                )

            # Swarm path: synthesize from parallel subtasks (no single endpoint call)
            if swarm_result is not None:
                resp_data: dict = {
                    "id": f"chatcmpl-swarm-{uuid.uuid4().hex[:12]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": "swarm-synthesis",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": swarm_result.synthesis},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
                ms_total = swarm_result.duration_ms
                response_ok = not swarm_result.aborted
                await asyncio.to_thread(
                    _finalize_decision,
                    decision_id,
                    ms_total,
                    response_ok,
                    swarm_result.abort_reason if swarm_result.aborted else None,
                    actual_cost_usd=swarm_result.total_cost_usd,
                )
                _record_request_metrics("swarm", "swarm", response_ok, ms_total)
                await asyncio.to_thread(
                    memory.update_session,
                    session_id, tenant_id, tier="swarm", vertical=r.vertical,
                    endpoint="swarm", response_ok=response_ok,
                    response_ms=ms_total,
                )
                try:
                    if memory_enabled and swarm_result.synthesis:
                        await asyncio.to_thread(
                            om.record_message,
                            resource_id=tenant_id,
                            thread_id=session_id,
                            role="assistant",
                            content=swarm_result.synthesis,
                            token_estimate=len(swarm_result.synthesis) // 4,
                            metadata={"tier": "swarm", "subtasks": swarm_result.subtask_count},
                            embed=recall_enabled,
                        )
                except Exception:
                    pass
                await asyncio.to_thread(
                    reviewer.enqueue_for_review,
                    decision_id, tenant_id, cost_estimate=swarm_result.total_cost_usd, prompt_text=text,
                )
                await asyncio.to_thread(
                    tenant_mgr.settle_usage,
                    tenant_id,
                    reserved_tokens_in=ctx.estimated_input_tokens,
                    reserved_tokens_out=ctx.estimated_output_tokens,
                    reserved_cost_usd=swarm_reserved_cost,
                    actual_tokens_in=ctx.estimated_input_tokens,
                    actual_tokens_out=ctx.estimated_output_tokens,
                    actual_cost_usd=swarm_result.total_cost_usd,
                    completed=True,
                    endpoint_name="swarm",
                )
                if memory_enabled:
                    _maybe_force_observe(request.app, tenant_id, session_id)
                events.emit_status(
                    events.EventSource.SWARM,
                    f"Swarm done ({swarm_result.subtask_count} subtasks, ${swarm_result.total_cost_usd:.4f})",
                    done=True,
                    tenant_id=tenant_id,
                    session_id=session_id,
                )
                return web.json_response(resp_data)

            # Forward to endpoint
            endpoint_cfg = conf.endpoint(decision.endpoint)
            tier_cfg = conf.tier(decision.tier)
            if not endpoint_cfg or not tier_cfg:
                return _openai_error("invalid_routing_decision", "No valid upstream route is available.", 500, error_type="server_error")

            # Assemble messages with memory context (Mastra-style pipeline)
            forwarded_body = dict(body)
            forwarded_body["messages"] = om.assemble_messages(
                body.get("messages", []),
                memory_ctx,
                conf,
            )

            # Prompt template auto-injection by vertical category
            prompt_cfg = conf.config.get("prompts", {}) or {}
            if prompt_cfg.get("enabled", True) and prompt_cfg.get("auto_inject", False):
                category = prompt_mod.category_for_vertical(r.vertical)
                if category:
                    try:
                        templates = await asyncio.to_thread(
                            memory.list_prompt_templates,
                            True,  # enabled_only
                            category,
                            tenant_id,  # tenant filter (includes __all__)
                        )
                        if templates:
                            # Prefer a tenant-specific template over global.
                            tmpl = templates[0]
                            for candidate in templates:
                                if candidate.get("tenant_id") == tenant_id:
                                    tmpl = candidate
                                    break
                            variables = {
                                "vertical": r.vertical,
                                "complexity": str(r.complexity),
                                "tenant_id": tenant_id,
                                "session_id": session_id,
                            }
                            rendered = prompt_mod.render_template(
                                tmpl["template_text"], variables
                            )
                            # Prepend as system message (don't override existing)
                            existing_msgs = forwarded_body["messages"]
                            forwarded_body["messages"] = [
                                {"role": "system", "content": rendered}
                            ] + existing_msgs
                    except Exception as e:
                        log.debug("prompt auto-inject skipped: %s", e)

            # Translation mode: rewrite system prompt for translation intents
            if trans_mode in ("rewrite", "dedicated_endpoint") and trans_intent is not None and trans_intent.is_translation:
                override = translation.build_translation_rewrite(trans_intent)
                if override:
                    forwarded_body = translation.apply_to_payload(forwarded_body, override)

            transcoded = transcoder.transcode(endpoint_cfg, tier_cfg, forwarded_body)
            stream = bool(body.get("stream", False))

        # Atomically reserve estimated budget + tokens before dispatch.
            # Pull per-model token limits for the chosen endpoint (if configured).
            model_limit = await asyncio.to_thread(
                memory.get_model_token_limit, tenant_id, decision.endpoint,
            )
            model_token_limit = int(model_limit.get("daily_token_limit", 0)) if model_limit else 0
            model_usd_limit = float(model_limit.get("daily_usd_limit", 0)) if model_limit else 0.0
            max_request_tokens = int(model_limit.get("max_request_tokens", 0)) if model_limit else 0
            try:
                await asyncio.to_thread(
                    tenant_mgr.reserve_usage,
                    tenant_id,
                    estimated_tokens_in=ctx.estimated_input_tokens,
                    estimated_tokens_out=ctx.estimated_output_tokens,
                    estimated_cost_usd=decision.cost_usd,
                    endpoint_name=decision.endpoint,
                    model_token_limit=model_token_limit,
                    model_usd_limit=model_usd_limit,
                    max_request_tokens=max_request_tokens,
                )
            except tenant.BudgetExceeded as e:
                await asyncio.to_thread(_finalize_decision, decision_id, 0.0, False, str(e))
                return _openai_error("insufficient_quota", str(e), 429, error_type="insufficient_quota")
            except tenant.RateLimited as e:
                await asyncio.to_thread(
                    _finalize_decision, decision_id, 0.0, False, "tokens-per-minute limit exceeded",
                )
                return _openai_error(
                    "tokens_per_minute_exceeded", "Token rate limit exceeded.", 429,
                    error_type="rate_limit_error", headers={"Retry-After": str(max(1, int(e.retry_after_seconds)))},
                )

            t0 = time.time()
            if stream:
                # Streaming has NO fallback ladder (unlike the non-stream path
                # via _forward_with_fallback): a partial SSE stream cannot be
                # replayed, so the first endpoint error fails the stream.
                response = web.StreamResponse(
                    status=200,
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "X-Gateway-Tier": decision.tier,
                        "X-Gateway-Source": decision.source,
                        "X-Gateway-Decision-Id": str(decision_id),
                        "X-Gateway-Ms-Decision": f"{r.ms_classify:.1f}",
                        "X-Gateway-Memory-Tokens": str(memory_ctx.total_tokens_estimate),
                    },
                )
                stream_iter = endpoints.stream_passthrough(decision.endpoint, transcoded, tenant_id=tenant_id).__aiter__()
                try:
                    first_chunk = await anext(stream_iter)
                except Exception as e:
                    ms_total = (time.time() - t0) * 1000
                    await asyncio.to_thread(_finalize_decision, decision_id, ms_total, False, str(e))
                    _record_request_metrics(decision.tier, decision.endpoint, False, ms_total)
                    await asyncio.to_thread(
                        tenant_mgr.settle_usage,
                        tenant_id,
                        reserved_tokens_in=ctx.estimated_input_tokens,
                        reserved_tokens_out=ctx.estimated_output_tokens,
                        reserved_cost_usd=decision.cost_usd,
                        actual_tokens_in=0,
                        actual_tokens_out=0,
                        actual_cost_usd=0.0,
                        completed=False,
                        endpoint_name=decision.endpoint,
                    )
                    status = e.status if isinstance(e, endpoints.EndpointHTTPError) and 400 <= e.status < 500 else 502
                    return _openai_error(
                        "upstream_failed", "The selected upstream rejected the streaming request.", status,
                        error_type="server_error" if status >= 500 else "invalid_request_error",
                    )
                await response.prepare(request)
                try:
                    collected_text = ""
                    await response.write(first_chunk)
                    first_text = first_chunk.decode("utf-8", errors="ignore")
                    first_match = re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', first_text)
                    if first_match:
                        collected_text += json.loads(f'"{first_match.group(1)}"')
                    async for chunk in stream_iter:
                        await response.write(chunk)
                        # Try to extract text from SSE chunks for memory recording
                        try:
                            s = chunk.decode("utf-8", errors="ignore")
                            if '"content":"' in s:
                                m = re.search(r'"content":"((?:[^"\\]|\\.)*)"', s)
                                if m:
                                    collected_text += m.group(1).encode().decode("unicode_escape", errors="ignore")
                        except Exception:
                            pass
                    await response.write_eof()
                    response_ok = True
                    error = None
                except Exception as e:
                    log.warning("endpoint stream failed: %s", e)
                    response_ok = False
                    error = str(e)
                ms_total = (time.time() - t0) * 1000
                await asyncio.to_thread(
                    _finalize_decision,
                    decision_id,
                    ms_total,
                    response_ok,
                    error,
                    actual_cost_usd=decision.cost_usd if response_ok else None,
                )
                _record_request_metrics(decision.tier, decision.endpoint, response_ok, ms_total)
                await asyncio.to_thread(
                    tenant_mgr.settle_usage,
                    tenant_id,
                    reserved_tokens_in=ctx.estimated_input_tokens,
                    reserved_tokens_out=ctx.estimated_output_tokens,
                    reserved_cost_usd=decision.cost_usd,
                    actual_tokens_in=ctx.estimated_input_tokens if response_ok else 0,
                    actual_tokens_out=ctx.estimated_output_tokens if response_ok else 0,
                    actual_cost_usd=decision.cost_usd if response_ok else 0.0,
                    completed=response_ok,
                    endpoint_name=decision.endpoint,
                )
                await asyncio.to_thread(
                    memory.update_session,
                    session_id, tenant_id, tier=decision.tier, vertical=r.vertical,
                    endpoint=decision.endpoint, response_ok=response_ok,
                    response_ms=ms_total,
                )
                if memory_enabled and response_ok and collected_text:
                    try:
                        await asyncio.to_thread(
                            om.record_message,
                            resource_id=tenant_id,
                            thread_id=session_id,
                            role="assistant",
                            content=collected_text,
                            token_estimate=len(collected_text) // 4,
                            metadata={"tier": decision.tier, "model_version": r.model_version, "streamed": True},
                            embed=recall_enabled,
                        )
                    except Exception:
                        pass
                if response_ok:
                    await asyncio.to_thread(
                        reviewer.enqueue_for_review,
                        decision_id, tenant_id, cost_estimate=decision.cost_usd, prompt_text=text,
                    )
                if memory_enabled:
                    _maybe_force_observe(request.app, tenant_id, session_id)
                events.emit_status(
                    events.EventSource.ROUTING,
                    f"Done ({ms_total:.0f}ms, {decision.tier})",
                    done=True,
                    tenant_id=tenant_id,
                    session_id=session_id,
                )
                return response
            else:
                try:
                    resp_data, actual_endpoint, actual_tier, attempts = await _forward_with_fallback(
                        request.app, conf, decision, forwarded_body, breaker_states, tenant_id,
                    )
                except Exception as e:
                    log.warning("all endpoints failed for decision %d: %s", decision_id, e)
                    ms_total = (time.time() - t0) * 1000
                    await asyncio.to_thread(_finalize_decision, decision_id, ms_total, False, str(e))
                    _record_request_metrics(decision.tier, decision.endpoint, False, ms_total)
                    await asyncio.to_thread(
                        tenant_mgr.settle_usage,
                        tenant_id,
                        reserved_tokens_in=ctx.estimated_input_tokens,
                        reserved_tokens_out=ctx.estimated_output_tokens,
                        reserved_cost_usd=decision.cost_usd,
                        actual_tokens_in=0,
                        actual_tokens_out=0,
                        actual_cost_usd=0.0,
                        completed=False,
                        endpoint_name=decision.endpoint,
                    )
                    return _openai_error(
                        "upstream_failed", "All eligible upstream providers failed.", 502,
                        error_type="server_error",
                    )
                ms_total = (time.time() - t0) * 1000
                used_fallback = actual_endpoint != decision.endpoint
                usage = resp_data.get("usage", {})
                actual_cost = _actual_endpoint_cost(conf, actual_endpoint, usage)
                await asyncio.to_thread(
                    _finalize_decision, decision_id, ms_total, True, None, actual_cost_usd=actual_cost,
                )
                _record_request_metrics(actual_tier, actual_endpoint, True, ms_total)
                if used_fallback:
                    await asyncio.to_thread(
                        memory.update_routing_decision,
                        decision_id, endpoint=actual_endpoint, tier=actual_tier, fallback_used=True,
                    )
                await asyncio.to_thread(
                    memory.update_session,
                    session_id, tenant_id, tier=actual_tier, vertical=r.vertical,
                    endpoint=actual_endpoint, response_ok=True,
                    response_ms=ms_total,
                )
                # Record assistant message in memory domain (for OM to compress)
                if memory_enabled:
                    try:
                        assistant_text = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        if assistant_text:
                            await asyncio.to_thread(
                                om.record_message,
                                resource_id=tenant_id,
                                thread_id=session_id,
                                role="assistant",
                                content=assistant_text,
                                token_estimate=len(assistant_text) // 4,
                                metadata={"tier": actual_tier, "model_version": r.model_version, "attempts": attempts},
                                embed=recall_enabled,
                            )
                    except Exception as e:
                        log.warning("record assistant message failed: %s", e)
                # Enqueue for async review
                await asyncio.to_thread(
                    reviewer.enqueue_for_review,
                    decision_id, tenant_id, cost_estimate=actual_cost, prompt_text=text,
                )
                # Record usage
                await asyncio.to_thread(
                    tenant_mgr.settle_usage,
                    tenant_id,
                    reserved_tokens_in=ctx.estimated_input_tokens,
                    reserved_tokens_out=ctx.estimated_output_tokens,
                    reserved_cost_usd=decision.cost_usd,
                    actual_tokens_in=int(usage.get("prompt_tokens", 0)),
                    actual_tokens_out=int(usage.get("completion_tokens", 0)),
                    actual_cost_usd=actual_cost,
                    completed=True,
                    endpoint_name=actual_endpoint,
                )
                if memory_enabled:
                    _maybe_force_observe(request.app, tenant_id, session_id)
                events.emit_status(
                    events.EventSource.ROUTING,
                    f"Done ({ms_total:.0f}ms, {actual_tier})",
                    done=True,
                    tenant_id=tenant_id,
                    session_id=session_id,
                )
                return web.json_response(resp_data)


def _finalize_decision(
    decision_id: int,
    ms_total: float,
    response_ok: bool,
    error: str | None,
    actual_cost_usd: float | None = None,
):
    from sqlalchemy import update as sql_update
    with memory.begin() as conn:
        conn.execute(
            sql_update(memory.routing_log)
            .where(memory.routing_log.c.id == decision_id)
            .values(ms_total=ms_total, response_ok=response_ok, error=error, actual_cost_usd=actual_cost_usd)
        )


def _record_request_metrics(tier: str, endpoint: str, response_ok: bool, ms_total: float):
    """Prometheus counters for completed chat requests."""
    reg = metrics_mod.registry()
    reg.inc("glint_requests_total", {"tier": tier, "endpoint": endpoint, "status": "ok" if response_ok else "error"})
    reg.inc("glint_request_duration_ms", {"tier": tier}, value=ms_total)
    if not response_ok:
        reg.inc("glint_upstream_failures_total", {"endpoint": endpoint})


def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # OpenAI multi-part: extract text parts
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return " ".join(parts)
    return ""


def _has_image(content) -> bool:
    if isinstance(content, list):
        return any(isinstance(p, dict) and p.get("type") == "image_url" for p in content)
    return False


def _estimate_input_tokens(messages: list) -> int:
    total_chars = sum(len(_extract_text(m.get("content", ""))) for m in messages)
    return total_chars // 4  # rough


def _estimate_output_tokens(r: router_mod.RouterOutput) -> int:
    if r.flags.get("long_output"):
        return 1024
    if r.complexity >= 4:
        return 768
    return 256


def _estimate_endpoint_cost(
    conf: cfg_mod.Config,
    endpoint_name: str,
    ctx: policy_mod.RequestContext,
    fit: float = 1.0,
) -> float:
    ep = conf.endpoint(endpoint_name)
    if not ep:
        return 0.0
    pricing = ep.get("pricing", {})
    retry_penalty = conf.config.get("routing", {}).get("cost_first", {}).get("retry_penalty_multiplier", 5.0)
    return policy_mod.expected_cost(
        fixed_per_request=float(pricing.get("fixed_per_request", 0.0)),
        in_per_1k=float(pricing.get("in_per_1k_tokens", 0.0)),
        out_per_1k=float(pricing.get("out_per_1k_tokens", 0.0)),
        estimated_in_tokens=ctx.estimated_input_tokens,
        estimated_out_tokens=ctx.estimated_output_tokens,
        fit=fit if fit > 0 else 1.0,
        retry_penalty_multiplier=float(retry_penalty),
    )


def _actual_endpoint_cost(conf: cfg_mod.Config, endpoint_name: str, usage: dict) -> float:
    ep = conf.endpoint(endpoint_name)
    if not ep:
        return 0.0
    pricing = ep.get("pricing", {})
    return (
        float(pricing.get("fixed_per_request", 0.0))
        + (int(usage.get("prompt_tokens", 0)) / 1000.0) * float(pricing.get("in_per_1k_tokens", 0.0))
        + (int(usage.get("completion_tokens", 0)) / 1000.0) * float(pricing.get("out_per_1k_tokens", 0.0))
    )


def _direct_model_route(
    conf: cfg_mod.Config,
    requested_model,
    tenant_mgr: tenant.TenantManager,
    tenant_id: str,
    breaker_states: dict[str, str],
) -> policy_mod.RoutingDecision | None:
    if requested_model in (None, "", "gateway") or not isinstance(requested_model, str):
        return None
    endpoints_by_name = {ep["name"]: ep for ep in conf.config.get("endpoints", [])}
    endpoint_cfg = endpoints_by_name.get(requested_model)
    if endpoint_cfg is None:
        endpoint_cfg = next(
            (ep for ep in conf.config.get("endpoints", []) if ep.get("model_alias") == requested_model),
            None,
        )
    if endpoint_cfg is None or breaker_states.get(endpoint_cfg["name"]) == "OPEN":
        return None
    for tier_cfg in conf.config.get("tiers", []):
        if endpoint_cfg["name"] not in tier_cfg.get("endpoints", []):
            continue
        if not tenant_mgr.can_access_tier(tenant_id, tier_cfg["name"]):
            continue
        return policy_mod.RoutingDecision(
            tier=tier_cfg["name"],
            endpoint=endpoint_cfg["name"],
            source="model_direct",
            rationale=f"explicit model={requested_model}",
        )
    return None


def _ensure_context_capacity(
    conf: cfg_mod.Config,
    decision: policy_mod.RoutingDecision,
    ctx: policy_mod.RequestContext,
    tenant_mgr: tenant.TenantManager,
    tenant_id: str,
    breaker_states: dict[str, str],
) -> bool:
    reserve_pct = float(conf.policy.get("ladder", {}).get("context_reserve_pct", 25))

    def capacity(tier_cfg: dict, endpoint_cfg: dict) -> int:
        configured = min(
            int(tier_cfg.get("max_context", 32768)),
            int(endpoint_cfg.get("max_context", tier_cfg.get("max_context", 32768))),
        )
        return int(configured * max(0.0, 1.0 - reserve_pct / 100.0))

    current_tier = conf.tier(decision.tier)
    current_endpoint = conf.endpoint(decision.endpoint)
    if current_tier and current_endpoint:
        required = ctx.estimated_input_tokens + max(
            ctx.estimated_output_tokens, int(current_tier.get("max_tokens_bump", 0)),
        )
        if required <= capacity(current_tier, current_endpoint):
            return True
    else:
        required = ctx.estimated_input_tokens + ctx.estimated_output_tokens

    candidates = []
    current_max = int(current_tier.get("max_context", 0)) if current_tier else 0
    for tier_cfg in conf.config.get("tiers", []):
        if tier_cfg.get("override_only") or int(tier_cfg.get("max_context", 0)) < current_max:
            continue
        if not tenant_mgr.can_access_tier(tenant_id, tier_cfg["name"]):
            continue
        for endpoint_cfg in conf.endpoints_for_tier(tier_cfg["name"]):
            if breaker_states.get(endpoint_cfg["name"]) == "OPEN":
                continue
            tier_required = ctx.estimated_input_tokens + max(
                ctx.estimated_output_tokens, int(tier_cfg.get("max_tokens_bump", 0)),
            )
            usable = capacity(tier_cfg, endpoint_cfg)
            if tier_required <= usable:
                candidates.append((usable, tier_cfg["name"], endpoint_cfg["name"]))
    if not candidates:
        return False
    _, tier_name, endpoint_name = min(candidates)
    if tier_name != decision.tier or endpoint_name != decision.endpoint:
        decision.rationale += f"; context_redirect:{decision.tier}->{tier_name}"
        decision.source = f"{decision.source}_context"
        decision.tier = tier_name
        decision.endpoint = endpoint_name
        decision.escalated = True
    return True


def _first_endpoint_for_tier(conf: cfg_mod.Config, tier_name: str, breaker_states: dict[str, str]) -> str:
    endpoints = conf.endpoints_for_tier(tier_name)
    for ep in endpoints:
        if breaker_states.get(ep["name"]) != "OPEN":
            return ep["name"]
    return endpoints[0]["name"] if endpoints else ""


def _breaker_config(endpoint_cfg: dict) -> circuit.BreakerConfig:
    breaker = endpoint_cfg.get("breaker", {})
    return circuit.BreakerConfig(
        failure_threshold=int(breaker.get("failure_threshold", 3)),
        open_duration_seconds=float(breaker.get("open_duration_seconds", 60)),
        half_open_max_probes=int(breaker.get("half_open_max_probes", 1)),
    )


async def _forward_with_fallback(
    app: web.Application,
    conf: cfg_mod.Config,
    decision: policy_mod.RoutingDecision,
    forwarded_body: dict,
    breaker_states: dict[str, str],
    tenant_id: str,
) -> tuple[dict, str, str, int]:
    """Try the chosen endpoint, then other endpoints in the same tier, then
    the configured fallback endpoint. Returns (resp_data, endpoint, tier, attempts).
    Raises the last exception if all candidates fail."""
    pool = app["endpoint_pool"]
    fallback_name = conf.config.get("routing", {}).get("cost_first", {}).get("fallback_endpoint", "intel")

    # Build candidate list: (endpoint_name, tier_name)
    candidates: list[tuple[str, str]] = [(decision.endpoint, decision.tier)]
    for ep in conf.endpoints_for_tier(decision.tier):
        if ep["name"] != decision.endpoint and breaker_states.get(ep["name"]) != "OPEN":
            candidates.append((ep["name"], decision.tier))
    # Fallback endpoint (find its tier, or just use its own config)
    if fallback_name and fallback_name not in [c[0] for c in candidates]:
        fb_tier = None
        for t in conf.config.get("tiers", []):
            if fallback_name in t.get("endpoints", []):
                fb_tier = t["name"]
                break
        if fb_tier:
            candidates.append((fallback_name, fb_tier))

    last_error: Exception = RuntimeError("no candidates available")
    attempts = 0
    for ep_name, tier_name in candidates:
        ep_cfg = conf.endpoint(ep_name)
        tier_cfg = conf.tier(tier_name)
        if not ep_cfg or not tier_cfg:
            continue
        br = circuit.registry().get(ep_name, _breaker_config(ep_cfg))
        if not br.allow():
            continue
        attempts += 1
        try:
            transcoded = transcoder.transcode(ep_cfg, tier_cfg, forwarded_body)
            client = pool.get(ep_name)
            resp_data = await client.send(transcoded, stream=False, tenant_id=tenant_id)
            if not isinstance(resp_data, dict):
                raise RuntimeError(f"endpoint {ep_name} returned a non-dict response")
            if attempts > 1:
                log.info("fallback succeeded: %s/%s (attempt %d) for decision tier=%s",
                         ep_name, tier_name, attempts, decision.tier)
            return resp_data, ep_name, tier_name, attempts
        except Exception as e:
            last_error = e
            log.warning("endpoint %s failed (attempt %d): %s", ep_name, attempts, e)
            if isinstance(e, endpoints.EndpointHTTPError) and not e.retryable:
                raise
            continue

    raise last_error


def _maybe_force_observe(app: web.Application, tenant_id: str, session_id: str):
    """If memory.force_observe_on_close is set and the thread crossed the
    message-token threshold, fire the Observer immediately in the background."""
    try:
        conf = app["conf_mgr"].current()
        mem_cfg = conf.config.get("memory", {})
        if not mem_cfg.get("force_observe_on_close", False):
            return
        om_cfg = conf.policy.get("memory", {})
        threshold = om_cfg.get("message_tokens", 12000)
        if om.get_thread_token_total(tenant_id, session_id) < threshold:
            return
        worker = observer_worker.worker()
        loop = asyncio.get_running_loop()

        async def _observe():
            await worker.observe(session_id, tenant_id, om_cfg)

        loop.create_task(_observe())
    except Exception as e:
        log.warning("force_observe_on_close failed: %s", e)


# ============================================================
# Other endpoints
# ============================================================


def _request_is_admin(request: web.Request) -> bool:
    auth_mgr = request.app.get("auth_manager")
    return bool(auth_mgr is not None and auth_mgr.enabled and "admin" in (request.get("auth_scope") or set()))


def _request_tenant_filter(request: web.Request, requested: str | None = None) -> str | None:
    """Tenant filter for user-visible data; None means unrestricted/admin."""
    auth_mgr = request.app.get("auth_manager")
    if auth_mgr is None or not auth_mgr.enabled:
        return requested
    if _request_is_admin(request):
        return requested
    return request.get("tenant_id") or "anonymous"


def _require_resource_owner(request: web.Request, resource_id: str) -> None:
    auth_mgr = request.app.get("auth_manager")
    if auth_mgr is None or not auth_mgr.enabled or _request_is_admin(request):
        return
    if resource_id != (request.get("tenant_id") or "anonymous"):
        raise web.HTTPForbidden(
            text=json.dumps({"error": "forbidden", "detail": "resource belongs to another tenant"}),
            content_type="application/json",
        )


async def list_models(request: web.Request):
    conf = request.app["conf_mgr"].current()
    tenant_mgr = request.app["tenant_mgr"]
    tenant_id = request.get("tenant_id") or request.headers.get("X-User-Id", "anonymous")
    models = [{
        "id": "gateway",
        "object": "model",
        "created": int(time.time()),
        "owned_by": "glint-v2",
        "capabilities": {
            "tiers": [t["name"] for t in conf.config.get("tiers", [])],
            "reviewer_model": conf.reviewer().get("model"),
            "router_version": request.app["router"].model_version(),
        },
    }]
    seen = {"gateway"}
    for endpoint_cfg in conf.config.get("endpoints", []):
        accessible_tiers = [
            tier["name"] for tier in conf.config.get("tiers", [])
            if endpoint_cfg["name"] in tier.get("endpoints", [])
            and tenant_mgr.can_access_tier(tenant_id, tier["name"])
        ]
        if not accessible_tiers:
            continue
        model_id = endpoint_cfg.get("model_alias") or endpoint_cfg["name"]
        if model_id in seen:
            model_id = endpoint_cfg["name"]
        seen.add(model_id)
        models.append({
            "id": model_id,
            "object": "model",
            "created": int(time.time()),
            "owned_by": endpoint_cfg.get("kind", "provider"),
            "gateway_endpoint": endpoint_cfg["name"],
            "gateway_tiers": accessible_tiers,
        })
    return web.json_response({
        "object": "list",
        "data": models,
    })


async def get_stats(request: web.Request):
    pool = request.app["endpoint_pool"]
    breaker_states = circuit.registry().all_states()
    health_map = request.app.get("endpoint_health") or {}
    inflight = pool.all_inflight()
    return web.json_response({
        "ts": datetime.now(UTC).isoformat(),
        "endpoints": {
            name: {
                "kind": ep.cfg.get("kind"),
                "health": health_map.get(name) if name in health_map else breaker_states.get(name, "CLOSED"),
                "breaker": breaker_states.get(name, "CLOSED"),
                "inflight": inflight.get(name, 0),
                "max_concurrency": ep.cfg.get("concurrency", 4),
            }
            for name, ep in pool.clients().items()
        },
        "router_version": request.app["router"].model_version(),
        "policy_version": request.app["conf_mgr"].current().version,
        "config_version": request.app["conf_mgr"].current().version,
    })


async def get_config(request: web.Request):
    conf = request.app["conf_mgr"].current()
    # Strip secrets
    cfg_copy = json.loads(json.dumps(conf.config))
    for ep in cfg_copy.get("endpoints", []):
        if "api_key_env" in ep:
            ep["api_key_env"] = "**redacted**" if ep.get("api_key_env") else None
    if "api_key_env" in cfg_copy.get("reviewer", {}):
        cfg_copy["reviewer"]["api_key_env"] = "**redacted**" if cfg_copy["reviewer"].get("api_key_env") else None
    if "keys" in cfg_copy.get("auth", {}):
        n_keys = len(cfg_copy["auth"]["keys"])
        cfg_copy["auth"]["keys"] = {"**redacted**": f"{n_keys} keys configured"}
    return web.json_response({
        "config": cfg_copy,
        "policy": conf.policy,
        "taxonomy": conf.taxonomy,
        "prototypes": conf.prototypes,
        "version": conf.version,
    })


async def post_reload(request: web.Request):
    conf_mgr = request.app["conf_mgr"]
    try:
        conf = conf_mgr.reload()
        await _apply_runtime_config(request.app, conf)
        return web.json_response({"reloaded": True, "version": conf.version})
    except Exception as e:
        return web.json_response({"reloaded": False, "error": str(e)}, status=500)


async def get_trace(request: web.Request):
    limit = int(request.query.get("limit", 100))
    session_id = request.query.get("session")
    vertical = request.query.get("vertical")
    tenant_id = _request_tenant_filter(request, request.query.get("tenant_id"))
    since_raw = request.query.get("since_hours")
    since_hours = float(since_raw) if since_raw else None
    decisions = memory.get_decisions(
        limit=limit, session_id=session_id, vertical=vertical,
        tenant_id=tenant_id, since_hours=since_hours,
    )
    return web.json_response({"decisions": decisions, "count": len(decisions)})


async def post_feedback(request: web.Request):
    try:
        body = await request.json()
        decision_id = int(body["decision_id"])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return web.json_response({"error": "decision_id required"}, status=400)
    owner = memory.decision_tenant(decision_id)
    if owner is None:
        return web.json_response({"error": "decision not found"}, status=404)
    tenant_filter = _request_tenant_filter(request)
    if tenant_filter is not None and owner != tenant_filter:
        return web.json_response({"error": "forbidden"}, status=403)
    correct = bool(body.get("correct", True))
    suggested_tier = body.get("suggested_tier")
    comment = body.get("comment")
    memory.record_feedback(decision_id, correct, suggested_tier, comment)
    return web.json_response({"recorded": True})


async def get_accuracy(request: web.Request):
    since_raw = request.query.get("since_hours")
    since = float(since_raw) if since_raw else None
    return web.json_response(memory.accuracy_report(since_hours=since, tenant_id=_request_tenant_filter(request)))


async def get_export(request: web.Request):
    decisions = memory.get_decisions(
        limit=int(request.query.get("limit", 1000)), tenant_id=_request_tenant_filter(request),
    )
    out = []
    for d in decisions:
        out.append({
            "text": d["query_preview"],
            "vertical": d["vertical"],
            "complexity": d["complexity"],
            "code": d["flags_code"],
            "math": d["flags_math"],
            "reasoning": d["flags_reasoning"],
            "long_output": d["flags_long_output"],
        })
    return web.json_response({"samples": out, "count": len(out)})


async def get_memory(request: web.Request):
    return web.json_response(memory.session_stats(tenant_id=_request_tenant_filter(request)))


async def get_verticals(request: web.Request):
    since_raw = request.query.get("since_hours")
    since = float(since_raw) if since_raw else None
    tenant_id = _request_tenant_filter(request)
    dist = memory.vertical_distribution(since_hours=since, tenant_id=tenant_id)
    accuracy = memory.accuracy_report(since_hours=since, tenant_id=tenant_id)
    return web.json_response({"distribution": dist, "accuracy_by_vertical": accuracy.get("per_vertical", {})})


async def get_cost(request: web.Request):
    since_raw = request.query.get("since_hours")
    since = float(since_raw) if since_raw else None
    return web.json_response(memory.cost_breakdown(since_hours=since, tenant_id=_request_tenant_filter(request)))


async def get_review_stats(request: web.Request):
    return web.json_response(memory.review_stats(tenant_id=_request_tenant_filter(request)))


async def post_retrain(request: web.Request):
    qs = request.query
    allow_embedding_finetune = qs.get("allow_embedding_finetune") == "true"
    confirm_drift = qs.get("confirm_drift") == "true"
    tw = request.app["trainer_worker"]
    if confirm_drift:
        tw.clear_drift_alarm()
    tw.manual_retrain(allow_embedding_finetune=allow_embedding_finetune)
    return web.json_response({"retrain_triggered": True, "embedding_finetune": allow_embedding_finetune})


async def get_registry(request: web.Request):
    return web.json_response({
        "checkpoints": memory.checkpoint_history(),
        "active_version": memory.active_model_version(),
    })


async def get_model_card(request: web.Request):
    version = request.match_info["version"]
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", version):
        return web.json_response({"error": "invalid_version"}, status=400)
    card_path = Path(f"./router_model/checkpoints/{version}_MODEL_CARD.md")
    if not card_path.exists():
        return web.json_response({"error": "not_found"}, status=404)
    return web.Response(text=card_path.read_text(encoding="utf-8"), content_type="text/markdown")


async def list_users(request: web.Request):
    return web.json_response({"tenants": request.app["tenant_mgr"].all_states()})


async def create_user(request: web.Request):
    body = await request.json()
    tenant_id = body["tenant_id"]
    defaults = {k: v for k, v in body.items() if k != "tenant_id"}
    user = memory.get_or_create_user(tenant_id, defaults=defaults, overwrite=True)
    request.app["tenant_mgr"].refresh(tenant_id)
    return web.json_response(user)


async def get_user_budget(request: web.Request):
    tenant_id = request.match_info["tenant_id"]
    spent = memory.get_today_spend(tenant_id)
    tokens_today = memory.get_today_token_spend(tenant_id)
    tenant_st = request.app["tenant_mgr"].get_or_create(tenant_id)
    return web.json_response({
        "tenant_id": tenant_id,
        "spent_today_usd": spent,
        "tokens_today": tokens_today,
        "daily_usd_limit": tenant_st.budget_usd_per_day,
        "daily_token_limit": tenant_st.daily_token_limit,
        "remaining_tokens_today": request.app["tenant_mgr"].remaining_tokens_today(tenant_id),
        "target_success_probability": tenant_st.target_success_probability,
    })


async def set_user_budget(request: web.Request):
    tenant_id = request.match_info["tenant_id"]
    body = await request.json()
    defaults = {"budget_usd_per_day": body.get("budget_usd_per_day", 1.0)}
    if "tier_access" in body:
        defaults["tier_access"] = body["tier_access"]
    if "daily_token_limit" in body:
        defaults["daily_token_limit"] = int(body["daily_token_limit"])
    if "target_success_probability" in body:
        defaults["target_success_probability"] = float(body["target_success_probability"])
    if "tokens_per_min" in body:
        defaults["tokens_per_min"] = int(body["tokens_per_min"])
    if "rps_limit" in body:
        defaults["rps_limit"] = int(body["rps_limit"])
    if "concurrent_limit" in body:
        defaults["concurrent_limit"] = int(body["concurrent_limit"])
    memory.get_or_create_user(tenant_id, defaults=defaults, overwrite=True)
    request.app["tenant_mgr"].refresh(tenant_id)
    return web.json_response({"updated": True, "tenant_id": tenant_id})


async def get_user_stats(request: web.Request):
    from collections import Counter
    tenant_id = request.match_info["tenant_id"]
    decisions = memory.get_decisions(limit=10000, tenant_id=tenant_id)
    return web.json_response({
        "tenant_id": tenant_id,
        "decision_count": len(decisions),
        "tier_distribution": dict(Counter(d["tier"] for d in decisions)),
        "spent_today_usd": memory.get_today_spend(tenant_id),
    })


# ---- Subscription plans + per-model token limits (admin) ----


async def admin_list_plans(request: web.Request):
    return web.json_response({"plans": await asyncio.to_thread(memory.list_plans)})


async def admin_create_plan(request: web.Request):
    body = await request.json()
    plan_id = body.get("plan_id")
    if not plan_id:
        raise web.HTTPBadRequest(reason="plan_id required")
    plan = await asyncio.to_thread(
        memory.upsert_plan,
        plan_id,
        name=body.get("name"),
        daily_token_limit=body.get("daily_token_limit"),
        daily_usd_limit=body.get("daily_usd_limit"),
        required_success_probability=body.get("required_success_probability"),
        allowed_models=body.get("allowed_models"),
    )
    return web.json_response(plan, status=201)


async def admin_update_plan(request: web.Request):
    plan_id = request.match_info["plan_id"]
    body = await request.json()
    plan = await asyncio.to_thread(
        memory.upsert_plan,
        plan_id,
        name=body.get("name"),
        daily_token_limit=body.get("daily_token_limit"),
        daily_usd_limit=body.get("daily_usd_limit"),
        required_success_probability=body.get("required_success_probability"),
        allowed_models=body.get("allowed_models"),
    )
    return web.json_response(plan)


async def admin_assign_plan(request: web.Request):
    tenant_id = request.match_info["tenant_id"]
    body = await request.json()
    plan_id = body.get("plan_id")
    if not plan_id:
        raise web.HTTPBadRequest(reason="plan_id required")
    plan = await asyncio.to_thread(memory.get_plan, plan_id)
    if not plan:
        raise web.HTTPNotFound(reason=f"plan {plan_id} not found")
    binding = await asyncio.to_thread(
        memory.assign_tenant_plan, tenant_id, plan_id, notes=body.get("notes"),
    )
    request.app["tenant_mgr"].refresh(tenant_id)
    return web.json_response(binding)


async def admin_unassign_plan(request: web.Request):
    tenant_id = request.match_info["tenant_id"]
    # Unbinding = removing the plan_id reference on the user row.
    await asyncio.to_thread(
        memory.get_or_create_user,
        tenant_id,
        defaults={"plan_id": None},
        overwrite=True,
    )
    request.app["tenant_mgr"].refresh(tenant_id)
    return web.json_response({"updated": True, "tenant_id": tenant_id})


async def admin_get_plan(request: web.Request):
    tenant_id = request.match_info["tenant_id"]
    binding = await asyncio.to_thread(memory.get_tenant_plan, tenant_id)
    if not binding:
        return web.json_response({"tenant_id": tenant_id, "plan_id": None})
    quota = await asyncio.to_thread(memory.get_tenant_plan_quota, tenant_id)
    return web.json_response({"binding": binding, "quota": quota})


async def admin_set_daily_token_budget(request: web.Request):
    tenant_id = request.match_info["tenant_id"]
    body = await request.json()
    daily_token_limit = int(body.get("daily_token_limit", 0))
    target_success_probability = body.get("target_success_probability")
    defaults: dict = {"daily_token_limit": daily_token_limit}
    if target_success_probability is not None:
        defaults["target_success_probability"] = float(target_success_probability)
    await asyncio.to_thread(
        memory.get_or_create_user, tenant_id, defaults=defaults, overwrite=True,
    )
    request.app["tenant_mgr"].refresh(tenant_id)
    return web.json_response(
        {"tenant_id": tenant_id, "daily_token_limit": daily_token_limit,
         "target_success_probability": target_success_probability}
    )


async def admin_get_model_limits(request: web.Request):
    tenant_id = request.match_info["tenant_id"]
    endpoint = request.match_info["endpoint"]
    ml = await asyncio.to_thread(memory.get_model_token_limit, tenant_id, endpoint)
    if not ml:
        return web.json_response({"tenant_id": tenant_id, "endpoint": endpoint,
                                  "daily_token_limit": 0, "max_request_tokens": 0,
                                  "daily_usd_limit": 0.0})
    return web.json_response(ml)


async def admin_set_model_limits(request: web.Request):
    tenant_id = request.match_info["tenant_id"]
    endpoint = request.match_info["endpoint"]
    body = await request.json()
    ml = await asyncio.to_thread(
        memory.upsert_model_token_limit,
        tenant_id,
        endpoint,
        daily_token_limit=body.get("daily_token_limit"),
        daily_usd_limit=body.get("daily_usd_limit"),
        max_request_tokens=body.get("max_request_tokens"),
    )
    return web.json_response(ml)


async def admin_get_all_limits(request: web.Request):
    tenant_id = request.match_info["tenant_id"]
    tenant_st = request.app["tenant_mgr"].get_or_create(tenant_id)
    plan_quota = await asyncio.to_thread(memory.get_tenant_plan_quota, tenant_id)
    model_limits = await asyncio.to_thread(memory.list_model_token_limits, tenant_id)
    return web.json_response({
        "tenant_id": tenant_id,
        "daily_token_limit": tenant_st.daily_token_limit,
        "daily_usd_limit": tenant_st.budget_usd_per_day,
        "rps_limit": tenant_st.rps_limit,
        "concurrent_limit": tenant_st.concurrent_limit,
        "tokens_per_min": tenant_st.tokens_per_min,
        "target_success_probability": tenant_st.target_success_probability,
        "plan_id": tenant_st.plan_id,
        "plan_quota": plan_quota,
        "model_limits": model_limits,
        "remaining_tokens_today": request.app["tenant_mgr"].remaining_tokens_today(tenant_id),
    })


async def admin_list_quality_profiles(request: web.Request):
    from sqlalchemy import select

    with memory.engine().connect() as conn:
        rows = conn.execute(
            select(memory.model_quality_profiles)
            .order_by(
                memory.model_quality_profiles.c.endpoint_name,
                memory.model_quality_profiles.c.vertical,
                memory.model_quality_profiles.c.complexity_min,
            )
        ).all()
    return web.json_response({
        "profiles": [
            {
                **dict(r._mapping),
                "success_rate": (
                    int(r.success_count) / int(r.total_count)
                    if int(r.total_count) > 0
                    else 0.0
                ),
            }
            for r in rows
        ]
    })


async def admin_record_quality_sample(request: web.Request):
    body = await request.json()
    endpoint_name = body.get("endpoint_name")
    vertical = body.get("vertical")
    complexity = int(body.get("complexity", 1))
    success = bool(body.get("success", False))
    if not endpoint_name or not vertical:
        raise web.HTTPBadRequest(reason="endpoint_name and vertical required")
    await asyncio.to_thread(
        memory.record_quality_sample, endpoint_name, vertical, complexity, success,
    )
    return web.json_response({"recorded": True})


# ---- User-facing usage + limits ----


async def get_my_usage(request: web.Request):
    tenant_id = request.get("tenant_id") or "anonymous"
    spent = memory.get_today_spend(tenant_id)
    tokens_today = memory.get_today_token_spend(tenant_id)
    return web.json_response({
        "tenant_id": tenant_id,
        "spent_today_usd": spent,
        "tokens_today": tokens_today,
    })


async def get_my_limits(request: web.Request):
    tenant_id = request.get("tenant_id") or "anonymous"
    tenant_st = request.app["tenant_mgr"].get_or_create(tenant_id)
    remaining = request.app["tenant_mgr"].remaining_tokens_today(tenant_id)
    return web.json_response({
        "tenant_id": tenant_id,
        "daily_token_limit": tenant_st.daily_token_limit,
        "daily_usd_limit": tenant_st.budget_usd_per_day,
        "rps_limit": tenant_st.rps_limit,
        "concurrent_limit": tenant_st.concurrent_limit,
        "tokens_per_min": tenant_st.tokens_per_min,
        "target_success_probability": tenant_st.target_success_probability,
        "remaining_tokens_today": remaining,
    })


async def get_provider_presets(request: web.Request):
    """Return the provider preset catalog for the Setup wizard."""
    conf = request.app["conf_mgr"].current()
    return web.json_response(conf.provider_presets)


# ---- Provider / Key / Tier CRUD ----


async def admin_add_endpoint(request: web.Request):
    """Add a new provider endpoint."""
    body = await request.json()
    mgr: admin_mod.OverlayManager = request.app["overlay_manager"]
    try:
        ep = mgr.add_endpoint(body)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    await _apply_runtime_config(request.app, request.app["conf_mgr"].current())
    return web.json_response({"created": True, "endpoint": ep}, status=201)


async def admin_update_endpoint(request: web.Request):
    """Edit an existing endpoint."""
    name = request.match_info["name"]
    body = await request.json()
    mgr = request.app["overlay_manager"]
    try:
        ep = mgr.update_endpoint(name, body)
    except KeyError as e:
        return web.json_response({"error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    await _apply_runtime_config(request.app, request.app["conf_mgr"].current())
    return web.json_response({"updated": True, "endpoint": ep})


async def admin_delete_endpoint(request: web.Request):
    """Remove an endpoint."""
    name = request.match_info["name"]
    mgr = request.app["overlay_manager"]
    try:
        ok = mgr.remove_endpoint(name)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=409)
    if not ok:
        return web.json_response({"error": "not found"}, status=404)
    await _apply_runtime_config(request.app, request.app["conf_mgr"].current())
    return web.json_response({"deleted": True, "name": name})


async def admin_generate_key(request: web.Request):
    """Generate a new gateway API key. Returns the key ONCE."""
    body = await request.json()
    tenant_id = body.get("tenant_id", "anonymous")
    scope = body.get("scope", ["user"])
    mgr = request.app["overlay_manager"]
    try:
        key = mgr.generate_key(tenant_id, scope)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    await _apply_runtime_config(request.app, request.app["conf_mgr"].current())
    return web.json_response({"key": key, "tenant_id": tenant_id, "scope": scope}, status=201)


async def admin_list_keys(request: web.Request):
    """List all API keys (masked)."""
    mgr = request.app["overlay_manager"]
    return web.json_response({"keys": mgr.list_keys()})


async def admin_revoke_key(request: web.Request):
    """Revoke an API key by full value or unique prefix."""
    key = request.match_info["key"]
    mgr = request.app["overlay_manager"]
    try:
        ok = mgr.revoke_key(key)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=409)
    if not ok:
        return web.json_response({"error": "key not found"}, status=404)
    await _apply_runtime_config(request.app, request.app["conf_mgr"].current())
    return web.json_response({"revoked": True})


async def admin_update_tier(request: web.Request):
    """Edit a tier."""
    name = request.match_info["name"]
    body = await request.json()
    mgr = request.app["overlay_manager"]
    try:
        tier = mgr.update_tier(name, body)
    except KeyError as e:
        return web.json_response({"error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=409)
    await _apply_runtime_config(request.app, request.app["conf_mgr"].current())
    return web.json_response({"updated": True, "tier": tier})


async def admin_assign_endpoint(request: web.Request):
    """Add an endpoint to a tier."""
    tier_name = request.match_info["name"]
    endpoint_name = request.match_info["endpoint"]
    mgr = request.app["overlay_manager"]
    try:
        tier = mgr.assign_endpoint_to_tier(tier_name, endpoint_name)
    except KeyError as e:
        return web.json_response({"error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=409)
    await _apply_runtime_config(request.app, request.app["conf_mgr"].current())
    return web.json_response({"assigned": True, "tier": tier})


async def admin_unassign_endpoint(request: web.Request):
    """Remove an endpoint from a tier."""
    tier_name = request.match_info["name"]
    endpoint_name = request.match_info["endpoint"]
    mgr = request.app["overlay_manager"]
    try:
        tier = mgr.remove_endpoint_from_tier(tier_name, endpoint_name)
    except KeyError as e:
        return web.json_response({"error": str(e)}, status=404)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=409)
    await _apply_runtime_config(request.app, request.app["conf_mgr"].current())
    return web.json_response({"unassigned": True, "tier": tier})


async def admin_probe_local(request: web.Request):
    """Scan localhost for running LLM servers (Ollama, LM Studio, vLLM, llama.cpp).
    Returns discovered services with model lists for one-click add."""
    results = await discovery.probe_local()
    return web.json_response({"discovered": results, "count": len(results)})


async def admin_test_endpoint(request: web.Request):
    """Test a provider connection by sending a tiny prompt.
    Body: {base_url, kind, model_alias, api_key_env?, api_key?}
    Returns: {ok, latency_ms, response_preview, error?}"""
    body = await request.json()
    result = await discovery.test_endpoint(
        base_url=body.get("base_url", ""),
        kind=body.get("kind", "openai"),
        model_alias=body.get("model_alias", ""),
        api_key_env=body.get("api_key_env"),
        api_key=body.get("api_key"),
    )
    status = 200 if result.get("ok") else 502
    return web.json_response(result, status=status)


async def admin_revert(request: web.Request):
    """Revert the router to a previously promoted model version."""
    version = request.match_info["version"]
    tw = request.app["trainer_worker"]
    if tw.training_in_progress:
        return web.json_response({"error": "training in progress; wait for completion"}, status=409)
    ok = tw.revert_to_version(version)
    if ok:
        return web.json_response({"reverted": True, "version": version})
    return web.json_response({"error": "revert failed — checkpoint files not found"}, status=404)


async def admin_training_status(request: web.Request):
    """Current training status for the dashboard."""
    tw = request.app["trainer_worker"]
    return web.json_response({
        "training_in_progress": tw.training_in_progress,
        "drift_alarm": tw._drift_alarm_active,
        "router_version": request.app["router"].model_version(),
        "is_stub": request.app["router"].is_stub(),
    })


async def admin_flywheel_graph(request: web.Request):
    """Graph data for the Obsidian-style flywheel visualization.

    Returns nodes (verticals with curated counts + accuracy) and edges
    (confusion pairs from the latest promoted checkpoint).
    """
    from sqlalchemy import func, select

    conf = request.app["conf_mgr"].current()
    verticals = conf.verticals()

    # Curated sample counts per vertical
    curated_counts: dict[str, int] = {}
    try:
        with memory.engine().connect() as conn:
            rows = conn.execute(
                select(memory.curated_samples.c.vertical, func.count())
                .group_by(memory.curated_samples.c.vertical)
            ).all()
            curated_counts = {r[0]: int(r[1]) for r in rows}
    except Exception:
        pass

    # Accuracy per vertical
    accuracy_data = memory.accuracy_report(since_hours=168)
    per_vert = accuracy_data.get("per_vertical", {})

    # Build nodes
    nodes = []
    for v in verticals:
        name = v["name"]
        nodes.append({
            "id": name,
            "label": v.get("display", name),
            "curated": curated_counts.get(name, 0),
            "accuracy": per_vert.get(name, {}).get("accuracy"),
            "correct": per_vert.get(name, {}).get("correct", 0),
            "wrong": per_vert.get(name, {}).get("wrong", 0),
        })

    # Confusion edges from latest promoted checkpoint
    edges: list[dict] = []
    try:
        checkpoints = memory.checkpoint_history(limit=1)
        if checkpoints:
            cp = checkpoints[0]
            if cp.get("confusion_top20"):
                import json as _json
                confusions = _json.loads(cp["confusion_top20"]) if isinstance(cp["confusion_top20"], str) else cp["confusion_top20"]
                for c in confusions[:20]:
                    edges.append({
                        "source": c.get("true", ""),
                        "target": c.get("pred", ""),
                        "count": c.get("count", 1),
                    })
    except Exception:
        pass

    return web.json_response({
        "nodes": nodes,
        "edges": edges,
        "total_curated": sum(curated_counts.values()),
        "active_version": memory.active_model_version(),
    })


async def list_flags(request: web.Request):
    limit = int(request.query.get("limit", 100))
    reason = request.query.get("reason")
    return web.json_response({"flags": memory.list_flagged(limit=limit, reason=reason)})


# ============================================================
# Security Hub — /admin/security/*
# ============================================================


async def admin_security_events(request: web.Request):
    """List security events. Filters: tenant_id, event_type, severity, since, limit."""
    from datetime import datetime as _dt
    tenant_id = request.query.get("tenant_id") or None
    event_type = request.query.get("event_type") or None
    severity = request.query.get("severity") or None
    limit = min(int(request.query.get("limit", 100)), 1000)
    since_raw = request.query.get("since")
    since = None
    if since_raw:
        try:
            since = _dt.fromisoformat(since_raw.replace("Z", "+00:00"))
        except ValueError:
            return _openai_error(
                "invalid_since",
                "since must be ISO 8601.",
                400,
                error_type="invalid_request_error",
            )
    events = await asyncio.to_thread(
        memory.list_security_events,
        tenant_id=tenant_id,
        event_type=event_type,
        severity=severity,
        since=since,
        limit=limit,
    )
    return web.json_response({"events": events, "count": len(events)})


async def admin_security_stats(request: web.Request):
    """Aggregated security stats for the dashboard."""
    tenant_id = request.query.get("tenant_id") or None
    window_days = int(request.query.get("window_days", 7))
    stats = await asyncio.to_thread(
        memory.security_event_stats,
        tenant_id=tenant_id,
        window_days=window_days,
    )
    return web.json_response(stats)


async def admin_list_injection_profiles(request: web.Request):
    enabled_only = request.query.get("enabled_only", "false").lower() in ("1", "true", "yes")
    profiles = await asyncio.to_thread(memory.list_injection_profiles, enabled_only)
    return web.json_response({"profiles": profiles, "count": len(profiles)})


async def admin_create_injection_profile(request: web.Request):
    try:
        body = await request.json()
    except Exception:
        return _openai_error("invalid_body", "JSON body required.", 400, error_type="invalid_request_error")
    name = body.get("name")
    regexes = body.get("regexes") or []
    severity = body.get("severity", "medium")
    action = body.get("action", "alert")
    enabled = bool(body.get("enabled", True))
    if not name or not isinstance(regexes, list):
        return _openai_error(
            "invalid_profile",
            "name (str) and regexes (list[str]) are required.",
            400,
            error_type="invalid_request_error",
        )
    try:
        pid = await asyncio.to_thread(
            memory.create_injection_profile,
            name=name,
            regexes=regexes,
            severity=severity,
            action=action,
            enabled=enabled,
            is_builtin=False,
        )
    except ValueError as e:
        return _openai_error("invalid_profile", str(e), 400, error_type="invalid_request_error")
    return web.json_response({"id": pid, "name": name}, status=201)


async def admin_update_injection_profile(request: web.Request):
    profile_id = int(request.match_info["profile_id"])
    try:
        body = await request.json()
    except Exception:
        return _openai_error("invalid_body", "JSON body required.", 400, error_type="invalid_request_error")
    try:
        ok = await asyncio.to_thread(
            memory.update_injection_profile,
            profile_id,
            name=body.get("name"),
            regexes=body.get("regexes"),
            severity=body.get("severity"),
            action=body.get("action"),
            enabled=body.get("enabled"),
        )
    except ValueError as e:
        return _openai_error("invalid_profile", str(e), 400, error_type="invalid_request_error")
    if not ok:
        return _openai_error("not_found", f"profile {profile_id} not found.", 404)
    # Refresh in-memory profiles
    global INJECTION_PROFILES
    try:
        db_profiles = await asyncio.to_thread(memory.list_injection_profiles, True)
        INJECTION_PROFILES = [
            security.InjectionProfile.from_config(
                name=row["name"],
                regexes=row.get("regexes", []),
                severity=row.get("severity", "medium"),
                action=row.get("action", "alert"),
                enabled=row.get("enabled", True),
                is_builtin=row.get("is_builtin", False),
            )
            for row in db_profiles
        ]
    except Exception:
        pass
    return web.json_response({"id": profile_id, "updated": True})


async def admin_delete_injection_profile(request: web.Request):
    profile_id = int(request.match_info["profile_id"])
    try:
        ok = await asyncio.to_thread(memory.delete_injection_profile, profile_id)
    except ValueError as e:
        return _openai_error("cannot_delete", str(e), 400, error_type="invalid_request_error")
    if not ok:
        return _openai_error("not_found", f"profile {profile_id} not found.", 404)
    global INJECTION_PROFILES
    try:
        db_profiles = await asyncio.to_thread(memory.list_injection_profiles, True)
        INJECTION_PROFILES = [
            security.InjectionProfile.from_config(
                name=row["name"],
                regexes=row.get("regexes", []),
                severity=row.get("severity", "medium"),
                action=row.get("action", "alert"),
                enabled=row.get("enabled", True),
                is_builtin=row.get("is_builtin", False),
            )
            for row in db_profiles
        ]
    except Exception:
        pass
    return web.json_response({"id": profile_id, "deleted": True})


async def admin_list_provider_allowlist(request: web.Request):
    tenant_id = request.query.get("tenant_id") or None
    rules = await asyncio.to_thread(memory.list_provider_allowlist, tenant_id)
    return web.json_response({"rules": rules, "count": len(rules)})


async def admin_upsert_provider_allowlist(request: web.Request):
    try:
        body = await request.json()
    except Exception:
        return _openai_error("invalid_body", "JSON body required.", 400, error_type="invalid_request_error")
    tenant_id = body.get("tenant_id", "*")
    pattern = body.get("pattern") or body.get("domain_pattern")
    action = body.get("action", "allow")
    notes = body.get("notes")
    if not pattern:
        return _openai_error(
            "invalid_rule",
            "pattern (str) is required.",
            400,
            error_type="invalid_request_error",
        )
    try:
        await asyncio.to_thread(
            memory.upsert_provider_allowlist,
            tenant_id, pattern, action, notes,
        )
    except ValueError as e:
        return _openai_error("invalid_rule", str(e), 400, error_type="invalid_request_error")
    # Reload enforcer
    global FIREWALL
    if FIREWALL is not None:
        try:
            fw_cfg = request.app["conf_mgr"].current().config.get(
                "security", {},
            ).get("provider_allowlist", {})
            FIREWALL.load_from_config(fw_cfg)
            db_rules = await asyncio.to_thread(memory.list_provider_allowlist)
            FIREWALL.load_from_db(db_rules)
            # Re-sync host firewall
            if HOST_FIREWALL and HOST_FIREWALL.enabled:
                patterns = [r.pattern for r in FIREWALL.list_rules() if r.tenant_id == "*"]
                HOST_FIREWALL.sync(patterns)
        except Exception:
            pass
    return web.json_response({"tenant_id": tenant_id, "pattern": pattern, "action": action}, status=201)


async def admin_delete_provider_allowlist(request: web.Request):
    tenant_id = request.match_info["tenant_id"]
    pattern = request.match_info["pattern"]
    ok = await asyncio.to_thread(
        memory.delete_provider_allowlist, tenant_id, pattern,
    )
    if not ok:
        return _openai_error("not_found", "rule not found.", 404)
    global FIREWALL
    if FIREWALL is not None:
        try:
            fw_cfg = request.app["conf_mgr"].current().config.get(
                "security", {},
            ).get("provider_allowlist", {})
            FIREWALL.load_from_config(fw_cfg)
            db_rules = await asyncio.to_thread(memory.list_provider_allowlist)
            FIREWALL.load_from_db(db_rules)
        except Exception:
            pass
    return web.json_response({"deleted": True, "tenant_id": tenant_id, "pattern": pattern})


async def admin_sync_host_firewall(request: web.Request):
    """Manually trigger host firewall sync."""
    if HOST_FIREWALL is None:
        return _openai_error("no_firewall", "host firewall not configured.", 400)
    try:
        patterns = [r.pattern for r in (FIREWALL.list_rules() if FIREWALL else []) if r.tenant_id == "*"]
        HOST_FIREWALL.sync(patterns)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)
    state = HOST_FIREWALL.state
    return web.json_response({
        "ok": state.in_sync,
        "enabled": state.enabled,
        "platform": state.platform,
        "rules": [
            {"pattern": r.pattern, "ip": r.ip, "rule_name": r.rule_name}
            for r in state.rules
        ],
        "last_sync_error": state.last_sync_error,
    })


async def admin_security_status(request: web.Request):
    """Single-shot status for the security tab: enforcer stats + host fw state + DB counts."""
    enforcer_stats = FIREWALL.stats if FIREWALL else None
    host_state = HOST_FIREWALL.state if HOST_FIREWALL else None
    fw_cfg = request.app["conf_mgr"].current().config.get(
        "security", {},
    ).get("provider_allowlist", {})
    return web.json_response({
        "firewall": {
            "enabled": bool(fw_cfg.get("enabled", False)),
            "default_action": fw_cfg.get("default_action", "block"),
            "in_process_stats": {
                "checks_total": enforcer_stats.checks_total if enforcer_stats else 0,
                "blocks_total": enforcer_stats.blocks_total if enforcer_stats else 0,
                "alerts_total": enforcer_stats.alerts_total if enforcer_stats else 0,
            } if enforcer_stats else None,
            "rules_count": len(FIREWALL.list_rules()) if FIREWALL else 0,
            "host_firewall": {
                "enabled": host_state.enabled if host_state else False,
                "platform": host_state.platform if host_state else "auto",
                "in_sync": host_state.in_sync if host_state else False,
                "rules_count": len(host_state.rules) if host_state else 0,
                "last_sync_error": host_state.last_sync_error if host_state else None,
            } if host_state else None,
        },
        "injection_profiles_loaded": len(INJECTION_PROFILES),
    })


async def admin_security_test(request: web.Request):
    """Test endpoint — checks a URL/domain against the in-process firewall without
    actually making an HTTP request. Useful for debugging."""
    try:
        body = await request.json()
    except Exception:
        return _openai_error("invalid_body", "JSON body required.", 400, error_type="invalid_request_error")
    url = body.get("url")
    tenant_id = body.get("tenant_id") or "anonymous"
    if not url:
        return _openai_error(
            "invalid_request",
            "url (str) is required.",
            400,
            error_type="invalid_request_error",
        )
    if FIREWALL is None:
        return _openai_error("no_firewall", "firewall not initialized.", 400)
    result = FIREWALL.check_outbound(url, tenant_id)
    return web.json_response({
        "url": url,
        "tenant_id": tenant_id,
        "allowed": result.allowed,
        "action": result.action,
        "matched_pattern": result.matched_pattern,
        "reason": result.reason,
    })


async def health(request: web.Request):
    return web.json_response({"status": "ok", "ts": datetime.now(UTC).isoformat()})


async def ready(request: web.Request):
    """Readiness for LB: OK only when DB reachable AND at least one endpoint
    is healthy (live health probe, falling back to breaker state)."""
    try:
        memory.engine().connect().close()
    except Exception:
        return web.json_response({"ready": False, "reason": "db_unavailable"}, status=503)
    if request.app["conf_mgr"].current().config.get("embedding", {}).get("require_real_model", False):
        if request.app["router"].is_stub():
            return web.json_response({"ready": False, "reason": "router_model_unavailable"}, status=503)
    health_map = request.app.get("endpoint_health") or {}
    breaker_states = circuit.registry().all_states()
    for ep in request.app["conf_mgr"].current().config.get("endpoints", []):
        name = ep["name"]
        live = health_map.get(name)
        if live is None:
            live = breaker_states.get(name, "CLOSED") != "OPEN"
        if live:
            return web.json_response({"ready": True, "endpoints": len(health_map) or len(request.app["conf_mgr"].current().config.get("endpoints", []))})
    return web.json_response({"ready": False, "reason": "no_healthy_endpoints"}, status=503)


async def get_metrics(request: web.Request):
    """Prometheus text exposition."""
    reg = metrics_mod.registry()
    # Live gauges
    pool = request.app["endpoint_pool"]
    for name, count in pool.all_inflight().items():
        reg.set_gauge("glint_endpoint_inflight", {"endpoint": name}, count)
    for name, state in circuit.registry().all_states().items():
        reg.set_gauge("glint_breaker_open", {"endpoint": name}, 1.0 if state == "OPEN" else 0.0)
    try:
        reg.set_gauge("glint_review_queue_depth", {}, float(memory.review_stats().get("pending", 0)))
    except Exception:
        pass
    return web.Response(text=reg.render(), content_type="text/plain; version=0.0.4")


async def root_index(request: web.Request):
    return web.Response(text="Glint-V2 Gateway. See /dashboard for the live SPA.", content_type="text/plain")


async def dashboard_spa(request: web.Request):
    return web.Response(text=(Path(__file__).parent / "dashboard" / "index.html").read_text(encoding="utf-8"), content_type="text/html")


# ============================================================
# Observational memory + event endpoints
# ============================================================


async def get_memory_context(request: web.Request):
    """Assemble the three-tier memory context for a thread."""
    default_resource: str = str(
        request.get("tenant_id") or request.headers.get("X-User-Id", "anonymous")
    )
    resource_id: str = str(request.query.get("resource_id") or default_resource)
    _require_resource_owner(request, resource_id)
    thread_id: str = str(request.query.get("thread_id") or request.headers.get("X-Session-Id", "default"))
    query = request.query.get("query", "")
    conf = request.app["conf_mgr"].current()
    ctx = om.load_memory_context(
        conf=conf,
        resource_id=resource_id,
        thread_id=thread_id,
        query_text=query,
    )
    return web.json_response({
        "resource_id": ctx.resource_id,
        "thread_id": ctx.thread_id,
        "tier_breakdown": ctx.tier_breakdown,
        "total_tokens_estimate": ctx.total_tokens_estimate,
        "recency_count": len(ctx.recency_messages),
        "working_memory_chars": len(ctx.working_memory_content or ""),
        "observations_chars": len(ctx.observations or ""),
        "reflection_chars": len(ctx.reflection or ""),
        "recalled_count": len(ctx.recalled_messages),
        "has_observations": ctx.observations is not None,
        "has_reflection": ctx.reflection is not None,
    })


async def get_working_memory(request: web.Request):
    resource_id = request.match_info["resource_id"]
    _require_resource_owner(request, resource_id)
    om.ensure_working_memory(resource_id)
    wm = om._load_working_memory(resource_id)
    return web.json_response({
        "resource_id": resource_id,
        "content": wm,
    })


async def update_working_memory_ep(request: web.Request):
    resource_id = request.match_info["resource_id"]
    _require_resource_owner(request, resource_id)
    body = await request.json()
    content = body.get("content", "")
    if not content:
        return web.json_response({"error": "content required"}, status=400)
    om.update_working_memory(resource_id, content, source=body.get("source", "admin"))
    return web.json_response({"updated": True, "resource_id": resource_id})


async def get_observations_ep(request: web.Request):
    resource_id = request.match_info["resource_id"]
    _require_resource_owner(request, resource_id)
    thread_id = request.match_info["thread_id"]
    obs = om._load_latest_observations(resource_id, thread_id)
    refl = om._load_latest_reflection(resource_id, thread_id)
    return web.json_response({
        "resource_id": resource_id,
        "thread_id": thread_id,
        "observations": obs,
        "reflection": refl,
    })


async def get_reflection_ep(request: web.Request):
    resource_id = request.match_info["resource_id"]
    _require_resource_owner(request, resource_id)
    thread_id = request.match_info["thread_id"]
    refl = om._load_latest_reflection(resource_id, thread_id)
    return web.json_response({"resource_id": resource_id, "thread_id": thread_id, "reflection": refl})


async def force_observe_ep(request: web.Request):
    """Manually trigger an observation pass on a thread."""
    body = await request.json()
    thread_id = body.get("thread_id")
    resource_id = body.get("resource_id")
    if not thread_id or not resource_id:
        return web.json_response({"error": "thread_id and resource_id required"}, status=400)
    _require_resource_owner(request, resource_id)
    conf = request.app["conf_mgr"].current()
    om_cfg = conf.policy.get("memory", {})
    worker = observer_worker.worker()
    await worker.observe(thread_id, resource_id, om_cfg)
    return web.json_response({"observed": True, "thread_id": thread_id})


async def force_reflect_ep(request: web.Request):
    """Manually trigger a reflection pass on a thread."""
    body = await request.json()
    thread_id = body.get("thread_id")
    resource_id = body.get("resource_id")
    if not thread_id or not resource_id:
        return web.json_response({"error": "thread_id and resource_id required"}, status=400)
    _require_resource_owner(request, resource_id)
    conf = request.app["conf_mgr"].current()
    om_cfg = conf.policy.get("memory", {})
    worker = observer_worker.worker()
    await worker.reflect(resource_id, thread_id, om_cfg)
    return web.json_response({"reflected": True, "thread_id": thread_id})


def _events_tenant_filter(request: web.Request) -> str | None:
    """When auth is enabled, non-admin viewers only see their own tenant's
    events (events without a tenant are always visible). Admin scope sees all."""
    auth_mgr = request.app.get("auth_manager")
    if auth_mgr is not None and auth_mgr.enabled:
        scope = request.get("auth_scope") or set()
        if "admin" not in scope:
            return request.get("tenant_id") or "anonymous"
    return None


async def events_stream(request: web.Request):
    """Server-Sent Events stream of gateway events (for live UI)."""
    source_filter = request.query.get("source")
    src = None
    if source_filter:
        try:
            src = events.EventSource(source_filter)
        except ValueError:
            pass
    tenant_filter = _events_tenant_filter(request)

    response = web.StreamResponse(
        status=200,
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)
    bus = events.bus()
    try:
        async for ev in bus.stream(src, tenant_id=tenant_filter):
            await response.write(f"id: {ev.id}\n".encode())
            await response.write(f"event: {ev.type}\n".encode())
            await response.write(f"data: {ev.to_json()}\n\n".encode())
    except (asyncio.CancelledError, ConnectionResetError):
        pass
    finally:
        try:
            await response.write_eof()
        except (ConnectionResetError, RuntimeError):
            pass
    return response


async def events_recent(request: web.Request):
    """Return recent events from the ring buffer (for non-streaming clients)."""
    limit = int(request.query.get("limit", 50))
    source_filter = request.query.get("source")
    src = None
    if source_filter:
        try:
            src = events.EventSource(source_filter)
        except ValueError:
            pass
    tenant_filter = _events_tenant_filter(request)
    recent = events.bus().recent(limit=limit, source=src, tenant_id=tenant_filter)
    return web.json_response({
        "events": [e.to_dict() for e in recent],
        "count": len(recent),
    })


# ============================================================
# Plugin admin handlers
# ============================================================


def _jsonify(value):
    """Convert a value to a JSON-safe form. Datetimes -> ISO strings."""
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonify(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


async def admin_list_plugins(request: web.Request):
    return web.json_response({"plugins": [_jsonify(p) for p in memory.list_plugins()]})


async def admin_get_plugin(request: web.Request):
    name = request.match_info["name"]
    plugin = memory.get_plugin(name)
    if plugin is None:
        return web.json_response({"error": "plugin not found"}, status=404)
    return web.json_response(_jsonify(plugin))


async def admin_upsert_plugin(request: web.Request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    name = body.get("name") or request.match_info.get("name")
    if not name:
        return web.json_response({"error": "name required"}, status=400)
    row = memory.upsert_plugin(
        name=name,
        version=body.get("version"),
        description=body.get("description"),
        prefix=body.get("prefix"),
        module_path=body.get("module_path"),
        config=body.get("config"),
        enabled=bool(body.get("enabled", True)),
        is_builtin=bool(body.get("is_builtin", False)),
    )
    return web.json_response(_jsonify(row))


async def admin_delete_plugin(request: web.Request):
    name = request.match_info["name"]
    if not memory.delete_plugin(name):
        return web.json_response({"error": "plugin is builtin or not found"}, status=400)
    return web.json_response({"deleted": name})


async def admin_reload_plugin(request: web.Request):
    name = request.match_info["name"]
    loader = plugin_mod.loader()
    if loader is None:
        return web.json_response({"error": "plugin loader not initialized"}, status=400)
    ok = loader.reload(name)
    if not ok:
        return web.json_response({"error": "plugin not loaded"}, status=404)
    return web.json_response({"reloaded": name})


async def admin_enable_plugin(request: web.Request):
    name = request.match_info["name"]
    memory.set_plugin_enabled(name, True)
    return web.json_response({"name": name, "enabled": True})


async def admin_disable_plugin(request: web.Request):
    name = request.match_info["name"]
    memory.set_plugin_enabled(name, False)
    return web.json_response({"name": name, "enabled": False})


# ============================================================
# A2A admin handlers
# ============================================================


async def admin_list_a2a_agents(request: web.Request):
    # Admin sees all; tenant_id query param optionally filters.
    tenant_id = request.query.get("tenant_id")
    return web.json_response({"agents": [_jsonify(a) for a in memory.list_a2a_agents(tenant_id=tenant_id)]})


async def admin_create_a2a_agent(request: web.Request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    if not body.get("name") or not body.get("endpoint_url") or not body.get("agent_type"):
        return web.json_response({"error": "name, endpoint_url, agent_type required"}, status=400)
    row = memory.upsert_a2a_agent(
        name=body["name"],
        endpoint_url=body["endpoint_url"],
        agent_type=body["agent_type"],
        description=body.get("description", ""),
        auth_type=body.get("auth_type", "none"),
        auth_value=body.get("auth_value"),
        protocol_version=body.get("protocol_version"),
        capabilities=body.get("capabilities"),
        config=body.get("config"),
        tags=body.get("tags"),
        enabled=bool(body.get("enabled", True)),
        tenant_id=body.get("tenant_id", "__all__"),
    )
    if "error" in row:
        return web.json_response(row, status=400)
    return web.json_response(_jsonify(row))


async def admin_get_a2a_agent(request: web.Request):
    agent_id = int(request.match_info["agent_id"])
    row = memory.get_a2a_agent(agent_id)
    if row is None:
        return web.json_response({"error": "agent not found"}, status=404)
    return web.json_response(_jsonify(row))


async def admin_update_a2a_agent(request: web.Request):
    agent_id = int(request.match_info["agent_id"])
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    existing = memory.get_a2a_agent(agent_id)
    if existing is None:
        return web.json_response({"error": "agent not found"}, status=404)
    row = memory.upsert_a2a_agent(
        name=existing["name"],
        endpoint_url=body.get("endpoint_url", existing["endpoint_url"]),
        agent_type=body.get("agent_type", existing["agent_type"]),
        description=body.get("description", existing.get("description", "")),
        auth_type=body.get("auth_type", existing.get("auth_type", "none")),
        auth_value=body.get("auth_value"),
        protocol_version=body.get("protocol_version", existing.get("protocol_version")),
        capabilities=body.get("capabilities"),
        config=body.get("config"),
        tags=body.get("tags"),
        enabled=bool(body.get("enabled", existing.get("enabled", True))),
        tenant_id=body.get("tenant_id", existing.get("tenant_id", "__all__")),
    )
    return web.json_response(_jsonify(row))


async def admin_delete_a2a_agent(request: web.Request):
    agent_id = int(request.match_info["agent_id"])
    if not memory.delete_a2a_agent(agent_id):
        return web.json_response({"error": "delete failed"}, status=400)
    return web.json_response({"deleted": agent_id})


async def admin_invoke_a2a_agent(request: web.Request):
    agent_id = int(request.match_info["agent_id"])
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    agent = a2a_mod.get_agent(agent_id)
    if agent is None:
        return web.json_response({"error": "agent not found"}, status=404)
    if not agent.enabled:
        return web.json_response({"error": "agent disabled"}, status=400)
    timeout = body.get("timeout_seconds", 30.0)
    tenant_id = body.get("tenant_id") or "admin"
    result = await a2a_mod.invoke_agent(
        agent=agent,
        parameters=body.get("parameters", {}),
        timeout_seconds=float(timeout),
        tenant_id=str(tenant_id),
        interaction_type="admin_invoke",
    )
    return web.json_response(
        {
            "agent_id": agent_id,
            "success": result.success,
            "latency_ms": result.latency_ms,
            "status_code": result.status_code,
            "error": result.error,
            "response": result.response,
        }
    )


async def admin_a2a_agent_metrics(request: web.Request):
    agent_id = int(request.match_info["agent_id"])
    return web.json_response(a2a_mod.metrics_summary(agent_id))


async def admin_list_a2a_servers(request: web.Request):
    return web.json_response({"servers": [_jsonify(s) for s in a2a_mod.list_virtual_servers()]})


async def admin_create_a2a_server(request: web.Request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    if not body.get("name"):
        return web.json_response({"error": "name required"}, status=400)
    row = a2a_mod.register_virtual_server(
        name=body["name"],
        description=body.get("description", ""),
        associated_agents=body.get("associated_agents", []),
        enabled=bool(body.get("enabled", True)),
    )
    if row is None:
        return web.json_response({"error": "create failed"}, status=400)
    return web.json_response(_jsonify(row))


async def admin_delete_a2a_server(request: web.Request):
    server_id = int(request.match_info["server_id"])
    if not a2a_mod.delete_virtual_server(server_id):
        return web.json_response({"error": "delete failed"}, status=400)
    return web.json_response({"deleted": server_id})


# ============================================================
# ContextForge admin handlers
# ============================================================


async def admin_contextforge_sync(request: web.Request):
    client = cf_mod.client()
    if client is None:
        return web.json_response({"error": "ContextForge client not enabled"}, status=400)
    results = await client.sync_all()
    return web.json_response(
        {
            sync_type: {
                "items_synced": r.items_synced,
                "items_added": r.items_added,
                "items_updated": r.items_updated,
                "errors": r.errors,
                "duration_ms": r.duration_ms,
            }
            for sync_type, r in results.items()
        }
    )


async def admin_contextforge_sync_log(request: web.Request):
    return web.json_response({"sync_log": [_jsonify(s) for s in memory.list_contextforge_sync_log(limit=100)]})


async def admin_contextforge_tools(request: web.Request):
    tenant_id = request.query.get("tenant_id")
    return web.json_response({"tools": [_jsonify(t) for t in memory.list_federated_tools(tenant_id=tenant_id)]})


# ============================================================
# MCP discovery admin handler
# ============================================================


async def admin_mcp_discover(request: web.Request):
    try:
        body = await request.json() if request.body_exists else {}
    except Exception:
        body = {}
    hosts = body.get("hosts")
    ports = body.get("ports")
    auto_register = bool(body.get("auto_register", True))
    results = await mcp_disc_mod.discover(
        hosts=hosts, ports=ports, auto_register=auto_register
    )
    return web.json_response(
        {
            "discovered": [
                {
                    "name": r.name,
                    "source_url": r.source_url,
                    "server_info": r.server_info,
                    "capabilities": r.capabilities,
                    "transport": r.transport,
                }
                for r in results
            ]
        }
    )


# ============================================================
# Prompt template admin handlers
# ============================================================


async def admin_list_prompts(request: web.Request):
    enabled_only = request.query.get("enabled_only", "").lower() in ("1", "true", "yes")
    category = request.query.get("category")
    tenant_id = request.query.get("tenant_id")
    return web.json_response(
        {"templates": [_jsonify(t) for t in memory.list_prompt_templates(enabled_only=enabled_only, category=category, tenant_id=tenant_id)]}
    )


async def admin_create_prompt(request: web.Request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    row = memory.upsert_prompt_template(
        name=body.get("name", ""),
        template_text=body.get("template_text", ""),
        description=body.get("description", ""),
        variables=body.get("variables"),
        category=body.get("category", "general"),
        enabled=bool(body.get("enabled", True)),
        source=body.get("source", "admin"),
        tenant_id=body.get("tenant_id", "__all__"),
    )
    if "error" in row:
        return web.json_response(row, status=400)
    return web.json_response(_jsonify(row))


async def admin_get_prompt(request: web.Request):
    template_id = int(request.match_info["template_id"])
    row = memory.get_prompt_template(template_id)
    if row is None:
        return web.json_response({"error": "template not found"}, status=404)
    return web.json_response(_jsonify(row))


async def admin_update_prompt(request: web.Request):
    template_id = int(request.match_info["template_id"])
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    existing = memory.get_prompt_template(template_id)
    if existing is None:
        return web.json_response({"error": "template not found"}, status=404)
    row = memory.upsert_prompt_template(
        name=existing["name"],
        template_text=body.get("template_text", existing["template_text"]),
        description=body.get("description", existing.get("description", "")),
        variables=body.get("variables"),
        category=body.get("category", existing.get("category", "general")),
        enabled=bool(body.get("enabled", existing.get("enabled", True))),
        tenant_id=body.get("tenant_id", existing.get("tenant_id", "__all__")),
    )
    return web.json_response(_jsonify(row))


async def admin_delete_prompt(request: web.Request):
    template_id = int(request.match_info["template_id"])
    if not memory.delete_prompt_template(template_id):
        return web.json_response({"error": "template is builtin or not found"}, status=400)
    return web.json_response({"deleted": template_id})


# ============================================================
# Webhook admin handlers
# ============================================================


async def admin_list_webhooks(request: web.Request):
    tenant_id = request.query.get("tenant_id")
    return web.json_response({"webhooks": [_jsonify(w) for w in memory.list_webhooks(tenant_id=tenant_id)]})


async def admin_create_webhook(request: web.Request):
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    if not body.get("name") or not body.get("url"):
        return web.json_response({"error": "name and url required"}, status=400)
    # max_retries: null = use global default; positive int = override.
    raw_retries = body.get("max_retries")
    max_retries: int | None = None
    if raw_retries is not None:
        try:
            max_retries = int(raw_retries)
        except (TypeError, ValueError):
            max_retries = None
    row = memory.upsert_webhook(
        name=body["name"],
        url=body["url"],
        events=body.get("events", ["*"]),
        secret=body.get("secret", ""),
        enabled=bool(body.get("enabled", True)),
        description=body.get("description", ""),
        max_retries=max_retries,
        tenant_id=body.get("tenant_id", "__all__"),
    )
    if "error" in row:
        return web.json_response(row, status=400)
    return web.json_response(_jsonify(row))


async def admin_delete_webhook(request: web.Request):
    webhook_id = int(request.match_info["webhook_id"])
    if not memory.delete_webhook(webhook_id):
        return web.json_response({"error": "delete failed"}, status=400)
    return web.json_response({"deleted": webhook_id})


async def admin_webhook_deliveries(request: web.Request):
    webhook_id_raw = request.query.get("webhook_id")
    webhook_id = int(webhook_id_raw) if webhook_id_raw else None
    limit = int(request.query.get("limit", "100"))
    return web.json_response({"deliveries": [_jsonify(d) for d in memory.list_webhook_deliveries(webhook_id=webhook_id, limit=limit)]})


# ============================================================
# Tool cache admin handlers
# ============================================================


async def admin_cache_stats(request: web.Request):
    cache = tool_cache_mod.cache()
    if cache is None:
        return web.json_response({"error": "tool cache disabled"}, status=400)
    return web.json_response({"stats": cache.stats(), "snapshot": cache.snapshot(limit=50)})


async def admin_cache_invalidate(request: web.Request):
    cache = tool_cache_mod.cache()
    if cache is None:
        return web.json_response({"error": "tool cache disabled"}, status=400)
    try:
        body = await request.json() if request.body_exists else {}
    except Exception:
        body = {}
    tool_name = body.get("tool_name")
    removed = cache.invalidate(tool_name=tool_name)
    return web.json_response({"invalidated": removed, "tool_name": tool_name})


# ============================================================
# Model catalog + sync admin handlers
# ============================================================


async def admin_models_sync(request: web.Request):
    """Trigger a model catalog sync from all enabled providers."""
    try:
        body = await request.json() if request.body_exists else {}
    except Exception:
        body = {}
    # Allow overriding provider selection from the request body.
    eng = model_sync_mod.engine()
    if eng is None:
        return web.json_response(
            {"error": "model sync engine not initialized (enable model_sync in config)"},
            status=400,
        )
    # Optionally override which providers to sync.
    if body.get("providers"):
        wanted = set(body["providers"])
        summaries: list[dict] = []
        if "openrouter" in wanted:
            s = await eng.sync_openrouter()
            summaries.append(s.to_dict())
        if "openai" in wanted:
            s = await eng.sync_openai()
            summaries.append(s.to_dict())
        if "anthropic" in wanted:
            s = await eng.sync_anthropic()
            summaries.append(s.to_dict())
        if "ollama" in wanted:
            s = await eng.sync_ollama()
            summaries.append(s.to_dict())
        return web.json_response({"results": summaries})
    # Full sync
    results = await eng.sync_all()
    return web.json_response({"results": [r.to_dict() for r in results]})


async def admin_models_catalog(request: web.Request):
    """List model catalog entries with optional filters."""
    provider = request.query.get("provider")
    enabled_only = request.query.get("enabled_only", "").lower() in ("1", "true", "yes")
    min_score = request.query.get("min_capability_score")
    supports_tools = request.query.get("supports_tools", "").lower() in ("1", "true", "yes") or None
    supports_vision = request.query.get("supports_vision", "").lower() in ("1", "true", "yes") or None
    supports_reasoning = request.query.get("supports_reasoning", "").lower() in ("1", "true", "yes") or None
    min_ctx = request.query.get("min_context_length")
    limit = int(request.query.get("limit", "500"))
    entries = memory.list_model_catalog(
        provider=provider,
        enabled_only=enabled_only,
        min_capability_score=float(min_score) if min_score else None,
        supports_tools=supports_tools if isinstance(supports_tools, bool) else None,
        supports_vision=supports_vision if isinstance(supports_vision, bool) else None,
        supports_reasoning=supports_reasoning if isinstance(supports_reasoning, bool) else None,
        min_context_length=int(min_ctx) if min_ctx else None,
        limit=limit,
    )
    return web.json_response({"models": [_jsonify(e) for e in entries], "count": len(entries)})


async def admin_models_stats(request: web.Request):
    """Return summary stats for the model catalog."""
    stats = memory.model_catalog_stats()
    return web.json_response(stats)


async def admin_models_set_tier(request: web.Request):
    """Manually set the tier assignment for a model."""
    model_id = request.match_info["model_id"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    tier = body.get("tier")
    provider = body.get("provider", "openrouter")
    ok = memory.set_model_catalog_tier(model_id, provider, tier)
    if not ok:
        return web.json_response({"error": "model not found"}, status=404)
    return web.json_response({"model_id": model_id, "provider": provider, "tier": tier})


async def admin_models_set_enabled(request: web.Request):
    """Enable/disable a model in the catalog."""
    model_id = request.match_info["model_id"]
    try:
        body = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON body"}, status=400)
    enabled = bool(body.get("enabled", True))
    provider = body.get("provider", "openrouter")
    ok = memory.set_model_catalog_enabled(model_id, provider, enabled)
    if not ok:
        return web.json_response({"error": "model not found"}, status=404)
    return web.json_response({"model_id": model_id, "provider": provider, "enabled": enabled})


# ============================================================
# Entry point
# ============================================================


def _setup_logging(conf: cfg_mod.Config):
    """Configure logging. JSON lines when config.logging.structured_json is true."""
    log_cfg = conf.config.get("logging", {})
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    if log_cfg.get("structured_json"):
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                payload = {
                    "ts": datetime.now(UTC).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "msg": record.getMessage(),
                }
                if record.exc_info:
                    payload["exc"] = self.formatException(record.exc_info)
                return json.dumps(payload, ensure_ascii=False, default=str)
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def main():
    import argparse
    import signal
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./gateway-config.json")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    async def _runner():
        app = await init_app(args.config)
        _setup_logging(app["conf_mgr"].current())
        conf = app["conf_mgr"].current()
        host = args.host or conf.config.get("host", "0.0.0.0")
        port = args.port or conf.config.get("port", 8076)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host, port)
        await site.start()
        log.info("gateway listening on %s:%d", host, port)

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, stop_event.set)
            except (NotImplementedError, RuntimeError):
                # Windows: signal handlers must be registered on the main thread
                signal.signal(sig, lambda *_: stop_event.set())

        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            pass
        log.info("shutting down: draining in-flight requests")
        await runner.cleanup()

    try:
        asyncio.run(_runner())
    except KeyboardInterrupt:
        log.info("shutting down")


if __name__ == "__main__":
    main()

# Glint-V2 — Agent Working Notes

## Verification commands (run before finishing any task)

```powershell
# Lint (installed via pip install ruff)
ruff check gateway tests router_model

# Type check (installed via pip install mypy)
mypy gateway

# Tests (unit + integration)
python -m unittest discover -s tests
```

Run the full suite — unit tests alone do NOT prove a pipeline works.
The integration suite (`tests/test_integration.py`) boots the real gateway
against in-process mock upstreams and is where wiring bugs (missing calls,
broken format strings, wrong column names) actually surface.

## Layout

- `gateway/` — aiohttp app. `app.py` routes; `policy.py` routing decisions;
  `endpoints.py` HTTP clients + fallback ladder; `reviewer.py` async labeler;
  `trainer_worker.py` auto-retrain; `memory.py` SQLAlchemy schema + helpers.
- `tests/` — `test_unit.py`, `test_memory.py`, `test_review_fixes.py` (unit),
  `test_integration.py` + `mock_endpoints.py` (end-to-end).

## Critical invariants (do not break)

1. **Routing is never altered by prompt-injection signals.** Injection is
   logged (`flagged_inputs`) but routing stays independent.
2. **`store_review_result` auto-curates** high-agreement samples
   (`all_fields_agree and trust >= min_trust_to_curate`) into
   `curated_samples` in the same transaction. This is the ONLY feeder for the
   trainer. If the flywheel dies, the whole self-training loop is dead.
   Negative human feedback (`record_feedback(correct=False)`) also curates the
   reviewer's labels via `curate_reviewed_correction` (source=human_reviewed).
3. **`REVIEW_SYSTEM_PROMPT` is `.format()`-ed.** Its literal JSON braces must
   stay escaped (`{{` / `}}`) or the reviewer crashes with
   `KeyError: '"vertical"'`. Custom prompts must use `{verticals}` (replaced
   with `.replace`, not `.format`) or it is appended.
4. **Streaming holds the tenant semaphore for the stream duration**
   (`concurrent_limit` semantics — intentional, do not "optimize" by
   releasing early).
5. **Stream sends must be awaited twice**: `async for chunk in await
   client.send(transcoded, stream=True)` — `send()` is a coroutine that
   returns the generator. Narrow the `dict | AsyncIterator[bytes]` union with
   `isinstance` before using either shape (mypy enforces this).
6. **`routing_log` column names are `vertical_top2_prob`, `flags_*`** — the
   reviewer reads `d.get("vertical_top2_prob")`, never indexing bare keys
   (old rows may lack it).
7. **`review_results` agreement columns are named `agreement_<field>`.**
   Do not rename to `<field>_agree` — that breaks the insert.
8. **Datetime serialization**: `get_decisions()`, `get_or_create_user`,
   `live_eval_samples`, `checkpoint_history()`, `list_flagged()` ISO-format
   datetimes — HTTP endpoints serialize them directly.
9. **Auth fails closed when enabled.** Every non-public path requires a valid
   key; `X-User-Id` fallback only when `auth.allow_unauthenticated_user` is
   true (trusted reverse proxy). `auth.public_paths` defaults to
   `["/", "/health", "/ready", "/dashboard"]`. Admin paths require `admin`
   scope. All tenant-facing reads (trace/accuracy/cost/review-stats/memory)
   are ownership-scoped; `/metrics` is an admin path.
10. **Generated API keys are stored hashed** (`sha256:<digest>`, key shown
    once). `AuthManager.resolve` hashes the presented token when the stored
    key starts with `sha256:`. Revocation accepts full key or prefix.
11. **Training pipeline contract**: `train.py --output-onnx` writes the
    deployable ONNX (copy of the frozen embedding) + sidecar
    `<stem>_heads.npz` + `metadata_json` embedded in the npz.
    `embedding.checksum_sha256` in gateway-config.json is verified on load
    (refuses on mismatch); `PLACEHOLDER*` values are treated as unset.
    Head artifacts are validated on load: vertical count and 5 ordinal
    complexity weights must match taxonomy.
12. **Curated labels**: `train.py` accepts BOTH `code`/`flag_code` style keys
    (`curated_samples` rows use `flag_*`). Base-eval rows are deduplicated out
    of the training mix (held-out integrity).
13. **Live eval**: `TrainerWorker._sample_live_eval` (hourly) seeds
    `live_eval_set` + `router_model/data/live-eval/live_eval.jsonl` from
    high-agreement review results. eval.py's live accuracy depends on it.
14. **Tenant edits**: `get_or_create_user(overwrite=True)` +
    `TenantManager.refresh(tenant_id)` are required for admin budget/tier
    changes to take effect. Per-user `tenants` entries in gateway-config.json
    are loaded as preconfigured overrides.
15. **Health**: `/ready` and `/stats` use live health probes
    (`http.health_poll_seconds`, default 30s), falling back to breaker state.
    The mock endpoints' `/health` honors `MOCK_FAIL_ENDPOINTS`. With
    `embedding.require_real_model: true`, `/ready` returns 503 while the
    router is the stub.
16. **Trainer never blocks the event loop**: `_run_training_run` uses an
    `asyncio.Lock` and runs subprocesses via `loop.run_in_executor`. Do not
    reintroduce `threading.Lock` or direct `subprocess.run` in async code.
17. **Complexity decode**: gateway `_RealModel._build_output` computes
    `complexity = clip(sum(c_probs > 0.5), 1, 5)` — MUST stay identical to
    `router_model/eval.py`. argmax is WRONG for ordinal heads.
18. **Reviewer caps count only `__reviewer__` spend**: `_caps_ok` filters
    `usage_counters.tenant_id == "__reviewer__"`. Never sum all tenants.
19. **Events tenant scoping**: when `auth.enabled`, non-admin `/events` +
    `/events/recent` subscribers only see their own tenant's events (plus
    tenantless ones); admin scope sees all. Filter via
    `EventBus.recent/stream(tenant_id=...)`.
20. **Stale reviews**: `memory.requeue_stale_reviews()` (run by the reviewer
    worker, max once/min) requeues `in_progress` items whose `started_at`
    is older than 5 min (not `ts_queued`).
21. **Quotas are transactional reservations**: `TenantManager.reserve_usage`
    atomically checks budget + rps + tokens-per-minute and inserts a usage
    row (request_count=1) before dispatch; `settle_usage` adjusts with the
    actual delta (or releases on failure). Never call `record_usage` for
    tenant traffic — that bypasses limits.
22. **Native provider adapters decode responses AND streams** to OpenAI
    shape: Ollama (`_decode_ollama_response`/`_decode_ollama_stream`),
    Anthropic SSE, Gemini SSE. Missing this breaks provider parity — the
    gateway contract is always OpenAI chat.completions.
23. **`_apply_runtime_config` is the single reload path**: it propagates a
    new snapshot to policy, pool, auth, reviewer/trainer/observer workers,
    tenant defaults, body limit, CORS, and reloads the router only when the
    embedding signature changes. Add new runtime subsystems there.
24. **Overlay mutations are single-instance only**: `OverlayManager` raises
    when `mode == "multi"` (runtime file writes are not shared). Multi mode
    requires `GLINT_DB_URL` and config shipped via image/volume.
25. **The event loop must never block on DB/ONNX**: chat_completions offloads
    memory, router predict, usage reservation/settlement, and all DB writes
    via `asyncio.to_thread`. Keep new sync DB calls inside `to_thread`.
26. **Router model artifact contract (v2, MLP heads)**: `heads.npz` now
    REQUIRES `W_trunk1`/`b_trunk1` (shared trunk, `Linear(384, hidden)` +
    ReLU) alongside the per-task `W_*`/`b_*` keys — `router.try_load_real`
    validates this and refuses artifacts missing them, plus shape-checks
    every head's input dim against the trunk's hidden dim. `train.py`,
    `eval.py`'s `predict_with_heads`, and `router.py`'s `_run_heads` MUST all
    compute the identical trunk-then-head forward pass. ReLU (not GELU) is
    deliberate — it's the one activation numpy can replicate bit-exact from
    the PyTorch training graph, so there's zero train/inference drift. No
    backward compatibility with pre-v2 (pure-linear) heads — none exist yet,
    train.py had never been exercised before this contract existed.
27. **Structural-prototype centroids are computed for real now**: `train.py`
    embeds each structural prototype's `centroid_seed_text`, runs it through
    the trained projection head, and writes the averaged unit vector into
    `prototypes.json`'s `centroid` field after every run. `policy.py`'s
    `_compute_prototype_scores`/`_eval_prototype_match` use cosine similarity
    against these when both `ctx.projection` and the centroid exist, else
    fall back to keyword overlap (stub model / never-trained yet).
    `RequestContext.projection` (from `RouterOutput.projection`, threaded
    through in `app.py`) defaults to `None` — existing callers unaffected.
28. **Embedding fine-tune has a built-in regression gate**:
    `embed_finetune.py` measures triplet accuracy on a held-out split before
    and after fine-tuning; if it does NOT improve, the script deliberately
    skips writing an ONNX file. `trainer_worker._run_embedding_finetune`
    treats a missing output as "no fine-tune happened" and falls back to
    heads-only training on the current embedding — expected/safe, not a bug.
29. **Embedding-finetune checksum handling**: when a run fine-tunes the
    embedding, the hot-swap in `_run_training_run` passes
    `checksum_sha256=None` (skips the static-checksum check — the artifact
    was just produced in-process, trusted provenance) instead of comparing
    against `gateway-config.json`'s old checksum, which would always
    mismatch after a real fine-tune and silently block promotion. Ordinary
    heads-only retrains still enforce the static checksum as before.
30. **Auto-retrain promotions persist across restarts**:
    `TrainerWorker._persist_as_boot_default` copies the newly promoted
    checkpoint to the static boot path (`embedding.onnx_path` / sibling
    `heads.npz`) and updates `embedding.checksum_sha256` in
    `gateway-config.json` after every successful hot-swap — single-instance
    mode only (mirrors invariant #24). Before this, `app.py.init_app()` only
    ever read the static path at startup with no notion of "latest promoted
    checkpoint," so a restart silently reverted to whatever was last
    manually exported. Best-effort/non-fatal: failure here is logged, not
    raised — the in-memory hot-swap already succeeded either way.
31. **`swarm.decompose()` is async and takes a 4th `pool` argument** —
    `app.py` calls `await swarm_mod.decompose(ctx, conf, swarm_cfg,
    request.app["endpoint_pool"])`. Needed for `llm_plan` mode's planner LLM
    call. `execute_swarm` runs subtasks in topological dependency layers
    (`_topological_layers`) instead of one flat `asyncio.gather`; chunked
    (non-llm_plan) subtasks have no `depends_on` so this collapses to the old
    fully-parallel behavior for the existing default path.
32. **Subscription token budgets + per-model limits + budget-aware routing**:
    - `users.daily_token_limit` (INTEGER, default 0 = unlimited) and
      `users.target_success_probability` (FLOAT, default 0.99) are the
      per-tenant knobs. `plan_quotas` (plan_id, daily_token_limit,
      daily_usd_limit, required_success_probability, allowed_models_json) and
      `tenant_plans` (tenant_id → plan_id) override them when a tenant is
      assigned a plan. `assign_tenant_plan` MUST create the user row first
      (insert with `plan_id`) — otherwise the override silently never lands.
    - `model_token_limits` (tenant_id + endpoint_name composite PK) carries
      per-model `daily_token_limit`, `daily_usd_limit`, `max_request_tokens`.
      `reserve_usage` checks tenant-wide daily token budget, daily USD budget,
      per-model token+USD limits, **and** `max_request_tokens` (rejects
      individual requests that exceed the per-call cap, not just daily totals).
    - `reserve_usage`/`settle_usage` now take `endpoint_name` and write it
      into `usage_counters.endpoint_name` so per-model spend queries
      (e.g. `get_today_token_spend(tenant, endpoint)`) work. Failed requests
      still release reserved tokens (negative delta) — never just leave them.
    - `policy.budget_aware_route` is the new router. It builds a cascade
      chain (`BudgetAwareCandidate[]`) ordered by ascending cost, walking
      candidates until `cascade_success_probability(chain) >=
      target_success_probability`. Throws `BudgetError("quality_target_unmet")`
      when no chain reaches the target — caller maps to 429 `insufficient_quota`.
      Falls back to `cost_first_route` when the tenant has no plan, no token
      budget, and no per-model limits.
    - `quality.estimate_success_probability` returns Wilson lower bound (95%
      confidence) of observed success rate, clamped to a conservative prior
      (0.5) when fewer than `min_samples` (=10) outcomes exist. This means
      a new model with zero history CANNOT claim 100% confidence — to meet a
      0.99 target via a single model you'd need > 300 samples with 0 failures.
    - `policy.cost_to_complete_p99` is the 99th-percentile cost estimate
      (geometric retry distribution, capped at `max_retries+1` attempts),
      surfaced in `routing_log.extra.achieved_success_probability` /
      `cost_to_complete_p99` for auditability.
    - `BudgetError` is mapped to HTTP 429 with `error.code` =
      `quality_target_unmet` and an `insufficient_quota` type by `app.py`.
    - Admin: `POST /admin/plans`, `PUT /admin/plans/{id}`,
      `POST /admin/users/{tenant}/subscription`, `PUT /admin/users/{tenant}/budget/tokens`,
      `PUT /admin/users/{tenant}/models/{endpoint}/limits`,
      `GET /admin/users/{tenant}/limits`. User-facing: `GET /usage`,
      `GET /usage/limits`. All limit/budget edits call `tenant_mgr.refresh`
      so the cached `TenantState` picks up the new values immediately.
33. **Security Hub — firewall + injection detection (gateway-level + host-level)**:
    - Three new tables: `security_events` (unified audit log: ts, tenant_id,
      event_type, severity, reason, matched_pattern, query_preview,
      endpoint_target, action_taken, request_metadata_json), `provider_allowlist`
      (tenant_id + domain_pattern composite PK, action=allow|block), and
      `injection_profiles` (DB-backed named regex sets with severity +
      action + enabled + is_builtin). `flagged_inputs` gained `severity`,
      `matched_profile`, `security_event_id` columns; `purge_old_flags` now
      also purges `security_events`.
    - `gateway.security.InjectionProfile.from_config()` validates severity
      (low|medium|high|critical) and action (block|alert|log) at construction.
      `check_injection_with_action(profiles)` returns the highest-severity
      match across all enabled profiles; the action of the highest-severity
      profile wins. `DEFAULT_INJECTION_PROFILES` (seeded at startup, marked
      `is_builtin=True` so they can't be deleted) covers jailbreak,
      role_override, context_escape, router_manipulation, data_exfiltration,
      semantic_dos — all `action=block` except `data_exfiltration=alert`.
    - Injection detection CAN now block requests — invariant #1 still
      holds (routing is never altered), but a `block`-action profile returns
      HTTP 400 with `error.code = injection_blocked` BEFORE any routing
      decision happens. The routing pipeline below the injection check still
      classifies normally when the prompt is allowed.
    - `gateway/firewall.py` provides two layers:
      - `DomainAllowlistEnforcer`: in-process. `load_from_config()` ingests
        `security.provider_allowlist` from gateway-config.json;
        `load_from_db()` layers DB-backed rules on top (config rules
        preserved unless shadowed by a same-key DB rule). Tenant-specific
        rules win over global rules; unknown domains fall back to
        `default_action` (default: block). Loopback (localhost/127.0.0.1)
        is always allowed to keep local ollama/mock endpoints reachable.
      - `HostFirewallManager`: optional, off by default. Windows uses
        `netsh advfirewall firewall add rule dir=out action=block
        remoteip=<ip>`; Linux uses `iptables -A OUTPUT -d <ip> -j DROP
        -m comment --comment <name>`. Both are idempotent on rule name
        (sha256-derived). Requires admin/root; gracefully degrades to
        disabled state when privileges are missing. `persist_on_shutdown`
        controls whether rules survive a gateway exit.
    - `EndpointClient.send(tenant_id=...)` runs the firewall check BEFORE
      the breaker so a blocked request doesn't trip the breaker.
      `stream_passthrough(tenant_id=...)` and `_forward_with_fallback`
      thread tenant_id through. Localhost URLs bypass the check.
    - `/admin/security/*` is a new admin sub-API:
      `events`, `events/stats`, `injection-profiles` (CRUD),
      `provider-allowlist` (CRUD), `sync-firewall` (manual host fw sync),
      `status` (combined enforcer + host-fw state), `test` (dry-run a URL
      against the enforcer). All require `admin` scope. CRUD on
      injection-profiles and provider-allowlist updates the in-memory
      enforcer/profiles immediately — no config reload required.
    - Dashboard has a new "Security" tab (`gateway/dashboard/index.html`)
      showing: 7-day event counts, critical/high severity count, in-process
      blocks, profile count, firewall status (in-process + host), allowlist
      CRUD, profile enable/disable, manual firewall sync, and the 20 most
      recent events. Pure vanilla JS — no external deps.
    - Config (gateway-config.json `security.provider_allowlist`):
      `enabled=false` (default off — gateway still enforces the admin's
      configured endpoints because each `EndpointPool.rebuild()` rebuilds
      clients with `firewall_enforcer` set, but with `enabled=false` the
      enforcer short-circuits to allow). `default_action='block'` is the
      safest default — only configured provider domains work. `host_firewall`
      sub-config controls system-level rules.
34. **Gateway extensions — plugin system, A2A registry, ContextForge
    connector, MCP discovery, prompt registry, webhooks, tool cache**:
    - Nine new tables in `memory.py`: `plugins` (name PK, version,
      enabled, manifest_json, config_json, status, error, paths, ts),
      `a2a_agents` (id, name, endpoint_url, agent_type, auth_type,
      auth_value, protocol_version, capabilities_json, config_json,
      tags_json, enabled), `a2a_virtual_servers` (id, name, description,
      associated_agents_json, enabled), `a2a_metrics` (id, agent_id,
      tenant_id, success, latency_ms, interaction_type, error, ts),
      `prompt_templates` (id, name, category, content, variables_json,
      enabled, is_builtin, position), `webhooks` (id, name, url,
      secret, events_json, enabled), `webhook_deliveries` (id, webhook_id,
      event_type, payload_json, status_code, attempt_count, next_retry_at,
      response_body, ts), `contextforge_sync_log` (id, direction, entity_type,
      entity_id, status, detail, ts), `federated_tools` (id, tool_name PK,
      source, description, input_schema_json, endpoint_url, enabled, tenant_id).
      Full CRUD for each: `upsert_*`, `get_*`, `list_*`, `set_*_enabled`,
      `delete_*` (builtins delete-refused where applicable), `record_*`,
      `*_summary`. `plugins`, `a2a_agents`, `prompt_templates`, `webhooks`,
      and `federated_tools` all carry a `tenant_id` column (default
      `"__all__"` = visible to every tenant); list/get paths filter by
      `(tenant_id IN ("__all__", <caller_tenant>))`. See invariant #35.
    - **Plugin system** (`gateway/plugin.py`): `manifest.yaml` +
      `plugin.py` contract adapted from `mcpchad` (which uses FastAPI's
      `APIRouter`; here plugins return `web.RouteTableDef`). Routes MUST
      include their full path (e.g. `@router.post("/integrations/foo/bar")`)
      — there is no `_prefix_route()` helper; aiohttp can't remove routes
      from a running app so reload re-imports module code but cannot
      replace already-registered routes. `PluginContext.emit_event()` uses
      the module-level `events_mod.emit()` (NOT a nonexistent
      `EventBus.emit()` — that was the P0 bug). `PluginLoader` starts a
      background `watch_loop` that re-scans `plugins.root` every
      `scan_interval_seconds`. Plugin event source: `EventSource.PLUGIN`.
    - **A2A registry** (`gateway/a2a_registry.py`): `invoke_agent()`
      supports `jsonrpc`/`openai`/`anthropic`/`custom` payload formats,
      `build_auth_headers()` (api_key/bearer/none), per-agent metrics.
      SSRF guard runs BEFORE the HTTP call with `allow_localhost=True,
      allow_private=True` (admin-configured agents reaching local
      upstreams is a supported case). `invoke_agent` catches all
      exceptions, records metrics, and returns an `A2AResult` — it never
      raises except for programming errors. Event source:
      `EventSource.A2A`.
    - **ContextForge connector** (`gateway/contextforge_client.py`):
      `mode` ∈ {`external`, `embedded`, `both`}. External mode fetches
      agents/servers/tools/prompts via REST and merges into local
      registries (`_merge_agents`/`_merge_servers`/`_merge_tools`/
      `_merge_prompts`). Embedded mode exposes
      `embedded_register_tool`/`embedded_register_prompt` for in-process
      use. `sync_loop()` periodically pulls. CLI/admin buttons trigger
      `sync_all()` on demand. SSRF guard runs before every outbound fetch
      with `allow_localhost=True, allow_private=True`. Event source:
      `EventSource.CONTEXTFORGE`.
    - **MCP discovery** (`gateway/mcp_discovery.py`):
      `DiscoveredServer` dataclass; `discover()` concurrently probes a
      list of `hosts × ports` via `_probe_well_known` (`/.well-known/
      mcp.json`, `/.well-known/agent.json`), `_probe_mcp_endpoint` (POST
      /mcp initialize), and `_probe_sse_endpoint` (GET /sse looking for
      `text/event-stream`). `watch_loop()` re-probes on
      `probe_interval_seconds`. Auto-registers discovered servers as
      `federated_tools`. SSRF-guarded per target with
      `allow_localhost=True, allow_private=True`. Event source:
      `EventSource.MCP`. `discovery.probe_mcp_servers()` wraps it.
    - **Unified MCP facade** (`POST /mcp`, `gateway/mcp_facade.py`):
      JSON-RPC 2.0 over HTTP. `initialize` returns protocol version
      `2024-11-05`, server name `glint-v2-gateway`. `tools/list` returns
      all enabled `federated_tools` plus every enabled `a2a_agent` exposed
      as a synthetic `a2a_<name>` tool. `tools/call` retrieves the
      federated tool definition (input schema) and, for A2A-backed
      tools, calls `invoke_agent` — **going through the tool cache** when
      initialized (cache key includes tool name + arguments hash + tenant).
      `prompts/list` + `prompts/get` read from `prompt_templates`.
    - **Prompt registry** (`gateway/prompt_registry.py`): DB-backed
      templates with `category` (code/translation/summarization/default/
      safety). Five builtins (`router_coder`, `router_reviewer`,
      `translator`, `summarizer`, `safety_refusal`) seeded via
      `seed_builtin_templates()` (idempotent, marked `is_builtin=1` —
      builtins can be disabled but not deleted). `render_template()`
      substitutes `{var}` placeholders found by `extract_variables()`.
      `inject_into_messages()` prepends/appends the rendered prompt as a
      system/user/assistant message (position configurable, with
      `replace_existing_system=True` to overwrite an existing system
      message). **Wired into `chat_completions()`**: when
      `prompts.auto_inject` is true, after memory assembly and before
      translation mode, the gateway looks up a template by
      `category_for_vertical(r.vertical)`, renders it with context
      variables, and prepends it as a system message. Event source:
      `EventSource.PROMPT`.
    - **Webhook dispatcher** (`gateway/webhook_dispatcher.py`): async
      fan-out with HMAC SHA-256 signing (`X-Glint-Signature: sha256=<hex>`),
      exponential backoff retry (`initial_backoff_seconds` ×
      `backoff_multiplier` ^ attempt, capped at
      `delivery_timeout_seconds`), bounded concurrency
      (`max_concurrent_deliveries`), and per-delivery logging into
      `webhook_deliveries`. Event-type filtering supports `*` wildcards
      (`_matches()`). `events.publish()` fans out to
      `dispatcher.dispatch()` for every published event when the
      dispatcher is initialized — so emitting an event from anywhere in
      the gateway also triggers registered webhooks. SSRF guard runs
      before each delivery with `allow_localhost=True,
      allow_private=True`. Event source: `EventSource.WEBHOOK`.
    - **Tool cache** (`gateway/tool_cache.py`): LRU + TTL cache, thread-
      safe. `ToolCache.get(key)` / `set(key, value, ttl)` /
      `invalidate(by_tool=...)` / `invalidate_all()` / `stats()` /
      `snapshot(tenant_id=...)`. Supports per-tool TTL overrides
      (`per_tool_ttl_seconds`), bypass keys (skips cache for sensitive
      ops like auth/login), and tenant isolation (keys are namespaced by
      `tenant_id` when provided). Initialized at startup via
      `init_cache(conf)` and refreshed by `_apply_runtime_config`.
      **Currently wired only into the MCP facade's `tools/call` path**;
      A2A `invoke_agent` does NOT consult the cache directly (the facade
      wraps it). See "Known unfinished" for direct A2A/plugin wiring.
    - **SSRF protection** (`gateway/ssrf.py`): `validate_url(url,
      allow_localhost=False, allow_private=False, allow_link_local=False,
      allowed_hosts=..., blocked_hosts=...)` raises `SSRFBlockedURL`
      with a reason string on block. `_is_blocked_ip` checks loopback,
      private (RFC 1918), link-local (169.254 — cloud metadata),
      reserved, multicast, and unspecified ranges. **Critical IPv6 quirk
      worked around**: `::1` is BOTH `is_loopback` AND `is_reserved` —
      when `allow_localhost=True`, the loopback short-circuit returns
      `None` (allowed) WITHOUT falling through to the `is_reserved`
      check; otherwise the IPv6 loopback would always be blocked even
      when the admin explicitly allowed localhost. `safe_url()` is the
      non-raising variant. Guard is applied at: A2A `invoke_agent`,
      ContextForge `_fetch`, MCP `_probe_one`, webhook
      `_deliver_with_retry` — all call with `allow_localhost=True,
      allow_private=True` (admin-configured integrations reaching
      in-cluster upstreams is a supported case).
    - **Dashboard**: 4 new tabs (`Plugins`, `A2A`, `Prompts`, `Webhooks`)
      added in `gateway/dashboard/index.html` with full CRUD UIs: plugin
      list/reload/enable/disable, A2A agent CRUD + invoke + test +
      virtual server CRUD + ContextForge sync button + MCP discover
      button, prompt template CRUD + edit + preview, webhook CRUD +
      deliveries table + tool cache stats/invalidate. `flash()` helper
      for transient status messages. `showTab()` dispatches the new
      loaders. All vanilla JS, no external deps.
    - **Config additions** (`gateway-config.json` top-level keys):
      `plugins` (enabled, root, scan_interval_seconds, auto_load),
      `a2a` (enabled, max_agents, default_timeout, max_retries,
      metrics_enabled), `contextforge` (enabled, mode, external_url,
      api_key, sync_interval_seconds, auto_sync, timeout_seconds),
      `mcp_discovery` (enabled, probe_interval_seconds, auto_register,
      hosts[], ports[]), `prompts` (enabled, auto_inject,
      default_category), `webhooks` (enabled, max_retries,
      initial_backoff_seconds, backoff_multiplier,
      delivery_timeout_seconds, max_concurrent_deliveries),
      `tool_cache` (enabled, max_entries, default_ttl_seconds,
      per_tool_ttl_seconds{}, bypass_keys[]).
    - **Sample plugin** (`gateway/plugins/microsoft_learn/`):
      `manifest.yaml` + `plugin.py` demonstrating the build_router
      contract with `/integrations/microsoft-learn/search` (POST) and
      `/integrations/microsoft-learn/modules` (GET) endpoints, event
      emission, and `context.get_setting()` for the `api_key` setting.
35. **Tenant scoping on the extension tables** (multi-tenant safety):
    - `plugins`, `a2a_agents`, `prompt_templates`, `webhooks`, and
      `federated_tools` all have a `tenant_id` column defaulting to
      `"__all__"`. The semantics:
      - `"__all__"` = global row, visible to every tenant. Admin-created
        builtins, ContextForge-imported tools, and MCP-discovered servers
        use this.
      - any other value = tenant-private row, visible only to that tenant.
    - List paths filter by `tenant_id IN ("__all__", <caller_tenant>)`:
      a tenant sees globals + its own privates, never another tenant's.
      Admin context (`tenant_id=None`) bypasses the filter and sees all
      rows — the admin REST routes accept a `?tenant_id=` query param
      for optional scoping.
    - **DB uniqueness is by `name` alone, not by `(tenant_id, name)`.**
      Two tenants CANNOT have a row with the same name — tenant-private
      rows must use distinct names (convention: prefix/suffix the tenant,
      e.g. `acme_coder`). The "preferred tenant → global fallback"
      pattern in `get_*_by_name(name, tenant_id=...)` returns a
      tenant-specific row if one exists, otherwise the global row of the
      same name, otherwise None. It does NOT allow name collisions.
    - **Prompt auto-inject** (`chat_completions`) lists templates by
      `category_for_vertical(r.vertical)` filtered by the caller's
      `tenant_id`, then prefers a tenant-specific template over a global
      one in the same category.
    - **MCP facade** (`POST /mcp`): `tools/list`, `prompts/list`,
      `tools/call`, and `prompts/get` all read `X-Tenant-Id` from the
      request header (default `"anonymous"`) and filter accordingly. A
      tenant cannot discover or invoke another tenant's private A2A
      agents or federated tools.
    - **Webhook dispatcher** (`gateway/webhook_dispatcher.py`):
      `dispatch(event_type, payload, tenant_id=...)` matches a webhook
      only when `webhook.tenant_id == "__all__"` OR
      `webhook.tenant_id == event_tenant_id`. Events published via
      `events.publish()` carry the originating tenant_id, so a webhook
      scoped to tenant `acme` only receives events that `acme` triggered.
    - **Per-webhook `max_retries` override**: `webhooks.max_retries` is
      a nullable int column. NULL = use the global
      `webhooks.max_retries` from gateway-config.json (default behavior).
      A positive int = this webhook gets that many retry attempts,
      overriding the global. `_deliver_with_retry` computes
      `effective_retries = webhook.max_retries if not None else
      dispatcher.max_retries` and clamps to >= 1.
    - **A2A `invoke_agent` tool cache integration**: when
      `interaction_type == "invoke"` (the default) and the global
      ToolCache singleton is initialized, `invoke_agent` checks the
      cache before the HTTP call (keyed by `a2a_<agent.name>`,
      `parameters`, `tenant_id`). A hit returns an `A2AResult` with
      `error="__cached__"` and `status_code=200`. Successful miss
      results are cached. The MCP facade calls `invoke_agent` with
      `interaction_type="mcp_call"`, which bypasses the in-invoke cache
      (the facade does its own cache check at the top of `tools/call`).
      `use_cache=False` disables the cache for a single call.
    - **MCP discovery `watch_loop` jitter + cache**: each iteration
      sleeps `interval_seconds + random(0, 10% of interval)` to
      desynchronize multiple gateways probing the same hosts. A
      per-host "last successful discovery" cache (in-memory dict, keyed
      by hostname) skips hosts that produced a result within the last
      `interval_seconds * 2` (min 120s) — avoids noisy re-probing of
      healthy known hosts while still re-probing hosts that previously
      returned nothing.


## Router model v2 — MLP heads, real embedding fine-tune, llm_plan swarm (UNVERIFIED)

Written in a session with no bash/execution access (sandbox VM failed to
start) — every file below was hand-verified for shape/contract consistency
but **never actually run**. Treat as a draft PR that needs the checks below
before it's trusted, same spirit as the rest of this section.

Changed: `gateway/router.py` (MLP trunk inference), `router_model/train.py`
(PyTorch trunk+heads training, class weighting, val split, early stopping,
supervised-contrastive projection loss, centroid write-back),
`router_model/eval.py` (mirrors the trunk forward pass), `router_model/
embed_finetune.py` (real contrastive fine-tune, was a stub), `gateway/
trainer_worker.py` (wires embedding fine-tune before heads training, boot-
default persistence), `gateway/policy.py` + `gateway/app.py` (real
cosine-similarity prototype matching), `gateway/swarm.py` + `gateway-
policy.json` (llm_plan dependency-aware subtask planning), `requirements.txt`
(torch uncommented — now required for training, still not required to run
the gateway itself).

Before trusting any of it:

```powershell
ruff check gateway tests router_model
mypy gateway
python -m unittest discover -s tests

# then a real end-to-end pass:
python router_model/generate_data.py --out router_model/data/base   # needs TEACHER_API_KEY
python router_model/train.py --base-data-dir router_model/data/base --output-heads router_model/checkpoints/v1_heads.npz --output-onnx router_model/checkpoints/v1_model.onnx --output-metadata router_model/checkpoints/v1_meta.json
python router_model/eval.py --heads router_model/checkpoints/v1_heads.npz --onnx router_model/checkpoints/v1_model.onnx --base-eval router_model/data/base/eval.jsonl --output-json router_model/checkpoints/v1_eval.json
```

Watch specifically for: PyTorch/numpy shape mismatches surfacing as
`try_load_real` raising `ValueError` on load (the shape-check invariants in
#26 are meant to catch this loudly rather than silently misrouting); the
`torch.onnx.export` I/O signature in `embed_finetune.py` actually matching
what `encode_with_embedding` feeds an ONNX session (`input_ids`,
`attention_mask`, optional `token_type_ids` -> `last_hidden_state`); and
whether `--projection-weight`'s supervised-contrastive loss actually helps
prototype separation or needs re-weighting once real data exists.

## Known unfinished (do not assume working)

- `router_model/train.py`/`eval.py`/`embed_finetune.py` need a HuggingFace
  download (bge-small-en-v1.5) and `pip install torch` — not exercised in
  CI, and not run by the agent that wrote the v2 rewrite above. See that
  section for exact verification commands.
- `tests/mock_endpoints.py` covers llama/openai chat + stream + failure
  scripting; add cases there for new endpoint behaviors, including swarm's
  `llm_plan` planner calls once that path gets exercised.
- The dashboard carries its API key in `sessionStorage` (`glint_api_key`);
  the event stream uses authenticated `fetch` streaming (EventSource cannot
  set headers). Named SSE events are consumed via `renderEvent`.
- The "Glint Roman Empire lesson" is referenced by name in four places
  (`gateway-config.json`, `gateway-policy.json`, `prototypes.json`,
  `generate_data.py`) as if written up in README.md, but the actual incident
  never got documented there — only the resulting rule did (topic prototypes
  forbidden; prototypes encode difficulty profile, not topic). If you know
  the real story, add it to README under structural prototypes.
- **Gateway extensions — known unfinished / do not assume working**:
  - `tenant_id` is NOT yet a column on `plugins`, `prompt_templates`,
    `webhooks`, or `federated_tools` — every tenant currently sees every
    row in those tables. Add a migration + filter-by-tenant on all
    list/get/upsert paths before exposing this to multi-tenant deployments.
    `a2a_agents` has `tenant_id` on metrics but not on the agent records
    themselves.
  - The tool cache is only consulted by the `/mcp` facade's `tools/call`
    handler — A2A `invoke_agent` and plugin route handlers do NOT consult
    the cache directly. If you want caching at the A2A layer, call
    `tool_cache.cache.get(key)`/`set(...)` inside `invoke_agent` or wrap it
    at the facade level (see the facade's existing pattern).
  - Plugin `reload()` re-imports the module and re-runs `build_router`,
    but already-registered aiohttp routes persist — aiohttp cannot remove
    routes from a running `Application.router`. The reload method warns
    about this and recommends a full gateway restart for route changes.
    Code/config-only updates (no new routes) work fine via reload.
  - `Microsoft Learn` sample plugin hits no real endpoint — it returns
    canned data to demonstrate the contract. Wire it to the real
    Microsoft Learn API before relying on its output.
  - ContextForge "embedded mode" registers tools/prompts into local tables
    but does not expose an in-process MCP server surface for them — they
    surface only through `/mcp` `tools/list`. An in-process dispatch path
    (skip the HTTP roundtrip) would be a natural follow-up.
  - The `/mcp` facade does not yet support `resources/*` or `roots/*`
    MCP method groups — only `initialize`, `tools/list`, `tools/call`,
    `prompts/list`, `prompts/get`. Add `resources/list` and `resources/read`
    when there's a concrete use case.
  - Webhook delivery does not yet respect a per-webhook `max_retries`
    override — all webhooks use the global `webhooks.max_retries` from
    config. Add a `max_retries` column to the `webhooks` table if
    per-webhook tuning is needed.
  - MCP discovery's `watch_loop` probes every host:port pair on every
    tick — no jitter, no result caching. With a large `hosts` list this
    could be noisy; consider adding jitter + a per-host probe cache.

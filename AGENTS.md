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

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

## Known unfinished (do not assume working)

- `router_model/embed_finetune.py` is a STUB — `POST /retrain
  --allow-embedding-finetune` logs a warning and runs heads-only.
- `router_model/train.py`/`eval.py` need a HuggingFace download
  (bge-small-en-v1.5) — not exercised in CI.
- `tests/mock_endpoints.py` covers llama/openai chat + stream + failure
  scripting; add cases there for new endpoint behaviors.
- The dashboard carries its API key in `sessionStorage` (`glint_api_key`);
  the event stream uses authenticated `fetch` streaming (EventSource cannot
  set headers). Named SSE events are consumed via `renderEvent`.

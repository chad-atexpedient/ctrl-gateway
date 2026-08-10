# Glint-V2

A semantic gateway / transcoder / router for LLM inference fleets. Routes requests to the **cheapest-sufficient** tier (cost = first gate) with **efficiency as the tiebreaker**, escalates on uncertainty to keep first-attempt correctness near 100%, and **auto-trains** itself from feedback + secondary model review.

## What this is

- **Gateway**: OpenAI-compatible reverse proxy with health checks, breakers, failover, concurrency control, streaming.
- **Transcoder**: rewrites payloads per endpoint kind (llama.cpp / Ollama / generic OpenAI), bumps `max_tokens` for thinking tiers, reformats vision content, passes SSE through.
- **Router**: custom hybrid model — frozen sentence embedding + small classifier heads. ~55 verticals, complexity scoring, code/math/reasoning flags, OOD detection. CPU-only, ~1-7 ms per request.
- **Policy engine**: deterministic pre-routes (vision, OWUI tasks, medical regex, freshness) → cost-first gate → uncertainty escalation → efficiency tie-break → tier ladder (context overflow, breaker, fallback).
- **Observability**: SQLite (single-instance) or Postgres (multi-instance). Every routing decision logged with model version + policy version + cost estimate.
- **Data flywheel**: async reviewer (post-response, never blocks the user) labels every prompt independently; high-agreement samples become training data; auto-retrain with eval gate, atomic hot-swap, auto-rollback on regression.
- **Multi-tenant**: per-user rate limits, cost budgets, tier access.
- **Security**: prompt-injection detection, sanitization, training-data trust scoring.
- **Dashboard**: live SPA at `/dashboard`.

## Quick start

```bash
# 1. Install
cd C:\Users\maris\glint-v2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 2. Configure
# Edit gateway-config.json with your endpoints + reviewer model.
# Edit gateway-policy.json with your thresholds/regexes.
# Set the reviewer API key (default model is "GPT-5.6 Luna"):
$env:TEACHER_API_KEY = "sk-..."
$env:TEACHER_BASE_URL = "https://api.your-provider.example/v1"

# 3. Generate seed training data
python router_model/generate_data.py --out router_model/data/base

# 4. Train the router (heads only; embedding is frozen)
python router_model/train.py
python router_model/export_onnx.py

# 5. Run
python -m gateway.app
# Gateway: http://localhost:8076  Dashboard: http://localhost:8076/dashboard
```

## Routing pipeline

```
request ──→ parse (vision? tools? stream?)
         ├─ pre-routes (deterministic, override-only)
         │   ├─ vision present → vision tier
         │   ├─ "### Task:" pattern → tier0
         │   ├─ medical keyword regex → tier_medical
         │   ├─ freshness + cx≤3 → tier0
         │   └─ structural prototype match → tier4
         ├─ router (frozen embedding + heads, ~1-7ms)
         │   ├─ vertical (top-2 + confidence) + complexity + flags
         │   ├─ OOD check (max-prob threshold)
         │   └─ inject scan (security.py) → sanitize, never trust injection
         ├─ policy (cost-first → escalate → ladder → fail)
         │   ├─ expected_cost = fixed + in×pin + est_out×pout + retry×(1−fit)
         │   ├─ fit(t,v) = sigmoid((capability_t,v − min_capability_v) × k)
         │   ├─ escalate: OOD | low-conf | top-2 close | cost within margin
         │   ├─ tie-break: speed + health + load
         │   └─ ladder: context overflow ↑, breaker open ↓, error → fallback
         ├─ budget-aware routing (when tenant has a plan or any token/model limit)
         │   ├─ capability × per-model quality profile (Wilson lower bound)
         │   ├─ remaining tenant/model token budget ≤ estimated tokens?
         │   ├─ cascade chain ≥ target_success_probability (default 0.99)
         │   └─ 99th-percentile cost to complete (geometric retry)
         ├─ transcode (per-endkind adapter: llamacpp / openai / ollama)
         └─ forward + stream (SSE passthrough)
                ↓
            response
                ↓
        async reviewer queue (post-response, batches of 10)
                ↓
        per-model quality sample recorded (success/failure)
                ↓
        cascade P(success) updates → future routing decisions
                ↓
        threshold met → auto-retrain heads → eval gate → atomic swap
```

## Subscription budgets, per-model limits, and budget-aware routing

Tenants can be assigned a **subscription plan** that bundles a daily token budget, a daily USD budget, a target success probability (default 99%), and a whitelist of allowed models. Additionally, admins can set per-model limits (daily tokens, daily USD, max-request-tokens) on top of any plan.

### How the router uses these signals

When a tenant has a plan, a daily token budget, or any per-model limit, the simple cost-first picker is replaced by `policy.budget_aware_route`. That routine:

1. Builds candidate models as usual (capability-fit × cost), filtering out anything that can't fit in the remaining tenant/model token budget or isn't on the plan's allowed-models list.
2. Estimates each candidate's P(success) from a per-model quality profile — Wilson lower bound (95% confidence) of observed success rate, capped to a conservative prior (0.5) until the model has ≥ 10 samples.
3. Walks the eligible candidates in ascending-cost order, building a **cascade chain** until the union P("at least one succeeds") ≥ target_success_probability.
4. Surfaces the 99th-percentile cost-to-complete (geometric retry model) in `routing_log.extra` for auditability.
5. If no chain reaches the target, returns HTTP 429 `quality_target_unmet` — never silently routes to an inadequate model.

### Admin endpoints

```bash
# Plans
POST   /admin/plans                                   # create a plan
PUT    /admin/plans/{plan_id}                         # update a plan
GET    /admin/plans                                   # list plans

# Per-tenant subscription
POST   /admin/users/{tenant_id}/subscription         # assign a plan (body: plan_id)
DELETE /admin/users/{tenant_id}/subscription         # unbind
GET    /admin/users/{tenant_id}/subscription         # read binding + quota

# Per-tenant daily token budget + target success probability
PUT    /admin/users/{tenant_id}/budget/tokens
       # body: {"daily_token_limit": 100000, "target_success_probability": 0.99}

# Per-model limits
PUT    /admin/users/{tenant_id}/models/{endpoint}/limits
       # body: {"daily_token_limit": 50000, "daily_usd_limit": 5.0, "max_request_tokens": 4096}
GET    /admin/users/{tenant_id}/models/{endpoint}/limits
GET    /admin/users/{tenant_id}/limits                # all limits at once

# Quality samples (recorded by the reviewer)
GET    /admin/models/quality
POST   /admin/models/quality
       # body: {"endpoint_name": "tier1", "vertical": "programming", "complexity": 3, "success": true}
```

### User-facing endpoints

```bash
GET /usage          # today's spend + tokens
GET /usage/limits    # the tenant's daily limits + remaining tokens
```

### Plan JSON example

```json
{
  "plans": {
    "free": {
      "daily_token_limit": 100000,
      "daily_usd_limit": 1.0,
      "required_success_probability": 0.99,
      "allowed_models": ["ollama_local"]
    },
    "pro": {
      "daily_token_limit": 1000000,
      "daily_usd_limit": 25.0,
      "required_success_probability": 0.99,
      "allowed_models": ["tier1_model", "tier2_model", "frontier"],
      "model_limits": {
        "frontier": {
          "daily_token_limit": 250000,
          "daily_usd_limit": 20.0,
          "max_request_tokens": 128000
        }
      }
    }
  }
}
```

### Closed-loop quality calibration

`quality.record_quality_sample(endpoint, vertical, complexity, success)` is called by the reviewer worker after every labelled decision. Once a model accumulates ≥ 10 samples on a `(vertical, complexity)` bucket, its Wilson lower bound replaces the prior of 0.5. Over time, the cascade chain naturally shortens to the cheapest-fit model whose profile still meets the tenant's target.


## Observational memory (Mastra pattern)

The gateway is the chokepoint for every chat completion, which means it can implement the three-tier memory + observational memory pattern at the routing layer — no per-app memory work needed.

**Three tiers** (`gateway/memory_observational.py`):

  - **L1 — Recency** (`last_messages: 20`): last N messages in the thread. Free, deterministic, always present.
  - **L2 — Working memory** (per-resource profile): durable per-user document the model rewrites via tool calls. Re-read into context every turn.
  - **L3 — Semantic recall** (optional, off by default): vector search over all past messages, `top_k: 3` with message range for context. Costs an embedding per stored message.

**Observational memory** (`gateway/observer_worker.py`): a three-agent loop mirroring Mastra's OM:

  - **Actor** — the chat completion that flows through the gateway. Never sees raw history past the recency window; sees observations + reflection + recency tail instead.
  - **Observer** — async background worker. Compresses raw unobserved messages into structured observations when token count crosses `message_tokens` (default 12K). Uses a small fast model (tier0 by default).
  - **Reflector** — async background worker. Condenses observations into a single reflection when observations cross `observation_tokens` (default 20K). Same model tier.

Both Observer and Reflector run **asynchronously, never blocking the hot path**. Buffered accumulation means the actor only ever sees a synchronous read of `observations + recency tail`, never a sync compression call.

**Compaction & redirect** (`gateway/policy.py`):

When the assembled memory context exceeds a tier's `max_context` × `compaction_token_threshold_pct` (default 75%), the gateway **redirects** to a higher tier with larger context window. Cost-first arithmetic still applies — we escalate only when no cheaper tier fits. This is the "semantically redirect and compact" pattern: the compacted view stays valid across handoffs.

**Resource vs thread scoping** mirrors Mastra's defaults: `resource_id = tenant_id` (stable per user), `thread_id = session_id` (per chat). Working memory is resource-scoped (survives across chats); observations are thread-scoped by default.

**Event emitter** (`gateway/events.py`): OWUI thinking-indicator pattern adapted. Every state change emits a typed event (routing, memory, observer, reflector, trainer, reviewer, breaker, security) that UI platforms can subscribe to via SSE at `/events`. The dashboard's "Live Event Stream" section (section 11) shows the full flow — observe the "Classifying request..." status emit, the routing decision, the memory context load, and the eventual "Done" — all streaming live.

```python
# Subscribe in your UI:
from gateway import events
events.emit_status(
    events.EventSource.OBSERVER,
    "Compressing 50 messages in thread abc123...",
    done=False,
    tenant_id="user-1",
    session_id="abc123",
)
```

**Manual triggers**:

```
POST /memory/observe  {thread_id, resource_id}   # force compression pass
POST /memory/reflect  {thread_id, resource_id}   # force consolidation
GET   /memory/context ?resource_id=X&thread_id=Y # preview assembled context
GET   /memory/working/{resource_id}              # read durable profile
POST  /memory/working/{resource_id} {"content": "..."}  # rewrite profile
GET   /memory/observations/{resource_id}/{thread_id}    # read observations + reflection
```

This is what makes the gateway a **memory-aware router**: not just "send to the cheapest tier", but "send to the cheapest tier that has the context this conversation needs, with the relevant history already compressed into it."

## Reviewer model selection

The reviewer is the **secondary model** that independently labels every prompt for the data flywheel. It is the most important model in the system — its labels become training data, so its quality caps your router's quality.

### Default

The default is **GPT-5.6 Luna**, the placeholder we ship with for getting started. It's a generic cloud LLM picked because it has reasonable agreement with most routing decisions at moderate cost. Swap it as soon as you have access to a better model.

### Swap the reviewer model

Edit `gateway-config.json` → `reviewer.model`:

```json
{
  "reviewer": {
    "endpoint": "https://api.your-provider.example/v1",
    "model": "your-preferred-model-id",
    "api_key_env": "TEACHER_API_KEY",
    ...
  }
}
```

Then `POST /reload` — no restart needed.

### What to look for in a reviewer model

The reviewer is called on every routing decision. It does a structured labeling task (JSON output with vertical, complexity, and four flags). The best reviewer models for this job share these traits:

| Trait | Why it matters | Priority |
|-------|---------------|----------|
| **Strong instruction-following / JSON-mode** | Must return valid JSON every time; invalid output = discarded + flagged | Critical |
| **Good calibration** | Should output confidence implicitly via its labels; we use agreement with the router, not the model's own confidence | High |
| **Decent classification ability** | Needs to assign one of ~55 verticals correctly; doesn't need to be a coding genius, but should recognize domains | High |
| **Long context support** | Some prompts have large attached code/data; reviewer should be able to see the full thing | Medium |
| **Cost per million tokens** | Called on every prompt — this is your biggest operational expense | Critical |
| **Latency** | Runs async, but latency caps throughput of the curation pipeline | Low |
| **Refuses correctly** | When asked to follow an injection, should refuse and still label (the reviewer system prompt explicitly tells it to label only, not follow user instructions) | High |

### Recommended reviewer models

Pick based on your cost ceiling and quality bar:

**Top tier (best quality, accept the cost):**
- **GPT-5-class or Claude-4-class API models** — best label agreement, lowest disagreement rate, lowest poison risk
- Use these if you have a meaningful budget for the flywheel

**Strong mid-tier (good balance):**
- **GPT-4-class / Claude-3.5-class** — solid labels, reasonable cost, widely available
- The default for most production deployments

**Budget tier (works, more disagreement to filter):**
- **GPT-4o-mini / Claude-3-haiku / Llama-3.1-70B via cheap API** — significantly cheaper, but expect higher disagreement rates; you'll need more curated samples to hit the same eval gate
- Good for getting started when budget is tight

**Self-hosted option (free, but you pay in latency and ops):**
- **Qwen3.5-9B / Qwen3.5-4B from your own fleet** — point the reviewer at one of your gateway endpoints
- Free at inference time; costs ~50-200ms per review; conflicts with live traffic if the box is busy
- Good for full-data-sovereignty deployments

**Avoid for this role:**
- Tiny instruct models (<3B) — too high disagreement rate, noisy labels
- Models without JSON-mode / function-calling — too many invalid outputs to be useful
- Models with strict content filters that refuse medical/clinical text — reviewer needs to label medical prompts correctly

### Context window constraints

The reviewer receives the full prompt text plus the system prompt and JSON schema. The prompt size depends on your traffic:

| Typical prompt size | Min context window |
|---------------------|-------------------|
| Short chat (<2K tokens) | 4K |
| Code generation (<8K tokens) | 16K |
| Long docs / RAG (<50K tokens) | 64K |
| Multi-file code analysis (<200K tokens) | 200K+ |

The reviewer gracefully degrades if a prompt exceeds its context: it labels with `truncated: true` and the system downweights those samples in training.

### Cost expectation

Rough budget planning (assumes 10K requests/day, avg prompt 500 tokens):

| Reviewer tier | Cost / 1M input tokens | Monthly cost (approx) |
|---------------|-----------------------|------------------------|
| Budget (mini model) | $0.15 | $25 |
| Mid-tier (4o-mini / 3.5-haiku) | $3 | $450 |
| Top tier (GPT-5 / Claude-4) | $15 | $2,250 |

Cap these with `reviewer.caps` in `gateway-config.json` — see config schema.

### Switching reviewer model in flight

1. Edit `gateway-config.json` → `reviewer.model`
2. `POST /reload` (or wait for the next config poll)
3. New routing decisions are reviewed by the new model
4. Old curated samples (labeled by the old reviewer) are tagged with `reviewer_model` in the pool — you can filter them out via the training mix ratio if you want a clean break

## Configuration reference

See `gateway-config.json` for full schema with comments. Key sections:
- `mode`: `single` (SQLite) or `multi` (Postgres)
- `tenants`: default + per-user overrides
- `endpoints`: list of `{name, kind, base_url, pricing, speed_tps, max_context, concurrency, breaker_config}`
- `tiers`: list of `{name, endpoints, max_context, capability_per_vertical, max_tokens_bump}`
- `reviewer`: `{endpoint, model, api_key_env, batch_size, caps}`
- `trainer`: `{trigger_threshold, target_accuracy, eval_gate_per_vertical, mix_ratio_base_curated, drift_alarm_threshold, embedding_finetune_manual_only}`

See `gateway-policy.json` for the rule layer:
- `overrides`: first-match-wins list
- `escalation`: confidence threshold, top-2 epsilon, cost margin
- `freshness_regex`, `medical_keyword_regex`, `owui_task_regex`, `injection_regex`
- `ood_threshold`: max probability below which we flag unknown verticals

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/chat/completions` | Main OpenAI-compatible proxy |
| GET | `/v1/models` | List models (single "gateway" with capabilities) |
| GET | `/stats` | Live: requests, errors, health, load, tier distribution |
| GET | `/config` | Current tier/endpoint configuration |
| POST | `/reload` | Hot-reload config + policy + prototypes + embedding + taxonomy |
| GET | `/trace` | Routing decision history (`?limit=N&session=X&vertical=Y`) |
| POST | `/feedback` | Mark routing decision correct/wrong |
| GET | `/accuracy` | Accuracy report from feedback |
| GET | `/export` | Export routing log as training data |
| GET | `/memory` | Session stats, popular queries |
| GET | `/verticals` | Per-vertical distribution + accuracy |
| GET | `/cost` | Per-tier cost / 1K requests, spend caps |
| GET | `/review-stats` | Reviewer queue depth, spend, agreement rate |
| GET | `/registry` | Router model versions + scores + rollback pointers |
| POST | `/retrain` | Manually trigger retraining (optional `--confirm-drift`, `--allow-embedding-finetune`) |
| GET | `/docs/model-card/{version}` | Auto-generated model card |
| GET | `/admin/users` | List/create/update tenants |
| GET | `/admin/users/{id}/budget` | Set daily budget |
| GET | `/admin/users/{id}/stats` | Per-tenant usage |
| GET | `/admin/flags` | Suspicious inputs + injected prompts |
| GET | `/health` | LB probe (multi-instance mode) |
| GET | `/ready` | Readiness: 200 only when DB reachable AND ≥1 endpoint not breaker-open |
| GET | `/metrics` | Prometheus text format: request counts, latency, failures, breaker state, queue depth |
| GET | `/dashboard` | Live SPA dashboard |

## Authentication (per-tenant API keys)

When `auth.enabled` is true in `gateway-config.json`, every request resolves its
tenant from `Authorization: Bearer <key>`:

```json
"auth": {
  "enabled": true,
  "keys": {
    "sha256:...": {"tenant_id": "alice", "scope": ["user"], "prefix": "glint-xxxx"}
  },
  "admin_paths": ["/admin", "/retrain", "/reload", "/config", "/export", "/registry", "/metrics"],
  "public_paths": ["/", "/health", "/ready", "/dashboard"],
  "allow_unauthenticated_user": false
}
```

- Auth **fails closed**: with `enabled: true`, every non-public path requires a
  valid key. The legacy `X-User-Id` header fallback only applies when
  `allow_unauthenticated_user: true` (trusted reverse proxy deployments).
- Admin paths require a key with `admin` scope (401 otherwise).
- Chat/other paths map the key to its `tenant_id` (rate limits, budgets, and
  all telemetry reads are keyed on it — `/trace`, `/accuracy`, `/cost`,
  `/review-stats`, and `/memory/*` are ownership-scoped).
- Keys are stored **hashed** (`sha256:<digest>`); the raw key is shown once at
  creation. Generate via the dashboard Keys tab, `POST /admin/keys`, or
  `GLINT_ADMIN_API_KEY` env (see docker-compose).
- All tenant-facing read endpoints are ownership-scoped; `/metrics` is admin.

Production hardening: body size cap (`http.max_body_bytes`), optional CORS
(`http.cors_origins`), security headers on every response, `/config` redacts
keys, SIGTERM/SIGINT graceful drain, JSON-line logging
(`logging.structured_json`).

## Model routing (OpenAI-compatible)

- `model: "gateway"` (default) → semantic routing (router + cost-first).
- `model: <endpoint-name-or-model-alias>` → direct routing to that provider;
  unknown models return an OpenAI-style `model_not_found` error (404).
- `GET /v1/models` lists `gateway` plus every endpoint the tenant can access.
- Errors on `/v1/chat/completions` use the OpenAI error envelope
  (`{"error": {"message", "type", "param", "code"}}`).

## Multi-instance (docker-compose)

Requires env: `POSTGRES_PASSWORD`, `GLINT_DB_URL`, `GLINT_ADMIN_API_KEY`.
Gateways run non-root, expose only the internal port, health-checked, and are
load-balanced through the nginx `lb` service. Runtime overlay mutations
(`/admin/*`) are disabled in `multi` mode — ship config via image/volume.

## Integration tests

`tests/test_integration.py` boots the real gateway against in-process mock
upstreams (`tests/mock_endpoints.py`) and verifies full-pipeline contracts:
routing→forward→log, SSE streaming, the retry/fallback ladder, the
review→curate flywheel, auth enforcement, and JSON-safe traces.

```bash
python -m unittest tests.test_integration   # integration
python -m unittest discover -s tests        # everything (unit + integration)
```

CI (`.github/workflows/ci.yml`) runs ruff, mypy, and the full suite on every push.

## Project layout

```
glint-v2/
├─ gateway/                aiohttp server
│  ├─ app.py               routes
│  ├─ router.py            embedding + heads, atomic swap
│  ├─ policy.py            pre-routes, cost-first, escalation
│  ├─ transcoder.py        per-endkind adapters
│  ├─ endpoints.py         llamacpp/openai/ollama adapters, semaphores
│  ├─ circuit.py           per-endpoint breakers
│  ├─ ood.py               out-of-distribution detector
│  ├─ tenant.py            rate/budget enforcement
│  ├─ security.py          injection detection, sanitization
│  ├─ reviewer.py          async queue, batching, caps
│  ├─ trainer_worker.py    auto-retrain, eval gate, hot-swap, rollback
│  ├─ memory.py            SQLAlchemy core, all tables
│  ├─ config.py            hot reload, mode-aware
│  └─ dashboard/           live SPA
├─ router_model/
│  ├─ taxonomy.yaml        ~55 verticals
│  ├─ prototypes.json      structural-only seed
│  ├─ generate_data.py     synthetic seed
│  ├─ train.py             heads training (frozen embedding)
│  ├─ eval.py              base + live eval
│  ├─ export_onnx.py       ONNX inference artifact
│  ├─ embed_finetune.py    contrastive, manual-gated
│  ├─ registry.json        checkpoint versions, scores
│  ├─ MODEL_CARD.md        auto-generated per checkpoint
│  └─ data/                base/, curated/, live-eval/, flagged/
├─ tests/                  pytest unit tests
├─ gateway-config.json
├─ gateway-policy.json
├─ docker-compose.yml      Postgres for multi-instance mode
└─ requirements.txt
```

## Verification

Run unit tests:
```bash
pytest tests/            # or: python -m unittest discover -s tests
```

Lint + type check (CI does this):
```bash
ruff check gateway tests router_model
mypy gateway
```

Run with mock endpoints (scripted prices/latency/errors/breaker states):
```bash
python tests/mock_endpoints.py &
python -m gateway.app
# Integration coverage lives in tests/test_integration.py
```

## Migration to multi-instance

```bash
# 1. Start Postgres
docker compose up -d postgres

# 2. Set mode in gateway-config.json
# "mode": "multi",
# "db_url": "postgresql+psycopg://user:pass@localhost:5432/glint"

# 3. Migrate existing SQLite data
python -m gateway.memory migrate --from sqlite:///glint-v2.db --to postgresql+psycopg://...

# 4. Run multiple gateway instances behind LB
docker compose up --scale gateway=3
```
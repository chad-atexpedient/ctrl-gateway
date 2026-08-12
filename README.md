# Glint-V2

A **semantic LLM gateway** that routes requests to the cheapest-sufficient model,
auto-discovers new models from all providers, learns from feedback, and manages
a multi-tenant inference fleet — with a plugin system, A2A agent orchestration,
MCP facade, and a real-time comparison dashboard.

Glint-V2 sits between your application and your LLM providers (OpenAI, Anthropic,
Google, Ollama, OpenRouter, Groq, Together, DeepSeek, and 15+ others). It
classifies every request by vertical + complexity, routes it to the cheapest
model that can handle it, falls back on failure, and learns from the outcomes
to improve future routing decisions.

---

## What this is

| Layer | What it does |
|-------|-------------|
| **Gateway** | OpenAI-compatible reverse proxy. Health checks, circuit breakers, streaming, concurrency control, per-tenant auth. |
| **Router** | Custom hybrid model — frozen sentence embedding + MLP classifier heads. 65 verticals, complexity scoring, code/math/reasoning flags, OOD detection. CPU-only, ~1–7 ms per request. |
| **Policy engine** | Deterministic pre-routes (vision, medical, freshness) → cost-first gate → uncertainty escalation → tier ladder. Budget-aware routing with Wilson confidence bounds. |
| **Model catalog** | Auto-discovers models from OpenRouter (300+), OpenAI, Anthropic, Ollama. Capability scoring, auto-tier assignment, spidergraph comparison. |
| **Data flywheel** | Async reviewer labels every prompt; high-agreement samples become training data; auto-retrain with eval gate, atomic hot-swap, auto-rollback on regression. |
| **Extensions** | Plugin system, A2A agent registry, IBM ContextForge connector, MCP discovery + unified facade, prompt templates, webhooks, tool cache, SSRF protection. |
| **Security** | Prompt-injection detection (block/alert/log), provider domain allowlist, host firewall, SSRF protection, audit log. |
| **Multi-tenant** | Per-user API keys, rate limits, token/USD budgets, tier access, plan-based quotas. Tenant scoping on all extension tables. |

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp gateway-config.json gateway-config.runtime.json  # edit as needed

# 3. Run
python -m gateway.app

# 4. Open the dashboard
#    http://localhost:8076/dashboard
```

The gateway starts with a local Ollama endpoint by default. Add cloud providers
via the dashboard's **Providers** tab or the `/admin/endpoints` API.

---

## Architecture

```
                          ┌──────────────────────────────────────────────┐
                          │              Glint-V2 Gateway                 │
                          │                                              │
  Client ──POST /v1/chat▶│  Auth → Injection Check → Router → Policy    │
  Client ──POST /mcp────▶│  MCP Facade (tools/list, tools/call)         │
                          │                    │                         │
                          │         ┌─────────┴──────────┐              │
                          │         ▼                    ▼              │
                          │   Cost-First Route    Budget-Aware Route    │
                          │   (tier ladder)       (cascade chain)       │
                          │         │                    │              │
                          │    Endpoint Pool ◀── Model Catalog          │
                          │    (breakers, health)  (discovered models)  │
                          │         │                                    │
                          │    Transcoder (OpenAI ↔ Ollama ↔ Anthropic) │
                          │         │                                    │
  OpenAI ◀───────────────│─────────┘                                    │
  Anthropic ◀────────────│───────────────┐                              │
  Ollama ◀───────────────│──────────┐    │                              │
  OpenRouter ◀───────────│──────────┼────┘                              │
                          │          │                                   │
                          │   Reviewer Worker ──▶ Curated Samples       │
                          │   Trainer Worker ────▶ Auto-Retrain         │
                          │   Observer Worker ───▶ Observational Memory │
                          │                                              │
                          │   Extensions: Plugins, A2A, Webhooks,       │
                          │   Prompt Templates, Tool Cache, SSRF         │
                          └──────────────────────────────────────────────┘
```

### Request lifecycle

1. **Auth**: per-tenant API key resolution (or `X-User-Id` when auth disabled).
2. **Injection check**: DB-backed regex profiles scan the prompt. `block`-action
   profiles return HTTP 400 before routing. `alert`/`log` profiles flag but allow.
3. **Router**: embedding + MLP heads classify into 65 verticals + complexity 1–5
   + code/math/reasoning/long_output flags. OOD detection for unknown inputs.
4. **Policy**: deterministic pre-routes (vision, medical regex, OWUI tasks) →
   cost-first tier selection OR budget-aware cascade chain → escalation paths
   (OOD, low confidence, top-2 close).
5. **Transcoder**: rewrites the payload for the target endpoint kind (OpenAI,
   Ollama, Anthropic, Gemini). Bumps `max_tokens` for thinking tiers.
6. **Dispatch**: sends with concurrency control, circuit breaker, retry/fallback
   ladder. Streaming responses pass through as SSE.
7. **Reviewer** (async, post-response): independently labels the prompt; high-
   agreement labels become curated training samples.
8. **Trainer** (periodic): retrains the router from curated samples, eval-gates
   the result, hot-swaps the model atomically, auto-rollbacks on regression.

---

## Model catalog + discovery

The gateway auto-discovers available models from all configured providers and
maintains a searchable, comparable catalog with capability scoring.

### Discovery sources

| Provider | Endpoint | Models | Metadata |
|----------|----------|--------|----------|
| **OpenRouter** | `/api/v1/models` | 300+ | Per-token pricing, context length, modality, reasoning, tool support |
| **OpenAI** | `/v1/models` | ~20 | Model IDs (pricing from presets) |
| **Anthropic** | `/v1/models` | ~5 | Claude family IDs |
| **Ollama** | `/api/tags` | varies | Locally installed models with family/quantization |
| **Any OpenAI-compatible** | `/v1/models` | varies | Groq, Together, DeepSeek, Mistral, xAI, etc. |

### Capability scoring

Each discovered model receives a 0.0–1.0 capability score — a transparent,
tunable weighted blend:

| Factor | Weight | Logic |
|--------|--------|-------|
| Context length | up to 0.30 | Log-scaled, capped at 200K |
| Supports tools | +0.15 | Boolean bonus |
| Supports vision | +0.10 | Boolean bonus |
| Supports reasoning | +0.20 | Boolean bonus |
| Inverse pricing | up to 0.25 | Cheaper = higher (log-scaled) |

### Auto tier assignment

| Score range | Tier | Description |
|-------------|------|-------------|
| ≥ 0.80 | tier4 | Frontier models |
| 0.65–0.80 | tier3 | Strong models |
| 0.50–0.65 | tier2 | Mid-range |
| 0.35–0.50 | tier1 | Budget |
| < 0.35 | tier0 | Weakest/cheapest |

### Auto-registration

`POST /admin/models/auto-register` creates gateway endpoint entries from
catalog models and assigns them to tiers — breaking the old 1:1 endpoint:model
coupling. One OpenRouter API key can register 300+ models into the routing
system in a single call.

### Spidergraph comparison

The dashboard's **Models** tab shows a radar chart comparing up to 5 models
across 6 axes: Context, Cost Efficiency, Capability, Observed Success Rate,
Observed Latency, Feature Breadth. The first three are static (from the
catalog); the latter two are dynamic (from the gateway's actual traffic data
when available).

### Admin API

```
POST /admin/models/sync              — trigger provider sync
GET  /admin/models/catalog           — list with filters (provider, score, features)
GET  /admin/models/stats             — summary counts
GET  /admin/models/comparison        — enriched radar data (catalog + quality)
POST /admin/models/auto-register     — create endpoints from catalog (dry_run + commit)
PUT  /admin/models/{id}/tier         — manual tier override
PUT  /admin/models/{id}/enabled      — enable/disable
```

### Configuration

```json
{
  "model_sync": {
    "enabled": true,
    "openrouter_enabled": true,
    "ollama_enabled": true,
    "anthropic_enabled": false,
    "providers": [],
    "sync_interval_seconds": 21600,
    "auto_sync": false,
    "timeout_seconds": 30.0
  }
}
```

---

## Subscription budgets + budget-aware routing

Per-tenant token/USD budgets with Wilson-confidence success probability
estimation and cascade-chain routing:

- **Plans** (`plan_quotas`): daily token/USD limits + required success
  probability + allowed model whitelist.
- **Per-model limits** (`model_token_limits`): daily token/USD caps +
  per-request max tokens for individual models.
- **Budget-aware router** (`policy.budget_aware_route`): builds a cascade chain
  ordered by ascending cost, walking candidates until
  `cascade_success_probability(chain) >= target_success_probability`. Raises
  `BudgetError("quality_target_unmet")` → HTTP 429 when no chain meets target.
- **Quality estimation** (`quality.estimate_success_probability`): Wilson lower
  bound (95% confidence) of observed success rate, clamped to a conservative
  0.5 prior when <10 samples exist. A new model with zero history CANNOT claim
  100% confidence.

```
POST   /admin/plans                           — create plan
PUT    /admin/plans/{id}                      — update plan
POST   /admin/users/{tenant}/subscription     — assign plan
PUT    /admin/users/{tenant}/budget/tokens    — set daily token budget
PUT    /admin/users/{tenant}/models/{ep}/limits — per-model limits
GET    /admin/users/{tenant}/limits           — all limits
GET    /usage                                 — user-facing usage
GET    /usage/limits                          — user-facing limits
```

---

## Security Hub

Three layers of protection, all configurable at runtime:

### Prompt-injection detection

DB-backed regex profiles (`injection_profiles` table) with severity
(low/medium/high/critical) and action (block/alert/log). Six built-in profiles
seeded at startup: jailbreak, role_override, context_escape, router_manipulation,
data_exfiltration, semantic_dos. `block`-action profiles return HTTP 400 before
routing. Custom profiles via `/admin/security/injection-profiles`.

### Provider domain allowlist

`DomainAllowlistEnforcer` governs which LLM provider domains the gateway may
talk to. Config + DB-backed rules; tenant-specific overrides; unknown domains
fall back to `default_action` (default: block). Optional `HostFirewallManager`
enforces at the OS level (Windows `netsh` / Linux `iptables`).

### SSRF protection

`gateway/ssrf.py` validates all outbound HTTP from A2A, ContextForge, MCP
discovery, and webhook delivery. Blocks loopback, private (RFC 1918), link-local
(169.254 — cloud metadata), reserved, and multicast IP ranges unless explicitly
allowed.

```
GET  /admin/security/events         — audit log
GET  /admin/security/events/stats   — 7-day summary
CRUD /admin/security/injection-profiles
CRUD /admin/security/provider-allowlist
POST /admin/security/sync-firewall   — sync host firewall rules
GET  /admin/security/status          — enforcer + host-fw state
GET  /admin/security/test            — dry-run a URL
```

---

## Gateway extensions

### Plugin system (`gateway/plugin.py`)

Plugins live under `gateway/plugins/<name>/` with a `manifest.yaml` + `plugin.py`
exposing `build_router(context) -> web.RouteTableDef`. The `PluginLoader` scans
on startup, hot-reloads on file changes, and provides `PluginContext` with
event emission, settings access, and memory. Sample plugin:
`gateway/plugins/microsoft_learn/`.

### A2A registry (`gateway/a2a_registry.py`)

Register and invoke external Agent-to-Agent endpoints. Supports `jsonrpc`,
`openai`, `anthropic`, and `custom` payload formats. Per-agent metrics,
virtual server grouping, tool-cache integration, SSRF-guarded invocation.

### IBM ContextForge connector (`gateway/contextforge_client.py`)

Three modes: `external` (pull from ContextForge REST), `embedded` (in-process
registration), `both`. Periodic sync loop. Merges agents, virtual servers,
tools, and prompts into local registries.

### MCP discovery + facade (`gateway/mcp_discovery.py`, `gateway/mcp_facade.py`)

Auto-discovers MCP servers via `/.well-known/mcp.json`, HTTP initialize probe,
and SSE endpoint scanning. Unified `POST /mcp` JSON-RPC 2.0 facade exposes
federated tools + A2A agents + prompt templates to any MCP-compatible client.
Jittered background probing with per-host last-seen cache.

### Prompt templates (`gateway/prompt_registry.py`)

DB-backed templates with category-based auto-injection. Five builtins (coder,
reviewer, translator, summarizer, safety_refusal). When `prompts.auto_inject`
is enabled, the gateway looks up a template by vertical category, renders it
with context variables, and prepends it as a system message.

### Webhook dispatcher (`gateway/webhook_dispatcher.py`)

Async fan-out with HMAC SHA-256 signing, exponential backoff retry, bounded
concurrency, and per-delivery logging. Every published event fans out to
matching webhooks automatically. Per-webhook `max_retries` override. Tenant-
scoped dispatch.

### Tool cache (`gateway/tool_cache.py`)

LRU + TTL cache with tenant isolation, per-tool TTL overrides, and bypass keys.
Wired into the MCP facade and A2A `invoke_agent`.

### Tenant scoping

All extension tables (`plugins`, `a2a_agents`, `prompt_templates`, `webhooks`,
`federated_tools`) carry a `tenant_id` column (default `"__all__"`). List/get
paths filter by `(tenant_id IN ("__all__", caller_tenant))`. The MCP facade,
webhook dispatcher, and prompt auto-inject all respect tenant scoping.

---

## Routing pipeline

```
Input → Router (embedding + MLP heads)
          ↓
      Vertical + Complexity + Flags
          ↓
    Policy Engine
    ├── Pre-routes (vision, medical, OWUI, freshness)
    ├── Cost-first gate (cheapest-sufficient tier)
    ├── Budget-aware route (cascade chain, Wilson confidence)
    ├── Uncertainty escalation (OOD, low confidence, top-2 close)
    └── Tier ladder (context overflow → breaker → fallback)
          ↓
    Endpoint Pool (concurrency, breaker, health)
          ↓
    Transcoder (OpenAI ↔ Ollama ↔ Anthropic ↔ Gemini)
          ↓
    Upstream Provider
```

---

## Observational memory (Mastra pattern)

Per-tenant working memory (`memory_observational.py`) maintains conversation
context across requests. Features include thread-scoped token counting,
automatic compaction when approaching context limits, and resource-level
metadata. Assembled into the message array before routing.

---

## Configuration reference

Top-level keys in `gateway-config.json`:

| Key | Description |
|-----|-------------|
| `mode` | `"single"` (SQLite) or `"multi"` (Postgres) |
| `db_url` | SQLAlchemy connection string |
| `endpoints` | Provider endpoint definitions |
| `tiers` | Tier ladder with capability scores |
| `tenants` | Per-user tier access, budgets, rate limits |
| `reviewer` | Async reviewer model config |
| `trainer` | Auto-retrain schedule + thresholds |
| `embedding` | Router embedding model (ONNX path, checksum) |
| `routing` | Router thresholds (confidence, OOD, top-2 margin) |
| `security` | Injection profiles, provider allowlist, host firewall |
| `auth` | Per-tenant API keys + public paths |
| `http` | CORS, max body size, health poll interval |
| `memory` | Observational memory config |
| `model_sync` | Provider discovery engine config |
| `plugins` | Plugin loader config |
| `a2a` | A2A registry config |
| `contextforge` | ContextForge connector config |
| `mcp_discovery` | MCP server discovery config |
| `prompts` | Prompt template auto-inject config |
| `webhooks` | Webhook dispatcher config |
| `tool_cache` | LRU + TTL cache config |

---

## API reference

### Public endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI-compatible chat (stream + non-stream) |
| GET | `/v1/models` | List available models (tier-scoped) |
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness probe (router loaded, endpoints healthy) |
| POST | `/mcp` | MCP JSON-RPC 2.0 facade |
| GET | `/dashboard` | Dashboard SPA |

### User-facing

| Method | Path | Description |
|--------|------|-------------|
| GET | `/usage` | Today's token/cost usage |
| GET | `/usage/limits` | Budget + rate limit status |
| GET | `/trace` | Recent routing decisions |
| GET | `/accuracy` | Router accuracy stats |
| GET | `/cost` | Cost breakdown |
| GET | `/memory` | Working memory state |
| GET | `/events/recent` | Recent events |

### Admin (requires `admin` scope)

| Method | Path | Description |
|--------|------|-------------|
| CRUD | `/admin/endpoints` | Provider endpoints |
| CRUD | `/admin/tiers` | Tier definitions + endpoint assignments |
| CRUD | `/admin/users/{tenant}/*` | Tenant budgets, limits, plans |
| CRUD | `/admin/keys` | API key generation + revocation |
| POST | `/admin/models/sync` | Trigger model catalog sync |
| GET | `/admin/models/catalog` | Browse discovered models |
| GET | `/admin/models/comparison` | Enriched spidergraph data |
| POST | `/admin/models/auto-register` | Create endpoints from catalog |
| CRUD | `/admin/plugins` | Plugin management |
| CRUD | `/admin/a2a/agents` | A2A agent registry |
| CRUD | `/admin/prompts` | Prompt templates |
| CRUD | `/admin/webhooks` | Webhook subscribers |
| GET | `/admin/cache/stats` | Tool cache stats |
| CRUD | `/admin/security/*` | Security profiles + allowlist + events |

---

## Authentication

Per-tenant API keys with hashed storage (`sha256:<digest>`). Keys are shown once
on generation. When `auth.enabled` is false, `X-User-Id` header identifies
tenants (trusted reverse proxy mode).

```bash
# Generate a key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Add to gateway-config.json
"auth": {
  "enabled": true,
  "keys": {
    "<generated-key>": {"tenant_id": "acme", "scope": ["admin"]}
  }
}
```

---

## Project layout

```
glint-v2/
├── gateway/
│   ├── app.py                    — aiohttp app, all routes, init/cleanup
│   ├── policy.py                 — routing decisions (cost-first, budget-aware)
│   ├── router.py                 — embedding + MLP heads inference (numpy)
│   ├── endpoints.py              — endpoint pool, clients, breakers
│   ├── transcoder.py             — payload rewriting per provider kind
│   ├── memory.py                 — SQLAlchemy schema + helpers (30+ tables)
│   ├── memory_observational.py   — working memory (Mastra pattern)
│   ├── config.py                 — config loading + validation
│   ├── auth.py                   — per-tenant API key auth
│   ├── tenant.py                 — tenant state + usage tracking
│   ├── admin.py                  — overlay manager (live config CRUD)
│   ├── events.py                 — event bus (pub/sub + webhooks)
│   ├── discovery.py              — local endpoint probing
│   ├── model_sync.py             — provider model discovery engine
│   ├── ssrf.py                   — SSRF protection
│   ├── firewall.py               — domain allowlist + host firewall
│   ├── security.py               — injection detection
│   ├── plugin.py                 — plugin loader
│   ├── a2a_registry.py           — A2A agent lifecycle
│   ├── contextforge_client.py    — IBM ContextForge connector
│   ├── mcp_discovery.py          — MCP server auto-discovery
│   ├── mcp_facade.py             — unified MCP JSON-RPC endpoint
│   ├── prompt_registry.py        — prompt template registry
│   ├── webhook_dispatcher.py     — webhook fan-out with HMAC + retry
│   ├── tool_cache.py             — LRU + TTL cache
│   ├── reviewer.py               — async post-response labeler
│   ├── trainer_worker.py         — auto-retrain pipeline
│   ├── observer_worker.py        — observational memory worker
│   ├── swarm.py                  — multi-step task decomposition
│   ├── ood.py                    — out-of-distribution detection
│   ├── quality.py                — Wilson confidence success estimation
│   ├── metrics.py                — Prometheus metrics
│   ├── provider_presets.json     — 22 provider connection templates
│   ├── dashboard/
│   │   └── index.html            — SPA dashboard (10 tabs)
│   └── plugins/
│       └── microsoft_learn/      — sample plugin
├── router_model/
│   ├── train.py                  — PyTorch MLP training (trunk + heads)
│   ├── eval.py                   — evaluation (numpy mirror)
│   ├── embed_finetune.py         — contrastive embedding fine-tune
│   ├── generate_data.py          — synthetic training data
│   ├── taxonomy.yaml             — 65 vertical definitions
│   └── prototypes.json           — structural prototype centroids
├── tests/
│   ├── test_unit.py              — unit tests
│   ├── test_memory.py            — memory + events
│   ├── test_review_fixes.py      — regression tests
│   ├── test_integration.py       — end-to-end with mock upstreams
│   ├── test_security.py          — security hub tests
│   ├── test_budgets.py           — budget-aware routing tests
│   ├── test_plugins.py           — extension module tests
│   └── test_model_catalog.py     — model discovery + catalog tests
├── gateway-config.json           — main config
├── gateway-policy.json           — routing policy config
├── AGENTS.md                     — agent working notes + invariants
└── requirements.txt
```

---

## Testing

```bash
# Unit + integration + extension tests
python -m unittest discover -s tests

# Or with pytest
python -m pytest tests/ -v

# Lint + type check
ruff check gateway tests router_model
mypy gateway
```

Test suite: **313+ tests** covering routing policy, budget enforcement, security
profiles, plugin system, A2A registry, model catalog, tenant scoping, and
end-to-end integration with mock upstreams.

---

## Multi-instance

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

---

## Dashboard

The dashboard at `/dashboard` has 11 tabs:

| Tab | Description |
|-----|-------------|
| **Live** | Real-time stats: decisions, model count, endpoint health, cost, trace |
| **Providers** | Add/edit/delete provider endpoints + tier assignments |
| **Keys** | API key generation + revocation |
| **Routing** | Tier ladder visualization + capability per vertical |
| **Models** | Model catalog table + sync + auto-register + spidergraph radar |
| **Flywheel** | Curated samples, training pipeline, router version, confusion graph |
| **Security** | Injection events, profiles, allowlist, firewall status |
| **Plugins** | Plugin management + reload + enable/disable |
| **A2A** | Agent registry + virtual servers + ContextForge sync + MCP discover |
| **Prompts** | Template CRUD + preview + auto-inject config |
| **Webhooks** | Webhook CRUD + delivery log + tool cache stats |

Pure vanilla JS — no external dependencies, no build step.

---

## Verification

```bash
ruff check gateway tests router_model
mypy gateway
python -m unittest discover -s tests
```

All three must pass before any change is trusted.

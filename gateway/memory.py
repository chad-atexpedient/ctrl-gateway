"""Glint-V2 data layer.

SQLAlchemy core (schema-only; no ORM) — same schema works for SQLite (single
mode) and Postgres (multi mode). Driver is selected from db_url.

Tables:
  routing_log      — every routing decision (immutable append-only)
  feedback         — human feedback (correct/wrong) for routing decisions
  sessions         — working memory: previous tier per session
  model_versions   — registry of router checkpoint versions
  checkpoints      — registry: per-version eval scores + rollback pointers
  flagged_inputs   — suspicious / injection-flagged prompts
  security_events  — unified audit log for all security signals
  provider_allowlist — per-tenant domain allowlist/block rules
  injection_profiles — configurable injection rule sets (DB-backed)
  users            — tenant config + current spend
  usage_counters   — token counts per user per period (for budget enforcement)
  plan_quotas      — subscription plan definitions
  tenant_plans     — tenant → plan assignment
  model_token_limits — per-model per-tenant daily token/USD caps
  model_quality_profiles — calibrated success rate per model+vertical+complexity
  breakers         — per-endpoint circuit breaker state
  review_queue     — async reviewer queue (Postgres LISTEN/NOTIFY in multi mode)
  review_results   — reviewer outputs (per-field labels)
  curated_samples  — high-agreement samples for training (versioned)
  live_eval_set    — periodically sampled live-traffic eval
  plugins          — gateway plugin registry (manifest-based connectors)
  a2a_agents       — external A2A agent registry (jsonrpc/openai/anthropic/custom)
  a2a_virtual_servers — named bundles of A2A agents
  a2a_metrics      — per-agent invocation history (success rate, latency)
  prompt_templates — DB-backed prompt template registry
  webhooks         — webhook subscribers (fan-out from event bus)
  webhook_deliveries — delivery log (success/failure/response)
  contextforge_sync_log — ContextForge connector sync history
  federated_tools  — tools federated from external sources (ContextForge, etc.)
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import threading
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    delete,
    func,
    insert,
    select,
    text,
    update,
)
from sqlalchemy.engine import Engine

log = logging.getLogger("glint.memory")

metadata = MetaData()

routing_log = Table(
    "routing_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime, default=lambda: datetime.now(UTC), index=True),
    Column("tenant_id", String(64), index=True),
    Column("session_id", String(64), index=True),
    Column("model_version", String(64), index=True),
    Column("policy_version", Integer),
    Column("query_hash", String(64), index=True),
    Column("query_preview", Text),
    Column("vertical", String(64), index=True),
    Column("vertical_top2_prob", Float),
    Column("complexity", Integer),
    Column("flags_code", Boolean),
    Column("flags_math", Boolean),
    Column("flags_reasoning", Boolean),
    Column("flags_long_output", Boolean),
    Column("tier", String(32), index=True),
    Column("endpoint", String(64)),
    Column("source", String(32), index=True),
    Column("ms_classify", Float),
    Column("ms_total", Float),
    Column("est_cost_usd", Float),
    Column("actual_cost_usd", Float),
    Column("escalated", Boolean, default=False),
    Column("fallback_used", Boolean, default=False),
    Column("error", Text),
    Column("truncated", Boolean, default=False),
    Column("has_image", Boolean, default=False),
    Column("has_injection_signal", Boolean, default=False),
    Column("response_ok", Boolean),
    Column("review_status", String(16), default="pending"),
    Column("extra", Text),
)

feedback = Table(
    "feedback",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime, default=lambda: datetime.now(UTC)),
    Column("decision_id", Integer, ForeignKey("routing_log.id"), index=True),
    Column("correct", Boolean),
    Column("suggested_tier", String(32)),
    Column("comment", Text),
    Column("source", String(16), default="human"),
)

sessions = Table(
    "sessions",
    metadata,
    Column("session_id", String(64), primary_key=True),
    Column("tenant_id", String(64), index=True),
    Column("last_tier", String(32)),
    Column("last_vertical", String(64)),
    Column("last_endpoint", String(64)),
    Column("last_response_ok", Boolean),
    Column("last_response_ms", Float),
    Column("last_used_at", DateTime, default=lambda: datetime.now(UTC)),
    Column("request_count", Integer, default=0),
)

model_versions = Table(
    "model_versions",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("parent_id", String(64)),
    Column("created_at", DateTime, default=lambda: datetime.now(UTC)),
    Column("embedding_model", String(128)),
    Column("heads_hash", String(64)),
    Column("active", Boolean, default=False),
    Column("created_by", String(32), default="trainer"),
)

checkpoints = Table(
    "checkpoints",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("model_version_id", String(64), ForeignKey("model_versions.id")),
    Column("eval_base_accuracy", Float),
    Column("eval_live_accuracy", Float),
    Column("per_vertical_accuracy", Text),
    Column("confusion_top20", Text),
    Column("policy_replay_drift_pct", Float),
    Column("promoted", Boolean, default=False),
    Column("rolled_back_at", DateTime),
    Column("rolled_back_reason", Text),
    Column("created_at", DateTime, default=lambda: datetime.now(UTC)),
)

flagged_inputs = Table(
    "flagged_inputs",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime, default=lambda: datetime.now(UTC)),
    Column("tenant_id", String(64), index=True),
    Column("decision_id", Integer),
    Column("reason", String(32), index=True),
    Column("severity", String(16), default="medium", index=True),
    Column("matched_profile", String(64)),
    Column("matched_regex", String(256)),
    Column("query_preview", Text),
    Column("action_taken", String(32)),
    Column("security_event_id", Integer, index=True),
)

security_events = Table(
    "security_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime, default=lambda: datetime.now(UTC), index=True),
    Column("tenant_id", String(64), index=True),
    Column("event_type", String(40), index=True),
    Column("severity", String(16), default="medium", index=True),
    Column("reason", Text),
    Column("matched_pattern", String(256)),
    Column("query_preview", Text),
    Column("endpoint_target", String(256)),
    Column("action_taken", String(32)),
    Column("request_metadata_json", Text),
)

provider_allowlist = Table(
    "provider_allowlist",
    metadata,
    Column("tenant_id", String(64), primary_key=True),
    Column("domain_pattern", String(256), primary_key=True),
    Column("action", String(16), default="allow"),
    Column("created_at", DateTime, default=lambda: datetime.now(UTC)),
    Column("notes", Text),
)

injection_profiles = Table(
    "injection_profiles",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(64), unique=True),
    Column("regexes_json", Text),
    Column("severity", String(16), default="medium"),
    Column("action", String(16), default="alert"),
    Column("enabled", Boolean, default=True),
    Column("is_builtin", Boolean, default=False),
    Column("created_at", DateTime, default=lambda: datetime.now(UTC)),
    Column("updated_at", DateTime, default=lambda: datetime.now(UTC)),
)

users = Table(
    "users",
    metadata,
    Column("tenant_id", String(64), primary_key=True),
    Column("tier_access", Text),
    Column("budget_usd_per_day", Float, default=1.0),
    Column("rps_limit", Integer, default=100),
    Column("concurrent_limit", Integer, default=20),
    Column("tokens_per_min", Integer, default=200000),
    Column("daily_token_limit", Integer, default=0),
    Column("plan_id", String(64), index=True),
    Column("target_success_probability", Float, default=0.99),
    Column("created_at", DateTime, default=lambda: datetime.now(UTC)),
)

usage_counters = Table(
    "usage_counters",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("tenant_id", String(64), index=True),
    Column("endpoint_name", String(64), index=True),
    Column("ts", DateTime, default=lambda: datetime.now(UTC), index=True),
    Column("period_hour", String(13), index=True),
    Column("period_day", String(10), index=True),
    Column("period_month", String(7), index=True),
    Column("tokens_in", Integer, default=0),
    Column("tokens_out", Integer, default=0),
    Column("cost_usd", Float, default=0.0),
    Column("request_count", Integer, default=0),
)

plan_quotas = Table(
    "plan_quotas",
    metadata,
    Column("plan_id", String(64), primary_key=True),
    Column("name", String(128)),
    Column("daily_token_limit", Integer, default=0),
    Column("daily_usd_limit", Float, default=0.0),
    Column("required_success_probability", Float, default=0.99),
    Column("allowed_models_json", Text),
    Column("created_at", DateTime, default=lambda: datetime.now(UTC)),
)

tenant_plans = Table(
    "tenant_plans",
    metadata,
    Column("tenant_id", String(64), primary_key=True),
    Column("plan_id", String(64), ForeignKey("plan_quotas.plan_id"), index=True),
    Column("effective_from", DateTime, default=lambda: datetime.now(UTC)),
    Column("notes", Text),
)

model_token_limits = Table(
    "model_token_limits",
    metadata,
    Column("tenant_id", String(64), primary_key=True),
    Column("endpoint_name", String(64), primary_key=True),
    Column("daily_token_limit", Integer, default=0),
    Column("daily_usd_limit", Float, default=0.0),
    Column("max_request_tokens", Integer, default=0),
    Column("period_day", String(10), index=True),
    Column("tokens_used", Integer, default=0),
    Column("cost_used_usd", Float, default=0.0),
    Column("updated_at", DateTime, default=lambda: datetime.now(UTC)),
)

model_quality_profiles = Table(
    "model_quality_profiles",
    metadata,
    Column("endpoint_name", String(64), primary_key=True),
    Column("vertical", String(64), primary_key=True),
    Column("complexity_min", Integer, primary_key=True),
    Column("complexity_max", Integer, primary_key=True),
    Column("success_count", Integer, default=0),
    Column("total_count", Integer, default=0),
    Column("calibration_samples", Integer, default=0),
    Column("last_updated", DateTime, default=lambda: datetime.now(UTC)),
)


breakers = Table(
    "breakers",
    metadata,
    Column("endpoint_name", String(64), primary_key=True),
    Column("state", String(16), default="CLOSED"),
    Column("consecutive_failures", Integer, default=0),
    Column("opened_at", DateTime),
    Column("last_failure_at", DateTime),
    Column("last_success_at", DateTime),
    Column("half_open_probes_remaining", Integer, default=0),
)

review_queue = Table(
    "review_queue",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts_queued", DateTime, default=lambda: datetime.now(UTC), index=True),
    Column("decision_id", Integer, ForeignKey("routing_log.id"), index=True),
    Column("tenant_id", String(64), index=True),
    Column("priority", Integer, default=0),
    Column("status", String(16), default="pending", index=True),
    Column("attempts", Integer, default=0),
    Column("started_at", DateTime),
    Column("last_error", Text),
    Column("cost_usd_estimate", Float, default=0.0),
    Column("prompt_text", Text),
)

review_results = Table(
    "review_results",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("decision_id", Integer, ForeignKey("routing_log.id"), index=True, unique=True),
    Column("reviewer_model", String(128)),
    Column("reviewer_endpoint", String(256)),
    Column("reviewed_at", DateTime, default=lambda: datetime.now(UTC)),
    Column("vertical_label", String(64)),
    Column("complexity_label", Integer),
    Column("flag_code_label", Boolean),
    Column("flag_math_label", Boolean),
    Column("flag_reasoning_label", Boolean),
    Column("flag_long_output_label", Boolean),
    Column("truncated", Boolean),
    Column("agreement_vertical", Boolean),
    Column("agreement_complexity", Boolean),
    Column("agreement_code", Boolean),
    Column("agreement_math", Boolean),
    Column("agreement_reasoning", Boolean),
    Column("agreement_long_output", Boolean),
    Column("all_fields_agree", Boolean, index=True),
    Column("router_confidence_at_decision", Float),
    Column("trust_score", Float, default=1.0),
    Column("curated", Boolean, default=False),
    Column("curated_at", DateTime),
    Column("curated_run_id", String(64), index=True),
    Column("meta_reviewed", Boolean, default=False),
    Column("meta_review_agreement", Float),
    Column("raw_response", Text),
    Column("prompt_text", Text),
    Column("cost_usd", Float),
)

curated_samples = Table(
    "curated_samples",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("decision_id", Integer, ForeignKey("routing_log.id"), index=True),
    Column("review_result_id", Integer, ForeignKey("review_results.id"), index=True),
    Column("curated_run_id", String(64), index=True),
    Column("query_hash", String(64), index=True),
    Column("text", Text),
    Column("vertical", String(64), index=True),
    Column("complexity", Integer),
    Column("flag_code", Boolean),
    Column("flag_math", Boolean),
    Column("flag_reasoning", Boolean),
    Column("flag_long_output", Boolean),
    Column("model_version", String(64)),
    Column("reviewer_model", String(128)),
    Column("trust_score", Float),
    Column("source", String(16), default="synthetic"),
    Column("created_at", DateTime, default=lambda: datetime.now(UTC)),
)

live_eval_set = Table(
    "live_eval_set",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("decision_id", Integer, ForeignKey("routing_log.id"), index=True, unique=True),
    Column("query_hash", String(64), index=True),
    Column("text", Text),
    Column("ground_truth_vertical", String(64)),
    Column("ground_truth_complexity", Integer),
    Column("ground_truth_flags", Text),
    Column("label_source", String(32)),
    Column("created_at", DateTime, default=lambda: datetime.now(UTC)),
)

trainer_state = Table(
    "trainer_state",
    metadata,
    Column("key", String(64), primary_key=True),
    Column("value", Text),
)


# ----- Plugin system, A2A registry, prompt templates, webhooks, cache -----


plugins = Table(
    "plugins",
    metadata,
    Column("name", String(64), primary_key=True),
    Column("version", String(32)),
    Column("description", Text),
    Column("prefix", String(128)),
    Column("module_path", String(256)),
    Column("config_json", Text),
    Column("enabled", Boolean, default=True),
    Column("loaded", Boolean, default=False),
    Column("loaded_at", DateTime),
    Column("error", Text),
    Column("is_builtin", Boolean, default=False),
    Column("created_at", DateTime, default=lambda: datetime.now(UTC)),
    Column("updated_at", DateTime, default=lambda: datetime.now(UTC)),
)


a2a_agents = Table(
    "a2a_agents",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(64), unique=True),
    Column("endpoint_url", String(256)),
    Column("agent_type", String(32)),
    Column("description", Text),
    Column("auth_type", String(32), default="none"),
    Column("auth_value_encrypted", Text),
    Column("enabled", Boolean, default=True),
    Column("protocol_version", String(16)),
    Column("capabilities_json", Text),
    Column("config_json", Text),
    Column("tags_json", Text),
    Column("created_at", DateTime, default=lambda: datetime.now(UTC)),
    Column("updated_at", DateTime, default=lambda: datetime.now(UTC)),
)


a2a_virtual_servers = Table(
    "a2a_virtual_servers",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(64), unique=True),
    Column("description", Text),
    Column("associated_agents_json", Text),
    Column("enabled", Boolean, default=True),
    Column("created_at", DateTime, default=lambda: datetime.now(UTC)),
    Column("updated_at", DateTime, default=lambda: datetime.now(UTC)),
)


a2a_metrics = Table(
    "a2a_metrics",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("agent_id", Integer, ForeignKey("a2a_agents.id"), index=True),
    Column("ts", DateTime, default=lambda: datetime.now(UTC), index=True),
    Column("tenant_id", String(64), index=True),
    Column("success", Boolean),
    Column("latency_ms", Float),
    Column("error", Text),
    Column("interaction_type", String(32)),
)


prompt_templates = Table(
    "prompt_templates",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(64), unique=True),
    Column("description", Text),
    Column("template_text", Text),
    Column("variables_json", Text),
    Column("category", String(64), index=True),
    Column("enabled", Boolean, default=True),
    Column("is_builtin", Boolean, default=False),
    Column("version", Integer, default=1),
    Column("source", String(32), default="manual"),
    Column("created_at", DateTime, default=lambda: datetime.now(UTC)),
    Column("updated_at", DateTime, default=lambda: datetime.now(UTC)),
)


webhooks = Table(
    "webhooks",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(64), unique=True),
    Column("url", String(512)),
    Column("events_json", Text),
    Column("secret", String(128)),
    Column("enabled", Boolean, default=True),
    Column("description", Text),
    Column("created_at", DateTime, default=lambda: datetime.now(UTC)),
    Column("updated_at", DateTime, default=lambda: datetime.now(UTC)),
)


webhook_deliveries = Table(
    "webhook_deliveries",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("webhook_id", Integer, ForeignKey("webhooks.id"), index=True),
    Column("ts", DateTime, default=lambda: datetime.now(UTC), index=True),
    Column("event_type", String(64), index=True),
    Column("tenant_id", String(64), index=True),
    Column("status_code", Integer),
    Column("response_body", Text),
    Column("payload_json", Text),
    Column("error", Text),
    Column("attempt", Integer, default=1),
    Column("duration_ms", Float),
)


contextforge_sync_log = Table(
    "contextforge_sync_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ts", DateTime, default=lambda: datetime.now(UTC), index=True),
    Column("sync_type", String(32), index=True),
    Column("source", String(256)),
    Column("items_synced", Integer, default=0),
    Column("items_added", Integer, default=0),
    Column("items_updated", Integer, default=0),
    Column("errors_json", Text),
    Column("duration_ms", Float),
)


federated_tools = Table(
    "federated_tools",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("name", String(128), unique=True),
    Column("source", String(32), index=True),
    Column("source_url", String(512)),
    Column("tool_json", Text),
    Column("enabled", Boolean, default=True),
    Column("last_synced", DateTime),
    Column("created_at", DateTime, default=lambda: datetime.now(UTC)),
)


# ----- Engine + connection management -----


_engine: Engine | None = None
_lock = threading.Lock()


def init_engine(db_url: str) -> Engine:
    """Create engine and create all tables. Idempotent."""
    global _engine
    with _lock:
        if _engine is not None:
            return _engine
        connect_args = {}
        if db_url.startswith("sqlite"):
            connect_args["check_same_thread"] = False
        _engine = create_engine(db_url, future=True, connect_args=connect_args)
        if db_url.startswith("sqlite"):
            from sqlalchemy import event

            @event.listens_for(_engine, "connect")
            def _sqlite_pragmas(dbapi_connection, _connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA busy_timeout=5000")
                cursor.close()
        metadata.create_all(_engine)
        _migrate(_engine)
        return _engine


def _migrate(engine: Engine):
    """Lightweight column migrations for pre-existing databases.

    create_all() does not alter existing tables, so columns added after the
    first deploy need an explicit ALTER TABLE. Safe to run on every boot.
    """
    try:
        with engine.connect() as conn:
            existing = _column_names(conn, "routing_log")
            if "vertical_top2_prob" not in existing:
                conn.execute(text("ALTER TABLE routing_log ADD COLUMN vertical_top2_prob FLOAT"))
                conn.commit()
            review_columns = _column_names(conn, "review_queue")
            if review_columns and "prompt_text" not in review_columns:
                conn.execute(text("ALTER TABLE review_queue ADD COLUMN prompt_text TEXT"))
                conn.commit()
            if review_columns and "started_at" not in review_columns:
                conn.execute(text("ALTER TABLE review_queue ADD COLUMN started_at DATETIME"))
                conn.commit()
            result_columns = _column_names(conn, "review_results")
            if result_columns and "prompt_text" not in result_columns:
                conn.execute(text("ALTER TABLE review_results ADD COLUMN prompt_text TEXT"))
                conn.commit()
    except Exception as e:
        log.warning("schema migration skipped: %s", e)


def _column_names(conn, table_name: str) -> set:
    import sqlalchemy
    insp = sqlalchemy.inspect(conn)
    try:
        return {c["name"] for c in insp.get_columns(table_name)}
    except Exception:
        return set()


def engine() -> Engine:
    if _engine is None:
        raise RuntimeError("engine not initialized — call init_engine first")
    return _engine


def close_engine() -> None:
    global _engine
    with _lock:
        if _engine is not None:
            _engine.dispose()
            _engine = None


@contextlib.contextmanager
def begin():
    """Transactional context. Commits on success, rolls back on exception."""
    with engine().begin() as conn:
        yield conn


# ----- High-level helpers -----


def log_decision(
    *,
    tenant_id: str,
    session_id: str,
    model_version: str,
    policy_version: int,
    query_hash: str,
    query_preview: str,
    vertical: str,
    complexity: int,
    flags: dict,
    tier: str,
    endpoint: str,
    source: str,
    ms_classify: float,
    ms_total: float,
    est_cost_usd: float,
    escalated: bool,
    fallback_used: bool,
    has_image: bool,
    has_injection_signal: bool,
    vertical_top2_prob: float | None = None,
    truncated: bool = False,
    error: str | None = None,
    response_ok: bool | None = None,
    actual_cost_usd: float | None = None,
    extra: dict | None = None,
) -> int:
    """Append a routing decision. Returns decision id."""
    with begin() as conn:
        result = conn.execute(
            insert(routing_log).values(
                tenant_id=tenant_id,
                session_id=session_id,
                model_version=model_version,
                policy_version=policy_version,
                query_hash=query_hash,
                query_preview=query_preview[:1000],
                vertical=vertical,
                vertical_top2_prob=vertical_top2_prob,
                complexity=complexity,
                flags_code=flags.get("code", False),
                flags_math=flags.get("math", False),
                flags_reasoning=flags.get("reasoning", False),
                flags_long_output=flags.get("long_output", False),
                tier=tier,
                endpoint=endpoint,
                source=source,
                ms_classify=ms_classify,
                ms_total=ms_total,
                est_cost_usd=est_cost_usd,
                actual_cost_usd=actual_cost_usd,
                escalated=escalated,
                fallback_used=fallback_used,
                has_image=has_image,
                has_injection_signal=has_injection_signal,
                truncated=truncated,
                error=error,
                response_ok=response_ok,
                extra=json.dumps(extra) if extra else None,
                review_status="pending",
            )
        )
        return int(result.inserted_primary_key[0])


def record_feedback(
    decision_id: int, correct: bool, suggested_tier: str | None = None,
    comment: str | None = None, source: str = "human",
):
    with begin() as conn:
        conn.execute(insert(feedback).values(
            decision_id=decision_id,
            correct=correct,
            suggested_tier=suggested_tier,
            comment=comment,
            source=source,
        ))
    if not correct:
        curate_reviewed_correction(decision_id)


def _insert_ignore(table: Table, **values):
    """Make an INSERT idempotent on conflict for SQLite and PostgreSQL."""
    dialect = engine().dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert as sqlite_insert
        return sqlite_insert(table).values(**values).on_conflict_do_nothing()
    if dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        return pg_insert(table).values(**values).on_conflict_do_nothing()
    return insert(table).values(**values)


def _session_storage_key(tenant_id: str, session_id: str) -> str:
    """Namespace externally supplied session IDs without changing the schema."""
    raw = f"{tenant_id}\0{session_id}".encode("utf-8", errors="ignore")
    return hashlib.sha256(raw).hexdigest()


def get_or_create_session(session_id: str, tenant_id: str) -> dict:
    """Return session row, creating if missing (race-safe)."""
    storage_key = _session_storage_key(tenant_id, session_id)
    with begin() as conn:
        row = conn.execute(
            select(sessions)
            .where(sessions.c.session_id == storage_key)
            .where(sessions.c.tenant_id == tenant_id)
        ).first()
        if row:
            out = dict(row._mapping)
            out["session_id"] = session_id
            return out
        conn.execute(_insert_ignore(sessions,
            session_id=storage_key,
            tenant_id=tenant_id,
            request_count=0,
        ))
        row = conn.execute(
            select(sessions)
            .where(sessions.c.session_id == storage_key)
            .where(sessions.c.tenant_id == tenant_id)
        ).first()
        if not row:
            return {"session_id": session_id, "tenant_id": tenant_id, "request_count": 0}
        out = dict(row._mapping)
        out["session_id"] = session_id
        return out


def update_session(
    session_id: str,
    tenant_id: str,
    *,
    tier: str,
    vertical: str,
    endpoint: str,
    response_ok: bool,
    response_ms: float,
):
    storage_key = _session_storage_key(tenant_id, session_id)
    with begin() as conn:
        conn.execute(
            update(sessions)
            .where(sessions.c.session_id == storage_key)
            .where(sessions.c.tenant_id == tenant_id)
            .values(
                last_tier=tier,
                last_vertical=vertical,
                last_endpoint=endpoint,
                last_response_ok=response_ok,
                last_response_ms=response_ms,
                last_used_at=datetime.now(UTC),
                request_count=sessions.c.request_count + 1,
            )
        )


def update_routing_decision(
    decision_id: int,
    *,
    endpoint: str | None = None,
    tier: str | None = None,
    fallback_used: bool | None = None,
):
    """Patch a logged routing decision (e.g. after a fallback succeeded)."""
    values: dict[str, Any] = {}
    if endpoint is not None:
        values["endpoint"] = endpoint
    if tier is not None:
        values["tier"] = tier
    if fallback_used is not None:
        values["fallback_used"] = fallback_used
    if not values:
        return
    with begin() as conn:
        conn.execute(
            update(routing_log)
            .where(routing_log.c.id == decision_id)
            .values(**values)
        )


def get_breaker_state(endpoint_name: str) -> dict:
    with engine().connect() as conn:
        row = conn.execute(
            select(breakers).where(breakers.c.endpoint_name == endpoint_name)
        ).first()
        if row:
            return dict(row._mapping)
        return {"endpoint_name": endpoint_name, "state": "CLOSED", "consecutive_failures": 0}


def set_breaker_state(endpoint_name: str, state: str, **fields):
    with begin() as conn:
        existing = conn.execute(
            select(breakers).where(breakers.c.endpoint_name == endpoint_name)
        ).first()
        values = {
            "endpoint_name": endpoint_name,
            "state": state,
            "consecutive_failures": fields.get("consecutive_failures", 0),
            "opened_at": fields.get("opened_at"),
            "last_failure_at": fields.get("last_failure_at"),
            "last_success_at": fields.get("last_success_at"),
            "half_open_probes_remaining": fields.get("half_open_probes_remaining", 0),
        }
        if existing:
            conn.execute(
                update(breakers)
                .where(breakers.c.endpoint_name == endpoint_name)
                .values(**values)
            )
        else:
            conn.execute(insert(breakers).values(**values))


def enqueue_review(
    decision_id: int,
    tenant_id: str,
    priority: int = 0,
    cost_estimate: float = 0.0,
    prompt_text: str | None = None,
):
    with begin() as conn:
        conn.execute(insert(review_queue).values(
            decision_id=decision_id,
            tenant_id=tenant_id,
            priority=priority,
            cost_usd_estimate=cost_estimate,
            prompt_text=prompt_text,
        ))


def dequeue_review() -> dict | None:
    """Pop the oldest pending review item. Returns row dict or None."""
    with begin() as conn:
        query = (
            select(review_queue)
            .where(review_queue.c.status == "pending")
            .order_by(review_queue.c.priority.desc(), review_queue.c.ts_queued.asc())
            .limit(1)
        )
        if engine().dialect.name == "postgresql":
            query = query.with_for_update(skip_locked=True)
        row = conn.execute(query).first()
        if not row:
            return None
        d = dict(row._mapping)
        conn.execute(
            update(review_queue)
            .where(review_queue.c.id == d["id"])
            .values(
                status="in_progress",
                attempts=review_queue.c.attempts + 1,
                started_at=datetime.now(UTC),
            )
        )
        return d


def complete_review(queue_id: int, status: str = "done", error: str | None = None):
    with begin() as conn:
        conn.execute(
            update(review_queue)
            .where(review_queue.c.id == queue_id)
            .values(status=status, last_error=error, prompt_text=None, started_at=None)
        )


def requeue_stale_reviews(stale_after_seconds: int = 300) -> int:
    """Requeue 'in_progress' review items stuck for too long (crash mid-batch).

    Returns the number of items requeued.
    """
    from datetime import timedelta
    cutoff = datetime.now(UTC) - timedelta(seconds=stale_after_seconds)
    try:
        with begin() as conn:
            result = conn.execute(
                update(review_queue)
                .where(review_queue.c.status == "in_progress")
                .where(review_queue.c.started_at < cutoff)
                .values(status="pending", last_error="requeued_stale", started_at=None)
            )
        return result.rowcount or 0
    except Exception as e:
        log.warning("requeue_stale_reviews failed: %s", e)
        return 0


def store_review_result(
    *,
    decision_id: int,
    reviewer_model: str,
    reviewer_endpoint: str,
    labels: dict,
    truncated: bool,
    router_labels: dict,
    router_confidence: float,
    cost_usd: float,
    raw_response: str,
    prompt_text: str | None = None,
    min_trust_to_curate: float = 0.3,
):
    """Compute per-field agreement + trust + persist.

    labels and router_labels keys: vertical, complexity, code, math, reasoning, long_output

    Auto-curates: when all fields agree AND trust >= min_trust_to_curate, the
    sample is written to curated_samples in the same transaction (curated_run_id="auto").
    This is what actually feeds the trainer — without it the data flywheel is dead.
    Returns the new review_result_id.
    """
    fields = ["vertical", "complexity", "code", "math", "reasoning", "long_output"]
    agreement = {}
    for f in fields:
        if f == "vertical":
            agreement[f"agreement_{f}"] = labels.get(f) == router_labels.get(f)
        else:
            r_lab = labels.get(f)
            if isinstance(r_lab, bool):
                agreement[f"agreement_{f}"] = bool(labels.get(f)) == bool(router_labels.get(f, False))
            else:
                agreement[f"agreement_{f}"] = labels.get(f) == router_labels.get(f)
    all_agree = all(agreement[f"agreement_{f}"] for f in fields)

    # Trust score: agreement rate × router confidence
    agree_count = sum(1 for v in agreement.values() if v)
    trust = (agree_count / len(fields)) * router_confidence

    with begin() as conn:
        result = conn.execute(insert(review_results).values(
            decision_id=decision_id,
            reviewer_model=reviewer_model,
            reviewer_endpoint=reviewer_endpoint,
            vertical_label=labels.get("vertical"),
            complexity_label=labels.get("complexity"),
            flag_code_label=labels.get("code"),
            flag_math_label=labels.get("math"),
            flag_reasoning_label=labels.get("reasoning"),
            flag_long_output_label=labels.get("long_output"),
            truncated=truncated,
            all_fields_agree=all_agree,
            router_confidence_at_decision=router_confidence,
            trust_score=trust,
            raw_response=raw_response[:50000],
            prompt_text=prompt_text,
            cost_usd=cost_usd,
            **agreement,
        ))
        review_result_id = int(result.inserted_primary_key[0])
        conn.execute(
            update(routing_log)
            .where(routing_log.c.id == decision_id)
            .values(review_status="done")
        )

        human_correction = conn.execute(
            select(feedback.c.id)
            .where(feedback.c.decision_id == decision_id)
            .where(feedback.c.correct.is_(False))
            .limit(1)
        ).first() is not None
        curate_trust = max(trust, 0.8) if human_correction else trust
        if ((all_agree and trust >= min_trust_to_curate) or human_correction) and not truncated:
            decision = conn.execute(select(routing_log).where(routing_log.c.id == decision_id)).first()
            if decision:
                _insert_curated_review(
                    conn,
                    decision=dict(decision._mapping),
                    review_result_id=review_result_id,
                    labels=labels,
                    reviewer_model=reviewer_model,
                    trust_score=curate_trust,
                    source="human_reviewed" if human_correction else "flywheel",
                    sample_text=prompt_text,
                )
                log.info("curated decision %d (trust=%.3f) into training pool", decision_id, curate_trust)
    return review_result_id


def _insert_curated_review(
    conn,
    *,
    decision: dict,
    review_result_id: int,
    labels: dict,
    reviewer_model: str,
    trust_score: float,
    source: str,
    sample_text: str | None = None,
):
    conn.execute(insert(curated_samples).values(
        decision_id=decision["id"],
        review_result_id=review_result_id,
        curated_run_id="auto",
        query_hash=decision["query_hash"],
        text=sample_text or decision["query_preview"],
        vertical=labels.get("vertical"),
        complexity=labels.get("complexity"),
        flag_code=bool(labels.get("code", False)),
        flag_math=bool(labels.get("math", False)),
        flag_reasoning=bool(labels.get("reasoning", False)),
        flag_long_output=bool(labels.get("long_output", False)),
        model_version=decision["model_version"],
        reviewer_model=reviewer_model,
        trust_score=trust_score,
        source=source,
    ))
    conn.execute(
        update(review_results)
        .where(review_results.c.id == review_result_id)
        .values(
            curated=True,
            curated_at=datetime.now(UTC),
            curated_run_id="auto",
            trust_score=trust_score,
            prompt_text=None,
        )
    )


def curate_reviewed_correction(decision_id: int) -> bool:
    """Curate a reviewer's correction after negative human feedback."""
    with begin() as conn:
        rr = conn.execute(
            select(review_results).where(review_results.c.decision_id == decision_id)
        ).first()
        if not rr or rr.curated or rr.truncated:
            return False
        decision = conn.execute(select(routing_log).where(routing_log.c.id == decision_id)).first()
        if not decision:
            return False
        labels = {
            "vertical": rr.vertical_label,
            "complexity": rr.complexity_label,
            "code": rr.flag_code_label,
            "math": rr.flag_math_label,
            "reasoning": rr.flag_reasoning_label,
            "long_output": rr.flag_long_output_label,
        }
        if not labels["vertical"] or not isinstance(labels["complexity"], int):
            return False
        _insert_curated_review(
            conn,
            decision=dict(decision._mapping),
            review_result_id=rr.id,
            labels=labels,
            reviewer_model=rr.reviewer_model,
            trust_score=max(float(rr.trust_score or 0.0), 0.8),
            source="human_reviewed",
            sample_text=rr.prompt_text,
        )
        return True


def curate_sample(review_result_id: int, run_id: str, model_version: str):
    """Move a high-agreement review result into curated pool."""
    with begin() as conn:
        rr = conn.execute(
            select(review_results).where(review_results.c.id == review_result_id)
        ).first()
        if not rr:
            return
        rr_d = dict(rr._mapping)
        if not rr_d["all_fields_agree"]:
            return
        decision = conn.execute(
            select(routing_log).where(routing_log.c.id == rr_d["decision_id"])
        ).first()
        if not decision:
            return
        d = dict(decision._mapping)
        conn.execute(insert(curated_samples).values(
            decision_id=rr_d["decision_id"],
            review_result_id=review_result_id,
            curated_run_id=run_id,
            query_hash=d["query_hash"],
            text=d["query_preview"],
            vertical=rr_d["vertical_label"],
            complexity=rr_d["complexity_label"],
            flag_code=rr_d["flag_code_label"],
            flag_math=rr_d["flag_math_label"],
            flag_reasoning=rr_d["flag_reasoning_label"],
            flag_long_output=rr_d["flag_long_output_label"],
            model_version=d["model_version"],
            reviewer_model=rr_d["reviewer_model"],
            trust_score=rr_d["trust_score"],
            source="flywheel",
        ))
        conn.execute(
            update(review_results)
            .where(review_results.c.id == review_result_id)
            .values(curated=True, curated_at=datetime.now(UTC), curated_run_id=run_id)
        )


def get_or_create_user(tenant_id: str, defaults: dict, overwrite: bool = False) -> dict:
    """Return user row, creating if missing.

    overwrite=True applies `defaults` to an existing row (admin edits).
    overwrite=False (lazy tenant creation) never clobbers existing config.
    """
    with begin() as conn:
        row = conn.execute(
            select(users).where(users.c.tenant_id == tenant_id)
        ).first()
        if row:
            if overwrite:
                values = {k: v for k, v in defaults.items() if k != "tenant_id"}
                if "tier_access" in values and isinstance(values["tier_access"], list):
                    values["tier_access"] = json.dumps(values["tier_access"])
                if values:
                    conn.execute(update(users).where(users.c.tenant_id == tenant_id).values(**values))
                row = conn.execute(
                    select(users).where(users.c.tenant_id == tenant_id)
                ).first()
            return _user_row_dict(row)
        values = {"tenant_id": tenant_id, **{k: v for k, v in defaults.items() if k != "tenant_id"}}
        if "tier_access" in values and isinstance(values["tier_access"], list):
            values["tier_access"] = json.dumps(values["tier_access"])
        conn.execute(_insert_ignore(users, **values))
        row = conn.execute(
            select(users).where(users.c.tenant_id == tenant_id)
        ).first()
        if not row:
            return {"tenant_id": tenant_id, **defaults}
        return _user_row_dict(row)


def _user_row_dict(row) -> dict:
    d = dict(row._mapping)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    if d.get("tier_access"):
        try:
            d["tier_access"] = json.loads(d["tier_access"])
        except (json.JSONDecodeError, TypeError):
            pass
    return d


def record_usage(tenant_id: str, tokens_in: int, tokens_out: int, cost_usd: float, endpoint_name: str = "unknown"):
    _insert_usage(tenant_id, tokens_in, tokens_out, cost_usd, request_count=1, endpoint_name=endpoint_name)


def _insert_usage(
    tenant_id: str,
    tokens_in: int,
    tokens_out: int,
    cost_usd: float,
    request_count: int,
    endpoint_name: str = "unknown",
    conn=None,
):
    now = datetime.now(UTC)
    values = dict(
            tenant_id=tenant_id,
            endpoint_name=endpoint_name,
            period_hour=now.strftime("%Y%m%d%H"),
            period_day=now.strftime("%Y%m%d"),
            period_month=now.strftime("%Y%m"),
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost_usd,
            request_count=request_count,
        )
    if conn is not None:
        conn.execute(insert(usage_counters).values(**values))
        return
    with begin() as own_conn:
        own_conn.execute(insert(usage_counters).values(**values))


# --- Plan quotas ---


def upsert_plan(
    plan_id: str,
    *,
    name: str | None = None,
    daily_token_limit: int | None = None,
    daily_usd_limit: float | None = None,
    required_success_probability: float | None = None,
    allowed_models: list[str] | None = None,
) -> dict:
    """Insert or update a plan quota definition."""
    with begin() as conn:
        existing = conn.execute(
            select(plan_quotas).where(plan_quotas.c.plan_id == plan_id)
        ).first()
        values: dict[str, Any] = {}
        if name is not None:
            values["name"] = name
        if daily_token_limit is not None:
            values["daily_token_limit"] = daily_token_limit
        if daily_usd_limit is not None:
            values["daily_usd_limit"] = daily_usd_limit
        if required_success_probability is not None:
            values["required_success_probability"] = required_success_probability
        if allowed_models is not None:
            values["allowed_models_json"] = json.dumps(allowed_models)
        if existing:
            if values:
                conn.execute(
                    update(plan_quotas).where(plan_quotas.c.plan_id == plan_id).values(**values)
                )
        else:
            insert_values = {
                "plan_id": plan_id,
                "name": name or plan_id,
                "daily_token_limit": daily_token_limit or 0,
                "daily_usd_limit": daily_usd_limit or 0.0,
                "required_success_probability": required_success_probability or 0.99,
                "allowed_models_json": json.dumps(allowed_models or []),
            }
            conn.execute(insert(plan_quotas).values(**insert_values))
        row = conn.execute(
            select(plan_quotas).where(plan_quotas.c.plan_id == plan_id)
        ).first()
    return _plan_row_dict(row)


def get_plan(plan_id: str) -> dict | None:
    with engine().connect() as conn:
        row = conn.execute(
            select(plan_quotas).where(plan_quotas.c.plan_id == plan_id)
        ).first()
    return _plan_row_dict(row) if row else None


def list_plans() -> list[dict]:
    with engine().connect() as conn:
        rows = conn.execute(select(plan_quotas).order_by(plan_quotas.c.plan_id)).all()
    return [_plan_row_dict(r) for r in rows]


def _plan_row_dict(row) -> dict:
    d = dict(row._mapping)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    if d.get("allowed_models_json"):
        try:
            d["allowed_models"] = json.loads(d.pop("allowed_models_json"))
        except (json.JSONDecodeError, TypeError):
            d["allowed_models"] = []
    else:
        d.pop("allowed_models_json", None)
        d["allowed_models"] = []
    return d


def assign_tenant_plan(tenant_id: str, plan_id: str, notes: str | None = None) -> dict:
    """Bind a tenant to a plan. Creates the tenant if missing."""
    with begin() as conn:
        existing = conn.execute(
            select(tenant_plans).where(tenant_plans.c.tenant_id == tenant_id)
        ).first()
        if existing:
            conn.execute(
                update(tenant_plans)
                .where(tenant_plans.c.tenant_id == tenant_id)
                .values(plan_id=plan_id, notes=notes)
            )
        else:
            conn.execute(
                insert(tenant_plans).values(
                    tenant_id=tenant_id, plan_id=plan_id, notes=notes
                )
            )
        # Ensure the user row exists so the plan_id can be persisted on it.
        user_existing = conn.execute(
            select(users.c.tenant_id).where(users.c.tenant_id == tenant_id)
        ).first()
        if user_existing:
            conn.execute(
                update(users)
                .where(users.c.tenant_id == tenant_id)
                .values(plan_id=plan_id)
            )
        else:
            conn.execute(
                _insert_ignore(users, tenant_id=tenant_id, plan_id=plan_id)
            )
        row = conn.execute(
            select(tenant_plans).where(tenant_plans.c.tenant_id == tenant_id)
        ).first()
    d = dict(row._mapping) if row else {"tenant_id": tenant_id, "plan_id": plan_id}
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


def get_tenant_plan(tenant_id: str) -> dict | None:
    with engine().connect() as conn:
        row = conn.execute(
            select(tenant_plans).where(tenant_plans.c.tenant_id == tenant_id)
        ).first()
    if not row:
        return None
    d = dict(row._mapping)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


def get_tenant_plan_quota(tenant_id: str) -> dict | None:
    """Combined tenant plan + plan_quota record. None if tenant has no plan."""
    tp = get_tenant_plan(tenant_id)
    if not tp:
        return None
    plan = get_plan(tp["plan_id"])
    if not plan:
        return None
    return {"tenant_id": tenant_id, "plan_id": tp["plan_id"], **{k: v for k, v in plan.items() if k != "plan_id"}}


# --- Per-model token limits ---


def upsert_model_token_limit(
    tenant_id: str,
    endpoint_name: str,
    *,
    daily_token_limit: int | None = None,
    daily_usd_limit: float | None = None,
    max_request_tokens: int | None = None,
) -> dict:
    """Set or update per-model token limits for a tenant."""
    today = datetime.now(UTC).strftime("%Y%m%d")
    with begin() as conn:
        existing = conn.execute(
            select(model_token_limits).where(
                (model_token_limits.c.tenant_id == tenant_id)
                & (model_token_limits.c.endpoint_name == endpoint_name)
            )
        ).first()
        values: dict[str, Any] = {"updated_at": datetime.now(UTC)}
        if daily_token_limit is not None:
            values["daily_token_limit"] = daily_token_limit
        if daily_usd_limit is not None:
            values["daily_usd_limit"] = daily_usd_limit
        if max_request_tokens is not None:
            values["max_request_tokens"] = max_request_tokens
        if existing:
            conn.execute(
                update(model_token_limits)
                .where(
                    (model_token_limits.c.tenant_id == tenant_id)
                    & (model_token_limits.c.endpoint_name == endpoint_name)
                )
                .values(**values)
            )
        else:
            insert_values = {
                "tenant_id": tenant_id,
                "endpoint_name": endpoint_name,
                "period_day": today,
                "tokens_used": 0,
                "cost_used_usd": 0.0,
                "daily_token_limit": daily_token_limit or 0,
                "daily_usd_limit": daily_usd_limit or 0.0,
                "max_request_tokens": max_request_tokens or 0,
            }
            conn.execute(insert(model_token_limits).values(**insert_values))
        row = conn.execute(
            select(model_token_limits).where(
                (model_token_limits.c.tenant_id == tenant_id)
                & (model_token_limits.c.endpoint_name == endpoint_name)
            )
        ).first()
    return _model_limit_row_dict(row) if row else {}


def get_model_token_limit(tenant_id: str, endpoint_name: str) -> dict | None:
    with engine().connect() as conn:
        row = conn.execute(
            select(model_token_limits).where(
                (model_token_limits.c.tenant_id == tenant_id)
                & (model_token_limits.c.endpoint_name == endpoint_name)
            )
        ).first()
    return _model_limit_row_dict(row) if row else None


def list_model_token_limits(tenant_id: str) -> list[dict]:
    with engine().connect() as conn:
        rows = conn.execute(
            select(model_token_limits)
            .where(model_token_limits.c.tenant_id == tenant_id)
            .order_by(model_token_limits.c.endpoint_name)
        ).all()
    return [_model_limit_row_dict(r) for r in rows]


def _model_limit_row_dict(row) -> dict:
    d = dict(row._mapping)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


def get_today_token_spend(tenant_id: str, endpoint_name: str | None = None) -> int:
    """Sum tokens_in + tokens_out for tenant today, optionally filtered by endpoint."""
    today = datetime.now(UTC).strftime("%Y%m%d")
    with engine().connect() as conn:
        q = (
            select(func.coalesce(func.sum(usage_counters.c.tokens_in + usage_counters.c.tokens_out), 0))
            .where(usage_counters.c.tenant_id == tenant_id)
            .where(usage_counters.c.period_day == today)
        )
        if endpoint_name is not None:
            q = q.where(usage_counters.c.endpoint_name == endpoint_name)
        row = conn.execute(q).first()
        return int(row[0]) if row else 0


def get_today_cost_spend(tenant_id: str, endpoint_name: str | None = None) -> float:
    """Sum cost_usd for tenant today, optionally filtered by endpoint."""
    today = datetime.now(UTC).strftime("%Y%m%d")
    with engine().connect() as conn:
        q = (
            select(func.coalesce(func.sum(usage_counters.c.cost_usd), 0.0))
            .where(usage_counters.c.tenant_id == tenant_id)
            .where(usage_counters.c.period_day == today)
        )
        if endpoint_name is not None:
            q = q.where(usage_counters.c.endpoint_name == endpoint_name)
        row = conn.execute(q).first()
        return float(row[0]) if row else 0.0


# --- Model quality profiles ---


def record_quality_sample(
    endpoint_name: str,
    vertical: str,
    complexity: int,
    success: bool,
):
    """Record a single (success/failure) outcome for a model on a vertical/complexity bucket.

    Quality profiles are used by the budget-aware router to estimate the success
    probability of a model on a given request. New combinations start with a
    conservative prior (no data) until enough samples accumulate.
    """
    complexity = max(1, min(5, int(complexity)))
    cx_lo = complexity
    cx_hi = complexity
    with begin() as conn:
        existing = conn.execute(
            select(model_quality_profiles).where(
                (model_quality_profiles.c.endpoint_name == endpoint_name)
                & (model_quality_profiles.c.vertical == vertical)
                & (model_quality_profiles.c.complexity_min == cx_lo)
                & (model_quality_profiles.c.complexity_max == cx_hi)
            )
        ).first()
        if existing:
            conn.execute(
                update(model_quality_profiles)
                .where(
                    (model_quality_profiles.c.endpoint_name == endpoint_name)
                    & (model_quality_profiles.c.vertical == vertical)
                    & (model_quality_profiles.c.complexity_min == cx_lo)
                    & (model_quality_profiles.c.complexity_max == cx_hi)
                )
                .values(
                    success_count=model_quality_profiles.c.success_count + (1 if success else 0),
                    total_count=model_quality_profiles.c.total_count + 1,
                    calibration_samples=model_quality_profiles.c.calibration_samples + 1,
                    last_updated=datetime.now(UTC),
                )
            )
        else:
            conn.execute(
                insert(model_quality_profiles).values(
                    endpoint_name=endpoint_name,
                    vertical=vertical,
                    complexity_min=cx_lo,
                    complexity_max=cx_hi,
                    success_count=1 if success else 0,
                    total_count=1,
                    calibration_samples=1,
                )
            )


def get_quality_profile(
    endpoint_name: str,
    vertical: str,
    complexity: int,
) -> dict | None:
    """Return {success_count, total_count, calibration_samples} for a model/vertical/complexity bucket."""
    complexity = max(1, min(5, int(complexity)))
    with engine().connect() as conn:
        row = conn.execute(
            select(model_quality_profiles).where(
                (model_quality_profiles.c.endpoint_name == endpoint_name)
                & (model_quality_profiles.c.vertical == vertical)
                & (model_quality_profiles.c.complexity_min <= complexity)
                & (model_quality_profiles.c.complexity_max >= complexity)
            )
        ).first()
    if not row:
        return None
    d = dict(row._mapping)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
    return d


def reserve_usage(
    tenant_id: str,
    *,
    budget_limit_usd: float,
    rps_limit: int,
    token_limit_per_minute: int,
    estimated_tokens_in: int,
    estimated_tokens_out: int,
    estimated_cost_usd: float,
    daily_token_limit: int = 0,
    endpoint_name: str = "unknown",
    model_token_limit: int = 0,
    model_usd_limit: float = 0.0,
    max_request_tokens: int = 0,
) -> tuple[bool, str | None]:
    """Atomically check quotas and reserve estimated usage for one request.

    Quotas (in order, all optional / 0 = unlimited):
      - daily_token_limit: tenant-wide daily token cap
      - budget_limit_usd: tenant-wide daily USD cap
      - model_token_limit: per-model daily token cap for this endpoint
      - model_usd_limit: per-model daily USD cap for this endpoint
      - max_request_tokens: per-request token cap (input + estimated output)
      - rps_limit: tenant-wide requests/sec
      - token_limit_per_minute: tenant-wide tokens/minute
    """
    from datetime import timedelta

    now = datetime.now(UTC)
    today = now.strftime("%Y%m%d")
    estimated_tokens = max(0, estimated_tokens_in) + max(0, estimated_tokens_out)

    if max_request_tokens > 0 and estimated_tokens > max_request_tokens:
        return False, f"max_request_tokens {max_request_tokens} exceeded (estimated {estimated_tokens})"

    with begin() as conn:
        user_q = select(users.c.tenant_id).where(users.c.tenant_id == tenant_id)
        if engine().dialect.name != "sqlite":
            user_q = user_q.with_for_update()
        conn.execute(user_q).first()

        if daily_token_limit > 0:
            tenant_tokens_today = conn.execute(
                select(func.coalesce(func.sum(usage_counters.c.tokens_in + usage_counters.c.tokens_out), 0))
                .where(usage_counters.c.tenant_id == tenant_id)
                .where(usage_counters.c.period_day == today)
            ).scalar_one()
            if int(tenant_tokens_today) + estimated_tokens > daily_token_limit:
                return False, (
                    f"daily token limit {daily_token_limit} exceeded "
                    f"(spent {int(tenant_tokens_today)} tokens)"
                )

        if budget_limit_usd > 0:
            spent = conn.execute(
                select(func.coalesce(func.sum(usage_counters.c.cost_usd), 0.0))
                .where(usage_counters.c.tenant_id == tenant_id)
                .where(usage_counters.c.period_day == today)
            ).scalar_one()
            if float(spent) + estimated_cost_usd > budget_limit_usd:
                return False, f"daily budget ${budget_limit_usd:.2f} exceeded (spent ${float(spent):.2f})"

        if model_token_limit > 0:
            model_tokens_today = conn.execute(
                select(func.coalesce(func.sum(usage_counters.c.tokens_in + usage_counters.c.tokens_out), 0))
                .where(usage_counters.c.tenant_id == tenant_id)
                .where(usage_counters.c.endpoint_name == endpoint_name)
                .where(usage_counters.c.period_day == today)
            ).scalar_one()
            if int(model_tokens_today) + estimated_tokens > model_token_limit:
                return False, (
                    f"model {endpoint_name} daily token limit {model_token_limit} exceeded "
                    f"(spent {int(model_tokens_today)} tokens)"
                )

        if model_usd_limit > 0:
            model_cost_today = conn.execute(
                select(func.coalesce(func.sum(usage_counters.c.cost_usd), 0.0))
                .where(usage_counters.c.tenant_id == tenant_id)
                .where(usage_counters.c.endpoint_name == endpoint_name)
                .where(usage_counters.c.period_day == today)
            ).scalar_one()
            if float(model_cost_today) + estimated_cost_usd > model_usd_limit:
                return False, (
                    f"model {endpoint_name} daily USD limit ${model_usd_limit:.2f} exceeded "
                    f"(spent ${float(model_cost_today):.2f})"
                )

        if rps_limit > 0:
            request_count = conn.execute(
                select(func.coalesce(func.sum(usage_counters.c.request_count), 0))
                .where(usage_counters.c.tenant_id == tenant_id)
                .where(usage_counters.c.ts >= now - timedelta(seconds=1))
            ).scalar_one()
            if int(request_count) >= rps_limit:
                return False, "requests-per-second limit exceeded"
        if token_limit_per_minute > 0:
            token_count = conn.execute(
                select(func.coalesce(func.sum(usage_counters.c.tokens_in + usage_counters.c.tokens_out), 0))
                .where(usage_counters.c.tenant_id == tenant_id)
                .where(usage_counters.c.ts >= now - timedelta(minutes=1))
            ).scalar_one()
            if int(token_count) + estimated_tokens > token_limit_per_minute:
                return False, "tokens-per-minute limit exceeded"

        _insert_usage(
            tenant_id,
            max(0, estimated_tokens_in),
            max(0, estimated_tokens_out),
            max(0.0, estimated_cost_usd),
            request_count=1,
            endpoint_name=endpoint_name,
            conn=conn,
        )
    return True, None


def settle_reserved_usage(
    tenant_id: str,
    *,
    reserved_tokens_in: int,
    reserved_tokens_out: int,
    reserved_cost_usd: float,
    actual_tokens_in: int,
    actual_tokens_out: int,
    actual_cost_usd: float,
    completed: bool,
    endpoint_name: str = "unknown",
):
    """Settle a reservation; failed requests release all reserved usage."""
    if completed:
        token_in_delta = max(0, actual_tokens_in) - max(0, reserved_tokens_in)
        token_out_delta = max(0, actual_tokens_out) - max(0, reserved_tokens_out)
        cost_delta = max(0.0, actual_cost_usd) - max(0.0, reserved_cost_usd)
        request_count = 0
    else:
        token_in_delta = -max(0, reserved_tokens_in)
        token_out_delta = -max(0, reserved_tokens_out)
        cost_delta = -max(0.0, reserved_cost_usd)
        request_count = -1
    _insert_usage(
        tenant_id,
        token_in_delta,
        token_out_delta,
        cost_delta,
        request_count=request_count,
        endpoint_name=endpoint_name,
    )


def get_today_spend(tenant_id: str) -> float:
    today = datetime.now(UTC).strftime("%Y%m%d")
    with engine().connect() as conn:
        row = conn.execute(
            select(func.coalesce(func.sum(usage_counters.c.cost_usd), 0.0))
            .where(usage_counters.c.tenant_id == tenant_id)
            .where(usage_counters.c.period_day == today)
        ).first()
        return float(row[0]) if row else 0.0


def get_decisions(
    limit: int = 100,
    session_id: str | None = None,
    vertical: str | None = None,
    tenant_id: str | None = None,
    since_hours: float | None = None,
) -> list[dict]:
    limit = max(1, min(int(limit), 10000))
    with engine().connect() as conn:
        q = select(routing_log).order_by(routing_log.c.id.desc()).limit(limit)
        if session_id:
            q = q.where(routing_log.c.session_id == session_id)
        if vertical:
            q = q.where(routing_log.c.vertical == vertical)
        if tenant_id:
            q = q.where(routing_log.c.tenant_id == tenant_id)
        if since_hours:
            from datetime import timedelta
            cutoff = datetime.now(UTC) - timedelta(hours=since_hours)
            q = q.where(routing_log.c.ts >= cutoff)
        rows = conn.execute(q).all()
        out = []
        for r in rows:
            d = dict(r._mapping)
            for k, v in d.items():
                if isinstance(v, datetime):
                    d[k] = v.isoformat()
            out.append(d)
        return out


def decision_tenant(decision_id: int) -> str | None:
    with engine().connect() as conn:
        return conn.execute(
            select(routing_log.c.tenant_id).where(routing_log.c.id == decision_id)
        ).scalar_one_or_none()


def accuracy_report(since_hours: float | None = None, tenant_id: str | None = None) -> dict:
    """Compute first-pass accuracy from feedback + fallback detection."""
    decisions = get_decisions(limit=10000, since_hours=since_hours, tenant_id=tenant_id)
    fb_by_dec = {}
    if decisions:
        decision_ids = [d["id"] for d in decisions]
        with engine().connect() as conn:
            fb_rows = conn.execute(
                select(feedback)
                .where(feedback.c.decision_id.in_(decision_ids))
            ).all()
        fb_by_dec = {f.decision_id: f for f in fb_rows}
    correct = 0
    wrong = 0
    per_vertical: dict[str, dict[str, int]] = {}
    for d in decisions:
        v = d["vertical"] or "unknown"
        per_vertical.setdefault(v, {"correct": 0, "wrong": 0})
        fb = fb_by_dec.get(d["id"])
        if fb is None:
            continue
        if fb.correct:
            correct += 1
            per_vertical[v]["correct"] += 1
        else:
            wrong += 1
            per_vertical[v]["wrong"] += 1
    total = correct + wrong
    return {
        "first_pass_accuracy": correct / total if total else None,
        "feedback_count": total,
        "per_vertical": {
            v: {
                "correct": s["correct"],
                "wrong": s["wrong"],
                "accuracy": s["correct"] / (s["correct"] + s["wrong"]) if (s["correct"] + s["wrong"]) else None,
            }
            for v, s in per_vertical.items()
        },
    }


def record_live_eval(
    *,
    decision_id: int,
    query_hash: str,
    text: str,
    ground_truth_vertical: str,
    ground_truth_complexity: int,
    ground_truth_flags: dict,
    label_source: str = "reviewer",
) -> bool:
    """Persist a labeled live-traffic sample for the live eval set.

    Idempotent per decision_id. Returns True if inserted (new sample).
    """
    try:
        with begin() as conn:
            exists = conn.execute(
                select(live_eval_set).where(live_eval_set.c.decision_id == decision_id)
            ).first()
            if exists:
                return False
            conn.execute(_insert_ignore(live_eval_set,
                decision_id=decision_id,
                query_hash=query_hash,
                text=text,
                ground_truth_vertical=ground_truth_vertical,
                ground_truth_complexity=ground_truth_complexity,
                ground_truth_flags=json.dumps(ground_truth_flags),
                label_source=label_source,
            ))
            return True
    except Exception as e:
        log.warning("record_live_eval failed: %s", e)
        return False


def live_eval_samples(limit: int = 500) -> list[dict]:
    """Most recent live-eval samples, most recent first."""
    with engine().connect() as conn:
        rows = conn.execute(
            select(live_eval_set).order_by(live_eval_set.c.id.desc()).limit(limit)
        ).all()
        out = []
        for r in rows:
            d = dict(r._mapping)
            for k, v in d.items():
                if isinstance(v, datetime):
                    d[k] = v.isoformat()
            if d.get("ground_truth_flags"):
                try:
                    d["ground_truth_flags"] = json.loads(d["ground_truth_flags"])
                except (json.JSONDecodeError, TypeError):
                    pass
            out.append(d)
        return out


def register_model_version(
    version_id: str, parent_id: str | None, embedding_model: str, heads_hash: str
) -> None:
    with begin() as conn:
        conn.execute(update(model_versions).values(active=False))
        existing = conn.execute(
            select(model_versions.c.id).where(model_versions.c.id == version_id)
        ).first()
        if existing:
            values = {"embedding_model": embedding_model, "heads_hash": heads_hash, "active": True}
            if parent_id is not None:
                values["parent_id"] = parent_id
            conn.execute(
                update(model_versions).where(model_versions.c.id == version_id).values(**values)
            )
        else:
            conn.execute(insert(model_versions).values(
                id=version_id,
                parent_id=parent_id,
                embedding_model=embedding_model,
                heads_hash=heads_hash,
                active=True,
            ))


def record_checkpoint(checkpoint_id: str, model_version_id: str) -> None:
    """Insert a checkpoint row so /registry has history. Idempotent per id."""
    with begin() as conn:
        conn.execute(_insert_ignore(checkpoints,
            id=checkpoint_id,
            model_version_id=model_version_id,
            promoted=False,
        ))


def active_model_version() -> str | None:
    with engine().connect() as conn:
        row = conn.execute(
            select(model_versions).where(model_versions.c.active.is_(True)).limit(1)
        ).first()
        return row.id if row else None


def model_version(version_id: str) -> dict | None:
    with engine().connect() as conn:
        row = conn.execute(
            select(model_versions).where(model_versions.c.id == version_id)
        ).first()
    return dict(row._mapping) if row else None


def checkpoint_history(limit: int = 20) -> list[dict]:
    with engine().connect() as conn:
        rows = conn.execute(
            select(checkpoints).order_by(checkpoints.c.created_at.desc()).limit(limit)
        ).all()
        out = []
        for row in rows:
            item = dict(row._mapping)
            for key, value in item.items():
                if isinstance(value, datetime):
                    item[key] = value.isoformat()
            out.append(item)
        return out


def mark_checkpoint_promoted(checkpoint_id: str, eval_results: dict):
    with begin() as conn:
        conn.execute(
            update(checkpoints)
            .where(checkpoints.c.id == checkpoint_id)
            .values(
                promoted=True,
                eval_base_accuracy=eval_results.get("base_accuracy"),
                eval_live_accuracy=eval_results.get("live_accuracy"),
                per_vertical_accuracy=json.dumps(eval_results.get("per_vertical_accuracy", {})),
                confusion_top20=json.dumps(eval_results.get("confusion_top20", [])),
                policy_replay_drift_pct=eval_results.get("policy_drift_pct"),
            )
        )


def get_trainer_state(key: str, default: str | None = None) -> str | None:
    with engine().connect() as conn:
        value = conn.execute(
            select(trainer_state.c.value).where(trainer_state.c.key == key)
        ).scalar_one_or_none()
    return value if value is not None else default


def set_trainer_state(key: str, value: str) -> None:
    with begin() as conn:
        existing = conn.execute(
            select(trainer_state.c.key).where(trainer_state.c.key == key)
        ).first()
        if existing:
            conn.execute(update(trainer_state).where(trainer_state.c.key == key).values(value=value))
        else:
            conn.execute(insert(trainer_state).values(key=key, value=value))


def curated_count_after(sample_id: int) -> tuple[int, int]:
    """Return (count, maximum id) for curated samples newer than sample_id."""
    with engine().connect() as conn:
        row = conn.execute(
            select(func.count(curated_samples.c.id), func.coalesce(func.max(curated_samples.c.id), sample_id))
            .where(curated_samples.c.id > sample_id)
        ).first()
    return (int(row[0]), int(row[1])) if row else (0, sample_id)


def latest_promoted_checkpoint() -> dict | None:
    with engine().connect() as conn:
        row = conn.execute(
            select(checkpoints)
            .where(checkpoints.c.promoted.is_(True))
            .where(checkpoints.c.rolled_back_at.is_(None))
            .order_by(checkpoints.c.created_at.desc())
            .limit(1)
        ).first()
    return dict(row._mapping) if row else None


def mark_checkpoint_rolled_back(checkpoint_id: str, reason: str):
    with begin() as conn:
        conn.execute(
            update(checkpoints)
            .where(checkpoints.c.id == checkpoint_id)
            .values(rolled_back_at=datetime.now(UTC), rolled_back_reason=reason)
        )


def flag_input(
    tenant_id: str, decision_id: int | None, reason: str,
    matched_regex: str, query_preview: str, action_taken: str,
    severity: str = "medium",
    matched_profile: str | None = None,
    security_event_id: int | None = None,
) -> int:
    with begin() as conn:
        result = conn.execute(insert(flagged_inputs).values(
            tenant_id=tenant_id,
            decision_id=decision_id,
            reason=reason,
            severity=severity,
            matched_profile=matched_profile,
            matched_regex=matched_regex,
            query_preview=query_preview[:1000],
            action_taken=action_taken,
            security_event_id=security_event_id,
        ))
        return int(result.inserted_primary_key[0])


def list_flagged(limit: int = 100, reason: str | None = None) -> list[dict]:
    with engine().connect() as conn:
        q = select(flagged_inputs).order_by(flagged_inputs.c.id.desc()).limit(limit)
        if reason:
            q = q.where(flagged_inputs.c.reason == reason)
        rows = conn.execute(q).all()
        out = []
        for row in rows:
            item = dict(row._mapping)
            for key, value in item.items():
                if isinstance(value, datetime):
                    item[key] = value.isoformat()
            out.append(item)
        return out


# ---------------------------------------------------------------------------
# Security events (unified audit log)
# ---------------------------------------------------------------------------


VALID_SECURITY_EVENT_TYPES = {
    "injection_blocked",
    "injection_alerted",
    "provider_blocked",
    "provider_alerted",
    "rate_violation",
    "budget_exhausted",
    "semantic_dos",
    "pii_detected",
    "firewall_sync",
}

VALID_SECURITY_SEVERITIES = {"low", "medium", "high", "critical"}


def record_security_event(
    tenant_id: str,
    event_type: str,
    severity: str,
    reason: str,
    matched_pattern: str | None = None,
    query_preview: str | None = None,
    endpoint_target: str | None = None,
    action_taken: str | None = None,
    request_metadata: dict | None = None,
) -> int:
    """Write a security event. Returns the new id.

    Validates event_type and severity are in the known sets; unknown values
    are stored verbatim but logged so we can spot typos. Returns 0 on insert
    failure rather than raising (callers are in the hot path).
    """
    if event_type not in VALID_SECURITY_EVENT_TYPES:
        log.warning("record_security_event: unknown event_type=%s", event_type)
    if severity not in VALID_SECURITY_SEVERITIES:
        log.warning("record_security_event: unknown severity=%s", severity)
    metadata_json = json.dumps(request_metadata) if request_metadata else None
    try:
        with begin() as conn:
            result = conn.execute(insert(security_events).values(
                tenant_id=tenant_id,
                event_type=event_type,
                severity=severity,
                reason=reason[:1000],
                matched_pattern=matched_pattern[:256] if matched_pattern else None,
                query_preview=query_preview[:1000] if query_preview else None,
                endpoint_target=endpoint_target[:256] if endpoint_target else None,
                action_taken=action_taken[:32] if action_taken else None,
                request_metadata_json=metadata_json,
            ))
            return int(result.inserted_primary_key[0])
    except Exception as e:
        log.warning("record_security_event failed: %s", e)
        return 0


def list_security_events(
    tenant_id: str | None = None,
    event_type: str | None = None,
    severity: str | None = None,
    since: datetime | None = None,
    limit: int = 100,
) -> list[dict]:
    with engine().connect() as conn:
        q = select(security_events).order_by(security_events.c.id.desc()).limit(limit)
        if tenant_id:
            q = q.where(security_events.c.tenant_id == tenant_id)
        if event_type:
            q = q.where(security_events.c.event_type == event_type)
        if severity:
            q = q.where(security_events.c.severity == severity)
        if since:
            q = q.where(security_events.c.ts >= since)
        rows = conn.execute(q).all()
        out = []
        for row in rows:
            item = dict(row._mapping)
            for key, value in item.items():
                if isinstance(value, datetime):
                    item[key] = value.isoformat()
            out.append(item)
        return out


def security_event_stats(
    tenant_id: str | None = None,
    window_days: int = 7,
) -> dict:
    """Return aggregated security stats: counts by type/severity, daily timeline, top patterns."""
    from datetime import timedelta
    cutoff = datetime.now(UTC) - timedelta(days=max(1, window_days))
    with engine().connect() as conn:
        base = select(security_events).where(security_events.c.ts >= cutoff)
        if tenant_id:
            base = base.where(security_events.c.tenant_id == tenant_id)

        # Count by type
        by_type_q = select(
            security_events.c.event_type,
            func.count(security_events.c.id).label("cnt"),
        ).where(security_events.c.ts >= cutoff)
        if tenant_id:
            by_type_q = by_type_q.where(security_events.c.tenant_id == tenant_id)
        by_type_q = by_type_q.group_by(security_events.c.event_type)
        by_type = {
            row.event_type: int(row.cnt)
            for row in conn.execute(by_type_q).all()
        }

        # Count by severity
        by_severity_q = select(
            security_events.c.severity,
            func.count(security_events.c.id).label("cnt"),
        ).where(security_events.c.ts >= cutoff)
        if tenant_id:
            by_severity_q = by_severity_q.where(security_events.c.tenant_id == tenant_id)
        by_severity_q = by_severity_q.group_by(security_events.c.severity)
        by_severity = {
            row.severity: int(row.cnt)
            for row in conn.execute(by_severity_q).all()
        }

        # Daily timeline (last N days)
        timeline_q = select(
            func.substr(security_events.c.ts.cast(type_=String), 1, 10).label("day"),
            security_events.c.event_type,
            func.count(security_events.c.id).label("cnt"),
        ).where(security_events.c.ts >= cutoff)
        if tenant_id:
            timeline_q = timeline_q.where(security_events.c.tenant_id == tenant_id)
        timeline_q = timeline_q.group_by("day", security_events.c.event_type)
        timeline_rows = conn.execute(timeline_q).all()
        timeline: dict = {}
        for row in timeline_rows:
            day = str(row.day)
            timeline.setdefault(day, {})[row.event_type] = int(row.cnt)

        # Top patterns (limit 10)
        patterns_q = (
            select(
                security_events.c.matched_pattern,
                func.count(security_events.c.id).label("cnt"),
            )
            .where(security_events.c.ts >= cutoff)
            .where(security_events.c.matched_pattern.is_not(None))
        )
        if tenant_id:
            patterns_q = patterns_q.where(security_events.c.tenant_id == tenant_id)
        patterns_q = (
            patterns_q.group_by(security_events.c.matched_pattern)
            .order_by(func.count(security_events.c.id).desc())
            .limit(10)
        )
        top_patterns = [
            {"pattern": str(row.matched_pattern or ""), "count": int(row.cnt)}
            for row in conn.execute(patterns_q).all()
        ]

        total_q = select(func.count(security_events.c.id)).where(
            security_events.c.ts >= cutoff
        )
        if tenant_id:
            total_q = total_q.where(security_events.c.tenant_id == tenant_id)
        total = int(conn.execute(total_q).scalar() or 0)

        return {
            "total": total,
            "by_type": by_type,
            "by_severity": by_severity,
            "timeline": timeline,
            "top_patterns": top_patterns,
            "window_days": window_days,
        }


# ---------------------------------------------------------------------------
# Provider allowlist
# ---------------------------------------------------------------------------


def upsert_provider_allowlist(
    tenant_id: str,
    domain_pattern: str,
    action: str = "allow",
    notes: str | None = None,
) -> None:
    """Insert or update a domain rule for a tenant. tenant_id='*' is global."""
    if action not in ("allow", "block"):
        raise ValueError(f"action must be 'allow' or 'block', got {action!r}")
    with begin() as conn:
        conn.execute(_insert_ignore(provider_allowlist,
            tenant_id=tenant_id,
            domain_pattern=domain_pattern,
            action=action,
            notes=notes,
        ))
        conn.execute(
            update(provider_allowlist)
            .where(provider_allowlist.c.tenant_id == tenant_id)
            .where(provider_allowlist.c.domain_pattern == domain_pattern)
            .values(action=action, notes=notes)
        )


def list_provider_allowlist(tenant_id: str | None = None) -> list[dict]:
    with engine().connect() as conn:
        q = select(provider_allowlist).order_by(
            provider_allowlist.c.tenant_id,
            provider_allowlist.c.domain_pattern,
        )
        if tenant_id:
            q = q.where(provider_allowlist.c.tenant_id == tenant_id)
        rows = conn.execute(q).all()
        out = []
        for row in rows:
            item = dict(row._mapping)
            for key, value in item.items():
                if isinstance(value, datetime):
                    item[key] = value.isoformat()
            out.append(item)
        return out


def delete_provider_allowlist(tenant_id: str, domain_pattern: str) -> bool:
    with begin() as conn:
        result = conn.execute(
            delete(provider_allowlist)
            .where(provider_allowlist.c.tenant_id == tenant_id)
            .where(provider_allowlist.c.domain_pattern == domain_pattern)
        )
        return (result.rowcount or 0) > 0


# ---------------------------------------------------------------------------
# Injection profiles (DB-backed, overrides config defaults)
# ---------------------------------------------------------------------------


def list_injection_profiles(enabled_only: bool = False) -> list[dict]:
    with engine().connect() as conn:
        q = select(injection_profiles).order_by(injection_profiles.c.id)
        if enabled_only:
            q = q.where(injection_profiles.c.enabled == True)  # noqa: E712
        rows = conn.execute(q).all()
        out = []
        for row in rows:
            item = dict(row._mapping)
            for key, value in item.items():
                if isinstance(value, datetime):
                    item[key] = value.isoformat()
            # Parse regexes_json for convenience
            if item.get("regexes_json"):
                try:
                    item["regexes"] = json.loads(item["regexes_json"])
                except (json.JSONDecodeError, TypeError):
                    item["regexes"] = []
            else:
                item["regexes"] = []
            out.append(item)
        return out


def get_injection_profile(profile_id: int) -> dict | None:
    with engine().connect() as conn:
        row = conn.execute(
            select(injection_profiles).where(injection_profiles.c.id == profile_id)
        ).first()
        if not row:
            return None
        item = dict(row._mapping)
        for key, value in item.items():
            if isinstance(value, datetime):
                item[key] = value.isoformat()
        if item.get("regexes_json"):
            try:
                item["regexes"] = json.loads(item["regexes_json"])
            except (json.JSONDecodeError, TypeError):
                item["regexes"] = []
        else:
            item["regexes"] = []
        return item


def create_injection_profile(
    name: str,
    regexes: list[str],
    severity: str = "medium",
    action: str = "alert",
    enabled: bool = True,
    is_builtin: bool = False,
) -> int:
    if severity not in VALID_SECURITY_SEVERITIES:
        raise ValueError(f"invalid severity: {severity}")
    if action not in ("block", "alert", "log"):
        raise ValueError(f"action must be block|alert|log, got {action!r}")
    with begin() as conn:
        result = conn.execute(insert(injection_profiles).values(
            name=name,
            regexes_json=json.dumps(regexes),
            severity=severity,
            action=action,
            enabled=enabled,
            is_builtin=is_builtin,
        ))
        return int(result.inserted_primary_key[0])


def update_injection_profile(
    profile_id: int,
    name: str | None = None,
    regexes: list[str] | None = None,
    severity: str | None = None,
    action: str | None = None,
    enabled: bool | None = None,
) -> bool:
    values: dict = {"updated_at": datetime.now(UTC)}
    if name is not None:
        values["name"] = name
    if regexes is not None:
        values["regexes_json"] = json.dumps(regexes)
    if severity is not None:
        if severity not in VALID_SECURITY_SEVERITIES:
            raise ValueError(f"invalid severity: {severity}")
        values["severity"] = severity
    if action is not None:
        if action not in ("block", "alert", "log"):
            raise ValueError(f"action must be block|alert|log, got {action!r}")
        values["action"] = action
    if enabled is not None:
        values["enabled"] = enabled
    with begin() as conn:
        result = conn.execute(
            update(injection_profiles).where(injection_profiles.c.id == profile_id).values(**values)
        )
        return (result.rowcount or 0) > 0


def delete_injection_profile(profile_id: int) -> bool:
    """Delete a profile. Refuses to delete built-in profiles (is_builtin=True)."""
    with begin() as conn:
        row = conn.execute(
            select(injection_profiles).where(injection_profiles.c.id == profile_id)
        ).first()
        if not row:
            return False
        if row.is_builtin:
            raise ValueError("cannot delete built-in injection profile")
        result = conn.execute(
            delete(injection_profiles).where(injection_profiles.c.id == profile_id)
        )
        return (result.rowcount or 0) > 0


def seed_default_injection_profiles(defaults: list[dict]) -> int:
    """Seed builtin profiles. Idempotent (skips if name already exists).

    Each entry in `defaults` is a dict with keys: name, regexes (list[str]),
    severity, action. Returns the number of profiles inserted.
    """
    inserted = 0
    with begin() as conn:
        for entry in defaults:
            existing = conn.execute(
                select(injection_profiles).where(injection_profiles.c.name == entry["name"])
            ).first()
            if existing:
                continue
            conn.execute(insert(injection_profiles).values(
                name=entry["name"],
                regexes_json=json.dumps(entry.get("regexes", [])),
                severity=entry.get("severity", "medium"),
                action=entry.get("action", "alert"),
                enabled=entry.get("enabled", True),
                is_builtin=True,
            ))
            inserted += 1
    return inserted


def review_stats(tenant_id: str | None = None) -> dict:
    with engine().connect() as conn:
        tenant_filter = review_queue.c.tenant_id == tenant_id if tenant_id else None

        def count_for(status: str | None = None):
            q = select(func.count(review_queue.c.id))
            if status:
                q = q.where(review_queue.c.status == status)
            if tenant_filter is not None:
                q = q.where(tenant_filter)
            return conn.execute(q).scalar()

        total = count_for()
        pending = count_for("pending")
        done = count_for("done")
        failed = count_for("failed")
        cost_q = select(func.coalesce(func.sum(review_queue.c.cost_usd_estimate), 0.0))
        if tenant_filter is not None:
            cost_q = cost_q.where(tenant_filter)
        total_cost = conn.execute(cost_q).scalar()
    return {
        "total": total,
        "pending": pending,
        "done": done,
        "failed": failed,
        "est_total_cost_usd": float(total_cost or 0.0),
    }


def cost_breakdown(since_hours: float | None = None, tenant_id: str | None = None) -> dict:
    decisions = get_decisions(limit=10000, since_hours=since_hours, tenant_id=tenant_id)
    by_tier: dict[str, dict[str, float | int]] = {}
    for d in decisions:
        t = d["tier"]
        by_tier.setdefault(t, {"count": 0, "total_cost": 0.0})
        by_tier[t]["count"] += 1
        by_tier[t]["total_cost"] += d.get("actual_cost_usd") or d.get("est_cost_usd") or 0.0
    return {
        "by_tier": by_tier,
        "total_count": sum(s["count"] for s in by_tier.values()),
        "total_cost": sum(s["total_cost"] for s in by_tier.values()),
    }


def vertical_distribution(since_hours: float | None = None, tenant_id: str | None = None) -> dict:
    decisions = get_decisions(limit=10000, since_hours=since_hours, tenant_id=tenant_id)
    dist: dict[str, int] = {}
    for d in decisions:
        v = d["vertical"] or "unknown"
        dist[v] = dist.get(v, 0) + 1
    return dist


def session_stats(tenant_id: str | None = None) -> dict:
    from datetime import timedelta
    cutoff = datetime.now(UTC) - timedelta(hours=1)
    with engine().connect() as conn:
        total_q = select(func.count(sessions.c.session_id))
        active_q = select(func.count(sessions.c.session_id)).where(sessions.c.last_used_at > cutoff)
        if tenant_id:
            total_q = total_q.where(sessions.c.tenant_id == tenant_id)
            active_q = active_q.where(sessions.c.tenant_id == tenant_id)
        total = conn.execute(total_q).scalar()
        active = conn.execute(active_q).scalar()
    return {"total_sessions": total, "active_last_hour": active}


def purge_old_traces(days: int) -> int:
    """Delete routing_log + feedback rows older than `days`. Returns rows purged."""
    if days <= 0:
        return 0
    from datetime import timedelta
    cutoff = datetime.now(UTC) - timedelta(days=days)
    try:
        with engine().connect() as conn:
            old_ids = conn.execute(
                select(routing_log.c.id).where(routing_log.c.ts < cutoff)
            ).all()
            ids = [r[0] for r in old_ids]
        if not ids:
            return 0
        with begin() as conn:
            review_ids = [
                row[0] for row in conn.execute(
                    select(review_results.c.id).where(review_results.c.decision_id.in_(ids))
                ).all()
            ]
            conn.execute(
                update(curated_samples)
                .where(curated_samples.c.decision_id.in_(ids))
                .values(decision_id=None, review_result_id=None)
            )
            conn.execute(delete(live_eval_set).where(live_eval_set.c.decision_id.in_(ids)))
            conn.execute(delete(review_queue).where(review_queue.c.decision_id.in_(ids)))
            if review_ids:
                conn.execute(delete(review_results).where(review_results.c.id.in_(review_ids)))
            conn.execute(delete(feedback).where(feedback.c.decision_id.in_(ids)))
            conn.execute(delete(routing_log).where(routing_log.c.id.in_(ids)))
        return len(ids)
    except Exception as e:
        log.warning("purge_old_traces failed: %s", e)
        return 0


def purge_old_flags(days: int) -> int:
    """Delete flagged_inputs AND security_events older than `days` (config logging.flagged_retention_days)."""
    if days <= 0:
        return 0
    from datetime import timedelta
    cutoff = datetime.now(UTC) - timedelta(days=days)
    try:
        with begin() as conn:
            result = conn.execute(
                delete(flagged_inputs).where(flagged_inputs.c.ts < cutoff)
            )
            events_result = conn.execute(
                delete(security_events).where(security_events.c.ts < cutoff)
            )
        return (result.rowcount or 0) + (events_result.rowcount or 0)
    except Exception as e:
        log.warning("purge_old_flags failed: %s", e)
        return 0


# =====================================================================
# Plugin system — CRUD for gateway plugins (manifest-based connectors)
# =====================================================================


def upsert_plugin(
    name: str,
    version: str | None = None,
    description: str | None = None,
    prefix: str | None = None,
    module_path: str | None = None,
    config: dict | None = None,
    enabled: bool = True,
    is_builtin: bool = False,
) -> dict:
    """Insert or update a plugin record. Returns the resulting row as a dict."""
    config_json = json.dumps(config or {})
    now = datetime.now(UTC)
    try:
        with begin() as conn:
            existing = conn.execute(
                select(plugins).where(plugins.c.name == name)
            ).mappings().first()
            if existing:
                conn.execute(
                    update(plugins)
                    .where(plugins.c.name == name)
                    .values(
                        version=version or existing["version"],
                        description=description or existing["description"],
                        prefix=prefix or existing["prefix"],
                        module_path=module_path or existing["module_path"],
                        config_json=config_json,
                        enabled=enabled,
                        is_builtin=is_builtin or existing["is_builtin"],
                        updated_at=now,
                    )
                )
            else:
                conn.execute(
                    insert(plugins).values(
                        name=name,
                        version=version or "0.0.0",
                        description=description or "",
                        prefix=prefix or "",
                        module_path=module_path or "",
                        config_json=config_json,
                        enabled=enabled,
                        loaded=False,
                        is_builtin=is_builtin,
                        created_at=now,
                        updated_at=now,
                    )
                )
        row = get_plugin(name)
        return row or {}
    except Exception as e:
        log.warning("upsert_plugin failed: %s", e)
        return {}


def get_plugin(name: str) -> dict | None:
    try:
        with begin() as conn:
            row = conn.execute(
                select(plugins).where(plugins.c.name == name)
            ).mappings().first()
            return dict(row) if row else None
    except Exception as e:
        log.warning("get_plugin failed: %s", e)
        return None


def list_plugins(enabled_only: bool = False) -> list[dict]:
    try:
        with begin() as conn:
            stmt = select(plugins)
            if enabled_only:
                stmt = stmt.where(plugins.c.enabled.is_(True))
            stmt = stmt.order_by(plugins.c.name.asc())
            return [dict(r) for r in conn.execute(stmt).mappings().all()]
    except Exception as e:
        log.warning("list_plugins failed: %s", e)
        return []


def set_plugin_loaded(
    name: str,
    loaded: bool,
    loaded_at: datetime | None = None,
    error: str | None = None,
) -> bool:
    try:
        with begin() as conn:
            result = conn.execute(
                update(plugins)
                .where(plugins.c.name == name)
                .values(
                    loaded=loaded,
                    loaded_at=loaded_at,
                    error=error,
                    updated_at=datetime.now(UTC),
                )
            )
        return bool(result.rowcount and result.rowcount > 0)
    except Exception as e:
        log.warning("set_plugin_loaded failed: %s", e)
        return False


def set_plugin_enabled(name: str, enabled: bool) -> bool:
    try:
        with begin() as conn:
            result = conn.execute(
                update(plugins)
                .where(plugins.c.name == name)
                .values(enabled=enabled, updated_at=datetime.now(UTC))
            )
        return bool(result.rowcount and result.rowcount > 0)
    except Exception as e:
        log.warning("set_plugin_enabled failed: %s", e)
        return False


def delete_plugin(name: str) -> bool:
    """Refuse to delete builtin plugins."""
    try:
        with begin() as conn:
            row = conn.execute(
                select(plugins.c.is_builtin).where(plugins.c.name == name)
            ).first()
            if row and row[0]:
                return False
            conn.execute(delete(plugins).where(plugins.c.name == name))
            return True
    except Exception as e:
        log.warning("delete_plugin failed: %s", e)
        return False


# =====================================================================
# A2A agent registry — CRUD for external A2A agents
# =====================================================================


VALID_A2A_AGENT_TYPES = {"jsonrpc", "openai", "anthropic", "custom"}
VALID_A2A_AUTH_TYPES = {"none", "api_key", "bearer", "oauth"}


def upsert_a2a_agent(
    name: str,
    endpoint_url: str,
    agent_type: str,
    description: str = "",
    auth_type: str = "none",
    auth_value: str | None = None,
    protocol_version: str | None = None,
    capabilities: dict | None = None,
    config: dict | None = None,
    tags: list[str] | None = None,
    enabled: bool = True,
) -> dict:
    """Insert or update an A2A agent. auth_value is stored as-is (callers
    should encrypt out-of-band if needed — gateway reads plaintext).
    """
    if agent_type not in VALID_A2A_AGENT_TYPES:
        return {"error": f"invalid agent_type: {agent_type}"}
    if auth_type not in VALID_A2A_AUTH_TYPES:
        return {"error": f"invalid auth_type: {auth_type}"}
    now = datetime.now(UTC)
    capabilities_json = json.dumps(capabilities or {})
    config_json = json.dumps(config or {})
    tags_json = json.dumps(tags or [])
    try:
        with begin() as conn:
            existing = conn.execute(
                select(a2a_agents).where(a2a_agents.c.name == name)
            ).mappings().first()
            if existing:
                conn.execute(
                    update(a2a_agents)
                    .where(a2a_agents.c.name == name)
                    .values(
                        endpoint_url=endpoint_url,
                        agent_type=agent_type,
                        description=description,
                        auth_type=auth_type,
                        auth_value_encrypted=auth_value or existing["auth_value_encrypted"],
                        protocol_version=protocol_version or existing["protocol_version"],
                        capabilities_json=capabilities_json,
                        config_json=config_json,
                        tags_json=tags_json,
                        enabled=enabled,
                        updated_at=now,
                    )
                )
            else:
                conn.execute(
                    insert(a2a_agents).values(
                        name=name,
                        endpoint_url=endpoint_url,
                        agent_type=agent_type,
                        description=description,
                        auth_type=auth_type,
                        auth_value_encrypted=auth_value or "",
                        protocol_version=protocol_version or "1.0",
                        capabilities_json=capabilities_json,
                        config_json=config_json,
                        tags_json=tags_json,
                        enabled=enabled,
                        created_at=now,
                        updated_at=now,
                    )
                )
        row = get_a2a_agent_by_name(name)
        return row or {}
    except Exception as e:
        log.warning("upsert_a2a_agent failed: %s", e)
        return {"error": str(e)}


def get_a2a_agent(agent_id: int) -> dict | None:
    try:
        with begin() as conn:
            row = conn.execute(
                select(a2a_agents).where(a2a_agents.c.id == agent_id)
            ).mappings().first()
            return dict(row) if row else None
    except Exception as e:
        log.warning("get_a2a_agent failed: %s", e)
        return None


def get_a2a_agent_by_name(name: str) -> dict | None:
    try:
        with begin() as conn:
            row = conn.execute(
                select(a2a_agents).where(a2a_agents.c.name == name)
            ).mappings().first()
            return dict(row) if row else None
    except Exception as e:
        log.warning("get_a2a_agent_by_name failed: %s", e)
        return None


def list_a2a_agents(enabled_only: bool = False) -> list[dict]:
    try:
        with begin() as conn:
            stmt = select(a2a_agents)
            if enabled_only:
                stmt = stmt.where(a2a_agents.c.enabled.is_(True))
            stmt = stmt.order_by(a2a_agents.c.name.asc())
            return [dict(r) for r in conn.execute(stmt).mappings().all()]
    except Exception as e:
        log.warning("list_a2a_agents failed: %s", e)
        return []


def set_a2a_agent_enabled(agent_id: int, enabled: bool) -> bool:
    try:
        with begin() as conn:
            result = conn.execute(
                update(a2a_agents)
                .where(a2a_agents.c.id == agent_id)
                .values(enabled=enabled, updated_at=datetime.now(UTC))
            )
        return bool(result.rowcount and result.rowcount > 0)
    except Exception as e:
        log.warning("set_a2a_agent_enabled failed: %s", e)
        return False


def delete_a2a_agent(agent_id: int) -> bool:
    try:
        with begin() as conn:
            conn.execute(delete(a2a_metrics).where(a2a_metrics.c.agent_id == agent_id))
            conn.execute(delete(a2a_agents).where(a2a_agents.c.id == agent_id))
            return True
    except Exception as e:
        log.warning("delete_a2a_agent failed: %s", e)
        return False


def record_a2a_metric(
    agent_id: int,
    tenant_id: str,
    success: bool,
    latency_ms: float,
    interaction_type: str = "invoke",
    error: str | None = None,
) -> int:
    try:
        with begin() as conn:
            result = conn.execute(
                insert(a2a_metrics).values(
                    agent_id=agent_id,
                    tenant_id=tenant_id,
                    success=success,
                    latency_ms=latency_ms,
                    interaction_type=interaction_type,
                    error=error or "",
                )
            )
            return int(result.inserted_primary_key[0]) if result.inserted_primary_key else 0
    except Exception as e:
        log.warning("record_a2a_metric failed: %s", e)
        return 0


def a2a_agent_metrics_summary(agent_id: int) -> dict:
    try:
        with begin() as conn:
            total = conn.execute(
                select(func.count()).select_from(a2a_metrics)
                .where(a2a_metrics.c.agent_id == agent_id)
            ).scalar() or 0
            ok = conn.execute(
                select(func.count()).select_from(a2a_metrics)
                .where(a2a_metrics.c.agent_id == agent_id, a2a_metrics.c.success.is_(True))
            ).scalar() or 0
            row = conn.execute(
                select(
                    func.min(a2a_metrics.c.latency_ms),
                    func.max(a2a_metrics.c.latency_ms),
                    func.avg(a2a_metrics.c.latency_ms),
                ).where(a2a_metrics.c.agent_id == agent_id)
            ).first()
            last_ts = conn.execute(
                select(func.max(a2a_metrics.c.ts)).where(a2a_metrics.c.agent_id == agent_id)
            ).scalar()
            return {
                "agent_id": agent_id,
                "total_invocations": int(total),
                "successful": int(ok),
                "success_rate": float(ok / total) if total else 0.0,
                "latency_min_ms": float(row[0]) if row and row[0] is not None else 0.0,
                "latency_max_ms": float(row[1]) if row and row[1] is not None else 0.0,
                "latency_avg_ms": float(row[2]) if row and row[2] is not None else 0.0,
                "last_interaction": last_ts.isoformat() if last_ts else None,
            }
    except Exception as e:
        log.warning("a2a_agent_metrics_summary failed: %s", e)
        return {}


# =====================================================================
# A2A virtual servers — group agents into named bundles
# =====================================================================


def upsert_a2a_virtual_server(
    name: str,
    description: str = "",
    associated_agents: list[int] | None = None,
    enabled: bool = True,
) -> dict:
    now = datetime.now(UTC)
    agents_json = json.dumps(associated_agents or [])
    try:
        with begin() as conn:
            existing = conn.execute(
                select(a2a_virtual_servers).where(a2a_virtual_servers.c.name == name)
            ).mappings().first()
            if existing:
                conn.execute(
                    update(a2a_virtual_servers)
                    .where(a2a_virtual_servers.c.name == name)
                    .values(
                        description=description,
                        associated_agents_json=agents_json,
                        enabled=enabled,
                        updated_at=now,
                    )
                )
            else:
                conn.execute(
                    insert(a2a_virtual_servers).values(
                        name=name,
                        description=description,
                        associated_agents_json=agents_json,
                        enabled=enabled,
                        created_at=now,
                        updated_at=now,
                    )
                )
        row = get_a2a_virtual_server_by_name(name)
        return row or {}
    except Exception as e:
        log.warning("upsert_a2a_virtual_server failed: %s", e)
        return {"error": str(e)}


def get_a2a_virtual_server(server_id: int) -> dict | None:
    try:
        with begin() as conn:
            row = conn.execute(
                select(a2a_virtual_servers).where(a2a_virtual_servers.c.id == server_id)
            ).mappings().first()
            return dict(row) if row else None
    except Exception as e:
        log.warning("get_a2a_virtual_server failed: %s", e)
        return None


def get_a2a_virtual_server_by_name(name: str) -> dict | None:
    try:
        with begin() as conn:
            row = conn.execute(
                select(a2a_virtual_servers).where(a2a_virtual_servers.c.name == name)
            ).mappings().first()
            return dict(row) if row else None
    except Exception as e:
        log.warning("get_a2a_virtual_server_by_name failed: %s", e)
        return None


def list_a2a_virtual_servers() -> list[dict]:
    try:
        with begin() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    select(a2a_virtual_servers).order_by(a2a_virtual_servers.c.name.asc())
                ).mappings().all()
            ]
    except Exception as e:
        log.warning("list_a2a_virtual_servers failed: %s", e)
        return []


def delete_a2a_virtual_server(server_id: int) -> bool:
    try:
        with begin() as conn:
            conn.execute(
                delete(a2a_virtual_servers).where(a2a_virtual_servers.c.id == server_id)
            )
            return True
    except Exception as e:
        log.warning("delete_a2a_virtual_server failed: %s", e)
        return False


# =====================================================================
# Prompt templates — DB-backed prompt template registry
# =====================================================================


def upsert_prompt_template(
    name: str,
    template_text: str,
    description: str = "",
    variables: list[str] | None = None,
    category: str = "general",
    enabled: bool = True,
    is_builtin: bool = False,
    source: str = "manual",
) -> dict:
    """Insert or update a prompt template."""
    if not name or not template_text:
        return {"error": "name and template_text required"}
    variables_json = json.dumps(variables or [])
    now = datetime.now(UTC)
    try:
        with begin() as conn:
            existing = conn.execute(
                select(prompt_templates).where(prompt_templates.c.name == name)
            ).mappings().first()
            if existing:
                conn.execute(
                    update(prompt_templates)
                    .where(prompt_templates.c.name == name)
                    .values(
                        description=description,
                        template_text=template_text,
                        variables_json=variables_json,
                        category=category,
                        enabled=enabled,
                        source=source,
                        version=existing["version"] + 1,
                        updated_at=now,
                    )
                )
            else:
                conn.execute(
                    insert(prompt_templates).values(
                        name=name,
                        description=description,
                        template_text=template_text,
                        variables_json=variables_json,
                        category=category,
                        enabled=enabled,
                        is_builtin=is_builtin,
                        version=1,
                        source=source,
                        created_at=now,
                        updated_at=now,
                    )
                )
        row = get_prompt_template_by_name(name)
        return row or {}
    except Exception as e:
        log.warning("upsert_prompt_template failed: %s", e)
        return {"error": str(e)}


def get_prompt_template(template_id: int) -> dict | None:
    try:
        with begin() as conn:
            row = conn.execute(
                select(prompt_templates).where(prompt_templates.c.id == template_id)
            ).mappings().first()
            return dict(row) if row else None
    except Exception as e:
        log.warning("get_prompt_template failed: %s", e)
        return None


def get_prompt_template_by_name(name: str) -> dict | None:
    try:
        with begin() as conn:
            row = conn.execute(
                select(prompt_templates).where(prompt_templates.c.name == name)
            ).mappings().first()
            return dict(row) if row else None
    except Exception as e:
        log.warning("get_prompt_template_by_name failed: %s", e)
        return None


def list_prompt_templates(
    enabled_only: bool = False, category: str | None = None
) -> list[dict]:
    try:
        with begin() as conn:
            stmt = select(prompt_templates)
            if enabled_only:
                stmt = stmt.where(prompt_templates.c.enabled.is_(True))
            if category:
                stmt = stmt.where(prompt_templates.c.category == category)
            stmt = stmt.order_by(prompt_templates.c.category.asc(), prompt_templates.c.name.asc())
            return [dict(r) for r in conn.execute(stmt).mappings().all()]
    except Exception as e:
        log.warning("list_prompt_templates failed: %s", e)
        return []


def set_prompt_template_enabled(template_id: int, enabled: bool) -> bool:
    try:
        with begin() as conn:
            result = conn.execute(
                update(prompt_templates)
                .where(prompt_templates.c.id == template_id)
                .values(enabled=enabled, updated_at=datetime.now(UTC))
            )
        return bool(result.rowcount and result.rowcount > 0)
    except Exception as e:
        log.warning("set_prompt_template_enabled failed: %s", e)
        return False


def delete_prompt_template(template_id: int) -> bool:
    """Refuse to delete builtin templates."""
    try:
        with begin() as conn:
            row = conn.execute(
                select(prompt_templates.c.is_builtin).where(
                    prompt_templates.c.id == template_id
                )
            ).first()
            if row and row[0]:
                return False
            conn.execute(
                delete(prompt_templates).where(prompt_templates.c.id == template_id)
            )
            return True
    except Exception as e:
        log.warning("delete_prompt_template failed: %s", e)
        return False


# =====================================================================
# Webhooks — fan-out event delivery
# =====================================================================


def upsert_webhook(
    name: str,
    url: str,
    events: list[str],
    secret: str = "",
    enabled: bool = True,
    description: str = "",
) -> dict:
    if not name or not url:
        return {"error": "name and url required"}
    events_json = json.dumps(events or [])
    now = datetime.now(UTC)
    try:
        with begin() as conn:
            existing = conn.execute(
                select(webhooks).where(webhooks.c.name == name)
            ).mappings().first()
            if existing:
                conn.execute(
                    update(webhooks)
                    .where(webhooks.c.name == name)
                    .values(
                        url=url,
                        events_json=events_json,
                        secret=secret,
                        enabled=enabled,
                        description=description,
                        updated_at=now,
                    )
                )
            else:
                conn.execute(
                    insert(webhooks).values(
                        name=name,
                        url=url,
                        events_json=events_json,
                        secret=secret,
                        enabled=enabled,
                        description=description,
                        created_at=now,
                        updated_at=now,
                    )
                )
        row = get_webhook_by_name(name)
        return row or {}
    except Exception as e:
        log.warning("upsert_webhook failed: %s", e)
        return {"error": str(e)}


def get_webhook(webhook_id: int) -> dict | None:
    try:
        with begin() as conn:
            row = conn.execute(
                select(webhooks).where(webhooks.c.id == webhook_id)
            ).mappings().first()
            return dict(row) if row else None
    except Exception as e:
        log.warning("get_webhook failed: %s", e)
        return None


def get_webhook_by_name(name: str) -> dict | None:
    try:
        with begin() as conn:
            row = conn.execute(
                select(webhooks).where(webhooks.c.name == name)
            ).mappings().first()
            return dict(row) if row else None
    except Exception as e:
        log.warning("get_webhook_by_name failed: %s", e)
        return None


def list_webhooks(enabled_only: bool = False) -> list[dict]:
    try:
        with begin() as conn:
            stmt = select(webhooks)
            if enabled_only:
                stmt = stmt.where(webhooks.c.enabled.is_(True))
            stmt = stmt.order_by(webhooks.c.name.asc())
            return [dict(r) for r in conn.execute(stmt).mappings().all()]
    except Exception as e:
        log.warning("list_webhooks failed: %s", e)
        return []


def delete_webhook(webhook_id: int) -> bool:
    try:
        with begin() as conn:
            conn.execute(
                delete(webhook_deliveries).where(webhook_deliveries.c.webhook_id == webhook_id)
            )
            conn.execute(delete(webhooks).where(webhooks.c.id == webhook_id))
            return True
    except Exception as e:
        log.warning("delete_webhook failed: %s", e)
        return False


def record_webhook_delivery(
    webhook_id: int,
    event_type: str,
    tenant_id: str | None,
    status_code: int | None,
    payload_json: str,
    response_body: str = "",
    error: str = "",
    attempt: int = 1,
    duration_ms: float = 0.0,
) -> int:
    try:
        with begin() as conn:
            result = conn.execute(
                insert(webhook_deliveries).values(
                    webhook_id=webhook_id,
                    event_type=event_type,
                    tenant_id=tenant_id,
                    status_code=status_code,
                    response_body=response_body[:4000] if response_body else "",
                    payload_json=payload_json[:8000],
                    error=error[:1000] if error else "",
                    attempt=attempt,
                    duration_ms=duration_ms,
                )
            )
            return int(result.inserted_primary_key[0]) if result.inserted_primary_key else 0
    except Exception as e:
        log.warning("record_webhook_delivery failed: %s", e)
        return 0


def list_webhook_deliveries(
    webhook_id: int | None = None, limit: int = 100
) -> list[dict]:
    try:
        with begin() as conn:
            stmt = select(webhook_deliveries)
            if webhook_id is not None:
                stmt = stmt.where(webhook_deliveries.c.webhook_id == webhook_id)
            stmt = stmt.order_by(webhook_deliveries.c.ts.desc()).limit(limit)
            return [dict(r) for r in conn.execute(stmt).mappings().all()]
    except Exception as e:
        log.warning("list_webhook_deliveries failed: %s", e)
        return []


# =====================================================================
# ContextForge sync log + federated tools
# =====================================================================


def record_contextforge_sync(
    sync_type: str,
    source: str,
    items_synced: int = 0,
    items_added: int = 0,
    items_updated: int = 0,
    errors: list[str] | None = None,
    duration_ms: float = 0.0,
) -> int:
    errors_json = json.dumps(errors or [])
    try:
        with begin() as conn:
            result = conn.execute(
                insert(contextforge_sync_log).values(
                    sync_type=sync_type,
                    source=source,
                    items_synced=items_synced,
                    items_added=items_added,
                    items_updated=items_updated,
                    errors_json=errors_json,
                    duration_ms=duration_ms,
                )
            )
            return int(result.inserted_primary_key[0]) if result.inserted_primary_key else 0
    except Exception as e:
        log.warning("record_contextforge_sync failed: %s", e)
        return 0


def list_contextforge_sync_log(limit: int = 50) -> list[dict]:
    try:
        with begin() as conn:
            return [
                dict(r)
                for r in conn.execute(
                    select(contextforge_sync_log)
                    .order_by(contextforge_sync_log.c.ts.desc())
                    .limit(limit)
                ).mappings().all()
            ]
    except Exception as e:
        log.warning("list_contextforge_sync_log failed: %s", e)
        return []


def upsert_federated_tool(
    name: str,
    source: str,
    source_url: str,
    tool: dict,
    enabled: bool = True,
) -> dict:
    """Insert or update a federated tool record (from ContextForge, etc.)."""
    tool_json = json.dumps(tool)
    now = datetime.now(UTC)
    try:
        with begin() as conn:
            existing = conn.execute(
                select(federated_tools).where(federated_tools.c.name == name)
            ).mappings().first()
            if existing:
                conn.execute(
                    update(federated_tools)
                    .where(federated_tools.c.name == name)
                    .values(
                        source=source,
                        source_url=source_url,
                        tool_json=tool_json,
                        enabled=enabled,
                        last_synced=now,
                    )
                )
            else:
                conn.execute(
                    insert(federated_tools).values(
                        name=name,
                        source=source,
                        source_url=source_url,
                        tool_json=tool_json,
                        enabled=enabled,
                        last_synced=now,
                        created_at=now,
                    )
                )
        row = get_federated_tool(name)
        return row or {}
    except Exception as e:
        log.warning("upsert_federated_tool failed: %s", e)
        return {"error": str(e)}


def get_federated_tool(name: str) -> dict | None:
    try:
        with begin() as conn:
            row = conn.execute(
                select(federated_tools).where(federated_tools.c.name == name)
            ).mappings().first()
            return dict(row) if row else None
    except Exception as e:
        log.warning("get_federated_tool failed: %s", e)
        return None


def list_federated_tools(enabled_only: bool = False) -> list[dict]:
    try:
        with begin() as conn:
            stmt = select(federated_tools)
            if enabled_only:
                stmt = stmt.where(federated_tools.c.enabled.is_(True))
            stmt = stmt.order_by(federated_tools.c.name.asc())
            return [dict(r) for r in conn.execute(stmt).mappings().all()]
    except Exception as e:
        log.warning("list_federated_tools failed: %s", e)
        return []


def delete_federated_tool(name: str) -> bool:
    try:
        with begin() as conn:
            conn.execute(delete(federated_tools).where(federated_tools.c.name == name))
            return True
    except Exception as e:
        log.warning("delete_federated_tool failed: %s", e)
        return False

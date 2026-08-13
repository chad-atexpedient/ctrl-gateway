"""Observer + Reflector background workers.

Implements Mastra's Observational Memory three-agent loop at the gateway level:

  Actor:    handled by app.py — every chat completion that flows through.
  Observer: this file. Compresses raw unobserved messages into structured
            observations when total unobserved tokens cross `message_tokens`.
  Reflector: this file. Compresses observations into a single reflection
             when observation tokens cross `observation_tokens`.

Both run async, never block the hot path. We buffer token accumulation
asynchronously (incrementing counters per message) and fire the heavy LLM
call when the threshold trips.

The Observer and Reflector use a small, fast model (per Mastra's guidance:
"Observer model is a cost lever, not a quality one"). The default tier is
tier0 (LFM2.5-1.2B-Thinking) for compression work. Configurable.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime

from sqlalchemy import func, insert, select, update

from . import config as cfg
from . import endpoints, events, transcoder
from . import memory as storage
from . import memory_observational as om

log = logging.getLogger("ctrl.observer")


OBSERVER_SYSTEM_PROMPT = (
    "You are a memory compression agent. Your only job is to read raw conversation "
    "messages and produce dense observations that preserve facts, user preferences, "
    "decisions, open threads, and any context the actor would need to continue the "
    "conversation naturally. Do NOT paraphrase the user's feelings. Do NOT add "
    "speculation. Output ONLY the observation text — no headers, no preamble, no "
    "markdown fences."
)

REFLECTOR_SYSTEM_PROMPT = (
    "You are a memory consolidation agent. You receive a sequence of observations "
    "and produce a condensed reflection that keeps only the durable, still-relevant "
    "facts. Drop resolved issues, dropped preferences, and ephemeral details. "
    "Preserve identity facts, open threads, and current context. Output ONLY the "
    "reflection text."
)


class ObserverReflectorWorker:
    """Background loop. Checks unobserved-token totals; fires Observer + Reflector."""

    def __init__(self, conf: cfg.Config, pool: endpoints.EndpointPool):
        self.conf = conf
        self.pool = pool
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()
        self._running = False
        self._last_check = 0.0
        self._work_lock = asyncio.Lock()

    async def start(self):
        if self._task and not self._task.done():
            return
        self._stop_event.clear()
        self._running = True
        self._task = asyncio.create_task(self._loop(), name="observer-reflector")
        # Ensure tables exist
        om.memory_metadata.create_all(storage.engine())
        log.info("observer/reflector worker started")

    async def stop(self):
        self._running = False
        self._stop_event.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):
                pass

    def update_config(self, conf: cfg.Config):
        self.conf = conf

    async def _loop(self):
        om_cfg = self.conf.policy.get("memory", {})
        poll_s = 10.0
        while self._running:
            try:
                if not self._work_lock.locked():
                    await self._tick(om_cfg)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.exception("observer/reflector tick failed: %s", e)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=poll_s)
            except TimeoutError:
                pass

    async def _tick(self, om_cfg: dict):
        message_threshold = om_cfg.get("message_tokens", 12000)
        observation_threshold = om_cfg.get("observation_tokens", 20000)
        if not message_threshold:
            return

        # Find threads with unobserved token totals >= threshold
        try:
            threads_over_threshold = self._find_threads_above(message_threshold)
        except Exception as e:
            log.warning("scan failed: %s", e)
            return

        if not threads_over_threshold:
            # Also check observations against reflection threshold
            await self._maybe_reflect(om_cfg, observation_threshold)
            return

        # Fire Observer for the largest one (single-flight)
        thread_id, resource_id, token_total = threads_over_threshold[0]
        async with self._work_lock:
            await self._observe_thread(thread_id, resource_id, om_cfg)
            await self._maybe_reflect(om_cfg, observation_threshold)

    async def observe(self, thread_id: str, resource_id: str, om_cfg: dict):
        async with self._work_lock:
            await self._observe_thread(thread_id, resource_id, om_cfg)

    async def reflect(self, resource_id: str, thread_id: str, om_cfg: dict):
        async with self._work_lock:
            await self._reflect_thread(resource_id, thread_id, om_cfg)

    def _find_threads_above(self, threshold: int) -> list[tuple[str, str, int]]:
        """Return [(thread_id, resource_id, unobserved_token_total)] sorted desc."""
        try:
            with storage.engine().connect() as conn:
                rows = conn.execute(
                    select(
                        om.message_history.c.thread_id,
                        om.message_history.c.resource_id,
                        func.sum(om.message_history.c.token_estimate).label("t"),
                    )
                    .where(om.message_history.c.observed_at.is_(None))
                    .group_by(om.message_history.c.thread_id, om.message_history.c.resource_id)
                    .having(func.sum(om.message_history.c.token_estimate) >= threshold)
                    .order_by(func.sum(om.message_history.c.token_estimate).desc())
                    .limit(5)
                ).all()
            return [(r.thread_id, r.resource_id, int(r.t)) for r in rows]
        except Exception as e:
            log.warning("find_threads_above failed: %s", e)
            return []

    async def _observe_thread(self, thread_id: str, resource_id: str, om_cfg: dict):
        """Pull unobserved messages, call Observer model, store observation."""
        try:
            with storage.engine().connect() as conn:
                rows = conn.execute(
                    select(om.message_history)
                    .where(om.message_history.c.resource_id == resource_id)
                    .where(om.message_history.c.thread_id == thread_id)
                    .where(om.message_history.c.observed_at.is_(None))
                    .order_by(om.message_history.c.id.asc())
                    .limit(200)
                ).all()
            if not rows:
                return
            msgs = [
                {"id": r._mapping["id"], "role": r._mapping["role"], "content": r._mapping["content"]}
                for r in rows
            ]
        except Exception as e:
            log.warning("observe_thread: fetch failed: %s", e)
            return

        events.emit_status(
            events.EventSource.OBSERVER,
            f"Compressing {len(msgs)} messages in thread {thread_id[:8]}...",
            done=False,
            tenant_id=resource_id,
            session_id=thread_id,
        )

        # Call observer model
        try:
            observation_text = await self._call_observer(msgs, om_cfg)
        except Exception as e:
            log.warning("observer call failed: %s", e)
            events.emit_status(
                events.EventSource.OBSERVER,
                f"Observer failed: {e}",
                done=True,
                tenant_id=resource_id,
                session_id=thread_id,
            )
            return

        if not observation_text:
            return

        # Store observation
        try:
            with storage.engine().begin() as conn:
                conn.execute(insert(om.observations).values(
                    resource_id=resource_id,
                    thread_id=thread_id,
                    content=observation_text,
                    token_estimate=len(observation_text) // 4,
                    source_message_ids=json.dumps([m["id"] for m in msgs]),
                    model=om_cfg.get("observer_model", "tier0"),
                ))
                # Mark messages as observed
                ids = [m["id"] for m in msgs]
                conn.execute(
                    update(om.message_history)
                    .where(om.message_history.c.id.in_(ids))
                    .values(observed_at=datetime.now(UTC))
                )
        except Exception as e:
            log.warning("observation persist failed: %s", e)
            return

        events.emit(
            events.EventSource.OBSERVER,
            "observation_created",
            {
                "thread_id": thread_id,
                "resource_id": resource_id,
                "messages_observed": len(msgs),
                "observation_chars": len(observation_text),
            },
            tenant_id=resource_id,
            session_id=thread_id,
        )
        events.emit_status(
            events.EventSource.OBSERVER,
            f"Compressed {len(msgs)} messages → {len(observation_text)} chars",
            done=True,
            tenant_id=resource_id,
            session_id=thread_id,
        )

    async def _maybe_reflect(self, om_cfg: dict, threshold: int):
        """If observation tokens cross threshold, condense via Reflector."""
        try:
            with storage.engine().connect() as conn:
                rows = conn.execute(
                    select(
                        om.observations.c.resource_id,
                        om.observations.c.thread_id,
                        func.sum(om.observations.c.token_estimate).label("t"),
                        func.max(om.observations.c.id).label("latest_id"),
                    )
                    .group_by(om.observations.c.resource_id, om.observations.c.thread_id)
                    .having(func.sum(om.observations.c.token_estimate) >= threshold)
                    .order_by(func.sum(om.observations.c.token_estimate).desc())
                    .limit(3)
                ).all()
            for r in rows:
                await self._reflect_thread(r.resource_id, r.thread_id, om_cfg)
        except Exception as e:
            log.warning("maybe_reflect scan failed: %s", e)

    async def _reflect_thread(self, resource_id: str, thread_id: str, om_cfg: dict):
        """Pull observations, call Reflector, store reflection."""
        try:
            with storage.engine().connect() as conn:
                rows = conn.execute(
                    select(om.observations)
                    .where(om.observations.c.resource_id == resource_id)
                    .where(om.observations.c.thread_id == thread_id)
                    .order_by(om.observations.c.id.asc())
                    .limit(100)
                ).all()
            if not rows:
                return
            obs_list = [r._mapping["content"] for r in rows if r._mapping.get("content")]
            if not obs_list:
                return
        except Exception as e:
            log.warning("reflect_thread: fetch failed: %s", e)
            return

        events.emit_status(
            events.EventSource.REFLECTOR,
            f"Consolidating {len(obs_list)} observations for thread {thread_id[:8]}",
            done=False,
            tenant_id=resource_id,
            session_id=thread_id,
        )

        try:
            reflection = await self._call_reflector(obs_list, om_cfg)
        except Exception as e:
            log.warning("reflector call failed: %s", e)
            events.emit_status(
                events.EventSource.REFLECTOR,
                f"Reflector failed: {e}",
                done=True,
                tenant_id=resource_id,
                session_id=thread_id,
            )
            return

        if not reflection:
            return

        try:
            with storage.engine().begin() as conn:
                # Mark old observations as superseded by the reflection
                obs_ids = [r.id for r in rows]
                conn.execute(insert(om.reflections).values(
                    resource_id=resource_id,
                    thread_id=thread_id,
                    content=reflection,
                    supersedes_observation_ids=json.dumps(obs_ids),
                    model=om_cfg.get("reflector_model", "tier0"),
                ))
        except Exception as e:
            log.warning("reflection persist failed: %s", e)
            return

        events.emit(
            events.EventSource.REFLECTOR,
            "reflection_created",
            {
                "thread_id": thread_id,
                "resource_id": resource_id,
                "observations_consolidated": len(obs_list),
                "reflection_chars": len(reflection),
            },
            tenant_id=resource_id,
            session_id=thread_id,
        )
        events.emit_status(
            events.EventSource.REFLECTOR,
            f"Reflection saved ({len(reflection)} chars)",
            done=True,
            tenant_id=resource_id,
            session_id=thread_id,
        )

    async def _call_observer(self, messages: list[dict], om_cfg: dict) -> str | None:
        """Call the Observer model. Default to tier0 (small/fast)."""
        model_tier = om_cfg.get("observer_tier", "tier0")
        endpoint_name = om_cfg.get("observer_endpoint")
        if not endpoint_name:
            # Pick first available endpoint in observer_tier
            tier_obj = self.conf.tier(model_tier)
            if not tier_obj:
                return None
            endpoints_list = self.conf.endpoints_for_tier(model_tier)
            if not endpoints_list:
                return None
            endpoint_name = endpoints_list[0]["name"]
        ep_cfg = self.conf.endpoint(endpoint_name)
        tier_cfg = self.conf.tier(model_tier)
        if not ep_cfg or not tier_cfg:
            return None

        # Build the messages
        conversation = "\n\n".join(
            f"[{m['role']}] {m['content']}" for m in messages
        )
        payload = {
            "model": ep_cfg.get("model_alias", "default"),
            "messages": [
                {"role": "system", "content": OBSERVER_SYSTEM_PROMPT},
                {"role": "user", "content": f"Conversation to observe:\n\n{conversation}\n\nObservation:"},
            ],
            "temperature": 0.0,
            "stream": False,
            "max_tokens": om_cfg.get("observer_max_tokens", 800),
        }
        transcoded = transcoder.transcode(ep_cfg, tier_cfg, payload)
        client = self.pool.get(endpoint_name)
        result = await client.send(transcoded, stream=False)
        if not isinstance(result, dict):
            return None
        return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

    async def _call_reflector(self, observations_list: list[str], om_cfg: dict) -> str | None:
        """Call the Reflector model."""
        model_tier = om_cfg.get("reflector_tier", "tier0")
        endpoint_name = om_cfg.get("reflector_endpoint")
        if not endpoint_name:
            tier_obj = self.conf.tier(model_tier)
            if not tier_obj:
                return None
            endpoints_list = self.conf.endpoints_for_tier(model_tier)
            if not endpoints_list:
                return None
            endpoint_name = endpoints_list[0]["name"]
        ep_cfg = self.conf.endpoint(endpoint_name)
        tier_cfg = self.conf.tier(model_tier)
        if not ep_cfg or not tier_cfg:
            return None

        joined = "\n\n---\n\n".join(observations_list)
        payload = {
            "model": ep_cfg.get("model_alias", "default"),
            "messages": [
                {"role": "system", "content": REFLECTOR_SYSTEM_PROMPT},
                {"role": "user", "content": f"Observations to consolidate:\n\n{joined}\n\nReflection:"},
            ],
            "temperature": 0.0,
            "stream": False,
            "max_tokens": om_cfg.get("reflector_max_tokens", 600),
        }
        transcoded = transcoder.transcode(ep_cfg, tier_cfg, payload)
        client = self.pool.get(endpoint_name)
        result = await client.send(transcoded, stream=False)
        if not isinstance(result, dict):
            return None
        return result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()


_worker: ObserverReflectorWorker | None = None


def init_worker(conf: cfg.Config, pool: endpoints.EndpointPool) -> ObserverReflectorWorker:
    global _worker
    _worker = ObserverReflectorWorker(conf, pool)
    return _worker


def worker() -> ObserverReflectorWorker:
    if _worker is None:
        raise RuntimeError("observer/reflector worker not initialized")
    return _worker

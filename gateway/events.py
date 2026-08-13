"""Event emitter system.

Adapts the Open WebUI thinking-indicator pattern (`__event_emitter__`) to a
generic pub/sub for our gateway. UI platforms (OWUI, custom dashboards,
webhooks) subscribe to typed events and stream them to the user.

Events are organized by:
  - source: routing, memory, observer, reflector, trainer, reviewer, breaker
  - severity: info, status, warn, error
  - audience: user-facing, admin, debug

Each event has:
  - type: free-form string
  - data: dict
  - timestamp
  - session_id, tenant_id, decision_id (when applicable)

Subscribers:
  - SSE endpoint at /events (live stream)
  - In-memory ring buffer for the dashboard
  - Webhook dispatcher (future)
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import asdict, dataclass, field
from enum import Enum

log = logging.getLogger("ctrl.events")


class EventSource(str, Enum):
    ROUTING = "routing"
    MEMORY = "memory"
    OBSERVER = "observer"
    REFLECTOR = "reflector"
    TRAINER = "trainer"
    REVIEWER = "reviewer"
    BREAKER = "breaker"
    SECURITY = "security"
    SWARM = "swarm"
    TRANSLATION = "translation"
    PLUGIN = "plugin"
    A2A = "a2a"
    WEBHOOK = "webhook"
    CONTEXTFORGE = "contextforge"
    MCP = "mcp"
    PROMPT = "prompt"
    SYSTEM = "system"


class EventSeverity(str, Enum):
    INFO = "info"
    STATUS = "status"  # UI status updates (thinking, done)
    WARN = "warn"
    ERROR = "error"
    DEBUG = "debug"


@dataclass
class Event:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    ts: float = field(default_factory=time.time)
    source: EventSource = EventSource.SYSTEM
    severity: EventSeverity = EventSeverity.INFO
    type: str = "generic"
    data: dict = field(default_factory=dict)
    tenant_id: str | None = None
    session_id: str | None = None
    decision_id: int | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["source"] = self.source.value if isinstance(self.source, EventSource) else self.source
        d["severity"] = self.severity.value if isinstance(self.severity, EventSeverity) else self.severity
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)


class EventBus:
    """Async pub/sub for gateway events.

    Subscribers receive every event via an asyncio.Queue (or a callback).
    A bounded ring buffer retains the last N events for late subscribers (dashboard).
    """

    def __init__(self, ring_buffer_size: int = 1000):
        self._subscribers: list[asyncio.Queue] = []
        self._callbacks: list[Callable[[Event], None]] = []
        self._ring: deque[Event] = deque(maxlen=ring_buffer_size)
        self._lock = asyncio.Lock()

    def publish(self, event: Event) -> None:
        """Synchronous publish — adds to ring buffer and notifies async subscribers.

        Safe to call from non-async code (e.g., the synchronous pre-route path).
        """
        self._ring.append(event)
        # Schedule async notifications on the running loop if any
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._notify(event))
            # Also fan-out to webhooks (if dispatcher initialized)
            try:
                from . import webhook_dispatcher
                disp = webhook_dispatcher.dispatcher()
                if disp is not None:
                    loop.create_task(
                        disp.dispatch(
                            event_type=event.type,
                            payload=event.to_dict(),
                            tenant_id=event.tenant_id,
                        )
                    )
            except Exception as e:
                log.debug("webhook dispatch skipped: %s", e)
        except RuntimeError:
            # No running loop; sync callbacks only (snapshot to avoid mutation during iteration)
            for cb in list(self._callbacks):
                try:
                    cb(event)
                except Exception as e:
                    log.warning("sync callback failed: %s", e)

    async def _notify(self, event: Event) -> None:
        async with self._lock:
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                # Subscriber is slow; drop this event for them, log it
                log.debug("subscriber queue full; dropped event %s", event.id)
        for cb in list(self._callbacks):
            try:
                cb(event)
            except Exception as e:
                log.warning("async callback failed: %s", e)

    async def subscribe(self, max_queue: int = 100) -> tuple[asyncio.Queue, Callable]:
        """Add a subscriber. Returns (queue, unsubscribe_fn)."""
        q: asyncio.Queue = asyncio.Queue(maxsize=max_queue)
        async with self._lock:
            self._subscribers.append(q)

        def unsubscribe():
            async def _remove():
                async with self._lock:
                    try:
                        self._subscribers.remove(q)
                    except ValueError:
                        pass
            try:
                asyncio.get_running_loop().create_task(_remove())
            except RuntimeError:
                try:
                    self._subscribers.remove(q)
                except ValueError:
                    pass

        return q, unsubscribe

    def subscribe_sync(self, callback: Callable[[Event], None]) -> Callable[[], None]:
        """Add a sync callback. Returns unsubscribe fn."""
        self._callbacks.append(callback)

        def unsubscribe():
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

        return unsubscribe

    def recent(
        self,
        limit: int = 100,
        source: EventSource | None = None,
        tenant_id: str | None = None,
    ) -> list[Event]:
        """Return the last N events (optionally filtered by source/tenant)."""
        out = []
        for ev in reversed(self._ring):
            if source is not None and ev.source != source:
                continue
            if tenant_id is not None and ev.tenant_id not in (None, tenant_id):
                continue
            out.append(ev)
            if len(out) >= limit:
                break
        return list(reversed(out))

    async def stream(
        self,
        source: EventSource | None = None,
        tenant_id: str | None = None,
    ) -> AsyncIterator[Event]:
        """Async iterator that yields events as they arrive. Includes replay of recent.

        tenant_id filters to events for that tenant (None = all tenants; an
        event with no tenant is always visible).
        """
        # Replay
        for ev in self.recent(limit=50, source=source, tenant_id=tenant_id):
            yield ev
        # Live
        q, unsub = await self.subscribe()
        try:
            while True:
                ev = await q.get()
                if source is not None and ev.source != source:
                    continue
                if tenant_id is not None and ev.tenant_id not in (None, tenant_id):
                    continue
                yield ev
        finally:
            unsub()


# Module-level singleton
_bus: EventBus | None = None


def bus() -> EventBus:
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def emit(
    source: EventSource | str,
    event_type: str,
    data: dict | None = None,
    severity: EventSeverity = EventSeverity.INFO,
    tenant_id: str | None = None,
    session_id: str | None = None,
    decision_id: int | None = None,
) -> Event:
    """Convenience function. Emit and return the Event."""
    src = source if isinstance(source, EventSource) else EventSource(source)
    sev = severity if isinstance(severity, EventSeverity) else EventSeverity(severity)
    ev = Event(
        source=src,
        severity=sev,
        type=event_type,
        data=data or {},
        tenant_id=tenant_id,
        session_id=session_id,
        decision_id=decision_id,
    )
    bus().publish(ev)
    return ev


def emit_status(
    source: EventSource | str,
    description: str,
    done: bool = False,
    tenant_id: str | None = None,
    session_id: str | None = None,
    decision_id: int | None = None,
) -> Event:
    """Convenience for OWUI-style thinking/finished status events."""
    return emit(
        source=source,
        event_type="status",
        data={"description": description, "done": done},
        severity=EventSeverity.STATUS,
        tenant_id=tenant_id,
        session_id=session_id,
        decision_id=decision_id,
    )

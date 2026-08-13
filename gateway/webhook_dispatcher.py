"""Webhook dispatcher.

Fulfills the `events.py` line-22 stub: "Webhook dispatcher (future)".

Each registered webhook subscribes to a list of event_type strings (or
"*" for all events). When an event is emitted from the gateway, every
matching webhook receives a POST with:

  - Content-Type: application/json
  - X-Gateway-Event-Type: <event_type>
  - X-Gateway-Event-Id: <id>
  - X-Gateway-Timestamp: <unix_seconds>
  - X-Gateway-Signature: hmac_sha256(secret, payload_json) (when secret is set)
  - Body: { id, type, source, severity, ts, tenant_id, data, ... }

Delivery is async with exponential backoff. Every attempt is logged into
webhook_deliveries. A delivery is considered failed when all retries are
exhausted; failed deliveries are visible via /admin/webhooks/deliveries.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import aiohttp

from . import memory

log = logging.getLogger("ctrl.webhook")


@dataclass
class WebhookRecord:
    id: int
    name: str
    url: str
    events: list[str]
    secret: str
    enabled: bool
    description: str
    # Per-webhook override of the dispatcher's max_retries. None means use
    # the dispatcher default (set in init_dispatcher from gateway-config).
    max_retries: int | None = None
    # tenant_id scopes which events this webhook receives. "__all__" = every
    # event regardless of origin; any other value = only events whose
    # tenant_id matches.
    tenant_id: str = "__all__"


def _row_to_webhook(row: dict) -> WebhookRecord:
    try:
        events = json.loads(row.get("events_json") or "[]")
    except (json.JSONDecodeError, TypeError):
        events = []
    if not isinstance(events, list):
        events = []
    return WebhookRecord(
        id=int(row["id"]),
        name=row["name"],
        url=row["url"],
        events=[str(e) for e in events],
        secret=row.get("secret", "") or "",
        enabled=bool(row.get("enabled", True)),
        description=row.get("description", ""),
        max_retries=row.get("max_retries"),
        tenant_id=row.get("tenant_id") or "__all__",
    )


def list_webhooks(
    enabled_only: bool = False, tenant_id: str | None = None
) -> list[WebhookRecord]:
    rows = memory.list_webhooks(enabled_only=enabled_only, tenant_id=tenant_id)
    return [_row_to_webhook(r) for r in rows]


def get_webhook(webhook_id: int) -> WebhookRecord | None:
    row = memory.get_webhook(webhook_id)
    return _row_to_webhook(row) if row else None


def get_webhook_by_name(
    name: str, tenant_id: str | None = None
) -> WebhookRecord | None:
    row = memory.get_webhook_by_name(name, tenant_id=tenant_id)
    return _row_to_webhook(row) if row else None


def upsert_webhook(
    name: str,
    url: str,
    events: list[str],
    secret: str = "",
    enabled: bool = True,
    description: str = "",
    max_retries: int | None = None,
    tenant_id: str = "__all__",
) -> WebhookRecord | None:
    row = memory.upsert_webhook(
        name=name,
        url=url,
        events=events,
        secret=secret,
        enabled=enabled,
        description=description,
        max_retries=max_retries,
        tenant_id=tenant_id,
    )
    if "error" in row:
        return None
    return get_webhook_by_name(name, tenant_id=tenant_id)


def delete_webhook(webhook_id: int) -> bool:
    return memory.delete_webhook(webhook_id)


def list_deliveries(webhook_id: int | None = None, limit: int = 100) -> list[dict]:
    return memory.list_webhook_deliveries(webhook_id=webhook_id, limit=limit)


def _sign(secret: str, body: str) -> str:
    return "sha256=" + hmac.new(
        secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def _matches(webhook: WebhookRecord, event_type: str) -> bool:
    if not webhook.enabled:
        return False
    if "*" in webhook.events:
        return True
    return event_type in webhook.events


class WebhookDispatcher:
    """Async fan-out of events to registered webhooks.

    Caller passes an event payload (already shaped as a JSON-serializable
    dict). The dispatcher fans out to all matching webhooks concurrently
    with bounded concurrency, retries, and delivery logging.
    """

    def __init__(
        self,
        max_retries: int = 3,
        initial_backoff_seconds: float = 1.0,
        backoff_multiplier: float = 2.0,
        delivery_timeout_seconds: float = 10.0,
        max_concurrent_deliveries: int = 16,
    ):
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self.backoff_multiplier = backoff_multiplier
        self.delivery_timeout_seconds = delivery_timeout_seconds
        self.max_concurrent_deliveries = max_concurrent_deliveries
        self._semaphore: asyncio.Semaphore | None = None
        self._session: aiohttp.ClientSession | None = None

    def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_deliveries)
        return self._semaphore

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            from . import ssrf
            # ssrf_client_kwargs() bundles the connect-time-DNS-rebinding
            # -safe connector (closing the TOCTOU/DNS-rebinding gap the
            # validate_url() call in _deliver_with_retry() below can't close
            # on its own — see ssrf.SSRFSafeResolver) with a redirect-target
            # guard (closing the separate gap where a webhook endpoint's 3xx
            # response redirects to a blocked target — see
            # ssrf.redirect_guard_trace_config). This session is reused
            # across deliveries to many different webhook URLs; policy is
            # the same fixed one for all of them, so one shared session is
            # fine.
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.delivery_timeout_seconds),
                **ssrf.ssrf_client_kwargs(allow_localhost=True, allow_private=True),
            )
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    async def dispatch(
        self,
        event_type: str,
        payload: dict[str, Any],
        tenant_id: str | None = None,
    ) -> list[int]:
        """Fan out `payload` to every matching webhook. Returns list of
        delivery record IDs.

        Tenant filtering: a webhook with tenant_id="__all__" always matches.
        A webhook with a specific tenant_id only matches when the event's
        tenant_id equals the webhook's tenant_id. If tenant_id is None
        (event without a tenant context), only "__all__" webhooks match.
        """
        all_hooks = list_webhooks(enabled_only=True)
        matching: list[WebhookRecord] = []
        for w in all_hooks:
            if not _matches(w, event_type):
                continue
            # Tenant scope check
            if w.tenant_id == "__all__":
                matching.append(w)
            elif tenant_id is not None and w.tenant_id == tenant_id:
                matching.append(w)
            # else: webhook is for a different tenant — skip
        if not matching:
            return []
        body_str = json.dumps(payload, default=str, ensure_ascii=False)
        semaphore = self._get_semaphore()

        async def _deliver_one(webhook: WebhookRecord) -> int:
            async with semaphore:
                return await self._deliver_with_retry(
                    webhook, event_type, payload, body_str, tenant_id
                )

        ids = await asyncio.gather(
            *(_deliver_one(w) for w in matching), return_exceptions=True
        )
        out: list[int] = []
        for result in ids:
            if isinstance(result, int):
                out.append(result)
        return out

    async def _deliver_with_retry(
        self,
        webhook: WebhookRecord,
        event_type: str,
        payload: dict[str, Any],
        body_str: str,
        tenant_id: str | None,
    ) -> int:
        # Per-webhook override takes precedence over dispatcher default.
        effective_retries = webhook.max_retries
        if effective_retries is None:
            effective_retries = self.max_retries
        if effective_retries < 1:
            effective_retries = 1
        session = await self._get_session()
        backoff = self.initial_backoff_seconds
        last_delivery_id = 0
        last_error = "no attempts"
        last_body = ""
        # SSRF protection: private networks and loopback are intentionally
        # allowed (self-hosted admins routinely point webhooks at internal
        # services); link-local/reserved/multicast/unspecified — notably the
        # 169.254.169.254 cloud-metadata endpoint — are always blocked.
        from . import ssrf
        try:
            ssrf.validate_url(webhook.url, allow_localhost=True, allow_private=True)
        except ssrf.SSRFBlockedURL as e:
            last_delivery_id = memory.record_webhook_delivery(
                webhook_id=webhook.id,
                event_type=event_type,
                tenant_id=tenant_id,
                status_code=None,
                payload_json=body_str,
                response_body="",
                error=f"ssrf_blocked: {e.reason}",
                attempt=1,
                duration_ms=0.0,
            )
            return last_delivery_id
        for attempt in range(1, effective_retries + 1):
            headers = {
                "Content-Type": "application/json",
                "X-Gateway-Event-Type": event_type,
                "X-Gateway-Event-Id": str(payload.get("id", "")),
                "X-Gateway-Timestamp": str(int(time.time())),
                "X-Gateway-Delivery-Attempt": str(attempt),
            }
            if webhook.secret:
                headers["X-Gateway-Signature"] = _sign(webhook.secret, body_str)
            started = time.monotonic()
            try:
                async with session.post(
                    webhook.url,
                    data=body_str,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.delivery_timeout_seconds),
                ) as resp:
                    duration_ms = (time.monotonic() - started) * 1000.0
                    last_body = (await resp.text())[:4000]
                    success = 200 <= resp.status < 300
                    last_error = "" if success else f"HTTP {resp.status}"
                    last_delivery_id = memory.record_webhook_delivery(
                        webhook_id=webhook.id,
                        event_type=event_type,
                        tenant_id=tenant_id,
                        status_code=resp.status,
                        payload_json=body_str,
                        response_body=last_body,
                        error=last_error,
                        attempt=attempt,
                        duration_ms=duration_ms,
                    )
                    if success:
                        return last_delivery_id
            except (TimeoutError, aiohttp.ClientError) as e:
                duration_ms = (time.monotonic() - started) * 1000.0
                last_error = str(e)
                last_delivery_id = memory.record_webhook_delivery(
                    webhook_id=webhook.id,
                    event_type=event_type,
                    tenant_id=tenant_id,
                    status_code=None,
                    payload_json=body_str,
                    response_body="",
                    error=last_error[:1000],
                    attempt=attempt,
                    duration_ms=duration_ms,
                )
            if attempt < effective_retries:
                await asyncio.sleep(backoff)
                backoff *= self.backoff_multiplier
        log.warning(
            "webhook %s delivery failed after %d attempts: %s",
            webhook.name,
            effective_retries,
            last_error,
        )
        return last_delivery_id


_default_dispatcher: WebhookDispatcher | None = None


def init_dispatcher(
    max_retries: int = 3,
    initial_backoff_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    delivery_timeout_seconds: float = 10.0,
    max_concurrent_deliveries: int = 16,
) -> WebhookDispatcher:
    global _default_dispatcher
    _default_dispatcher = WebhookDispatcher(
        max_retries=max_retries,
        initial_backoff_seconds=initial_backoff_seconds,
        backoff_multiplier=backoff_multiplier,
        delivery_timeout_seconds=delivery_timeout_seconds,
        max_concurrent_deliveries=max_concurrent_deliveries,
    )
    return _default_dispatcher


def dispatcher() -> WebhookDispatcher | None:
    return _default_dispatcher


async def emit_event(
    event_type: str,
    payload: dict[str, Any],
    tenant_id: str | None = None,
) -> list[int]:
    """Convenience helper: dispatch a single event to all matching webhooks."""
    disp = dispatcher()
    if disp is None:
        return []
    return await disp.dispatch(event_type, payload, tenant_id=tenant_id)

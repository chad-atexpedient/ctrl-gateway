"""Endpoint HTTP clients.

Per-endpoint async aiohttp client with:
  - Per-endpoint semaphore (concurrency cap)
  - Per-endpoint circuit breaker integration
  - SSE streaming passthrough
  - Health probes

Endpoints are looked up by name from gateway-config.json. The active
client pool is rebuilt on config reload (atomic ref swap).
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator

import aiohttp

from . import circuit, transcoder
from . import config as cfg

log = logging.getLogger("ctrl.endpoints")


class EndpointHTTPError(RuntimeError):
    def __init__(self, endpoint: str, status: int, detail: str = ""):
        self.endpoint = endpoint
        self.status = status
        self.retryable = status in (408, 409, 429) or status >= 500
        super().__init__(f"endpoint {endpoint} HTTP {status}: {detail[:300]}")


class EndpointClient:
    """Wraps a single endpoint. Holds semaphore + breaker reference."""

    def __init__(self, cfg_dict: dict, firewall_enforcer=None):
        self.cfg = cfg_dict
        self.name = cfg_dict["name"]
        self.semaphore = asyncio.Semaphore(cfg_dict.get("concurrency", 4))
        self._breaker_config = circuit.BreakerConfig(
            failure_threshold=cfg_dict.get("breaker", {}).get("failure_threshold", 3),
            open_duration_seconds=cfg_dict.get("breaker", {}).get("open_duration_seconds", 60),
            half_open_max_probes=cfg_dict.get("breaker", {}).get("half_open_max_probes", 1),
        )
        self._inflight = 0
        self._last_used = 0.0
        self._lock = asyncio.Lock()
        self._session: aiohttp.ClientSession | None = None
        # Optional reference to the in-process firewall enforcer. app.py
        # sets this on the EndpointPool after both are constructed. `None`
        # means no firewall (default — keeps tests simple).
        self._firewall = firewall_enforcer

    def set_firewall(self, enforcer) -> None:
        """Attach the in-process firewall enforcer to this client."""
        self._firewall = enforcer

    def breaker(self) -> circuit.CircuitBreaker:
        return circuit.registry().get(self.name, self._breaker_config)

    def inflight(self) -> int:
        return self._inflight

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.cfg.get("timeout_seconds", 120))
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    def _check_firewall(self, url: str, tenant_id: str | None) -> None:
        """Run the in-process firewall against an outbound URL.

        Raises FirewallBlockedRequest when the request is not allowed.
        Tenant context is optional — defaults to "anonymous" when absent.
        """
        if self._firewall is None:
            return
        # Localhost / 127.0.0.1 is always allowed (loopback) — we don't
        # want to block the gateway from talking to mock endpoints, local
        # ollama, etc.
        from . import firewall as fw
        domain = fw.extract_domain(url)
        if domain in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
            return
        result = self._firewall.check_outbound(url, tenant_id or "anonymous")
        if not result.allowed:
            raise fw.FirewallBlockedRequest(
                endpoint=self.name,
                target_url=url,
                reason=result.reason,
                matched_pattern=result.matched_pattern,
            )

    async def health_check(self) -> bool:
        """Cheap probe. Returns True if endpoint reachable."""
        try:
            session = await self._get_session()
            probe = self.cfg.get("health_probe", "/health")
            url = f"{self.cfg['base_url'].rstrip('/')}{probe}"
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return resp.status < 500
        except Exception as e:
            log.debug("health check %s failed: %s", self.name, e)
            return False

    async def send(
        self,
        transcoded: transcoder.TranscodedRequest,
        stream: bool,
        tenant_id: str | None = None,
    ) -> dict | AsyncIterator[bytes]:
        """Send a request to the endpoint. Returns parsed JSON or async iterator of SSE chunks.

        For streaming, the returned async generator owns the semaphore + inflight
        counter for the entire stream lifetime (not just until the first chunk).
        """
        # Firewall check runs synchronously and BEFORE breaker/semaphore so
        # a blocked request doesn't deplete the breaker failure counter.
        self._check_firewall(transcoded.url, tenant_id)
        br = self.breaker()
        if not br.allow():
            raise RuntimeError(f"breaker open for endpoint {self.name}")
        if stream:
            return self._stream_send(transcoded, br)

        async with self.semaphore:
            self._inflight += 1
            self._last_used = time.time()
            try:
                session = await self._get_session()
                return await self._single_response(session, transcoded, br)
            finally:
                self._inflight -= 1

    async def _stream_send(
        self,
        transcoded: transcoder.TranscodedRequest,
        br: circuit.CircuitBreaker,
    ) -> AsyncIterator[bytes]:
        """Async generator that holds the semaphore + inflight for stream duration."""
        async with self.semaphore:
            self._inflight += 1
            self._last_used = time.time()
            try:
                session = await self._get_session()
                async for chunk in self._stream_response(session, transcoded, br):
                    yield chunk
            finally:
                self._inflight -= 1

    async def _single_response(
        self,
        session: aiohttp.ClientSession,
        req: transcoder.TranscodedRequest,
        br: circuit.CircuitBreaker,
    ) -> dict:
        try:
            async with session.post(
                req.url, headers=req.headers, json=req.body
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    error = EndpointHTTPError(self.name, resp.status, text)
                    if error.retryable:
                        br.record_failure()
                    raise error
                data = await resp.json()
                # Unwrap native response (Anthropic/Gemini) to OpenAI format
                if req.response_decoder:
                    data = req.response_decoder(data)
                br.record_success()
                return data
        except EndpointHTTPError:
            raise
        except Exception:
            br.record_failure()
            raise

    async def _stream_response(
        self,
        session: aiohttp.ClientSession,
        req: transcoder.TranscodedRequest,
        br: circuit.CircuitBreaker,
    ) -> AsyncIterator[bytes]:
        """SSE streaming. Yields raw bytes chunks."""
        # We need to release the semaphore inside the generator, so this is
        # actually structured as an async context that owns the semaphore
        # for the duration of the stream.
        try:
            async with session.post(
                req.url, headers=req.headers, json=req.body
            ) as resp:
                if resp.status >= 400:
                    text = await resp.text()
                    error = EndpointHTTPError(self.name, resp.status, text)
                    if error.retryable:
                        br.record_failure()
                    raise error
                source = resp.content.iter_any()
                stream = req.stream_decoder(source) if req.stream_decoder else source
                async for chunk in stream:
                    yield chunk
                br.record_success()
        except EndpointHTTPError:
            raise
        except Exception:
            br.record_failure()
            raise


class EndpointPool:
    """Manages all endpoint clients."""

    def __init__(self, firewall_enforcer=None):
        self._clients: dict[str, EndpointClient] = {}
        self._lock = asyncio.Lock()
        self._firewall = firewall_enforcer

    def set_firewall(self, enforcer) -> None:
        """Attach (or replace) the firewall enforcer and propagate to clients."""
        self._firewall = enforcer
        for c in self._clients.values():
            c.set_firewall(enforcer)

    async def rebuild(self, conf: cfg.Config):
        """Rebuild pool from config (atomic ref swap)."""
        new_clients = {}
        for ep_cfg in conf.config.get("endpoints", []):
            name = ep_cfg["name"]
            existing = self._clients.get(name)
            if existing and existing.cfg == ep_cfg:
                new_clients[name] = existing
                existing.set_firewall(self._firewall)
            else:
                if existing:
                    await existing.close()
                new_clients[name] = EndpointClient(ep_cfg, firewall_enforcer=self._firewall)
        async with self._lock:
            old = self._clients
            self._clients = new_clients
        for name, c in old.items():
            if name not in new_clients:
                await c.close()

    def get(self, name: str) -> EndpointClient:
        c = self._clients.get(name)
        if not c:
            raise KeyError(f"unknown endpoint '{name}'")
        return c

    def all_inflight(self) -> dict[str, int]:
        return {n: c.inflight() for n, c in self._clients.items()}

    def clients(self) -> dict[str, EndpointClient]:
        """Snapshot of current clients (name -> client)."""
        return dict(self._clients)

    async def close_all(self):
        for c in self._clients.values():
            await c.close()


_pool: EndpointPool | None = None


def init_pool() -> EndpointPool:
    global _pool
    _pool = EndpointPool()
    return _pool


def pool() -> EndpointPool:
    if _pool is None:
        raise RuntimeError("endpoint pool not initialized")
    return _pool


async def stream_passthrough(
    endpoint_name: str,
    transcoded: transcoder.TranscodedRequest,
    tenant_id: str | None = None,
) -> AsyncIterator[bytes]:
    """Wrap the stream so the semaphore is held for the entire stream lifetime.

    This is a helper for the app.py request handler.
    """
    client = pool().get(endpoint_name)
    result = await client.send(transcoded, stream=True, tenant_id=tenant_id)
    if not isinstance(result, AsyncIterator):
        raise RuntimeError(f"stream request to {endpoint_name} returned a non-stream response")
    async for chunk in result:
        yield chunk

"""SSRF (Server-Side Request Forgery) protection.

Validates outbound URLs before an HTTP request is made. Blocks:
  - Loopback addresses (127.0.0.0/8, ::1) — unless explicitly allowed
  - Private networks (10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16)
  - Link-local (169.254.0.0/16 — includes AWS/GCP/Azure metadata endpoints)
  - Reserved/undocumented ranges

Used by A2A agent invocation, ContextForge sync, MCP discovery, and webhook
delivery. The gateway-level firewall (DomainAllowlistEnforcer) is a separate
concern: it governs which LLM providers the gateway may talk to. This module
governs which admin-configured integration URLs (agents, webhooks, sync
sources) the gateway may reach.

Usage:
    from gateway.ssrf import validate_url, SSRFBlockedURL

    try:
        validate_url("https://api.example.com/webhook", allow_localhost=False)
    except SSRFBlockedURL:
        return error_403
"""
from __future__ import annotations

import ipaddress
import logging
import socket
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    # Only for static typing (see SSRFSafeResolver's docstring for why this
    # module avoids a hard runtime import of aiohttp) — `from __future__
    # import annotations` above means this name is never evaluated at
    # runtime, so the TYPE_CHECKING guard is what keeps it import-free.
    from aiohttp.abc import AbstractResolver

log = logging.getLogger("ctrl.ssrf")


class SSRFBlockedURL(Exception):
    """Raised when a URL targets a blocked/private/loopback address."""

    def __init__(self, url: str, reason: str):
        self.url = url
        self.reason = reason
        super().__init__(f"SSRF blocked: {url} ({reason})")


@dataclass
class SSRFConfig:
    allow_localhost: bool = False
    allow_private: bool = False
    allow_link_local: bool = False
    allowed_hosts: set[str] | None = None
    blocked_hosts: set[str] | None = None


def _resolve_host(host: str) -> list[str]:
    """Resolve a hostname to its IP addresses. Returns [] on failure."""
    try:
        infos = socket.getaddrinfo(host, None, proto=socket.IPPROTO_TCP)
        return list({str(info[4][0]) for info in infos})
    except (socket.gaierror, socket.herror, OSError):
        return []


def _is_blocked_ip(ip_str: str, cfg: SSRFConfig) -> str | None:
    """Return a reason string if the IP is blocked, None if allowed."""
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return f"invalid IP: {ip_str}"
    # Loopback — when explicitly allowed, skip subsequent reserved/private
    # checks too: ::1 is both is_loopback AND is_reserved, but it is the
    # legitimate IPv6 loopback — allowing localhost while blocking ::1
    # based on is_reserved would make the allow_localhost flag useless.
    if ip.is_loopback:
        if cfg.allow_localhost:
            return None
        return "loopback address blocked"
    # Link-local (169.254.x.x — cloud metadata) — checked independently of
    # (and before) the general private-network check below. Python's
    # ipaddress module classifies 169.254.0.0/16 as BOTH is_private AND
    # is_link_local, so if the is_private branch ran first it would block
    # link-local addresses whenever allow_private=False, even if
    # allow_link_local=True — making allow_link_local unreachable/useless
    # under the common policy (allow_private=False). Evaluating
    # is_link_local on its own branch first keeps the two flags truly
    # independent.
    if ip.is_link_local:
        if cfg.allow_link_local:
            return None
        return "link-local address blocked (cloud metadata)"
    # Private (RFC 1918)
    if ip.is_private and not cfg.allow_private:
        return "private network blocked"
    # Reserved/multicast — note: loopback already handled above
    if ip.is_reserved:
        return "reserved address blocked"
    if ip.is_multicast:
        return "multicast address blocked"
    if ip.is_unspecified:
        return "unspecified address blocked"
    return None


def validate_url(
    url: str,
    allow_localhost: bool = False,
    allow_private: bool = False,
    allow_link_local: bool = False,
    allowed_hosts: set[str] | None = None,
    blocked_hosts: set[str] | None = None,
) -> None:
    """Validate that a URL does not target a blocked address.

    Raises SSRFBlockedURL if the URL resolves to a disallowed IP.

    Args:
        url: the full URL to validate
        allow_localhost: permit 127.0.0.0/8 and ::1
        allow_private: permit RFC 1918 private ranges
        allow_link_local: permit 169.254.0.0/16 (cloud metadata)
        allowed_hosts: if set, ONLY these hostnames are permitted
        blocked_hosts: if set, these hostnames are always blocked
    """
    if not url:
        raise SSRFBlockedURL("(empty)", "empty URL")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise SSRFBlockedURL(url, f"scheme not allowed: {parsed.scheme}")
    host = parsed.hostname
    if not host:
        raise SSRFBlockedURL(url, "no hostname in URL")
    host_lower = host.lower()
    # Host-level allow/block lists
    if blocked_hosts and host_lower in {h.lower() for h in blocked_hosts}:
        raise SSRFBlockedURL(url, f"host explicitly blocked: {host_lower}")
    if allowed_hosts is not None and host_lower not in {h.lower() for h in allowed_hosts}:
        raise SSRFBlockedURL(url, f"host not in allowlist: {host_lower}")
    # Resolve and check IPs
    cfg = SSRFConfig(
        allow_localhost=allow_localhost,
        allow_private=allow_private,
        allow_link_local=allow_link_local,
    )
    # If the host is already an IP literal, check directly
    try:
        ipaddress.ip_address(host)
        ips = [host]
    except ValueError:
        ips = _resolve_host(host)
    if not ips:
        # Can't resolve — be conservative for non-IP hostnames
        # Only block if it looks like a raw IP that we couldn't parse
        if any(c.isdigit() for c in host) and "." in host:
            raise SSRFBlockedURL(url, f"unresolvable host: {host}")
        # Allow unresolvable hostnames (DNS may resolve later, or it may be a valid hostname)
        return
    for ip_str in ips:
        reason = _is_blocked_ip(ip_str, cfg)
        if reason:
            raise SSRFBlockedURL(url, reason)


class SSRFSafeResolver:
    """aiohttp DNS resolver wrapper that re-applies the SSRF policy to every
    resolved IP at the moment aiohttp is actually about to connect.

    validate_url() below checks a hostname's IPs once, ahead of the real
    request. If the target hostname's DNS is attacker-controlled, the
    attacker can let that check pass against a safe IP and then "rebind"
    the name to a blocked address (most dangerously the cloud-metadata
    endpoint, 169.254.169.254) before the real connection happens — a plain
    aiohttp.ClientSession performs its own independent resolution at connect
    time, so there is a real gap between check and use. Passing an instance
    of this class via TCPConnector(resolver=...) (see safe_connector())
    makes that second, real resolution the one that gets policy-checked, so
    there is no gap left to race.

    Implements aiohttp.abc.AbstractResolver's async duck-typed interface
    (resolve()/close()) without subclassing it, so this module has no hard
    dependency on aiohttp at import time.
    """

    def __init__(self, cfg: SSRFConfig | None = None, inner: AbstractResolver | None = None):
        self._cfg = cfg or SSRFConfig()
        self._inner: AbstractResolver
        if inner is not None:
            self._inner = inner
        else:
            from aiohttp.resolver import ThreadedResolver
            self._inner = ThreadedResolver()

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET):
        hosts = await self._inner.resolve(host, port, family)
        safe = []
        for entry in hosts:
            reason = _is_blocked_ip(entry["host"], self._cfg)
            if reason:
                log.warning("SSRF resolver blocked %s -> %s: %s", host, entry["host"], reason)
                continue
            safe.append(entry)
        if not safe:
            raise OSError(f"SSRF blocked: no safe address remained for {host}")
        return safe

    async def close(self) -> None:
        await self._inner.close()


def safe_connector(
    allow_localhost: bool = False,
    allow_private: bool = False,
    allow_link_local: bool = False,
    **connector_kwargs,
):
    """Build an aiohttp.TCPConnector whose resolver enforces the SSRF policy
    at actual connect time (see SSRFSafeResolver), closing the TOCTOU /
    DNS-rebinding gap that a validate_url() pre-check alone can't close.

    Use alongside validate_url() (not instead of it) at each call site:
    validate_url() gives a fast, specific SSRFBlockedURL with a clear reason
    before a connection is even attempted; this connector is the actual
    last-line enforcement at request time, in case DNS changed in between.
    """
    import aiohttp
    cfg = SSRFConfig(
        allow_localhost=allow_localhost,
        allow_private=allow_private,
        allow_link_local=allow_link_local,
    )
    return aiohttp.TCPConnector(resolver=SSRFSafeResolver(cfg), **connector_kwargs)


def redirect_guard_trace_config(
    allow_localhost: bool = False,
    allow_private: bool = False,
    allow_link_local: bool = False,
):
    """Build an aiohttp.TraceConfig that re-validates every redirect target
    against the SSRF policy before aiohttp follows it.

    Neither validate_url() (a one-time pre-check) nor safe_connector() /
    SSRFSafeResolver (re-checks at DNS-resolution time) catch a redirect: an
    admin-configured URL can be safe on the initial request, then have the
    remote server respond with a 3xx to a blocked target (e.g. the cloud
    metadata IP, or an IP-literal private address) and aiohttp will follow it
    by default. An IP-literal redirect target is the most dangerous case,
    because aiohttp special-cases IP-literal hosts and skips the custom
    resolver entirely for them — SSRFSafeResolver alone can never see that
    hop, so this needs an independent guard at the redirect signal itself.

    Note: aiohttp's on_request_redirect trace signal hands you the
    *pre-redirect* request URL in TraceRequestRedirectParams.url, not the
    redirect target — the actual target has to be read from the response's
    Location/URI header and resolved against the request URL (relative
    redirects are legal), the same way aiohttp's own client internals
    resolve it before following the hop.

    Raising SSRFBlockedURL from the trace handler propagates out of the
    request and aborts it — aiohttp trace signals run synchronously as part
    of the request lifecycle, so an exception raised here is not swallowed.
    """
    import aiohttp
    from yarl import URL

    cfg = SSRFConfig(
        allow_localhost=allow_localhost,
        allow_private=allow_private,
        allow_link_local=allow_link_local,
    )

    async def _on_request_redirect(session, trace_ctx, params) -> None:
        location = params.response.headers.get("Location") or params.response.headers.get("URI")
        if not location:
            return
        target = URL(params.url).join(URL(location))
        reason = None
        try:
            validate_url(
                str(target),
                allow_localhost=cfg.allow_localhost,
                allow_private=cfg.allow_private,
                allow_link_local=cfg.allow_link_local,
            )
        except SSRFBlockedURL as e:
            reason = e.reason
        if reason:
            log.warning("SSRF blocked redirect %s -> %s: %s", params.url, target, reason)
            raise SSRFBlockedURL(str(target), f"redirect target blocked: {reason}")

    trace_config = aiohttp.TraceConfig()
    trace_config.on_request_redirect.append(_on_request_redirect)
    return trace_config


def ssrf_client_kwargs(
    allow_localhost: bool = False,
    allow_private: bool = False,
    allow_link_local: bool = False,
    **extra,
) -> dict:
    """Bundle a safe connector + redirect guard into kwargs ready to splat
    into aiohttp.ClientSession(**ssrf.ssrf_client_kwargs(...)).

    This is the recommended way to construct any ClientSession that talks to
    an admin-configured URL (agents, webhooks, sync sources): it covers all
    three enforcement points together — the pre-check callers still do
    separately via validate_url(), connect-time DNS-rebinding protection via
    safe_connector()/SSRFSafeResolver, and post-response redirect-target
    validation via redirect_guard_trace_config(). Any extra ClientSession
    kwargs (timeout, headers, etc.) can be passed through via **extra.
    """
    kwargs: dict = dict(extra)
    kwargs["connector"] = safe_connector(
        allow_localhost=allow_localhost,
        allow_private=allow_private,
        allow_link_local=allow_link_local,
    )
    trace_configs = list(kwargs.get("trace_configs") or [])
    trace_configs.append(
        redirect_guard_trace_config(
            allow_localhost=allow_localhost,
            allow_private=allow_private,
            allow_link_local=allow_link_local,
        )
    )
    kwargs["trace_configs"] = trace_configs
    return kwargs


def safe_url(
    url: str,
    allow_localhost: bool = False,
    allow_private: bool = False,
    allow_link_local: bool = False,
) -> bool:
    """Return True if the URL is safe, False if blocked. Never raises."""
    try:
        validate_url(
            url,
            allow_localhost=allow_localhost,
            allow_private=allow_private,
            allow_link_local=allow_link_local,
        )
        return True
    except SSRFBlockedURL as e:
        log.warning("SSRF blocked: %s", e)
        return False

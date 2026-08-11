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
from urllib.parse import urlparse

log = logging.getLogger("glint.ssrf")


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
    # Private (RFC 1918)
    if ip.is_private and not cfg.allow_private:
        return "private network blocked"
    # Link-local (169.254.x.x — cloud metadata)
    if ip.is_link_local and not cfg.allow_link_local:
        return "link-local address blocked (cloud metadata)"
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

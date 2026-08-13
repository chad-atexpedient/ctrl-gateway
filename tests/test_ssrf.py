"""Tests for gateway/ssrf.py — SSRF (Server-Side Request Forgery) protection.

Covers the pre-existing validate_url()/_is_blocked_ip() policy checks (which
previously had no dedicated test coverage at all) plus SSRFSafeResolver /
safe_connector(): the fix for a TOCTOU (time-of-check-to-time-of-use) gap
where validate_url() resolved and approved a hostname once, ahead of the
real request, but the actual aiohttp connection later performed its own,
separate DNS resolution -- if the target hostname's DNS is attacker
-controlled, the attacker could let the check pass against a safe IP and
then "rebind" the name to a blocked address (most dangerously the cloud
-metadata endpoint, 169.254.169.254) before the real connection fired.
"""
from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gateway import ssrf


def _run(coro):
    # asyncio.run() creates and tears down its own fresh event loop rather
    # than depending on ambient/leftover global loop state -- other test
    # files in this suite (e.g. IntegrationBase) explicitly set the current
    # event loop to None in tearDown, which breaks asyncio.get_event_loop()
    # for any test that runs afterward in the same process.
    return asyncio.run(coro)


class IsBlockedIPTests(unittest.TestCase):
    """_is_blocked_ip(): the shared policy check used by both validate_url()
    (pre-check) and SSRFSafeResolver (actual-connect-time enforcement)."""

    def test_loopback_blocked_by_default(self):
        cfg = ssrf.SSRFConfig()
        self.assertIsNotNone(ssrf._is_blocked_ip("127.0.0.1", cfg))
        self.assertIsNotNone(ssrf._is_blocked_ip("::1", cfg))

    def test_loopback_allowed_when_configured(self):
        cfg = ssrf.SSRFConfig(allow_localhost=True)
        self.assertIsNone(ssrf._is_blocked_ip("127.0.0.1", cfg))
        # ::1 is both is_loopback AND is_reserved; allow_localhost must cover
        # both or the flag would be useless for IPv6 loopback.
        self.assertIsNone(ssrf._is_blocked_ip("::1", cfg))

    def test_private_blocked_by_default(self):
        cfg = ssrf.SSRFConfig()
        self.assertIsNotNone(ssrf._is_blocked_ip("10.0.0.5", cfg))
        self.assertIsNotNone(ssrf._is_blocked_ip("192.168.1.1", cfg))
        self.assertIsNotNone(ssrf._is_blocked_ip("172.16.0.1", cfg))

    def test_private_allowed_when_configured(self):
        cfg = ssrf.SSRFConfig(allow_private=True)
        self.assertIsNone(ssrf._is_blocked_ip("10.0.0.5", cfg))

    def test_link_local_blocked_even_with_private_and_localhost_allowed(self):
        """169.254.169.254 (cloud metadata) must stay blocked under the
        exact policy every real call site in this codebase uses
        (allow_localhost=True, allow_private=True, allow_link_local=False)."""
        cfg = ssrf.SSRFConfig(allow_localhost=True, allow_private=True)
        reason = ssrf._is_blocked_ip("169.254.169.254", cfg)
        self.assertIsNotNone(reason)
        self.assertIn("link-local", reason)

    def test_link_local_allowed_when_configured(self):
        # Python's ipaddress module classifies 169.254.0.0/16 as BOTH
        # is_private and is_link_local (it's in IANA's special-purpose
        # registry under both). _is_blocked_ip checks is_private first, so
        # allow_link_local alone isn't sufficient to unblock it -- allow_private
        # must also be set. This matches every real call site in this codebase
        # (none use allow_link_local=True with allow_private=False), so it's
        # not exercised as a bug fix here, just pinned as documented behavior.
        cfg = ssrf.SSRFConfig(allow_link_local=True, allow_private=True)
        self.assertIsNone(ssrf._is_blocked_ip("169.254.169.254", cfg))

    def test_public_ip_always_allowed(self):
        cfg = ssrf.SSRFConfig()
        self.assertIsNone(ssrf._is_blocked_ip("93.184.216.34", cfg))

    def test_multicast_and_unspecified_blocked(self):
        cfg = ssrf.SSRFConfig(allow_localhost=True, allow_private=True, allow_link_local=True)
        self.assertIsNotNone(ssrf._is_blocked_ip("224.0.0.1", cfg))
        self.assertIsNotNone(ssrf._is_blocked_ip("0.0.0.0", cfg))

    def test_invalid_ip_string_blocked(self):
        cfg = ssrf.SSRFConfig()
        self.assertIsNotNone(ssrf._is_blocked_ip("not-an-ip", cfg))


class ValidateUrlTests(unittest.TestCase):
    def test_rejects_non_http_scheme(self):
        with self.assertRaises(ssrf.SSRFBlockedURL):
            ssrf.validate_url("file:///etc/passwd")

    def test_rejects_empty_url(self):
        with self.assertRaises(ssrf.SSRFBlockedURL):
            ssrf.validate_url("")

    def test_ip_literal_loopback_blocked_by_default(self):
        with self.assertRaises(ssrf.SSRFBlockedURL):
            ssrf.validate_url("http://127.0.0.1:8080/x")

    def test_ip_literal_loopback_allowed_when_configured(self):
        ssrf.validate_url("http://127.0.0.1:8080/x", allow_localhost=True)  # no raise

    def test_ip_literal_metadata_blocked_even_with_private_and_localhost_allowed(self):
        with self.assertRaises(ssrf.SSRFBlockedURL):
            ssrf.validate_url(
                "http://169.254.169.254/latest/meta-data/",
                allow_localhost=True, allow_private=True,
            )

    def test_blocked_hosts_list(self):
        with self.assertRaises(ssrf.SSRFBlockedURL):
            ssrf.validate_url(
                "http://example.com/x",
                allow_localhost=True, allow_private=True,
                blocked_hosts={"example.com"},
            )

    def test_allowed_hosts_list_rejects_others(self):
        with self.assertRaises(ssrf.SSRFBlockedURL):
            ssrf.validate_url(
                "http://not-allowed.example.com/x",
                allowed_hosts={"allowed.example.com"},
            )

    def test_safe_url_never_raises(self):
        self.assertFalse(ssrf.safe_url("http://127.0.0.1/x"))
        self.assertTrue(ssrf.safe_url("http://127.0.0.1/x", allow_localhost=True))


class _FakeInnerResolver:
    """Stand-in for aiohttp's ThreadedResolver: returns whatever
    ResolveResult-shaped dicts the test configured, without touching real
    DNS. Also records whether close() was delegated to."""

    def __init__(self, entries: list[dict]):
        self._entries = entries
        self.closed = False

    async def resolve(self, host: str, port: int = 0, family: int = 0) -> list[dict]:
        return list(self._entries)

    async def close(self) -> None:
        self.closed = True


def _entry(ip: str, hostname: str = "example.test") -> dict:
    return {"hostname": hostname, "host": ip, "port": 443, "family": 0, "proto": 0, "flags": 0}


class SSRFSafeResolverTests(unittest.TestCase):
    """The actual TOCTOU fix: SSRFSafeResolver re-applies the SSRF policy at
    the same moment aiohttp performs its real, connect-time DNS resolution,
    instead of relying solely on validate_url()'s earlier, separate check."""

    def test_passes_through_safe_public_ip(self):
        inner = _FakeInnerResolver([_entry("93.184.216.34")])
        resolver = ssrf.SSRFSafeResolver(inner=inner)
        result = _run(resolver.resolve("example.test"))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["host"], "93.184.216.34")

    def test_blocks_link_local_metadata_ip_under_real_call_site_policy(self):
        """Matches the exact SSRFConfig every real call site uses
        (allow_localhost=True, allow_private=True, allow_link_local=False)."""
        cfg = ssrf.SSRFConfig(allow_localhost=True, allow_private=True)
        inner = _FakeInnerResolver([_entry("169.254.169.254")])
        resolver = ssrf.SSRFSafeResolver(cfg, inner=inner)
        with self.assertRaises(OSError):
            _run(resolver.resolve("example.test"))

    def test_filters_mixed_results_keeping_only_safe_entries(self):
        cfg = ssrf.SSRFConfig(allow_localhost=True, allow_private=True)
        inner = _FakeInnerResolver([_entry("169.254.169.254"), _entry("8.8.8.8")])
        resolver = ssrf.SSRFSafeResolver(cfg, inner=inner)
        result = _run(resolver.resolve("example.test"))
        self.assertEqual([e["host"] for e in result], ["8.8.8.8"])

    def test_allows_private_ip_when_configured(self):
        cfg = ssrf.SSRFConfig(allow_private=True)
        inner = _FakeInnerResolver([_entry("10.0.0.5")])
        resolver = ssrf.SSRFSafeResolver(cfg, inner=inner)
        result = _run(resolver.resolve("internal.test"))
        self.assertEqual(result[0]["host"], "10.0.0.5")

    def test_close_delegates_to_inner(self):
        inner = _FakeInnerResolver([])
        resolver = ssrf.SSRFSafeResolver(inner=inner)
        _run(resolver.close())
        self.assertTrue(inner.closed)

    def test_toctou_scenario_is_closed(self):
        """The scenario this bug fix exists for: a validate_url() pre-check
        resolves and approves a hostname against a safe IP; DNS then
        "rebinds" the same hostname to a blocked IP (169.254.169.254)
        before the real connection fires. The pre-check alone can't see
        that -- it already ran and passed. SSRFSafeResolver represents
        aiohttp's actual, later, connect-time resolution, which must
        independently catch the now-blocked address.
        """
        # "Check time": validate_url() resolves and approves a safe IP.
        original_resolve_host = ssrf._resolve_host
        ssrf._resolve_host = lambda host: ["93.184.216.34"]
        try:
            ssrf.validate_url(
                "http://rebind.example.test/", allow_localhost=True, allow_private=True,
            )  # passes -- no raise
        finally:
            ssrf._resolve_host = original_resolve_host

        # "Connect time": DNS has since rebound to the metadata endpoint.
        # This is what TCPConnector(resolver=...) would now see.
        cfg = ssrf.SSRFConfig(allow_localhost=True, allow_private=True)
        inner = _FakeInnerResolver([_entry("169.254.169.254", hostname="rebind.example.test")])
        resolver = ssrf.SSRFSafeResolver(cfg, inner=inner)
        with self.assertRaises(OSError):
            _run(resolver.resolve("rebind.example.test"))


class SafeConnectorTests(unittest.TestCase):
    def test_builds_tcp_connector_with_safe_resolver(self):
        import aiohttp

        async def build():
            connector = ssrf.safe_connector(allow_localhost=True, allow_private=True)
            try:
                self.assertIsInstance(connector, aiohttp.TCPConnector)
                self.assertIsInstance(connector._resolver, ssrf.SSRFSafeResolver)
                self.assertTrue(connector._resolver._cfg.allow_localhost)
                self.assertTrue(connector._resolver._cfg.allow_private)
                self.assertFalse(connector._resolver._cfg.allow_link_local)
            finally:
                await connector.close()

        _run(build())


if __name__ == "__main__":
    unittest.main()

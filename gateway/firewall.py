"""Gateway-level firewall + host-level firewall manager.

Layered defense:

  1. **DomainAllowlistEnforcer** (in-process): intercepts every outbound
     HTTP request from the gateway and rejects/flags requests destined
     for domains not in the admin-approved allowlist. Pure Python — no
     system changes — works in containers, runs everywhere.

  2. **HostFirewallManager** (system-level): on a host with admin/root
     privileges, syncs Windows Firewall (netsh) or iptables rules so
     that even non-gateway processes on the host cannot reach blocked
     domains. Optional; disabled by default.

Domain matching: supports exact matches (`api.openai.com`) and wildcard
prefixes (`*.anthropic.com` matches `api.anthropic.com` and
`console.anthropic.com`, but NOT `anthropic.com` itself — wildcard
domains must have a non-empty subdomain).
"""
from __future__ import annotations

import ipaddress
import logging
import re
import socket
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Literal
from urllib.parse import urlparse

log = logging.getLogger("glint.firewall")


# ---------------------------------------------------------------------------
# Domain pattern matching
# ---------------------------------------------------------------------------


class DomainPatternError(ValueError):
    """Raised when a domain pattern is malformed."""


def _validate_pattern(pattern: str) -> str:
    """Validate and normalize a domain pattern. Returns it lowercased.

    Accepts:
      - exact: api.openai.com
      - wildcard: *.anthropic.com

    Rejects anything with scheme, path, or empty label.
    """
    p = pattern.strip().lower()
    if not p:
        raise DomainPatternError("empty domain pattern")
    if "/" in p or ":" in p or "@" in p:
        raise DomainPatternError(f"invalid domain pattern: {pattern!r}")
    labels = p.split(".")
    if any(not label for label in labels):
        raise DomainPatternError(f"invalid domain pattern: {pattern!r}")
    if not all(re.fullmatch(r"[a-z0-9-]{1,63}", label.replace("*", "a")) for label in labels):
        raise DomainPatternError(f"invalid domain pattern: {pattern!r}")
    return p


def domain_matches(pattern: str, host: str) -> bool:
    """Return True if `host` matches `pattern` (exact or *.wildcard)."""
    pat = pattern.strip().lower()
    h = host.strip().lower()
    if not pat or not h:
        return False
    if pat.startswith("*."):
        # Wildcard: matches any subdomain of the rest
        suffix = pat[2:]
        if not suffix:
            return False
        # h must END with .suffix and have at least one subdomain label
        if h.endswith("." + suffix) and h != "." + suffix:
            return True
        return False
    # Exact match
    return pat == h


def extract_domain(url: str) -> str:
    """Extract the hostname (domain) from a URL. Returns '' on failure."""
    try:
        parsed = urlparse(url)
        return (parsed.hostname or "").lower()
    except Exception:
        return ""


def resolve_domain_to_ips(domain: str) -> list[str]:
    """Resolve a domain to its IPs via DNS. Returns [] on failure."""
    try:
        infos = socket.getaddrinfo(domain, None)
    except (socket.gaierror, Exception):
        return []
    seen = set()
    out: list[str] = []
    for info in infos:
        try:
            addr = info[4][0]
            # addr may be str (IPv4/v6 hostname) or tuple (sockaddr)
            if isinstance(addr, tuple):
                addr = addr[0]
        except (IndexError, TypeError):
            continue
        if not isinstance(addr, str):
            continue
        if addr in seen:
            continue
        seen.add(addr)
        out.append(addr)
    return out


# ---------------------------------------------------------------------------
# DomainAllowlistEnforcer (in-process)
# ---------------------------------------------------------------------------


@dataclass
class AllowlistRule:
    pattern: str
    action: Literal["allow", "block"]
    tenant_id: str
    notes: str | None = None


@dataclass
class BlockResult:
    allowed: bool
    action: Literal["allow", "block"]
    matched_pattern: str | None = None
    reason: str = ""


@dataclass
class AllowlistStats:
    checks_total: int = 0
    blocks_total: int = 0
    alerts_total: int = 0


def _default_global_patterns_from_endpoints(endpoints: list[dict]) -> list[str]:
    """Build the default global allowlist from currently configured endpoints.

    Each endpoint's base_url is parsed for its domain. This means the
    default enforcer allows exactly the providers the admin has configured
    — and blocks everything else.
    """
    seen: set[str] = set()
    out: list[str] = []
    for ep in endpoints or []:
        url = ep.get("base_url") if isinstance(ep, dict) else None
        if not url:
            continue
        domain = extract_domain(url)
        if not domain or domain in seen:
            continue
        seen.add(domain)
        out.append(domain)
    return out


class DomainAllowlistEnforcer:
    """In-process firewall that enforces a per-tenant + global domain allowlist.

    Rule precedence (most specific wins):
      1. tenant_id-specific rule for the matched domain
      2. global rule (tenant_id="*") for the matched domain
      3. default_action

    `enabled=False` short-circuits everything to allowed (no enforcement).
    """

    def __init__(self, enabled: bool = False, default_action: str = "block"):
        self.enabled = bool(enabled)
        self.default_action = default_action if default_action in ("allow", "block") else "block"
        self._rules: list[AllowlistRule] = []
        self._by_pattern: dict[str, AllowlistRule] = {}
        self.stats = AllowlistStats()

    def load_from_config(self, config: dict) -> None:
        """Reload rules from the gateway-config.json security.provider_allowlist section."""
        self.enabled = bool(config.get("enabled", False))
        self.default_action = config.get("default_action", "block")
        if self.default_action not in ("allow", "block"):
            self.default_action = "block"
        self._rules.clear()
        seen_keys: set[tuple[str, str]] = set()

        # Global patterns (tenant_id="*")
        for pat in config.get("global_patterns", []) or []:
            try:
                normalized = _validate_pattern(pat)
            except DomainPatternError as e:
                log.warning("skipping invalid global pattern %r: %s", pat, e)
                continue
            key = ("*", normalized)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            self._rules.append(AllowlistRule(
                pattern=normalized, action="allow", tenant_id="*",
                notes="global_default",
            ))

        # Tenant overrides
        for tenant_id, override in (config.get("tenant_overrides", {}) or {}).items():
            action = override.get("action", "block")
            if action not in ("allow", "block"):
                action = "block"
            for pat in override.get("patterns", []) or []:
                try:
                    normalized = _validate_pattern(pat)
                except DomainPatternError as e:
                    log.warning("skipping invalid tenant pattern %r: %s", pat, e)
                    continue
                key = (tenant_id, normalized)
                if key in seen_keys:
                    continue
                seen_keys.add(key)
                self._rules.append(AllowlistRule(
                    pattern=normalized, action=action, tenant_id=tenant_id,
                ))

        # Rebuild index
        self._by_pattern = {
            f"{r.tenant_id}::{r.pattern}": r for r in self._rules
        }

    def load_from_db(self, db_rows: list[dict]) -> None:
        """Layer in DB-backed rules (typically per-tenant). tenant_id='*' is global.

        Replaces any rules previously loaded from the DB (config-loaded rules
        are preserved). DB rules take precedence over config rules of the
        same key because they appear later in `self._rules`.
        """
        # Build a set of (tenant, pattern) keys for config rules so we can
        # tell which entries came from the DB.
        db_keys: set[tuple[str, str]] = set()
        new_rules: list[AllowlistRule] = []
        for row in db_rows or []:
            pattern = row.get("domain_pattern")
            tenant_id = row.get("tenant_id")
            action = row.get("action", "allow")
            if not pattern or not tenant_id:
                continue
            if action not in ("allow", "block"):
                continue
            try:
                normalized = _validate_pattern(pattern)
            except DomainPatternError:
                continue
            key = (tenant_id, normalized)
            db_keys.add(key)
            rule = AllowlistRule(
                pattern=normalized, action=action, tenant_id=tenant_id,
                notes=row.get("notes"),
            )
            new_rules.append(rule)
            self._by_pattern[f"{tenant_id}::{normalized}"] = rule
        # Keep config rules that aren't shadowed by DB rules, then append DB rules
        kept = [r for r in self._rules if (r.tenant_id, r.pattern) not in db_keys]
        self._rules = kept + new_rules

    def check_outbound(self, url: str, tenant_id: str) -> BlockResult:
        """Check whether a request to `url` is allowed for the given tenant.

        Returns:
          - BlockResult(allowed=True, action='allow', ...) if the request
            is allowed.
          - BlockResult(allowed=False, action='block', matched_pattern=...,
            reason=...) if the request is blocked.
        """
        self.stats.checks_total += 1
        if not self.enabled:
            return BlockResult(allowed=True, action="allow", reason="enforcer_disabled")

        host = extract_domain(url)
        if not host:
            # Cannot extract hostname — fail closed (block) under default_action=block,
            # fail open under default_action=allow.
            if self.default_action == "block":
                self.stats.blocks_total += 1
                return BlockResult(
                    allowed=False, action="block", reason="unparseable_hostname",
                )
            return BlockResult(allowed=True, action="allow", reason="unparseable_hostname_default_allow")

        # 1. Tenant-specific rule
        tenant_rule = self._match_for(host, tenant_id)
        if tenant_rule is not None:
            return self._to_result(tenant_rule, host)

        # 2. Global rule
        global_rule = self._match_for(host, "*")
        if global_rule is not None:
            return self._to_result(global_rule, host)

        # 3. Default
        if self.default_action == "block":
            self.stats.blocks_total += 1
            return BlockResult(
                allowed=False, action="block", reason="not_in_allowlist",
            )
        return BlockResult(allowed=True, action="allow", reason="default_allow")

    def _match_for(self, host: str, tenant_id: str) -> AllowlistRule | None:
        """Find the first matching rule for a (tenant_id, host) pair."""
        # Tenant-specific rules first
        for rule in self._rules:
            if rule.tenant_id == tenant_id and domain_matches(rule.pattern, host):
                return rule
        # Global rules (tenant_id="*")
        for rule in self._rules:
            if rule.tenant_id == "*" and domain_matches(rule.pattern, host):
                return rule
        return None

    def _to_result(self, rule: AllowlistRule, host: str) -> BlockResult:
        if rule.action == "block":
            self.stats.blocks_total += 1
            return BlockResult(
                allowed=False, action="block", matched_pattern=rule.pattern,
                reason=f"domain {host} blocked by rule",
            )
        return BlockResult(
            allowed=True, action="allow", matched_pattern=rule.pattern,
            reason=f"domain {host} allowed by rule",
        )

    def list_rules(self) -> list[AllowlistRule]:
        return list(self._rules)


# ---------------------------------------------------------------------------
# Host-level firewall (Windows netsh / Linux iptables)
# ---------------------------------------------------------------------------


class HostFirewallUnsupported(RuntimeError):
    """Raised when the platform doesn't support our firewall implementations."""


class HostFirewallUnavailable(RuntimeError):
    """Raised when the host firewall can't be modified (no privileges)."""


@dataclass
class HostFirewallRule:
    pattern: str
    ip: str
    rule_name: str


@dataclass
class HostFirewallState:
    enabled: bool = False
    platform: str = "auto"
    last_sync_at: float = 0.0
    last_sync_error: str | None = None
    rules: list[HostFirewallRule] = field(default_factory=list)
    in_sync: bool = False


def _detect_platform() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    if sys.platform == "darwin":
        return "macos"
    return "unknown"


def _make_rule_name(prefix: str, pattern: str) -> str:
    """Windows netsh rule names are limited to ~64 chars. Use a stable hash."""
    import hashlib
    h = hashlib.sha256(pattern.encode()).hexdigest()[:10]
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", pattern)[:30]
    return f"{prefix}_{safe}_{h}"


def _is_admin() -> bool:
    """Best-effort check for admin/root privileges."""
    try:
        if sys.platform.startswith("win"):
            import ctypes
            return bool(ctypes.windll.shell32.IsUserAnAdmin())
        if sys.platform.startswith("linux") or sys.platform == "darwin":
            return os.geteuid() == 0
    except Exception:
        return False
    return False


# Top-level import for `os` (we need it for both Windows and POSIX).
import os  # noqa: E402


def _run(cmd: list[str], timeout: float = 15.0) -> tuple[int, str, str]:
    """Run a subprocess with timeout. Returns (rc, stdout, stderr)."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as e:
        return 1, "", f"timeout: {e}"
    except Exception as e:
        return 1, "", str(e)


class HostFirewallManager:
    """Manages host-level firewall rules for blocked provider domains.

    Windows: uses `netsh advfirewall firewall add rule` to block outbound traffic.
    Linux: uses `iptables -A OUTPUT` to block outbound traffic.

    Calls are idempotent on rule name. Failed syncs are logged but never
    raise from `sync()` — the enforcer still works at the gateway level.
    """

    RULE_PREFIX = "Glint-Block"

    def __init__(self, enabled: bool = False, platform: str = "auto",
                 persist_on_shutdown: bool = False):
        self.enabled = bool(enabled)
        self.platform = platform if platform != "auto" else _detect_platform()
        self.persist_on_shutdown = bool(persist_on_shutdown)
        self.state = HostFirewallState(
            enabled=self.enabled,
            platform=self.platform,
        )

    def sync(self, patterns: Iterable[str]) -> None:
        """Sync host firewall rules to match `patterns` (block all these domains).

        Idempotent. Errors are captured into self.state.last_sync_error.
        """
        self.state.last_sync_at = _now_epoch()
        if not self.enabled:
            self.state.in_sync = True
            self.state.last_sync_error = "disabled"
            return
        if not _is_admin():
            self.state.last_sync_error = "insufficient privileges (need admin/root)"
            self.state.in_sync = False
            log.warning("host firewall disabled: %s", self.state.last_sync_error)
            return

        # Resolve domains to IPs and create rules
        new_rules: list[HostFirewallRule] = []
        for pat in patterns:
            if pat.startswith("*."):
                base = pat[2:]
                ips = resolve_domain_to_ips(base)
            else:
                ips = resolve_domain_to_ips(pat)
            for ip in ips:
                try:
                    ipaddress.ip_address(ip)
                except ValueError:
                    continue
                new_rules.append(HostFirewallRule(
                    pattern=pat,
                    ip=ip,
                    rule_name=_make_rule_name(self.RULE_PREFIX, pat + "|" + ip),
                ))

        # Sync: remove rules not in new set, add missing ones
        try:
            if self.platform == "windows":
                self._sync_windows(new_rules)
            elif self.platform == "linux":
                self._sync_linux(new_rules)
            else:
                self.state.last_sync_error = f"unsupported platform: {self.platform}"
                self.state.in_sync = False
                return
            self.state.rules = new_rules
            self.state.in_sync = True
            self.state.last_sync_error = None
        except Exception as e:
            self.state.last_sync_error = str(e)
            self.state.in_sync = False
            log.warning("host firewall sync failed: %s", e)

    def clear(self) -> None:
        """Remove all managed rules (best-effort). Safe to call on shutdown."""
        if not self.enabled:
            return
        if not self.state.rules:
            return
        try:
            if self.platform == "windows":
                self._clear_windows()
            elif self.platform == "linux":
                self._clear_linux()
            self.state.rules = []
            self.state.in_sync = True
        except Exception as e:
            log.warning("host firewall clear failed: %s", e)
            self.state.last_sync_error = str(e)

    # ---- Windows implementation ----

    def _sync_windows(self, new_rules: list[HostFirewallRule]) -> None:
        existing = self._list_windows_rules()
        existing_by_name = {r.rule_name: r for r in existing}
        target_names = {r.rule_name for r in new_rules}
        # Remove managed rules no longer in target
        for name in existing_by_name:
            if name not in target_names:
                self._delete_windows_rule(name)
        # Add new ones
        for r in new_rules:
            if r.rule_name not in existing_by_name:
                self._add_windows_rule(r)

    def _list_windows_rules(self) -> list[HostFirewallRule]:
        rc, stdout, stderr = _run([
            "netsh", "advfirewall", "firewall", "show", "rule", f"name={self.RULE_PREFIX}",
        ])
        if rc != 0:
            return []
        out: list[HostFirewallRule] = []
        # Parse: "Rule Name: Glint-Block_xxx" and "Remote IP: 1.2.3.4"
        name = None
        ip = None
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("Rule Name:"):
                name = line.split(":", 1)[1].strip()
            elif line.startswith("Remote IP:"):
                ip = line.split(":", 1)[1].strip()
                if name and ip:
                    out.append(HostFirewallRule(pattern="", rule_name=name, ip=ip))
                name = None
                ip = None
        return out

    def _add_windows_rule(self, rule: HostFirewallRule) -> None:
        _run([
            "netsh", "advfirewall", "firewall", "add", "rule",
            f"name={rule.rule_name}",
            "dir=out",
            "action=block",
            f"remoteip={rule.ip}",
        ])

    def _delete_windows_rule(self, name: str) -> None:
        _run([
            "netsh", "advfirewall", "firewall", "delete", "rule", f"name={name}",
        ])

    def _clear_windows(self) -> None:
        for r in list(self.state.rules):
            self._delete_windows_rule(r.rule_name)

    # ---- Linux implementation ----

    def _sync_linux(self, new_rules: list[HostFirewallRule]) -> None:
        # We use a comment tag to identify our rules.
        current = self._list_linux_rules()
        current_by_name = {r.rule_name: r for r in current}
        target_names = {r.rule_name for r in new_rules}
        for name in current_by_name:
            if name not in target_names:
                self._delete_linux_rule(name)
        for r in new_rules:
            if r.rule_name not in current_by_name:
                self._add_linux_rule(r)

    def _list_linux_rules(self) -> list[HostFirewallRule]:
        rc, stdout, stderr = _run(["iptables", "-L", "OUTPUT", "-n", "--line-numbers"])
        if rc != 0:
            return []
        out: list[HostFirewallRule] = []
        for line in stdout.splitlines():
            # Look for our comment marker
            if self.RULE_PREFIX not in line:
                continue
            # Parse: "1  DROP  all  --  *  *  0.0.0.0/0  1.2.3.4  /* Glint-Block_xxx_xxx */"
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                comment_idx = next(
                    i for i, p in enumerate(parts) if p.startswith(f"/*{self.RULE_PREFIX}")
                )
            except StopIteration:
                continue
            comment = parts[comment_idx].strip("/ *")
            ip = parts[parts.index("0.0.0.0/0") + 1]
            out.append(HostFirewallRule(pattern="", rule_name=comment, ip=ip))
        return out

    def _add_linux_rule(self, rule: HostFirewallRule) -> None:
        _run([
            "iptables", "-A", "OUTPUT",
            "-d", rule.ip,
            "-j", "DROP",
            "-m", "comment", "--comment", rule.rule_name,
        ])

    def _delete_linux_rule(self, name: str) -> None:
        _run([
            "iptables", "-D", "OUTPUT",
            "-j", "DROP",
            "-m", "comment", "--comment", name,
        ])

    def _clear_linux(self) -> None:
        for r in list(self.state.rules):
            self._delete_linux_rule(r.rule_name)


def _now_epoch() -> float:
    import time
    return time.time()


# ---------------------------------------------------------------------------
# FirewallBlockedRequest exception (raised by endpoints.py)
# ---------------------------------------------------------------------------


class FirewallBlockedRequest(RuntimeError):
    """Raised when an outbound request is blocked by the in-process firewall."""

    def __init__(self, endpoint: str, target_url: str, reason: str,
                 matched_pattern: str | None = None):
        self.endpoint = endpoint
        self.target_url = target_url
        self.reason = reason
        self.matched_pattern = matched_pattern
        super().__init__(f"firewall blocked request to {target_url}: {reason}")

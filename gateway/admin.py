"""Live admin operations: add/edit/remove providers, generate/revoke API keys,
edit tiers — all persisted to a runtime overlay file (gateway-config.runtime.json)
that survives restarts without mutating the base config.

Every mutation triggers a config reload + endpoint pool rebuild + auth refresh,
so changes take effect immediately.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import threading
from pathlib import Path

log = logging.getLogger("ctrl.admin")


class OverlayManager:
    """Manages runtime config mutations via an overlay file.

    The overlay is a JSON file with the same structure as gateway-config.json.
    On reload, the overlay is merged onto the base config (dicts merge one level
    deep; lists replace entirely). Admin mutations write to the overlay, then
    trigger a reload so the gateway picks up the change immediately.
    """

    def __init__(self, conf_mgr, overlay_path: str | Path | None = None):
        self.conf_mgr = conf_mgr
        self.overlay_path = Path(overlay_path or conf_mgr.overlay_path)
        self._lock = threading.RLock()  # RLock: add_endpoint calls _set_section which also locks

    # ---- overlay file I/O ----

    def _read_overlay(self) -> dict:
        if self.overlay_path.exists():
            try:
                return json.loads(self.overlay_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
        return {}

    def _write_overlay(self, data: dict):
        self.overlay_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.overlay_path.with_suffix(self.overlay_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp_path, self.overlay_path)

    def _set_section(self, key: str, value):
        """Replace a top-level section in the overlay."""
        with self._lock:
            overlay = self._read_overlay()
            overlay[key] = value
            self._write_overlay(overlay)

    def _set_nested(self, section: str, key: str, value):
        """Set a nested key within a section."""
        with self._lock:
            overlay = self._read_overlay()
            overlay.setdefault(section, {})[key] = value
            self._write_overlay(overlay)

    def _reload(self):
        """Trigger config reload (re-reads base + overlay)."""
        self.conf_mgr.reload()

    def _ensure_mutable(self):
        if self._config.get("mode") == "multi":
            raise ValueError("runtime file mutations are disabled in multi-instance mode; update shared config instead")

    # ---- current state helpers ----

    @property
    def _config(self) -> dict:
        return self.conf_mgr.current().config

    @property
    def _endpoints(self) -> list[dict]:
        return json.loads(json.dumps(self._config.get("endpoints", [])))

    @property
    def _auth_keys(self) -> dict:
        return json.loads(json.dumps(self._config.get("auth", {}).get("keys", {})))

    @property
    def _tiers(self) -> list[dict]:
        return json.loads(json.dumps(self._config.get("tiers", [])))

    # ---- endpoint CRUD ----

    def add_endpoint(self, ep: dict) -> dict:
        """Add a new provider endpoint. Fills sensible defaults."""
        with self._lock:
            self._ensure_mutable()
            ep = self._validated_endpoint(ep)
            names = {e["name"] for e in self._endpoints}
            if ep["name"] in names:
                raise ValueError(f"endpoint '{ep['name']}' already exists")
            ep.setdefault("model_alias", ep.get("name", "default"))
            ep.setdefault("concurrency", 4)
            ep.setdefault("max_context", 32768)
            ep.setdefault("pricing", {"in_per_1k_tokens": 0.0, "out_per_1k_tokens": 0.0, "fixed_per_request": 0.0})
            ep.setdefault("breaker", {"failure_threshold": 3, "open_duration_seconds": 60, "half_open_max_probes": 1})
            ep.setdefault("health_probe", "/health")
            endpoints = self._endpoints + [ep]
            self._set_section("endpoints", endpoints)
            self._reload()
            log.info("endpoint added: %s (%s @ %s)", ep["name"], ep["kind"], ep["base_url"])
            return ep

    def update_endpoint(self, name: str, updates: dict) -> dict:
        """Edit an existing endpoint by name."""
        with self._lock:
            self._ensure_mutable()
            endpoints = self._endpoints
            found = False
            for i, ep in enumerate(endpoints):
                if ep["name"] == name:
                    endpoints[i] = self._validated_endpoint(
                        {**ep, **{k: v for k, v in updates.items() if k != "name"}}
                    )
                    found = True
                    break
            if not found:
                raise KeyError(f"endpoint '{name}' not found")
            self._set_section("endpoints", endpoints)
            self._reload()
            log.info("endpoint updated: %s", name)
            return endpoints[i]

    @staticmethod
    def _validated_endpoint(ep: dict) -> dict:
        ep = dict(ep)
        name = ep.get("name")
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", name):
            raise ValueError("endpoint name must be 1-64 letters, numbers, dot, underscore, or dash")
        if ep.get("kind") not in {"openai", "anthropic", "gemini", "ollama", "llamacpp"}:
            raise ValueError("unsupported endpoint kind")
        base_url = ep.get("base_url")
        if not isinstance(base_url, str) or not re.match(r"^https?://[^\s]+$", base_url):
            raise ValueError("endpoint base_url must be an http(s) URL")
        if "api_key" in ep or "_api_key" in ep:
            raise ValueError("store provider credentials in an environment variable, not endpoint config")
        for key in ("concurrency", "max_context"):
            if key in ep and (isinstance(ep[key], bool) or not isinstance(ep[key], int) or ep[key] <= 0):
                raise ValueError(f"endpoint {key} must be a positive integer")
        return ep

    def remove_endpoint(self, name: str) -> bool:
        """Remove an endpoint by name."""
        with self._lock:
            self._ensure_mutable()
            endpoints = self._endpoints
            new_endpoints = [e for e in endpoints if e["name"] != name]
            if len(new_endpoints) == len(endpoints):
                return False
            # Also remove from any tier endpoint lists
            tiers = self._tiers
            for t in tiers:
                if name in t.get("endpoints", []):
                    t["endpoints"] = [e for e in t["endpoints"] if e != name]
            self._set_section("endpoints", new_endpoints)
            if tiers != self._tiers:
                self._set_section("tiers", tiers)
            self._reload()
            log.info("endpoint removed: %s", name)
            return True

    # ---- API key management ----

    # Recommended/default key kinds -- reuses the same taxonomy as endpoint
    # `kind` (see _validated_endpoint above), but `generate_key` does NOT
    # restrict to only these: any short identifier is accepted so new wire
    # formats/upstream families don't require a code change here.
    KNOWN_KEY_KINDS = ("openai", "anthropic", "gemini", "ollama", "llamacpp")

    def generate_key(self, tenant_id: str, scope: list[str] | None = None, kind: str = "openai") -> str:
        """Generate a new API key. Returns the key (only shown once).

        `kind` is informational/organizational metadata recording which wire
        format or upstream family this key is meant for (e.g. "openai",
        "anthropic" -- see KNOWN_KEY_KINDS for the recommended set). It does
        NOT restrict which endpoint or route the key may actually call --
        auth/routing enforcement is unchanged. Any short identifier is
        accepted (not just KNOWN_KEY_KINDS) so this stays generally
        applicable as new wire formats/providers are added, rather than
        hardcoded to a fixed pair.
        """
        key = "ctrl-" + secrets.token_urlsafe(32)
        scope = scope or ["user"]
        if not isinstance(tenant_id, str) or not re.fullmatch(r"[A-Za-z0-9._:@-]{1,64}", tenant_id):
            raise ValueError("invalid tenant_id")
        if not isinstance(scope, list) or not set(scope).issubset({"user", "admin"}):
            raise ValueError("scope must contain only user/admin")
        if not isinstance(kind, str):
            raise ValueError("kind must be a string")
        kind = kind.strip().lower()
        if not re.fullmatch(r"[a-z0-9_-]{1,32}", kind):
            raise ValueError("kind must be 1-32 chars: lowercase letters, digits, underscore, or dash")
        with self._lock:
            self._ensure_mutable()
            keys = self._auth_keys
            import hashlib
            digest = "sha256:" + hashlib.sha256(key.encode()).hexdigest()
            keys[digest] = {"tenant_id": tenant_id, "scope": scope, "prefix": key[:12], "kind": kind}
            self._set_nested("auth", "keys", keys)
            self._reload()
            log.info("API key generated for tenant '%s' (scope=%s, kind=%s)", tenant_id, scope, kind)
            return key

    def list_keys(self) -> list[dict]:
        """List all keys with masked values."""
        keys = self._auth_keys
        return [
            {
                "key_masked": v.get("prefix", k[:12]) + "...",
                "tenant_id": v.get("tenant_id"),
                "scope": v.get("scope", ["user"]),
                "kind": v.get("kind", "openai"),
            }
            for k, v in keys.items()
        ]

    def revoke_key(self, key_or_prefix: str) -> bool:
        """Revoke a key by full value or unique prefix."""
        with self._lock:
            self._ensure_mutable()
            keys = self._auth_keys
            target = key_or_prefix
            if target.startswith("ctrl-"):
                import hashlib
                digest = "sha256:" + hashlib.sha256(target.encode()).hexdigest()
                if digest in keys:
                    target = digest
            if target not in keys:
                matches = [
                    k for k, meta in keys.items()
                    if k.startswith(target) or str(meta.get("prefix", "")).startswith(target)
                ]
                if len(matches) == 1:
                    target = matches[0]
                else:
                    return False
            tenant = keys[target].get("tenant_id", "?")
            del keys[target]
            self._set_nested("auth", "keys", keys)
            self._reload()
            log.info("API key revoked (tenant=%s)", tenant)
            return True

    # ---- tier management ----

    def update_tier(self, name: str, updates: dict) -> dict:
        """Edit a tier (endpoints list, capability_per_vertical, max_context, etc.)."""
        with self._lock:
            self._ensure_mutable()
            tiers = self._tiers
            found = False
            for i, t in enumerate(tiers):
                if t["name"] == name:
                    tiers[i] = {**t, **{k: v for k, v in updates.items() if k != "name"}}
                    found = True
                    break
            if not found:
                raise KeyError(f"tier '{name}' not found")
            self._set_section("tiers", tiers)
            self._reload()
            log.info("tier updated: %s", name)
            return tiers[i]

    def assign_endpoint_to_tier(self, tier_name: str, endpoint_name: str) -> dict:
        """Add an endpoint to a tier's endpoint list."""
        with self._lock:
            self._ensure_mutable()
            if endpoint_name not in {ep["name"] for ep in self._endpoints}:
                raise KeyError(f"endpoint '{endpoint_name}' not found")
            tiers = self._tiers
            for t in tiers:
                if t["name"] == tier_name:
                    eps = t.get("endpoints", [])
                    if endpoint_name not in eps:
                        eps.append(endpoint_name)
                        t["endpoints"] = eps
                    self._set_section("tiers", tiers)
                    self._reload()
                    return t
            raise KeyError(f"tier '{tier_name}' not found")

    def remove_endpoint_from_tier(self, tier_name: str, endpoint_name: str) -> dict:
        """Remove an endpoint from a tier's endpoint list."""
        with self._lock:
            self._ensure_mutable()
            tiers = self._tiers
            for t in tiers:
                if t["name"] == tier_name:
                    t["endpoints"] = [e for e in t.get("endpoints", []) if e != endpoint_name]
                    self._set_section("tiers", tiers)
                    self._reload()
                    return t
            raise KeyError(f"tier '{tier_name}' not found")

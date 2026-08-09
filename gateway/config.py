"""Glint-V2 gateway configuration loader with hot reload.

Reads gateway-config.json + gateway-policy.json + router_model/taxonomy.yaml.
Hot-reloadable via /reload endpoint. Thread-safe atomic swap.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("glint.config")


class ConfigError(Exception):
    """Raised when config is invalid or files cannot be loaded."""


def _merge_overlay(base: dict, overlay: dict) -> dict:
    """Deep-merge the runtime overlay onto the base config.

    - Dict values are merged one level deep (overlay keys override base keys).
    - List values (endpoints, tiers) are replaced entirely by the overlay.
    - Keys starting with '_' are documentation — skipped.
    """
    result = dict(base)
    for k, v in overlay.items():
        if k.startswith("_"):
            continue
        if isinstance(v, dict) and isinstance(result.get(k), dict):
            result[k] = {**result[k], **v}
        else:
            result[k] = v
    return result


class Config:
    """Immutable snapshot of all configs at a point in time.

    A new snapshot is created on every reload. The active snapshot is held
    behind an immutable reference for thread-safe reads; writes (reload)
    atomically replace the reference.
    """

    __slots__ = (
        "config",
        "policy",
        "taxonomy",
        "prototypes",
        "provider_presets",
        "loaded_at",
        "version",
    )

    def __init__(
        self,
        config: dict,
        policy: dict,
        taxonomy: dict,
        prototypes: dict,
        version: int,
        provider_presets: dict | None = None,
    ):
        self.config = config
        self.policy = policy
        self.taxonomy = taxonomy
        self.prototypes = prototypes
        self.provider_presets = provider_presets or {}
        self.loaded_at = time.time()
        self.version = version

    def get(self, *path: str, default: Any = None) -> Any:
        """Walk dot-separated path through nested dicts. Returns default if missing."""
        node: Any = self.config
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def policy_get(self, *path: str, default: Any = None) -> Any:
        node: Any = self.policy
        for key in path:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def verticals(self) -> list[dict]:
        return self.taxonomy.get("verticals", [])

    def vertical(self, name: str) -> dict | None:
        for v in self.verticals():
            if v["name"] == name:
                return v
        return None

    def tier(self, name: str) -> dict | None:
        for t in self.config.get("tiers", []):
            if t["name"] == name:
                return t
        return None

    def endpoint(self, name: str) -> dict | None:
        for e in self.config.get("endpoints", []):
            if e["name"] == name:
                return e
        return None

    def endpoints_for_tier(self, tier_name: str) -> list[dict]:
        t = self.tier(tier_name)
        if not t:
            return []
        names = t.get("endpoints", [])
        return [e for e in (self.endpoint(n) for n in names) if e]

    def reviewer(self) -> dict:
        return self.config.get("reviewer", {})

    def is_override_only_tier(self, tier_name: str) -> bool:
        t = self.tier(tier_name)
        return bool(t and t.get("override_only"))


class ConfigManager:
    """Thread-safe manager for config snapshots.

    The current snapshot is exposed via .current. Reloading atomically
    replaces it. Reload is triggered by /reload endpoint, file mtime
    polling, or explicit reload() call.
    """

    def __init__(
        self,
        config_path: str | Path = "./gateway-config.json",
        policy_path: str | Path = "./gateway-policy.json",
        taxonomy_path: str | Path = "./router_model/taxonomy.yaml",
        prototypes_path: str | Path = "./router_model/prototypes.json",
        provider_presets_path: str | Path = "./gateway/provider_presets.json",
        overlay_path: str | Path | None = None,
    ):
        self.config_path = Path(config_path)
        self.policy_path = Path(policy_path)
        self.taxonomy_path = Path(taxonomy_path)
        self.prototypes_path = Path(prototypes_path)
        self.provider_presets_path = Path(provider_presets_path)
        self.overlay_path = Path(overlay_path) if overlay_path is not None else self.config_path.with_name("gateway-config.runtime.json")

        self._current: Config | None = None
        self._lock = threading.RLock()
        self._reload_count = 0
        self._last_mtime_check = 0.0

        self.reload()

    def reload(self) -> Config:
        """Reload all configs from disk. Returns the new snapshot."""
        with self._lock:
            try:
                config = self._load_json(self.config_path)
                policy = self._load_json(self.policy_path)
                taxonomy = self._load_yaml(self.taxonomy_path)
                prototypes = self._load_json(self.prototypes_path)
            except Exception as e:
                raise ConfigError(f"Failed to reload config: {e}") from e

            # Load provider presets (optional — not fatal if missing)
            provider_presets = {}
            try:
                provider_presets = self._load_json(self.provider_presets_path)
            except Exception:
                pass

            # Env overrides (docker-compose / k8s): GLINT_DB_URL, GLINT_MODE
            if os.environ.get("GLINT_DB_URL"):
                config["db_url"] = os.environ["GLINT_DB_URL"]
            if os.environ.get("GLINT_MODE"):
                config["mode"] = os.environ["GLINT_MODE"]
            if os.environ.get("GLINT_AUTH_ENABLED"):
                config.setdefault("auth", {})["enabled"] = os.environ["GLINT_AUTH_ENABLED"].lower() in (
                    "1", "true", "yes", "on",
                )
            if os.environ.get("GLINT_ADMIN_API_KEY"):
                raw_key = os.environ["GLINT_ADMIN_API_KEY"]
                digest = "sha256:" + hashlib.sha256(raw_key.encode()).hexdigest()
                config.setdefault("auth", {}).setdefault("keys", {})[digest] = {
                    "tenant_id": os.environ.get("GLINT_ADMIN_TENANT", "admin"),
                    "scope": ["admin", "user"],
                    "prefix": raw_key[:12],
                }

            # Runtime overlay: admin changes (new providers, keys, tier edits)
            # persist across restarts without mutating the base config file.
            if self.overlay_path.exists():
                try:
                    overlay = self._load_json(self.overlay_path)
                    config = _merge_overlay(config, overlay)
                except Exception as e:
                    log.warning("overlay merge failed: %s", e)

            self._validate(config, policy, taxonomy, prototypes)

            self._reload_count += 1
            snapshot = Config(
                config=config,
                policy=policy,
                taxonomy=taxonomy,
                prototypes=prototypes,
                version=self._reload_count,
                provider_presets=provider_presets,
            )
            self._current = snapshot
            return snapshot

    def current(self) -> Config:
        """Return the current snapshot. Raises if not loaded."""
        if self._current is None:
            raise ConfigError("Config not loaded yet")
        return self._current

    def check_mtime_and_reload(self, poll_interval_seconds: float = 5.0) -> bool:
        """Check file mtimes; reload if any changed since last check.

        Returns True if a reload happened. Cheap when no changes.
        """
        now = time.time()
        if now - self._last_mtime_check < poll_interval_seconds:
            return False
        self._last_mtime_check = now

        try:
            paths = [
                self.config_path,
                self.policy_path,
                self.taxonomy_path,
                self.prototypes_path,
                self.provider_presets_path,
            ]
            if self.overlay_path.exists():
                paths.append(self.overlay_path)
            latest = max(p.stat().st_mtime for p in paths)
            current = self._current.loaded_at if self._current else 0.0
            if latest > current:
                self.reload()
                return True
        except OSError:
            pass
        return False

    @staticmethod
    def _load_json(path: Path) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_yaml(path: Path) -> dict:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def _validate(config, policy, taxonomy, prototypes):
        """Sanity-check configs. Fail fast on bad config."""
        if "mode" not in config:
            raise ConfigError("config.mode missing")
        if config["mode"] not in ("single", "multi"):
            raise ConfigError(f"config.mode must be 'single' or 'multi', got {config['mode']}")
        if "db_url" not in config:
            raise ConfigError("config.db_url missing")
        if "endpoints" not in config or not config["endpoints"]:
            raise ConfigError("config.endpoints missing or empty")
        if "tiers" not in config or not config["tiers"]:
            raise ConfigError("config.tiers missing or empty")
        if "reviewer" not in config:
            raise ConfigError("config.reviewer missing")
        if not config["reviewer"].get("model"):
            raise ConfigError("config.reviewer.model missing — set to your chosen reviewer model id")
        if "api_key_env" not in config["reviewer"]:
            raise ConfigError("config.reviewer.api_key_env missing")
        if not os.environ.get(config["reviewer"]["api_key_env"]):
            # Not fatal at load (env may be set later by launcher), just warn.
            pass

        # Validate vertical names referenced by tiers exist in taxonomy
        vertical_names = {v["name"] for v in taxonomy.get("verticals", [])}
        endpoint_names = {e["name"] for e in config["endpoints"]}
        if len(endpoint_names) != len(config["endpoints"]):
            raise ConfigError("endpoint names must be unique")
        allowed_kinds = {"openai", "anthropic", "gemini", "ollama", "llamacpp"}
        for endpoint in config["endpoints"]:
            if endpoint.get("kind") not in allowed_kinds:
                raise ConfigError(f"endpoint '{endpoint.get('name')}' has unsupported kind")
            base_url = endpoint.get("base_url")
            if not isinstance(base_url, str) or not base_url.startswith(("http://", "https://")):
                raise ConfigError(f"endpoint '{endpoint.get('name')}' base_url must be http(s)")
            if int(endpoint.get("concurrency", 4)) <= 0:
                raise ConfigError(f"endpoint '{endpoint.get('name')}' concurrency must be positive")
        tier_names = [tier.get("name") for tier in config["tiers"]]
        if len(set(tier_names)) != len(tier_names):
            raise ConfigError("tier names must be unique")
        for tier in config["tiers"]:
            cap = tier.get("capability_per_vertical", {})
            for v_name in cap:
                if v_name != "_default" and v_name not in vertical_names:
                    raise ConfigError(
                        f"tier '{tier['name']}' references unknown vertical '{v_name}'"
                    )
            # Validate tier -> endpoint references
            for ep_name in tier.get("endpoints", []):
                if ep_name not in endpoint_names:
                    raise ConfigError(
                        f"tier '{tier['name']}' references unknown endpoint '{ep_name}'"
                    )

        # Validate prototype kinds are structural (Glint lesson)
        for proto in prototypes.get("prototypes", []):
            kind = proto.get("kind")
            if kind not in ("structural", "topic"):
                raise ConfigError(
                    f"prototype '{proto.get('name')}' has invalid kind '{kind}'"
                )
            if kind == "topic":
                raise ConfigError(
                    f"prototype '{proto.get('name')}' is topic kind — "
                    f"TOPIC PROTOTYPES ARE FORBIDDEN (Glint Roman Empire lesson). "
                    f"Use embedding + classifier head for topic verticals."
                )

        # Validate reviewer caps config
        caps = config["reviewer"].get("caps", {})
        for cap_key in ("per_request_usd", "per_hour_usd", "per_day_usd", "per_month_usd"):
            if cap_key in caps and caps[cap_key] <= 0:
                raise ConfigError(f"reviewer.caps.{cap_key} must be > 0")


# Singleton-ish accessor. Created once in app.py startup.
_manager: ConfigManager | None = None


def init(
    config_path: str | Path = "./gateway-config.json",
    policy_path: str | Path = "./gateway-policy.json",
    taxonomy_path: str | Path = "./router_model/taxonomy.yaml",
    prototypes_path: str | Path = "./router_model/prototypes.json",
) -> ConfigManager:
    global _manager
    _manager = ConfigManager(config_path, policy_path, taxonomy_path, prototypes_path)
    return _manager


def manager() -> ConfigManager:
    if _manager is None:
        raise ConfigError("ConfigManager not initialized — call init() first")
    return _manager

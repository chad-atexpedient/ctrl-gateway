"""Gateway plugin system.

Contract (adapted from mcpchad for aiohttp):

  Each plugin lives in its own directory under `plugin_root`.
  The directory must contain:
    manifest.yaml   - plugin metadata (name, version, prefix, routes, etc.)
    plugin.py       - module exposing `build_router(context) -> web.RouteTableDef`

  manifest.yaml schema:
    name: <str>                 (required)
    version: <str>              (semver-ish, optional)
    description: <str>          (optional)
    module: <str>               (default "plugin.py")
    prefix: <str>               (default ""; prepended to all routes)
    auth:
      scope: <str>              (admin | user | anonymous)
    routes:
      - path: <str>             (under prefix)
        method: <GET|POST|PUT|DELETE>
        summary: <str>
    events:
      - <event_type>            (events this plugin may emit)
    settings:
      <key>: <value>

  PluginContext provides:
    name, manifest, config (gateway cfg.Config), event_bus, memory (db helpers),
    emit_event(), get_setting().

  The loader scans `plugin_root` on a configurable interval and picks up new
  plugins without a restart. Errors loading a plugin are recorded but never
  crash the gateway.
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from aiohttp import web

from . import config as cfg_mod
from . import events as events_mod
from . import memory

log = logging.getLogger("ctrl.plugin")


@dataclass
class LoadedPlugin:
    name: str
    directory: Path
    manifest: dict[str, Any]
    routes: web.RouteTableDef | None
    prefix: str
    enabled: bool
    loaded: bool
    error: str | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.manifest.get("version", "0.0.0"),
            "description": self.manifest.get("description", ""),
            "prefix": self.prefix,
            "enabled": self.enabled,
            "loaded": self.loaded,
            "error": self.error,
            "routes": self.manifest.get("routes", []),
            "events": self.manifest.get("events", []),
            "settings": self.manifest.get("settings", {}),
        }


class PluginContext:
    """Context object handed to a plugin's build_router().

    Mirrors the mcpchad PluginContext surface (name, manifest, config,
    event_bus, emit_event, get_setting) but uses the gateway's own
    event_bus + Config types instead of the FastAPI/MCP equivalents.
    """

    def __init__(
        self,
        name: str,
        manifest: dict[str, Any],
        config: cfg_mod.Config,
        event_bus: events_mod.EventBus,
        memory_module: Any = memory,
    ):
        self.name = name
        self.manifest = manifest
        self.config = config
        self.event_bus = event_bus
        self.memory = memory_module

    async def emit_event(
        self,
        event_name: str,
        data: dict[str, Any] | None = None,
        tenant_id: str | None = None,
        severity: str = "info",
    ) -> None:
        try:
            events_mod.emit(
                source=events_mod.EventSource.PLUGIN,
                event_type=event_name,
                data=data or {},
                tenant_id=tenant_id,
                severity=events_mod.EventSeverity(severity) if severity in (
                    "info", "status", "warn", "error", "debug"
                ) else events_mod.EventSeverity.INFO,
            )
        except Exception as e:
            log.warning("plugin %s emit_event failed: %s", self.name, e)

    def get_setting(self, key: str, default: Any = None) -> Any:
        return self.manifest.get("settings", {}).get(key, default)


class PluginLoader:
    """Scans a plugin directory, loads each plugin, attaches its routes."""

    def __init__(
        self,
        app: web.Application,
        config: cfg_mod.Config,
        event_bus: events_mod.EventBus,
        plugin_root: str | Path = "./gateway/plugins",
        scan_interval_seconds: int = 30,
        auto_load: bool = True,
    ):
        self.app = app
        self.config = config
        self.event_bus = event_bus
        self.plugin_root = Path(plugin_root)
        self.scan_interval_seconds = scan_interval_seconds
        self.auto_load = auto_load
        self.plugins: dict[str, LoadedPlugin] = {}
        self._lock = threading.Lock()
        self._watch_task: asyncio.Task | None = None

    def load_all(self) -> int:
        """One-shot scan + load. Returns number of plugins loaded."""
        loaded_count = 0
        if not self.plugin_root.exists():
            log.debug("plugin root missing: %s", self.plugin_root)
            return 0
        for candidate in self.plugin_root.iterdir():
            if not candidate.is_dir():
                continue
            manifest_path = candidate / "manifest.yaml"
            if not manifest_path.exists():
                continue
            try:
                manifest = yaml.safe_load(manifest_path.read_text()) or {}
            except Exception as e:
                log.warning("could not parse %s: %s", manifest_path, e)
                continue
            name = manifest.get("name") or candidate.name
            with self._lock:
                if name in self.plugins:
                    continue
            if self._load_plugin(name, candidate, manifest):
                loaded_count += 1
        return loaded_count

    def _load_plugin(
        self, name: str, directory: Path, manifest: dict[str, Any]
    ) -> bool:
        """Load a single plugin and attach its routes. Errors are logged
        into the DB so the dashboard can show them — never raised."""
        module_file = directory / manifest.get("module", "plugin.py")
        prefix = manifest.get("prefix", "")
        spec = importlib.util.spec_from_file_location(
            f"ctrl.plugins.{name}", module_file
        )
        if spec is None or spec.loader is None:
            self._record_error(name, directory, manifest, prefix, "could not load plugin module spec")
            return False
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as e:
            self._record_error(name, directory, manifest, prefix, f"import failed: {e}")
            return False
        builder = getattr(module, "build_router", None)
        if builder is None:
            self._record_error(name, directory, manifest, prefix, "plugin must expose build_router(context)")
            return False
        try:
            context = PluginContext(
                name=name,
                manifest=manifest,
                config=self.config,
                event_bus=self.event_bus,
            )
            routes = builder(context)
        except Exception as e:
            self._record_error(name, directory, manifest, prefix, f"build_router failed: {e}")
            return False
        if routes is None:
            self._record_error(name, directory, manifest, prefix, "build_router returned None")
            return False
        try:
            self.app.add_routes(list(routes))
        except Exception as e:
            self._record_error(name, directory, manifest, prefix, f"route registration failed: {e}")
            return False
        with self._lock:
            self.plugins[name] = LoadedPlugin(
                name=name,
                directory=directory,
                manifest=manifest,
                routes=routes,
                prefix=prefix,
                enabled=True,
                loaded=True,
                error=None,
            )
        # Persist to DB
        try:
            memory.upsert_plugin(
                name=name,
                version=manifest.get("version", "0.0.0"),
                description=manifest.get("description", ""),
                prefix=prefix,
                module_path=str(module_file),
                config=manifest.get("settings", {}),
                enabled=True,
                is_builtin=False,
            )
            memory.set_plugin_loaded(name, True, error=None)
        except Exception as e:
            log.debug("plugin DB upsert failed for %s: %s", name, e)
        log.info("loaded plugin %s (prefix=%s)", name, prefix)
        return True

    def _record_error(
        self,
        name: str,
        directory: Path,
        manifest: dict[str, Any],
        prefix: str,
        error: str,
    ) -> None:
        log.warning("plugin %s load error: %s", name, error)
        with self._lock:
            self.plugins[name] = LoadedPlugin(
                name=name,
                directory=directory,
                manifest=manifest,
                routes=None,
                prefix=prefix,
                enabled=False,
                loaded=False,
                error=error,
            )
        try:
            memory.upsert_plugin(
                name=name,
                version=manifest.get("version", "0.0.0"),
                description=manifest.get("description", ""),
                prefix=prefix,
                module_path=str(directory / manifest.get("module", "plugin.py")),
                config=manifest.get("settings", {}),
                enabled=False,
                is_builtin=False,
            )
            memory.set_plugin_loaded(name, False, error=error)
        except Exception as e:
            log.debug("plugin DB error upsert failed for %s: %s", name, e)

    def reload(self, name: str) -> bool:
        """Reload a plugin's module code and update in-memory state.

        Note: aiohttp does not support removing routes from a running
        application. Route paths registered at load time persist; reload
        only re-imports the module and refreshes the build_router output
        so handler logic picks up code changes. Changing route paths
        requires a gateway restart.
        """
        with self._lock:
            existing = self.plugins.get(name)
        if not existing:
            return False
        # Re-import the module
        module_file = existing.directory / existing.manifest.get("module", "plugin.py")
        spec = importlib.util.spec_from_file_location(
            f"ctrl.plugins.{name}", module_file
        )
        if spec is None or spec.loader is None:
            return False
        try:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            builder = getattr(module, "build_router", None)
            if builder is None:
                return False
            context = PluginContext(
                name=name,
                manifest=existing.manifest,
                config=self.config,
                event_bus=self.event_bus,
            )
            new_routes = builder(context)
        except Exception as e:
            log.warning("reload plugin %s failed: %s", name, e)
            return False
        with self._lock:
            self.plugins[name].routes = new_routes
            self.plugins[name].error = None
        try:
            memory.set_plugin_loaded(name, True, error=None)
        except Exception:
            pass
        log.info("reloaded plugin %s (route paths unchanged)", name)
        return True

    def set_enabled(self, name: str, enabled: bool) -> bool:
        with self._lock:
            plugin = self.plugins.get(name)
            if plugin:
                plugin.enabled = enabled
        try:
            return memory.set_plugin_enabled(name, enabled)
        except Exception:
            return False

    async def watch_loop(self) -> None:
        """Background task: re-scan periodically for new plugins."""
        while True:
            try:
                self.load_all()
            except Exception as e:
                log.warning("plugin watcher error: %s", e)
            await asyncio.sleep(self.scan_interval_seconds)

    def start_watch(self) -> None:
        """Spawn the background watch task (idempotent)."""
        if self._watch_task is not None and not self._watch_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
            self._watch_task = loop.create_task(self.watch_loop())
        except RuntimeError:
            log.debug("no running event loop; plugin watch loop not started")

    def stop_watch(self) -> None:
        if self._watch_task is not None and not self._watch_task.done():
            self._watch_task.cancel()
            self._watch_task = None


_default_loader: PluginLoader | None = None


def init_loader(
    app: web.Application,
    config: cfg_mod.Config,
    event_bus: events_mod.EventBus,
    plugin_root: str | Path = "./gateway/plugins",
    scan_interval_seconds: int = 30,
    auto_load: bool = True,
) -> PluginLoader:
    global _default_loader
    loader = PluginLoader(
        app=app,
        config=config,
        event_bus=event_bus,
        plugin_root=plugin_root,
        scan_interval_seconds=scan_interval_seconds,
        auto_load=auto_load,
    )
    _default_loader = loader
    if auto_load:
        loader.load_all()
    return loader


def loader() -> PluginLoader | None:
    return _default_loader

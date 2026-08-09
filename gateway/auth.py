"""Per-tenant API-key authentication + admin scoping.

Auth model: static keys configured in gateway-config.json -> auth:

    "auth": {
      "enabled": true,
      "keys": {
        "sk-admin-123": {"tenant_id": "admin", "scope": ["admin", "user"]},
        "sk-user-456": {"tenant_id": "alice", "scope": ["user"]}
      },
      "admin_paths": ["/admin", "/retrain", "/reload", "/config", "/export"]
    }

Behavior:
  - Middleware reads `Authorization: Bearer <key>`.
  - Resolves the key to a tenant_id; sets request["tenant_id"] and
    request["auth_scope"].
  - Requests to paths under admin_paths require scope "admin" (401 otherwise).
  - When auth is enabled, every non-public path requires a valid key.
    X-User-Id fallback is available only when explicitly enabled for a
    trusted reverse proxy deployment.
  - Keys are hot-reloaded with the rest of the config (ConfigManager).

X-User-Id remains a fallback identity so the chat proxy keeps working for
unauthenticated clients behind a reverse proxy that injects the header.
"""
from __future__ import annotations

import hashlib
import hmac
import logging

from aiohttp import web

from . import config as cfg

log = logging.getLogger("glint.auth")


DEFAULT_ADMIN_PATHS = (
    "/admin", "/retrain", "/reload", "/config", "/export", "/registry", "/metrics",
)
DEFAULT_PUBLIC_PATHS = ("/", "/health", "/ready", "/dashboard")


class AuthError(Exception):
    pass


class AuthManager:
    """Holds the key map. Reloaded on config reload."""

    def __init__(self, config: dict | None = None):
        self._keys: dict[str, dict] = {}
        self.enabled = False
        self.admin_paths = DEFAULT_ADMIN_PATHS
        self.public_paths = DEFAULT_PUBLIC_PATHS
        self.allow_unauthenticated_user = False
        if config:
            self.update(config)

    def update(self, auth_cfg: dict):
        self.enabled = bool(auth_cfg.get("enabled", False))
        self._keys = dict(auth_cfg.get("keys", {}) or {})
        self.admin_paths = tuple(auth_cfg.get("admin_paths", list(DEFAULT_ADMIN_PATHS)))
        self.public_paths = tuple(auth_cfg.get("public_paths", list(DEFAULT_PUBLIC_PATHS)))
        self.allow_unauthenticated_user = bool(auth_cfg.get("allow_unauthenticated_user", False))

    def resolve(self, token: str) -> dict | None:
        """Return key metadata {tenant_id, scope} or None if invalid."""
        if not token:
            return None
        token_digest = "sha256:" + hashlib.sha256(token.encode()).hexdigest()
        # Constant-time-ish compare against every key (avoid leaking which key is wrong)
        for k, meta in self._keys.items():
            candidate = token_digest if k.startswith("sha256:") else token
            if hmac.compare_digest(k, candidate):
                return {
                    "tenant_id": meta.get("tenant_id", "anonymous"),
                    "scope": set(meta.get("scope", ["user"])),
                }
        return None

    def is_admin_path(self, path: str) -> bool:
        return any(path.startswith(p) for p in self.admin_paths)

    def is_public_path(self, path: str) -> bool:
        if path == "/":
            return "/" in self.public_paths
        return any(p != "/" and path.startswith(p) for p in self.public_paths)


@web.middleware
async def auth_middleware(request: web.Request, handler):
    """Resolve API key -> tenant, enforce admin scope on admin paths."""
    auth_cfg = request.app.get("auth_manager")
    if auth_cfg is None or not auth_cfg.enabled:
        return await handler(request)

    path = request.path
    auth_header = request.headers.get("Authorization", "")
    token = ""
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:].strip()

    meta = auth_cfg.resolve(token) if token else None

    if auth_cfg.is_public_path(path):
        if meta is not None:
            request["tenant_id"] = meta["tenant_id"]
            request["auth_scope"] = meta["scope"]
        return await handler(request)

    # Admin paths always require a valid admin-scoped key
    if auth_cfg.is_admin_path(path):
        if meta is None or "admin" not in meta["scope"]:
            return web.json_response(
                {"error": "unauthorized", "detail": "admin scope required"},
                status=401,
            )
        request["tenant_id"] = meta["tenant_id"]
        request["auth_scope"] = meta["scope"]
        return await handler(request)

    # User-facing paths: key maps to tenant. Header fallback is intentionally
    # opt-in because otherwise any direct client can impersonate any tenant.
    if meta is not None:
        request["tenant_id"] = meta["tenant_id"]
        request["auth_scope"] = meta["scope"]
    elif auth_cfg.allow_unauthenticated_user:
        request["tenant_id"] = request.headers.get("X-User-Id", "anonymous")
        request["auth_scope"] = {"user"}
    else:
        return web.json_response(
            {"error": "unauthorized", "detail": "valid API key required"},
            status=401,
        )
    return await handler(request)


@web.middleware
async def body_size_middleware(request: web.Request, handler):
    """Cap request body size (config http.max_body_bytes, default 4 MB)."""
    max_bytes = request.app.get("max_body_bytes", 4 * 1024 * 1024)
    if request.content_length is not None and request.content_length > max_bytes:
        return web.json_response(
            {"error": "payload_too_large", "detail": f"max {max_bytes} bytes"},
            status=413,
        )
    return await handler(request)


@web.middleware
async def cors_middleware(request: web.Request, handler):
    """Optional CORS. Enabled when config http.cors_origins is a non-empty list."""
    origins = request.app.get("cors_origins", [])
    if not origins:
        return await handler(request)
    origin = request.headers.get("Origin")
    allow = "*" if "*" in origins else (origin if origin in origins else None)
    if allow is None:
        return await handler(request)
    if request.method == "OPTIONS":
        resp = web.Response(status=204)
    else:
        resp = await handler(request)
    resp.headers["Access-Control-Allow-Origin"] = allow
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Authorization, Content-Type, X-User-Id, X-Session-Id"
    resp.headers["Access-Control-Max-Age"] = "600"
    if allow != "*":
        resp.headers["Vary"] = "Origin"
    return resp


@web.middleware
async def security_headers_middleware(request: web.Request, handler):
    resp = await handler(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    if request.path.startswith("/dashboard"):
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; "
            "script-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'"
        )
    if request.secure:
        resp.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return resp


def init_manager(conf: cfg.Config) -> AuthManager:
    """Build the manager from the current config snapshot."""
    return AuthManager(conf.config.get("auth", {}))


_manager: AuthManager | None = None


def manager() -> AuthManager:
    if _manager is None:
        raise RuntimeError("auth manager not initialized")
    return _manager


def set_manager(m: AuthManager):
    global _manager
    _manager = m

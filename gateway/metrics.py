"""Tiny Prometheus-format metrics registry (no external dependency).

Exposes counters/gauges in the Prometheus text exposition format via
GET /metrics. Thread-safe via a lock.
"""
from __future__ import annotations

import threading
import time
from collections import defaultdict


class Metrics:
    def __init__(self):
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple], float] = defaultdict(float)
        self._gauges: dict[tuple[str, tuple], float] = {}
        self._start = time.time()

    def inc(self, name: str, labels: dict | None = None, value: float = 1.0):
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self._counters[key] += value

    def set_gauge(self, name: str, labels: dict | None, value: float):
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self._gauges[key] = value

    def render(self) -> str:
        lines = [
            "# HELP glint_uptime_seconds Process uptime.",
            "# TYPE glint_uptime_seconds gauge",
            f"glint_uptime_seconds {time.time() - self._start:.3f}",
        ]
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                label_str = _fmt_labels(labels)
                lines.append(f"{name}{label_str} {value}")
            for (name, labels), value in sorted(self._gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                label_str = _fmt_labels(labels)
                lines.append(f"{name}{label_str} {value}")
        return "\n".join(lines) + "\n"


def _fmt_labels(labels: tuple) -> str:
    if not labels:
        return ""
    def escaped(value) -> str:
        return str(value).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')

    inner = ",".join(f'{k}="{escaped(v)}"' for k, v in labels)
    return f"{{{inner}}}"


_registry: Metrics | None = None


def registry() -> Metrics:
    global _registry
    if _registry is None:
        _registry = Metrics()
    return _registry

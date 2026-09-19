"""Operational monitoring, drift history, alerts, and telemetry exports."""
from __future__ import annotations

import json
import time
import urllib.request
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

from .models import FileResult


class RollingMetricWindow:
    def __init__(self, size: int = 20):
        self.size = max(1, int(size))
        self._values: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.size))

    def add(self, name: str, value: float) -> dict:
        values = self._values[name]
        values.append(float(value))
        return {"name": name, "count": len(values), "mean": sum(values) / len(values), "latest": values[-1]}


class DriftHistory:
    def __init__(self, path: str | Path, *, max_entries: int = 1000):
        self.path = Path(path)
        self.max_entries = max(1, int(max_entries))
        self._data = self._load()

    def append(self, metric: str, value: float, *, timestamp: float | None = None) -> None:
        items = self._data.setdefault(metric, [])
        items.append({"timestamp": timestamp if timestamp is not None else time.time(), "value": float(value)})
        self._data[metric] = items[-self.max_entries :]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    def entries(self, metric: str) -> list[dict]:
        return list(self._data.get(metric, []))

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}


class AlertPolicy:
    def __init__(self, *, cooldown_seconds: float = 300.0):
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._last_emitted: dict[str, float] = {}

    def should_emit(self, key: str, *, now: float | None = None) -> bool:
        current = time.time() if now is None else float(now)
        previous = self._last_emitted.get(key)
        if previous is not None and current - previous < self.cooldown_seconds:
            return False
        self._last_emitted[key] = current
        return True


def prometheus_metrics(results: Iterable[FileResult]) -> str:
    lines = ["# HELP qualiax_metric Numeric audio quality metric.", "# TYPE qualiax_metric gauge"]
    for result in results:
        for metric in result.metrics:
            if isinstance(metric.value, (int, float)) and not isinstance(metric.value, bool):
                lines.append(
                    f'qualiax_metric{{file="{_escape(result.path)}",group="{_escape(metric.group)}",metric="{_escape(metric.name)}"}} {float(metric.value)}'
                )
    return "\n".join(lines) + "\n"


def send_webhook(url: str, payload: dict, *, timeout: float = 10.0) -> int:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "qualiax"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.status)


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")

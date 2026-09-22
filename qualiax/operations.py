"""Operational monitoring, drift history, alerts, and telemetry exports."""
from __future__ import annotations

import http.client
import json
import math
import numbers
import time
import urllib.parse
import urllib.request
import warnings
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable

from .models import FileResult


class RollingMetricWindow:
    """In-memory bounded window of the most recent values per metric.

    Deprecated: ``DriftHistory.window`` provides the same rolling view and persists it.
    """

    def __init__(self, size: int = 20):
        from .migrations import warn_deprecated

        warn_deprecated("RollingMetricWindow", removal_version="2.0.0", alternative="DriftHistory.window", stacklevel=3)
        self.size = max(1, int(size))
        self._values: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self.size))

    def add(self, name: str, value: float) -> dict:
        values = self._values[name]
        values.append(float(value))
        return self.summary(name)

    def summary(self, name: str) -> dict:
        return _window_summary(name, list(self._values.get(name, ())))


class DriftHistory:
    """Persistent, bounded per-metric history stored as JSON.

    Each metric keeps at most ``max_entries`` values. Writes are atomic (temporary
    file + rename). A history file that cannot be parsed is moved aside to
    ``<name>.corrupt`` with a warning instead of being silently overwritten.
    """

    def __init__(self, path: str | Path, *, max_entries: int = 1000):
        self.path = Path(path)
        self.max_entries = max(1, int(max_entries))
        self._data = self._load()

    def append(self, metric: str, value: float, *, timestamp: float | None = None) -> None:
        self._add(metric, value, timestamp)
        self._save()

    def record_results(
        self,
        results: Iterable[FileResult],
        *,
        metrics: Iterable[str] | None = None,
        timestamp: float | None = None,
    ) -> int:
        """Append every numeric metric (or only ``metrics``) from ``results`` in one write."""
        wanted = set(metrics) if metrics is not None else None
        stamp = time.time() if timestamp is None else float(timestamp)
        added = 0
        for result in results:
            if result.error:
                continue
            for metric in result.metrics:
                if wanted is not None and metric.name not in wanted:
                    continue
                if not _is_number(metric.value) or not math.isfinite(float(metric.value)):
                    continue
                self._add(metric.name, float(metric.value), stamp)
                added += 1
        if added:
            self._save()
        return added

    def entries(self, metric: str) -> list[dict]:
        return [dict(item) for item in self._data.get(metric, [])]

    def metrics(self) -> list[str]:
        return sorted(self._data)

    def window(self, metric: str, size: int = 20) -> dict:
        """Summary statistics over the ``size`` most recent persisted values."""
        size = max(1, int(size))
        values = [item["value"] for item in self._data.get(metric, [])[-size:]]
        return _window_summary(metric, values)

    def check_drift(
        self,
        metric: str,
        value: float,
        *,
        window: int = 20,
        z_threshold: float = 3.0,
        min_samples: int = 5,
    ) -> dict:
        """Compare ``value`` against the rolling window of persisted history.

        Drift is flagged when the value's z-score against the window exceeds
        ``z_threshold``. With fewer than ``min_samples`` values there is not enough
        history and ``drifted`` is ``False`` with ``status == "insufficient_history"``.
        The value is not appended; call :meth:`append` afterwards to record it.
        """
        stats = self.window(metric, window)
        value = float(value)
        report = {"metric": metric, "value": value, "window": stats, "z_threshold": float(z_threshold)}
        if stats["count"] < max(1, int(min_samples)):
            return {**report, "drifted": False, "zscore": None, "status": "insufficient_history"}
        spread = stats["stdev"]
        delta = value - stats["mean"]
        if spread == 0:
            zscore = 0.0 if delta == 0 else math.copysign(math.inf, delta)
        else:
            zscore = delta / spread
        drifted = abs(zscore) > float(z_threshold)
        return {
            **report,
            "drifted": drifted,
            "zscore": zscore if math.isinf(zscore) else round(zscore, 6),
            "status": "drift" if drifted else "ok",
        }

    def _add(self, metric: str, value: float, timestamp: float | None) -> None:
        items = self._data.setdefault(metric, [])
        items.append({"timestamp": timestamp if timestamp is not None else time.time(), "value": float(value)})
        if len(items) > self.max_entries:
            self._data[metric] = items[-self.max_entries :]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(f"{self.path.suffix}.tmp")
        temporary.write_text(json.dumps(self._data, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.path)

    def _load(self) -> dict:
        if not self.path.exists():
            return {}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("drift history must be a JSON object")
        except (OSError, ValueError) as exc:
            quarantine = self.path.with_name(f"{self.path.name}.corrupt")
            try:
                self.path.replace(quarantine)
                where = f"moved to {quarantine}"
            except OSError:
                where = "left in place"
            warnings.warn(
                f"Drift history {self.path} could not be read ({exc}); {where} and starting a new history.",
                RuntimeWarning,
                stacklevel=3,
            )
            return {}
        cleaned: dict[str, list[dict]] = {}
        for metric, items in payload.items():
            if not isinstance(items, list):
                continue
            kept = [
                {"timestamp": item.get("timestamp"), "value": float(item["value"])}
                for item in items
                if isinstance(item, dict) and _is_number(item.get("value"))
            ]
            cleaned[str(metric)] = kept[-self.max_entries :]
        return cleaned


class AlertPolicy:
    """Cooldown-based alert suppression.

    Pass ``state_path`` to persist the last-emitted times so suppression holds
    across separate runs (cron jobs, CI invocations), not just within one process.
    """

    def __init__(self, *, cooldown_seconds: float = 300.0, state_path: str | Path | None = None):
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.state_path = Path(state_path) if state_path is not None else None
        self._last_emitted: dict[str, float] = self._load_state()

    def is_suppressed(self, key: str, *, now: float | None = None) -> bool:
        current = time.time() if now is None else float(now)
        previous = self._last_emitted.get(key)
        return previous is not None and current - previous < self.cooldown_seconds

    def record(self, key: str, *, now: float | None = None) -> None:
        self._last_emitted[key] = time.time() if now is None else float(now)
        self._save_state()

    def should_emit(self, key: str, *, now: float | None = None) -> bool:
        """Return True and start the cooldown if ``key`` is not currently suppressed."""
        current = time.time() if now is None else float(now)
        if self.is_suppressed(key, now=current):
            return False
        self.record(key, now=current)
        return True

    def _load_state(self) -> dict[str, float]:
        if self.state_path is None or not self.state_path.exists():
            return {}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(payload, dict):
            return {}
        return {str(key): float(value) for key, value in payload.items() if _is_number(value)}

    def _save_state(self) -> None:
        if self.state_path is None:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(f"{self.state_path.suffix}.tmp")
        temporary.write_text(json.dumps(self._last_emitted, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self.state_path)


def prometheus_metrics(results: Iterable[FileResult]) -> str:
    """Render numeric metrics in the Prometheus text exposition format (v0.0.4).

    Each numeric metric becomes a ``qualiax_metric`` gauge labelled by file, group,
    and metric name. ``qualiax_analysis_error`` is 1 for files that failed to
    analyze, so failures are visible to alerting. Duplicate series (the same file
    listed twice) keep their first value, because Prometheus rejects scrapes with
    duplicate series.
    """
    metric_lines: list[str] = []
    error_lines: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    seen_files: set[str] = set()
    for result in results:
        file_label = _escape(result.path)
        if file_label not in seen_files:
            seen_files.add(file_label)
            error_lines.append(f'qualiax_analysis_error{{file="{file_label}"}} {1 if result.error else 0}')
        for metric in result.metrics:
            if not _is_number(metric.value):
                continue
            key = (file_label, _escape(metric.group), _escape(metric.name))
            if key in seen:
                continue
            seen.add(key)
            metric_lines.append(
                f'qualiax_metric{{file="{key[0]}",group="{key[1]}",metric="{key[2]}"}} {_format_sample(metric.value)}'
            )
    lines = [
        "# HELP qualiax_metric Numeric audio quality metric.",
        "# TYPE qualiax_metric gauge",
        *metric_lines,
        "# HELP qualiax_analysis_error 1 if qualiax failed to analyze the file, else 0.",
        "# TYPE qualiax_analysis_error gauge",
        *error_lines,
    ]
    return "\n".join(lines) + "\n"


def send_webhook(url: str, payload: dict, *, timeout: float = 10.0) -> int:
    """POST ``payload`` as JSON to an http(s) URL and return the response status.

    Raises ``ValueError`` for non-http(s) URLs and ``urllib.error.URLError`` /
    ``HTTPError`` for network failures and 4xx/5xx responses.
    """
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"Webhook URL must use http or https, got {scheme or 'no scheme'!r}")
    request = urllib.request.Request(
        url,
        data=json.dumps(_json_safe(payload), allow_nan=False, default=str).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": "qualiax"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.status)


def deliver_alert(
    url: str,
    payload: dict,
    *,
    policy: AlertPolicy,
    key: str,
    timeout: float = 10.0,
    now: float | None = None,
) -> dict:
    """Send a webhook alert unless ``key`` is inside its cooldown window.

    The cooldown only starts after a successful delivery, so a failed POST is
    retried on the next call. Delivery errors are returned, not raised, so a
    flaky alert endpoint cannot break an analysis pipeline.
    """
    if policy.is_suppressed(key, now=now):
        return {"key": key, "sent": False, "suppressed": True, "status": None, "error": None}
    try:
        status = send_webhook(url, payload, timeout=timeout)
    except (OSError, ValueError, http.client.HTTPException) as exc:  # URLError is an OSError
        return {"key": key, "sent": False, "suppressed": False, "status": None, "error": str(exc)}
    policy.record(key, now=now)
    return {"key": key, "sent": True, "suppressed": False, "status": status, "error": None}


def _window_summary(name: str, values: list[float]) -> dict:
    if not values:
        return {"name": name, "count": 0, "mean": None, "stdev": None, "min": None, "max": None, "latest": None}
    return {
        "name": name,
        "count": len(values),
        "mean": mean(values),
        "stdev": pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "latest": values[-1],
    }


def _json_safe(value):
    """Replace NaN/Inf (invalid in strict JSON) with null, recursively."""
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _format_sample(value) -> str:
    number = float(value)
    if math.isnan(number):
        return "NaN"
    if math.isinf(number):
        return "+Inf" if number > 0 else "-Inf"
    return repr(number)


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

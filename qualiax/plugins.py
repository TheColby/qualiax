"""Versioned plugin discovery and isolated execution.

A plugin is an entry point in the ``qualiax.plugins`` group that resolves to a
module or object with a ``register(manager)`` function, or to a callable taking
the manager. It may declare the plugin API it targets with a
``QUALIAX_PLUGIN_API = "1.0"`` attribute; a plugin whose major version differs
from :data:`PLUGIN_API_VERSION`, or whose minor version is newer, is rejected.

Label providers take a :class:`FileResult` and return a list of label dicts that
contain at least ``"id"``. Reporters take a list of results and return ``str``.
"""
from __future__ import annotations

from importlib import metadata
from typing import Callable, Iterable

from .models import DiagnosticEntry, FileResult

PLUGIN_API_VERSION = "1.0"


class PluginManager:
    def __init__(self):
        self.label_providers: dict[str, Callable] = {}
        self.reporters: dict[str, Callable] = {}
        self.discovery_failures: list[dict] = []
        self.plugins: dict[str, dict] = {}

    def register_label_provider(self, name: str, provider: Callable) -> None:
        if not callable(provider):
            raise TypeError("label provider must be callable")
        self.label_providers[_name(name)] = provider

    def register_reporter(self, name: str, reporter: Callable) -> None:
        if not callable(reporter):
            raise TypeError("reporter must be callable")
        self.reporters[_name(name)] = reporter

    def discover(self, group: str = "qualiax.plugins") -> list[str]:
        """Load and register entry-point plugins; returns the names that registered.

        A plugin that fails to import, targets an incompatible API version, or
        raises during ``register`` is skipped and recorded in
        ``discovery_failures``; anything it registered before failing is rolled
        back. Other plugins still load.
        """
        discovered = []
        for entry_point in _entry_points(group):
            stage = "load"
            snapshot = (dict(self.label_providers), dict(self.reporters))
            try:
                plugin = entry_point.load()
                stage = "version"
                declared = getattr(plugin, "QUALIAX_PLUGIN_API", None)
                if declared is not None and not is_compatible_api(str(declared)):
                    raise RuntimeError(
                        f"plugin targets API {declared}, but this qualiax provides {PLUGIN_API_VERSION}"
                    )
                stage = "register"
                register = getattr(plugin, "register", plugin)
                if not callable(register):
                    raise TypeError("plugin must be callable or define register(manager)")
                register(self)
            except Exception as exc:
                self.label_providers, self.reporters = snapshot
                self.discovery_failures.append({"plugin": entry_point.name, "stage": stage, "error": str(exc)})
                continue
            self.plugins[entry_point.name] = {
                "api_version": str(declared) if declared is not None else None,
                "value": getattr(entry_point, "value", None),
            }
            discovered.append(entry_point.name)
        return discovered

    def run_label_providers(self, result: FileResult) -> tuple[list[dict], list[dict]]:
        labels = []
        failures = []
        for _provider, produced, failure in self._run_providers(result):
            if failure is not None:
                failures.append(failure)
            else:
                labels.extend(produced)
        return labels, failures

    def _run_providers(self, result: FileResult):
        for name, provider in self.label_providers.items():
            try:
                produced = provider(result)
                _validate_labels(produced)
            except Exception as exc:
                yield name, [], {"plugin": name, "error": str(exc)}
                continue
            yield name, produced, None

    def apply_label_providers(self, results: Iterable[FileResult]) -> dict:
        """Run every label provider over ``results`` and merge the output in place.

        Plugin labels are appended to ``result.insights["defect_labels"]`` with the
        standard label fields filled in (``severity`` defaults to ``"warn"``) and a
        ``source`` of ``"plugin:<name>"``. Provider failures become ``plugin``
        diagnostics on the affected result instead of exceptions.
        """
        label_count = 0
        failure_count = 0
        for result in results:
            labels, failures = self.collect_labels(result)
            failure_count += failures
            if labels:
                result.insights.setdefault("defect_labels", []).extend(labels)
                label_count += len(labels)
        return {"labels": label_count, "failures": failure_count}

    def collect_labels(self, result: FileResult) -> tuple[list[dict], int]:
        """Run every label provider on ``result`` without merging anything.

        Returns the normalized labels and the number of failed providers; each
        failure is recorded as a ``plugin`` diagnostic on ``result``.
        """
        labels: list[dict] = []
        failures = 0
        for name, produced, failure in self._run_providers(result):
            if failure is not None:
                failures += 1
                result.diagnostics.append(
                    DiagnosticEntry(
                        code="plugin_label_provider_failed",
                        severity="warn",
                        source="plugin",
                        message=f"Label provider {name!r} failed: {failure['error']}",
                        context={"plugin": name},
                    )
                )
                continue
            labels.extend(_normalize_label(label, name) for label in produced)
        return labels, failures

    def render(self, name: str, results: list[FileResult]) -> str:
        key = _name(name)
        if key not in self.reporters:
            raise KeyError(f"Unknown reporter plugin: {name}")
        rendered = self.reporters[key](results)
        if not isinstance(rendered, str):
            raise TypeError("reporter plugin must return str")
        return rendered

    def conformance_report(self) -> dict:
        """Exercise every registered plugin against a synthetic result.

        ``"ok"`` means the plugin ran and returned the right shape. The report also
        lists plugins that failed during discovery and a top-level ``passed`` flag.
        """
        provider_status = {}
        sample = FileResult(path="conformance.wav")
        for name, provider in self.label_providers.items():
            try:
                produced = provider(sample)
            except Exception as exc:
                provider_status[name] = f"error: {exc}"
                continue
            try:
                _validate_labels(produced)
                provider_status[name] = "ok"
            except TypeError as exc:
                provider_status[name] = f"invalid: {exc}"
        reporter_status = {}
        for name, reporter in self.reporters.items():
            try:
                reporter_status[name] = "ok" if isinstance(reporter([]), str) else "invalid: reporter must return str"
            except Exception as exc:
                reporter_status[name] = f"error: {exc}"
        passed = (
            not self.discovery_failures
            and all(status == "ok" for status in provider_status.values())
            and all(status == "ok" for status in reporter_status.values())
        )
        return {
            "api_version": PLUGIN_API_VERSION,
            "label_providers": provider_status,
            "reporters": reporter_status,
            "plugins": dict(self.plugins),
            "discovery_failures": list(self.discovery_failures),
            "passed": passed,
        }


def is_compatible_api(required: str) -> bool:
    """True when a plugin built for API ``required`` can run on this qualiax."""
    try:
        req_major, req_minor = _parse_version(required)
    except ValueError:
        return False
    major, minor = _parse_version(PLUGIN_API_VERSION)
    return req_major == major and req_minor <= minor


def _parse_version(value: str) -> tuple[int, int]:
    parts = value.strip().split(".")
    if not parts or not parts[0].isdigit():
        raise ValueError(f"invalid plugin API version: {value!r}")
    minor = parts[1] if len(parts) > 1 else "0"
    if not minor.isdigit():
        raise ValueError(f"invalid plugin API version: {value!r}")
    return int(parts[0]), int(minor)


def _entry_points(group: str):
    entry_points = metadata.entry_points()
    if hasattr(entry_points, "select"):
        return list(entry_points.select(group=group))
    return list(entry_points.get(group, []))  # Python 3.9


def _validate_labels(produced) -> None:
    if not isinstance(produced, list) or any(not isinstance(item, dict) or "id" not in item for item in produced):
        raise TypeError("label provider must return a list of objects containing id")


def _normalize_label(label: dict, plugin: str) -> dict:
    normalized = dict(label)
    normalized.setdefault("source", f"plugin:{plugin}")
    normalized["id"] = str(normalized["id"])
    normalized.setdefault("severity", "warn")
    normalized.setdefault("confidence", None)
    normalized.setdefault("evidence", "")
    normalized.setdefault("evidence_metrics", [])
    return normalized


def _name(value: str) -> str:
    key = value.strip().lower()
    if not key:
        raise ValueError("plugin name cannot be empty")
    return key

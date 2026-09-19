"""Versioned plugin discovery and isolated execution."""
from __future__ import annotations

from importlib import metadata
from typing import Callable

from .models import FileResult

PLUGIN_API_VERSION = "1.0"


class PluginManager:
    def __init__(self):
        self.label_providers: dict[str, Callable] = {}
        self.reporters: dict[str, Callable] = {}

    def register_label_provider(self, name: str, provider: Callable) -> None:
        self.label_providers[_name(name)] = provider

    def register_reporter(self, name: str, reporter: Callable) -> None:
        self.reporters[_name(name)] = reporter

    def discover(self, group: str = "qualiax.plugins") -> list[str]:
        discovered = []
        entry_points = metadata.entry_points()
        selected = entry_points.select(group=group) if hasattr(entry_points, "select") else entry_points.get(group, [])
        for entry_point in selected:
            plugin = entry_point.load()
            register = getattr(plugin, "register", plugin)
            register(self)
            discovered.append(entry_point.name)
        return discovered

    def run_label_providers(self, result: FileResult) -> tuple[list[dict], list[dict]]:
        labels = []
        failures = []
        for name, provider in self.label_providers.items():
            try:
                produced = provider(result)
                if not isinstance(produced, list) or any(not isinstance(item, dict) or "id" not in item for item in produced):
                    raise TypeError("label provider must return a list of objects containing id")
                labels.extend(produced)
            except Exception as exc:
                failures.append({"plugin": name, "error": str(exc)})
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
        provider_status = {}
        sample = FileResult(path="conformance.wav")
        for name, provider in self.label_providers.items():
            try:
                produced = provider(sample)
                provider_status[name] = "ok" if isinstance(produced, list) else "invalid"
            except Exception as exc:
                provider_status[name] = f"error: {exc}"
        reporter_status = {}
        for name, reporter in self.reporters.items():
            try:
                reporter_status[name] = "ok" if isinstance(reporter([]), str) else "invalid"
            except Exception as exc:
                reporter_status[name] = f"error: {exc}"
        return {"api_version": PLUGIN_API_VERSION, "label_providers": provider_status, "reporters": reporter_status}


def _name(value: str) -> str:
    key = value.strip().lower()
    if not key:
        raise ValueError("plugin name cannot be empty")
    return key

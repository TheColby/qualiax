"""Compatibility helpers for serialized reports and CLI automation."""
from __future__ import annotations

from copy import deepcopy
from enum import IntEnum
from pathlib import Path
import hashlib
import json
import warnings
from typing import Callable, Iterable, Mapping

from .version import OUTPUT_SCHEMA_VERSION, __version__


class ExitCode(IntEnum):
    OK = 0
    INVALID_INPUT = 1
    QUALITY_GATE_FAILED = 2
    ANALYSIS_FAILED = 3
    CONTRACT_VIOLATION = 4


Migration = Callable[[dict], dict]


class MigrationRegistry:
    def __init__(self):
        self._migrations: dict[str, Migration] = {}

    def register(self, source_version: str, migration: Migration) -> None:
        self._migrations[str(source_version)] = migration

    def migrate(self, item: Mapping) -> dict:
        result = deepcopy(dict(item))
        seen = set()
        while str(result.get("schema_version", "legacy")) != OUTPUT_SCHEMA_VERSION:
            version = str(result.get("schema_version", "legacy"))
            if version in seen or version not in self._migrations:
                result = _normalize_current(result)
                break
            seen.add(version)
            result = self._migrations[version](result)
        return _normalize_current(result)


def migrate_report(payload: Iterable[Mapping] | Mapping, registry: MigrationRegistry | None = None):
    active = registry or default_migration_registry()
    if isinstance(payload, Mapping):
        return active.migrate(payload)
    return [active.migrate(item) for item in payload]


def default_migration_registry() -> MigrationRegistry:
    registry = MigrationRegistry()
    for version in ("legacy", "1.0", "2.0", "3.0", "3.1", "3.2", "3.3", "3.4", "3.5"):
        registry.register(version, _normalize_current)
    return registry


def warn_deprecated(feature: str, *, removal_version: str, alternative: str = "") -> None:
    suffix = f" Use {alternative} instead." if alternative else ""
    warnings.warn(
        f"{feature} is deprecated and will be removed in {removal_version}.{suffix}",
        DeprecationWarning,
        stacklevel=2,
    )


def build_asset_lock(paths: Iterable[str | Path]) -> dict[str, str]:
    return {str(Path(path)): _sha256(Path(path)) for path in sorted(map(Path, paths), key=str)}


def verify_asset_lock(lock: Mapping[str, str]) -> dict:
    mismatches = []
    for name, expected in lock.items():
        path = Path(name)
        actual = _sha256(path) if path.exists() else None
        if actual != expected:
            mismatches.append({"path": name, "expected": expected, "actual": actual})
    return {"valid": not mismatches, "mismatches": mismatches}


def write_asset_lock(paths: Iterable[str | Path], output: str | Path) -> Path:
    destination = Path(output)
    destination.write_text(json.dumps(build_asset_lock(paths), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _normalize_current(item: dict) -> dict:
    normalized = deepcopy(item)
    if "path" not in normalized and "file" in normalized:
        normalized["path"] = normalized.pop("file")
    normalized.setdefault("path", "")
    normalized.setdefault("source_file", normalized["path"])
    normalized.setdefault("content_type", "unknown")
    normalized.setdefault("duration_s", 0.0)
    normalized.setdefault("sample_rate_hz", None)
    normalized.setdefault("channels", None)
    normalized.setdefault("status", "ok")
    normalized.setdefault("error", None)
    normalized.setdefault("diagnostics", [])
    normalized.setdefault("group_health", [])
    normalized.setdefault("metrics", [])
    normalized.setdefault("insights", {})
    normalized.setdefault("provenance", {})
    normalized["schema_version"] = OUTPUT_SCHEMA_VERSION
    normalized.setdefault("tool_version", __version__)
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

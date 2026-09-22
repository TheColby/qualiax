"""Compatibility helpers for serialized reports and CLI automation.

Report schema history (``schema_version`` in each JSON report item):

* legacy (no ``schema_version``, qualiax < 0.3): file, duration/sample-rate/channel
  fields, notes, error, and metrics with name/value/unit/description/group/
  higher_is_better/warning.
* ``3.0`` (0.3 - 0.10): adds schema_version, tool_version, content_type, and
  speech_confidence.
* ``3.4`` (0.11): adds segment fields, confidence_notes, diagnostics, group_health,
  provenance, and per-metric reference_range/confidence/calibration_note.
* ``3.5`` (0.17+): adds the ``insights`` object.

:func:`migrate_report` upgrades any of these to the current contract so the result
passes :func:`qualiax.validate_report_payload`.
"""
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
    """Process exit codes for automation.

    ``INVALID_INPUT``: usage errors, missing or undecodable audio, invalid rule configs.
    ``QUALITY_GATE_FAILED``: threshold-rule violations and failed ``--ci`` gates.
    ``ANALYSIS_FAILED``: analysis crashed after the audio loaded (including ``--strict``).
    ``CONTRACT_VIOLATION``: ``--validate-output`` or ``qualiax insights validate`` failed.
    A run with several outcomes reports the most severe: ANALYSIS_FAILED, then
    INVALID_INPUT, then QUALITY_GATE_FAILED.
    """

    OK = 0
    INVALID_INPUT = 1
    QUALITY_GATE_FAILED = 2
    ANALYSIS_FAILED = 3
    CONTRACT_VIOLATION = 4


class QualiaxDeprecationWarning(DeprecationWarning):
    """DeprecationWarning carrying machine-readable details about what is deprecated."""

    def __init__(self, message: str, *, feature: str, removal_version: str, alternative: str = ""):
        super().__init__(message)
        self.feature = feature
        self.removal_version = removal_version
        self.alternative = alternative

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "removal_version": self.removal_version,
            "alternative": self.alternative or None,
            "message": str(self),
        }


Migration = Callable[[dict], dict]


class MigrationRegistry:
    """Chain of per-version migrations ending at the current schema.

    Each migration receives a report item at ``source_version`` and must return it
    at a newer ``schema_version``. :meth:`migrate` follows the chain and then fills
    any remaining current-schema defaults.
    """

    def __init__(self):
        self._migrations: dict[str, Migration] = {}

    def register(self, source_version: str, migration: Migration) -> None:
        self._migrations[str(source_version)] = migration

    def versions(self) -> list[str]:
        return sorted(self._migrations)

    def migrate(self, item: Mapping) -> dict:
        result = deepcopy(dict(item))
        seen = set()
        while _version_of(result) != OUTPUT_SCHEMA_VERSION:
            version = _version_of(result)
            if version in seen:
                raise ValueError(f"Migration chain for schema_version {version!r} does not advance the version.")
            if version not in self._migrations:
                raise ValueError(
                    f"Unsupported report schema_version {version!r}; "
                    f"known versions: {', '.join(self.versions()) or 'none'}."
                )
            seen.add(version)
            result = self._migrations[version](result)
            if not isinstance(result, dict):
                raise TypeError(f"Migration for schema_version {version!r} must return a dict.")
        return _normalize_current(result)


def migrate_report(payload: Iterable[Mapping] | Mapping, registry: MigrationRegistry | None = None):
    """Upgrade one report item (a mapping) or a list of items to the current schema.

    Raises ``ValueError`` for schema versions it does not know, including reports
    written by a newer qualiax, instead of relabeling them.
    """
    active = registry or default_migration_registry()
    if isinstance(payload, Mapping):
        return active.migrate(payload)
    return [active.migrate(item) for item in payload]


def default_migration_registry() -> MigrationRegistry:
    registry = MigrationRegistry()
    # "1.0"/"2.0" and "3.1"-"3.3" never shipped; they are accepted as aliases of
    # the nearest real schema so hand-written or third-party payloads still load.
    for version in ("legacy", "1.0", "2.0"):
        registry.register(version, _migrate_legacy_to_3_0)
    for version in ("3.0", "3.1", "3.2", "3.3"):
        registry.register(version, _migrate_3_0_to_3_4)
    registry.register("3.4", _migrate_3_4_to_3_5)
    return registry


def warn_deprecated(feature: str, *, removal_version: str, alternative: str = "", stacklevel: int = 2) -> None:
    """Emit a :class:`QualiaxDeprecationWarning`; ``stacklevel`` counts from the caller of this function."""
    suffix = f" Use {alternative} instead." if alternative else ""
    warnings.warn(
        QualiaxDeprecationWarning(
            f"{feature} is deprecated and will be removed in {removal_version}.{suffix}",
            feature=feature,
            removal_version=removal_version,
            alternative=alternative,
        ),
        stacklevel=stacklevel,
    )


def build_asset_lock(paths: Iterable[str | Path], *, base_dir: str | Path | None = None) -> dict[str, str]:
    """Map each asset path to its SHA-256.

    With ``base_dir`` the keys are POSIX paths relative to it, so the lock file can
    be committed and verified from another checkout or machine.
    """
    base = Path(base_dir) if base_dir is not None else None
    lock = {}
    for path in sorted(map(Path, paths), key=str):
        key = path.resolve().relative_to(base.resolve()).as_posix() if base is not None else str(path)
        lock[key] = _sha256(path)
    return lock


def verify_asset_lock(lock: Mapping[str, str] | str | Path, *, base_dir: str | Path | None = None) -> dict:
    """Check files against a lock (a mapping, or the path of a JSON lock file).

    Missing and unreadable files are reported as mismatches with ``actual: None``.
    """
    entries = load_asset_lock(lock) if isinstance(lock, (str, Path)) else lock
    base = Path(base_dir) if base_dir is not None else None
    mismatches = []
    for name, expected in entries.items():
        path = base / name if base is not None and not Path(name).is_absolute() else Path(name)
        try:
            actual = _sha256(path) if path.is_file() else None
        except OSError:
            actual = None
        if actual != expected:
            mismatches.append({"path": name, "expected": expected, "actual": actual})
    return {"valid": not mismatches, "mismatches": mismatches}


def write_asset_lock(
    paths: Iterable[str | Path],
    output: str | Path,
    *,
    base_dir: str | Path | None = None,
) -> Path:
    destination = Path(output)
    destination.write_text(
        json.dumps(build_asset_lock(paths, base_dir=base_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_asset_lock(path: str | Path) -> dict[str, str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in payload.items()):
        raise ValueError(f"Asset lock {path} must be a JSON object mapping paths to SHA-256 strings.")
    return payload


def _version_of(item: Mapping) -> str:
    value = item.get("schema_version")
    return "legacy" if value is None else str(value)


def _migrate_legacy_to_3_0(item: dict) -> dict:
    migrated = _rename_known_aliases(item)
    migrated.setdefault("tool_version", "unknown")
    migrated.setdefault("content_type", None)
    migrated.setdefault("speech_confidence", None)
    migrated["schema_version"] = "3.0"
    return migrated


def _migrate_3_0_to_3_4(item: dict) -> dict:
    migrated = _rename_known_aliases(item)
    for key in ("source_file", "segment_index", "total_segments", "segment_start_s", "segment_end_s", "provenance"):
        migrated.setdefault(key, None)
    for key in ("confidence_notes", "diagnostics", "group_health"):
        migrated.setdefault(key, [])
    migrated["metrics"] = [_migrate_metric(metric) for metric in migrated.get("metrics") or []]
    migrated["schema_version"] = "3.4"
    return migrated


def _migrate_3_4_to_3_5(item: dict) -> dict:
    migrated = _rename_known_aliases(item)
    migrated.setdefault("insights", {})
    migrated["schema_version"] = "3.5"
    return migrated


def _migrate_metric(metric):
    if not isinstance(metric, dict):
        return metric
    migrated = dict(metric)
    migrated.setdefault("unit", "")
    migrated.setdefault("description", "")
    migrated.setdefault("group", "")
    migrated.setdefault("higher_is_better", None)
    migrated.setdefault("warning", None)
    migrated.setdefault("reference_range", None)
    migrated.setdefault("confidence", None)
    migrated.setdefault("calibration_note", None)
    return migrated


def _rename_known_aliases(item: dict) -> dict:
    """Undo field names that are not part of any qualiax schema.

    qualiax 1.0.0's migrate_report wrote ``path``/``sample_rate_hz``/``status``
    instead of the real field names; reports saved from it are repaired here.
    """
    migrated = dict(item)
    if "file" not in migrated and "path" in migrated:
        migrated["file"] = migrated.pop("path")
    elif "path" in migrated and migrated.get("path") == migrated.get("file"):
        migrated.pop("path")
    if "sample_rate_hz" in migrated:
        value = migrated.pop("sample_rate_hz")
        migrated.setdefault("sample_rate", value)
    if migrated.get("status") == "ok":
        migrated.pop("status")
    return migrated


def _normalize_current(item: dict) -> dict:
    normalized = _rename_known_aliases(deepcopy(item))
    normalized.setdefault("file", "")
    normalized.setdefault("source_file", None)
    normalized.setdefault("segment_index", None)
    normalized.setdefault("total_segments", None)
    normalized.setdefault("segment_start_s", None)
    normalized.setdefault("segment_end_s", None)
    for key, default in (("duration_s", 0.0), ("sample_rate", 0), ("channels", 0)):
        if normalized.get(key) is None:
            normalized[key] = default
    normalized.setdefault("bit_depth", None)
    normalized.setdefault("content_type", None)
    normalized.setdefault("speech_confidence", None)
    for key in ("notes", "confidence_notes", "diagnostics", "group_health"):
        if not isinstance(normalized.get(key), list):
            normalized[key] = []
    if normalized.get("provenance") == {}:
        normalized["provenance"] = None
    normalized.setdefault("provenance", None)
    if not isinstance(normalized.get("insights"), dict):
        normalized["insights"] = {}
    normalized.setdefault("error", None)
    normalized["metrics"] = [_migrate_metric(metric) for metric in normalized.get("metrics") or []]
    normalized["schema_version"] = OUTPUT_SCHEMA_VERSION
    normalized.setdefault("tool_version", __version__)
    return normalized


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

"""Release qualification and reproducibility metadata."""
from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable, Mapping

from .version import OUTPUT_SCHEMA_VERSION, __version__

_RELEASE_VERSION = re.compile(r"\d+\.\d+\.\d+")
_PYTHON_VERSION = re.compile(r"3\.\d+")
_REQUIREMENT_NAME = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_EXTRA_MARKER = re.compile(r"extra\s*==\s*['\"]([^'\"]+)['\"]")


def release_readiness(
    *,
    version: str,
    tests_passed: bool,
    schemas_valid: bool,
    docs_present: bool,
    supported_python: Iterable[str],
    security_findings: int = 0,
    extra_checks: Mapping[str, bool] | None = None,
) -> dict:
    """Combine release gates into a single ready / not-ready decision.

    Besides the caller-supplied results (tests, schemas, docs, security, and any
    ``extra_checks``), two gates are checked here: ``version`` must be a final
    ``MAJOR.MINOR.PATCH`` release number, and ``support_matrix`` must list at least
    one ``3.N`` Python version and nothing else.
    """
    supported = [str(item) for item in supported_python]
    checks = {
        "tests": bool(tests_passed),
        "schemas": bool(schemas_valid),
        "documentation": bool(docs_present),
        "security": int(security_findings) == 0,
        "version": bool(_RELEASE_VERSION.fullmatch(str(version).strip())),
        "support_matrix": bool(supported) and all(_PYTHON_VERSION.fullmatch(item) for item in supported),
    }
    for name, passed in (extra_checks or {}).items():
        if name in checks:
            raise ValueError(f"extra check {name!r} collides with a built-in release check")
        checks[str(name)] = bool(passed)
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    return {
        "version": version,
        "ready": all(checks.values()),
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if not passed),
        "security_findings": int(security_findings),
        "support_matrix": {"python": supported, "running": running, "running_supported": running in supported},
    }


def dependency_inventory(package: str | None = None) -> list[dict]:
    """Inventory installed distributions.

    With no argument, every distribution in the environment is listed (``name`` and
    ``version``). With ``package`` (for example ``"qualiax"``), only that package's
    declared requirements are listed, each with its ``requirement`` string, the
    ``extra`` that pulls it in (``None`` for core dependencies), and the installed
    ``version`` (``None`` when it is not installed).
    """
    if package is not None:
        return _declared_requirements(package)
    inventory: dict[str, dict] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name and name.lower() not in inventory:
            inventory[name.lower()] = {"name": name, "version": distribution.version}
    return sorted(inventory.values(), key=lambda item: item["name"].lower())


def reproducibility_manifest(paths: Iterable[str | Path] = ()) -> dict:
    """Describe the interpreter, qualiax build, core dependency versions, and file hashes."""
    files = {}
    for value in sorted(map(Path, paths), key=str):
        files[str(value)] = _sha256(value)
    try:
        dependencies = [
            {"name": item["name"], "version": item["version"]}
            for item in _declared_requirements("qualiax")
            if item["extra"] is None
        ]
    except metadata.PackageNotFoundError:
        dependencies = []
    return {
        "qualiax_version": __version__,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "dependencies": dependencies,
        "files": files,
    }


def write_reproducibility_manifest(paths: Iterable[str | Path], output: str | Path) -> Path:
    destination = Path(output)
    destination.write_text(json.dumps(reproducibility_manifest(paths), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _declared_requirements(package: str) -> list[dict]:
    requirements = metadata.requires(package) or []
    inventory: list[dict] = []
    seen: set[tuple[str, str | None]] = set()
    for requirement in requirements:
        match = _REQUIREMENT_NAME.match(requirement)
        if not match:
            continue
        name = match.group(1)
        extra_match = _EXTRA_MARKER.search(requirement)
        extra = extra_match.group(1) if extra_match else None
        key = (name.lower(), extra)
        if key in seen:
            continue
        seen.add(key)
        try:
            version = metadata.version(name)
        except metadata.PackageNotFoundError:
            version = None
        inventory.append(
            {"name": name, "requirement": requirement.split(";")[0].strip(), "extra": extra, "version": version}
        )
    return sorted(inventory, key=lambda item: (item["extra"] or "", item["name"].lower()))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

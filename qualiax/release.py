"""Release qualification and reproducibility metadata."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Iterable


def release_readiness(
    *,
    version: str,
    tests_passed: bool,
    schemas_valid: bool,
    docs_present: bool,
    supported_python: Iterable[str],
    security_findings: int = 0,
) -> dict:
    checks = {
        "tests": bool(tests_passed),
        "schemas": bool(schemas_valid),
        "documentation": bool(docs_present),
        "security": int(security_findings) == 0,
    }
    return {
        "version": version,
        "ready": all(checks.values()),
        "checks": checks,
        "security_findings": int(security_findings),
        "support_matrix": {"python": list(supported_python)},
    }


def dependency_inventory() -> list[dict[str, str]]:
    inventory = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            inventory.append({"name": name, "version": distribution.version})
    return sorted(inventory, key=lambda item: item["name"].lower())


def reproducibility_manifest(paths: Iterable[str | Path] = ()) -> dict:
    files = {}
    for value in sorted(map(Path, paths), key=str):
        files[str(value)] = _sha256(value)
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "files": files,
    }


def write_reproducibility_manifest(paths: Iterable[str | Path], output: str | Path) -> Path:
    destination = Path(output)
    destination.write_text(json.dumps(reproducibility_manifest(paths), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

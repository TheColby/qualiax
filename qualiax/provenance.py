"""
Runtime and model provenance helpers.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from functools import lru_cache
from importlib import metadata
from pathlib import Path

from .assets import MODEL_ASSETS, asset_status, model_dir
from .models import MetricResult, ProvenanceInfo
from .version import __version__
from . import gpu


# Libraries whose versions can change metric values. Their versions feed the
# runtime fingerprint so a dependency upgrade shows up as a provenance change.
_CALIBRATION_LIBRARIES = ("numpy", "scipy", "pesq", "pystoi", "onnxruntime", "crepe", "torch", "torchaudio")

_DIGEST_CACHE: dict[tuple[str, int, int], str] = {}


def build_provenance(metrics: list[MetricResult]) -> ProvenanceInfo:
    """Describe the backend, runtime, and model assets behind a set of metrics.

    ``runtime_fingerprint`` hashes the visible fields plus the qualiax version and
    the installed versions of metric-affecting libraries, so two results with the
    same fingerprint were produced by the same calibration-relevant stack.
    """
    model_assets = _collect_model_assets()
    payload = {
        "compute_backend": gpu.get_device_str(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "model_runtime": _detect_model_runtime(metrics),
        "model_assets": model_assets,
    }
    fingerprint_source = {
        **payload,
        "qualiax_version": __version__,
        "libraries": _library_versions(),
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return ProvenanceInfo(
        compute_backend=payload["compute_backend"],
        python_version=payload["python_version"],
        platform=payload["platform"],
        runtime_fingerprint=fingerprint,
        model_runtime=payload["model_runtime"],
        model_assets=model_assets,
    )


def model_asset_paths() -> list[Path]:
    """Paths of the registered model files that are downloaded, e.g. for ``write_asset_lock``."""
    model_dir = _model_dir()
    return [
        model_dir / asset.relative_path
        for asset in MODEL_ASSETS.values()
        if (model_dir / asset.relative_path).is_file()
    ]


def _model_dir() -> Path:
    return model_dir()


def _detect_model_runtime(metrics: list[MetricResult]) -> str:
    if any(metric.confidence == "model" for metric in metrics):
        try:
            import onnxruntime as ort  # type: ignore

            return f"onnxruntime {ort.__version__}"
        except Exception:
            return "model runtime unavailable"
    return "none"


@lru_cache(maxsize=1)
def _library_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in _CALIBRATION_LIBRARIES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _collect_model_assets() -> list[dict[str, object]]:
    directory = _model_dir()
    collected: list[dict[str, object]] = []
    for asset in MODEL_ASSETS.values():
        path = directory / asset.relative_path
        if not path.is_file():
            continue
        collected.append(
            {
                "name": asset.name,
                "file": asset.relative_path,
                "bytes": path.stat().st_size,
                "sha256": _sha256_prefix(path),
                "verified": asset_status(asset.name, directory) == "ok",
                "source": asset.source,
            }
        )
    return collected


def _sha256_prefix(path: Path) -> str:
    # Model files can be large and provenance is built for every result, so reuse
    # the digest until the file's size or modification time changes.
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _DIGEST_CACHE.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()[:16]
    _DIGEST_CACHE[key] = value
    return value

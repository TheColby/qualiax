"""
Runtime and model provenance helpers.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

from .models import MetricResult, ProvenanceInfo
from . import gpu


_KNOWN_MODEL_FILES = [
    "dnsmos_sig.onnx",
    "dnsmos_bak.onnx",
    "dnsmos_ovrl.onnx",
    "aecmos.onnx",
    "utmos.onnx",
    "sheet.onnx",
]


def build_provenance(metrics: list[MetricResult]) -> ProvenanceInfo:
    model_assets = _collect_model_assets()
    payload = {
        "compute_backend": gpu.get_device_str(),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "model_runtime": _detect_model_runtime(metrics),
        "model_assets": model_assets,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]
    return ProvenanceInfo(
        compute_backend=payload["compute_backend"],
        python_version=payload["python_version"],
        platform=payload["platform"],
        runtime_fingerprint=fingerprint,
        model_runtime=payload["model_runtime"],
        model_assets=model_assets,
    )


def _detect_model_runtime(metrics: list[MetricResult]) -> str:
    if any(metric.confidence == "model" for metric in metrics):
        try:
            import onnxruntime as ort  # type: ignore

            return f"onnxruntime {ort.__version__}"
        except Exception:
            return "model runtime unavailable"
    return "none"


def _collect_model_assets() -> list[dict[str, object]]:
    model_dir = Path(__file__).parent
    assets: list[dict[str, object]] = []
    for filename in _KNOWN_MODEL_FILES:
        path = model_dir / filename
        if not path.exists():
            continue
        assets.append(
            {
                "name": filename,
                "bytes": path.stat().st_size,
                "sha256": _sha256_prefix(path),
            }
        )
    return assets


def _sha256_prefix(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]

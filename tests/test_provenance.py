import hashlib
import sys

import pytest

from qualiax import provenance
from qualiax.migrations import verify_asset_lock, write_asset_lock
from qualiax.models import FileResult, MetricResult
from qualiax.validation import validate_report_payload


@pytest.fixture
def model_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "_model_dir", lambda: tmp_path)
    monkeypatch.setattr(provenance, "_DIGEST_CACHE", {})
    return tmp_path


def test_no_models_installed(model_dir):
    info = provenance.build_provenance([MetricResult(name="SNR", value=1.0, confidence="measured")])

    assert info.model_assets == []
    assert info.model_runtime == "none"
    assert info.python_version == sys.version.split()[0]
    assert len(info.runtime_fingerprint) == 16
    assert provenance.model_asset_paths() == []


def test_model_assets_are_reported_with_digests(model_dir):
    (model_dir / "dnsmos").mkdir()
    (model_dir / "dnsmos" / "sig_bak_ovr.onnx").write_bytes(b"not-the-real-weights")
    (model_dir / "dnsmos" / "unrelated.onnx").write_bytes(b"ignored")

    info = provenance.build_provenance([])

    assert info.model_assets == [
        {
            "name": "dnsmos-p835",
            "file": "dnsmos/sig_bak_ovr.onnx",
            "bytes": 20,
            "sha256": hashlib.sha256(b"not-the-real-weights").hexdigest()[:16],
            "verified": False,
            "source": provenance.MODEL_ASSETS["dnsmos-p835"].source,
        }
    ]
    assert [path.name for path in provenance.model_asset_paths()] == ["sig_bak_ovr.onnx"]


def test_fingerprint_tracks_model_changes(model_dir):
    (model_dir / "dnsmos").mkdir()
    asset = model_dir / "dnsmos" / "model_v8.onnx"
    asset.write_bytes(b"v1")
    first = provenance.build_provenance([])
    again = provenance.build_provenance([])
    asset.write_bytes(b"v2-longer")
    changed = provenance.build_provenance([])

    assert first.runtime_fingerprint == again.runtime_fingerprint
    assert changed.runtime_fingerprint != first.runtime_fingerprint
    assert changed.model_assets[0]["sha256"] == hashlib.sha256(b"v2-longer").hexdigest()[:16]


def test_fingerprint_tracks_qualiax_and_library_versions(model_dir, monkeypatch):
    baseline = provenance.build_provenance([]).runtime_fingerprint

    monkeypatch.setattr(provenance, "__version__", "1.0.1")
    bumped = provenance.build_provenance([]).runtime_fingerprint
    monkeypatch.setattr(provenance, "__version__", "1.0.0")
    monkeypatch.setattr(provenance, "_library_versions", lambda: {"numpy": "0.0.1"})
    other_numpy = provenance.build_provenance([]).runtime_fingerprint

    assert len({baseline, bumped, other_numpy}) == 3


def test_model_runtime_reports_missing_onnxruntime(model_dir, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    info = provenance.build_provenance([MetricResult(name="DNSMOS OVRL", value=3.1, confidence="model")])

    assert info.model_runtime == "model runtime unavailable"


def test_provenance_satisfies_report_schema(model_dir):
    (model_dir / "dnsmos").mkdir()
    (model_dir / "dnsmos" / "model_v8.onnx").write_bytes(b"p808")
    result = FileResult(path="a.wav", provenance=provenance.build_provenance([]))

    assert validate_report_payload([result.to_dict()]) == []


def test_model_assets_can_be_locked_and_verified(model_dir, tmp_path):
    (model_dir / "dnsmos").mkdir()
    (model_dir / "dnsmos" / "sig_bak_ovr.onnx").write_bytes(b"ovrl")
    lock = write_asset_lock(provenance.model_asset_paths(), tmp_path / "models.lock", base_dir=model_dir)

    assert verify_asset_lock(lock, base_dir=model_dir)["valid"] is True
    (model_dir / "dnsmos" / "sig_bak_ovr.onnx").write_bytes(b"swapped")
    assert verify_asset_lock(lock, base_dir=model_dir)["valid"] is False

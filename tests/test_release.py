import hashlib
import json
from importlib import metadata

import pytest

from qualiax.release import (
    dependency_inventory,
    release_readiness,
    reproducibility_manifest,
    write_reproducibility_manifest,
)
from qualiax.version import __version__


def _ready(**overrides):
    arguments = {
        "version": "1.0.0",
        "tests_passed": True,
        "schemas_valid": True,
        "docs_present": True,
        "supported_python": ("3.9", "3.10", "3.11", "3.12"),
    }
    arguments.update(overrides)
    return release_readiness(**arguments)


def test_release_ready_when_every_gate_passes():
    report = _ready()
    assert report["ready"] is True
    assert report["failed_checks"] == []
    assert report["support_matrix"]["python"] == ["3.9", "3.10", "3.11", "3.12"]


@pytest.mark.parametrize(
    "overrides, failed",
    [
        ({"tests_passed": False}, ["tests"]),
        ({"security_findings": 2}, ["security"]),
        ({"version": "1.0.0rc1"}, ["version"]),
        ({"version": "1.0"}, ["version"]),
        ({"supported_python": ()}, ["support_matrix"]),
        ({"supported_python": ("3.11", "2.7-ish")}, ["support_matrix"]),
        ({"docs_present": False, "schemas_valid": False}, ["documentation", "schemas"]),
        ({"extra_checks": {"changelog": False, "wheel_built": True}}, ["changelog"]),
    ],
)
def test_release_gates_block_on_failures(overrides, failed):
    report = _ready(**overrides)
    assert report["ready"] is False
    assert report["failed_checks"] == failed


def test_extra_checks_cannot_shadow_builtin_gates():
    with pytest.raises(ValueError):
        _ready(extra_checks={"tests": True})


def test_dependency_inventory_for_a_package_lists_declared_requirements():
    inventory = dependency_inventory("qualiax")
    by_name = {(item["name"].lower(), item["extra"]): item for item in inventory}

    for core in ("numpy", "scipy", "click"):
        entry = by_name[(core, None)]
        assert entry["version"] == metadata.version(core)
        assert entry["requirement"].lower().startswith(core)
    assert ("pesq", "perceptual") in by_name
    assert ("onnxruntime", "ml") in by_name
    pesq = by_name[("pesq", "perceptual")]
    try:
        expected = metadata.version("pesq")
    except metadata.PackageNotFoundError:
        expected = None
    assert pesq["version"] == expected
    assert "extra" not in pesq["requirement"]


def test_dependency_inventory_for_environment_is_unique_and_sorted():
    inventory = dependency_inventory()
    names = [item["name"].lower() for item in inventory]
    assert names == sorted(names)
    assert len(names) == len(set(names))
    assert "numpy" in names


def test_reproducibility_manifest_records_build_and_file_hashes(tmp_path):
    asset = tmp_path / "model.onnx"
    asset.write_bytes(b"weights")
    output = write_reproducibility_manifest([asset], tmp_path / "manifest.json")

    manifest = json.loads(output.read_text())

    assert manifest == reproducibility_manifest([asset])
    assert manifest["qualiax_version"] == __version__
    assert manifest["files"] == {str(asset): hashlib.sha256(b"weights").hexdigest()}
    dependencies = {item["name"].lower(): item["version"] for item in manifest["dependencies"]}
    assert dependencies["numpy"] == metadata.version("numpy")
    assert "pesq" not in dependencies  # optional extras are not part of the core stack

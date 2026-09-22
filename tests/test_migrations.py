import json
import warnings

import pytest

from qualiax.insights import enrich_results
from qualiax.migrations import (
    ExitCode,
    MigrationRegistry,
    QualiaxDeprecationWarning,
    build_asset_lock,
    load_asset_lock,
    migrate_report,
    verify_asset_lock,
    warn_deprecated,
    write_asset_lock,
)
from qualiax.models import DiagnosticEntry, FileResult, GroupHealth, MetricResult, ProvenanceInfo
from qualiax.reporter import JsonReporter
from qualiax.validation import validate_report_payload
from qualiax.version import OUTPUT_SCHEMA_VERSION


def _current_payload():
    result = FileResult(
        path="call.wav",
        duration_s=2.0,
        sample_rate=16000,
        channels=1,
        bit_depth=16,
        content_type="speech",
        speech_confidence=0.9,
        diagnostics=[DiagnosticEntry(code="c", severity="warn", source="runtime", message="m")],
        group_health=[GroupHealth(group="noise", status="ok", metric_count=1)],
        provenance=ProvenanceInfo(compute_backend="cpu", model_assets=[]),
        metrics=[MetricResult(name="Estimated SNR", value=20.0, unit="dB", group="noise", reference_range=(15.0, 60.0))],
    )
    enrich_results([result])
    return json.loads(JsonReporter().render([result]))


_LEGACY_ITEM = {
    "file": "old.wav",
    "duration_s": 1.5,
    "sample_rate": 8000,
    "channels": 1,
    "bit_depth": 16,
    "notes": ["pre-0.3 output"],
    "error": None,
    "metrics": [
        {"name": "SNR", "value": 12.0, "unit": "dB", "description": "", "group": "noise", "higher_is_better": True, "warning": None}
    ],
}


def test_current_reports_round_trip_unchanged():
    payload = _current_payload()
    assert validate_report_payload(payload) == []

    migrated = migrate_report(payload)

    assert migrated == payload
    assert validate_report_payload(migrated) == []


def test_legacy_reports_migrate_to_a_valid_current_report():
    migrated = migrate_report([_LEGACY_ITEM])

    assert validate_report_payload(migrated) == []
    item = migrated[0]
    assert item["schema_version"] == OUTPUT_SCHEMA_VERSION
    assert item["file"] == "old.wav"
    assert item["tool_version"] == "unknown"
    assert item["sample_rate"] == 8000
    assert item["notes"] == ["pre-0.3 output"]
    assert item["insights"] == {}
    assert item["provenance"] is None
    assert item["metrics"][0]["confidence"] is None
    assert item["metrics"][0]["higher_is_better"] is True


def test_schema_3_0_and_3_4_reports_migrate():
    v30 = {**_LEGACY_ITEM, "schema_version": "3.0", "tool_version": "0.10.0", "content_type": "speech", "speech_confidence": 0.8}
    v34 = migrate_report(v30)
    v34_input = {key: value for key, value in v34.items() if key != "insights"}
    v34_input["schema_version"] = "3.4"

    assert validate_report_payload([v34]) == []
    assert v34["tool_version"] == "0.10.0"  # producer version is preserved
    assert v34["content_type"] == "speech"
    assert migrate_report(v34_input)["insights"] == {}
    assert validate_report_payload([migrate_report(v34_input)]) == []


def test_output_of_the_1_0_0_migrator_is_repaired():
    # qualiax 1.0.0's migrate_report produced these invalid field names.
    broken = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "path": "a.wav",
        "source_file": "a.wav",
        "content_type": "unknown",
        "duration_s": 0.0,
        "sample_rate_hz": None,
        "channels": None,
        "status": "ok",
        "error": None,
        "diagnostics": [],
        "group_health": [],
        "metrics": [],
        "insights": {},
        "provenance": {},
        "tool_version": "1.0.0",
    }

    repaired = migrate_report(broken)

    assert validate_report_payload([repaired]) == []
    assert repaired["file"] == "a.wav"
    assert "path" not in repaired and "status" not in repaired and "sample_rate_hz" not in repaired
    assert repaired["sample_rate"] == 0


def test_unknown_or_future_versions_are_rejected():
    with pytest.raises(ValueError, match="Unsupported report schema_version '9.9'"):
        migrate_report({"schema_version": "9.9", "file": "future.wav"})


def test_custom_registry_chain_and_cycle_detection():
    registry = MigrationRegistry()
    registry.register("0.1", lambda item: {"schema_version": "3.4", "file": item["name"]})
    custom_insights = {"repair_suggestions": ["custom step ran"]}
    registry.register("3.4", lambda item: {**item, "schema_version": OUTPUT_SCHEMA_VERSION, "insights": custom_insights})

    migrated = registry.migrate({"schema_version": "0.1", "name": "x.wav"})

    assert migrated["file"] == "x.wav"
    assert migrated["insights"] == custom_insights
    assert migrated["schema_version"] == OUTPUT_SCHEMA_VERSION
    assert validate_report_payload([migrated]) == []
    assert registry.versions() == ["0.1", "3.4"]

    stuck = MigrationRegistry()
    stuck.register("0.1", lambda item: item)
    with pytest.raises(ValueError, match="does not advance"):
        stuck.migrate({"schema_version": "0.1"})
    with pytest.raises(ValueError, match="known versions: none"):
        MigrationRegistry().migrate({"schema_version": "0.1"})


def test_migrate_report_accepts_single_items_and_does_not_mutate_input():
    item = dict(_LEGACY_ITEM)
    migrated = migrate_report(item)
    assert isinstance(migrated, dict)
    assert "schema_version" not in item


def test_exit_codes_are_stable():
    assert {code.name: int(code) for code in ExitCode} == {
        "OK": 0,
        "INVALID_INPUT": 1,
        "QUALITY_GATE_FAILED": 2,
        "ANALYSIS_FAILED": 3,
        "CONTRACT_VIOLATION": 4,
    }


def test_warn_deprecated_emits_structured_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_deprecated("analyze(strict=False)", removal_version="2.0.0", alternative="strict=True")

    warning = caught[0]
    assert issubclass(warning.category, DeprecationWarning)
    assert issubclass(warning.category, QualiaxDeprecationWarning)
    assert warning.message.to_dict() == {
        "feature": "analyze(strict=False)",
        "removal_version": "2.0.0",
        "alternative": "strict=True",
        "message": "analyze(strict=False) is deprecated and will be removed in 2.0.0. Use strict=True instead.",
    }
    assert warning.filename == __file__


def test_asset_lock_round_trip_with_portable_relative_paths(tmp_path, monkeypatch):
    models = tmp_path / "checkout" / "models"
    models.mkdir(parents=True)
    (models / "a.onnx").write_bytes(b"model-a")
    (models / "b.onnx").write_bytes(b"model-b")
    lock_path = write_asset_lock(sorted(models.glob("*.onnx")), tmp_path / "models.lock", base_dir=models)

    lock = load_asset_lock(lock_path)
    assert sorted(lock) == ["a.onnx", "b.onnx"]
    assert len(lock["a.onnx"]) == 64

    copy = tmp_path / "elsewhere"
    copy.mkdir()
    (copy / "a.onnx").write_bytes(b"model-a")
    (copy / "b.onnx").write_bytes(b"tampered")
    monkeypatch.chdir(tmp_path)

    result = verify_asset_lock(lock_path, base_dir=copy)
    assert result["valid"] is False
    assert [item["path"] for item in result["mismatches"]] == ["b.onnx"]
    assert verify_asset_lock(lock, base_dir=models)["valid"] is True


def test_verify_asset_lock_reports_missing_and_directory_entries(tmp_path):
    asset = tmp_path / "m.onnx"
    asset.write_bytes(b"x")
    lock = build_asset_lock([asset])
    lock[str(tmp_path)] = "0" * 64
    asset.unlink()

    result = verify_asset_lock(lock)

    assert result["valid"] is False
    assert {item["actual"] for item in result["mismatches"]} == {None}


def test_load_asset_lock_rejects_malformed_files(tmp_path):
    bad = tmp_path / "bad.lock"
    bad.write_text(json.dumps({"a.onnx": 5}))
    with pytest.raises(ValueError):
        load_asset_lock(bad)

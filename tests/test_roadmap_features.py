import pytest
import qualiax
from qualiax.migrations import QualiaxDeprecationWarning
from qualiax.calibration import CalibrationCase, calibrate_labels, confidence_interval, consistency_report
from qualiax.dataset_intelligence import audit_dataset
from qualiax.migrations import ExitCode, build_asset_lock, migrate_report, verify_asset_lock
from qualiax.models import FileResult, MetricResult
from qualiax.operations import AlertPolicy, DriftHistory, prometheus_metrics
from qualiax.performance import (
    AnalysisCache,
    LocalExecutorAdapter,
    benchmark_budget,
    incremental_sources,
    iter_audio_chunks,
)
from qualiax.plugins import PluginManager
from qualiax.release import release_readiness, reproducibility_manifest
from qualiax.repair import apply_repair_plan, build_repair_plan, evaluate_repair
from qualiax.review import ReviewStore


def _result(path: str, *, snr=24.0, peak=-2.0, content="speech", speaker=None):
    metrics = [
        MetricResult(name="SNR", value=snr, unit="dB", group="noise"),
        MetricResult(name="True Peak", value=peak, unit="dBTP", group="loudness"),
    ]
    if speaker:
        metrics.append(MetricResult(name="Speaker Cluster", value=speaker, group="speaker"))
    return FileResult(path=path, content_type=content, duration_s=1.0, metrics=metrics)


def test_calibration_reports_quality_and_consistency():
    cases = [
        CalibrationCase("a", {"noisy_floor"}, {"noisy_floor"}),
        CalibrationCase("b", set(), {"noisy_floor"}),
        CalibrationCase("c", {"clipping_risk"}, {"clipping_risk"}),
    ]
    report = calibrate_labels(cases)
    interval = confidence_interval([1.0, 2.0, 3.0, 4.0])
    consistency = consistency_report({"wav": 3.2, "mp3": 3.0, "aac": 3.1}, tolerance=0.25)

    assert report["labels"]["noisy_floor"]["precision"] == 0.5
    assert report["labels"]["clipping_risk"]["recall"] == 1.0
    assert interval[0] < interval[1]
    assert consistency["consistent"] is True


def test_dataset_intelligence_finds_duplicates_imbalance_leakage_and_remediation():
    results = [
        _result("train/a.wav", snr=10, speaker="s1"),
        _result("train/b.wav", snr=10, speaker="s1"),
        _result("validation/a-copy.wav", snr=10, speaker="s2"),
        _result("validation/music.wav", content="music", speaker="s1"),
    ]
    results[0].insights = {"quality_fingerprint": {"signature": "dup"}, "defect_labels": [{"id": "noisy_floor"}]}
    results[1].insights = {"quality_fingerprint": {"signature": "dup"}, "defect_labels": [{"id": "noisy_floor"}]}
    results[2].insights = {"quality_fingerprint": {"signature": "dup"}, "defect_labels": [{"id": "noisy_floor"}]}
    results[3].insights = {"quality_fingerprint": {"signature": "music"}, "defect_labels": []}

    audit = audit_dataset(results, expected_content_type="speech")

    assert audit["near_duplicate_groups"]
    assert audit["split_leakage"]
    assert audit["speaker_distribution"]["s1"] == 3
    assert audit["label_mismatch_risks"]
    assert audit["remediation_plan"]


def test_operations_persist_history_suppress_alerts_and_export_prometheus(tmp_path):
    history_path = tmp_path / "history.json"
    history = DriftHistory(history_path, max_entries=2)
    history.append("snr", 20.0, timestamp=100.0)
    history.append("snr", 15.0, timestamp=200.0)
    history.append("snr", 10.0, timestamp=300.0)

    policy = AlertPolicy(cooldown_seconds=60)
    assert policy.should_emit("noise", now=100.0)
    assert not policy.should_emit("noise", now=120.0)
    assert policy.should_emit("noise", now=170.0)

    text = prometheus_metrics([_result("a.wav", snr=12)])
    assert len(history.entries("snr")) == 2
    assert "qualiax_metric" in text
    assert 'metric="SNR"' in text


def test_repair_plan_is_safe_and_can_be_evaluated(tmp_path, monkeypatch):
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    result = _result(str(source), snr=7, peak=0.2)
    result.insights = {"defect_labels": [{"id": "noisy_floor"}, {"id": "clipping_risk"}]}
    plan = build_repair_plan(result, output_path=tmp_path / "repaired.wav")

    monkeypatch.setattr("qualiax.repair.subprocess.run", lambda *args, **kwargs: type("P", (), {"returncode": 0, "stderr": ""})())
    applied = apply_repair_plan(plan, dry_run=True)
    comparison = evaluate_repair(_result("before", snr=7), _result("after", snr=20))

    assert plan.source != plan.output
    assert any("afftdn" in argument for argument in plan.ffmpeg_args)
    assert applied["executed"] is False
    assert comparison["improvements"][0]["delta"] == 13.0


def test_plugin_manager_isolates_failures_and_checks_conformance():
    manager = PluginManager()
    manager.register_label_provider("good", lambda result: [{"id": "custom", "severity": "warn"}])
    manager.register_label_provider("bad", lambda result: 1 / 0)
    manager.register_reporter("text", lambda results: "ok")

    labels, failures = manager.run_label_providers(_result("a.wav"))
    conformance = manager.conformance_report()

    assert labels[0]["id"] == "custom"
    assert failures[0]["plugin"] == "bad"
    assert conformance["label_providers"]["good"] == "ok"
    assert manager.render("text", []) == "ok"


def test_review_store_persists_annotations_and_decisions(tmp_path):
    store = ReviewStore(tmp_path / "reviews.json")
    store.annotate("a.wav", "Needs a second listen", reviewer="alex", segment_index=2)
    store.decide("a.wav", "reject", reviewer="alex")
    reloaded = ReviewStore(tmp_path / "reviews.json")

    assert reloaded.for_file("a.wav")["decision"] == "reject"
    assert reloaded.for_file("a.wav")["annotations"][0]["segment_index"] == 2


def test_performance_cache_incremental_chunks_and_budget(tmp_path):
    source = tmp_path / "a.wav"
    source.write_bytes(b"abc")
    cache = AnalysisCache(tmp_path / "cache")
    cache.put(source, {"score": 1})

    assert cache.get(source) == {"score": 1}
    source.write_bytes(b"changed")
    assert cache.get(source) is None
    assert incremental_sources([source], {str(source): (0, 0)}) == [source]
    assert list(iter_audio_chunks(list(range(10)), 4)) == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert benchmark_budget({"runtime_s": 1.0}, {"runtime_s": 1.1}, max_regression=0.2)["passed"]
    with pytest.warns(QualiaxDeprecationWarning):
        assert LocalExecutorAdapter(max_workers=2).map(abs, [-1, -2]) == [1, 2]


def test_migration_and_release_readiness(tmp_path):
    old = [{"file": "a.wav", "metrics": [], "schema_version": "2.0"}]
    migrated = migrate_report(old)
    readiness = release_readiness(
        version="1.0.0",
        tests_passed=True,
        schemas_valid=True,
        docs_present=True,
        supported_python=("3.9", "3.10", "3.11", "3.12"),
    )

    assert migrated[0]["schema_version"] == qualiax.OUTPUT_SCHEMA_VERSION
    assert migrated[0]["diagnostics"] == []
    assert qualiax.validate_report_payload(migrated) == []
    assert ExitCode.QUALITY_GATE_FAILED == 2
    assert readiness["ready"] is True
    assert readiness["support_matrix"]["python"] == ["3.9", "3.10", "3.11", "3.12"]


def test_asset_locks_reproducibility_and_public_api(tmp_path):
    asset = tmp_path / "model.onnx"
    asset.write_bytes(b"stable model")
    lock = build_asset_lock([asset])
    manifest = reproducibility_manifest([asset])

    assert verify_asset_lock(lock)["valid"] is True
    assert manifest["files"][str(asset)] == lock[str(asset)]
    asset.write_bytes(b"changed model")
    assert verify_asset_lock(lock)["valid"] is False
    assert qualiax.audit_dataset is audit_dataset
    assert qualiax.__version__ == "1.1.0"

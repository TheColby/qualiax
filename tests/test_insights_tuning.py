"""Baseline, drift, and true-peak behaviour of --insights."""
from __future__ import annotations

import pytest

from qualiax.insights import enrich_results
from qualiax.models import FileResult, MetricResult


def _result(path="a.wav", *, snr=30.0, lufs=-16.0, peak=-3.0, centroid=1500.0, f0=120.0):
    return FileResult(
        path=path,
        duration_s=3.0,
        sample_rate=16_000,
        channels=1,
        content_type="speech",
        metrics=[
            MetricResult("Estimated SNR", snr, "dB", group="noise"),
            MetricResult("Integrated Loudness (LUFS)", lufs, "LUFS", group="loudness"),
            MetricResult("True Peak", peak, "dBTP", group="loudness"),
            MetricResult("Spectral Centroid", centroid, "Hz", group="spectral"),
            MetricResult("F0 Mean", f0, "Hz", group="prosody"),
        ],
    )


def _comparisons(current, baseline):
    enrich_results([current], baseline_results=baseline)
    comparison = current.insights["baseline_comparison"]
    return comparison, {item["feature"]: item for item in comparison["comparisons"]}


@pytest.mark.parametrize(
    ("change", "feature", "regressed"),
    [
        ({"lufs": -10.0}, "loudness.integrated_lufs", True),   # louder: no better direction
        ({"lufs": -22.0}, "loudness.integrated_lufs", True),   # quieter
        ({"f0": 160.0}, "prosody.f0_mean", True),               # pitch moved up
        ({"centroid": 1100.0}, "spectral.centroid", True),
        ({"snr": 45.0}, "noise.snr", False),                    # cleaner is not a regression
        ({"snr": 20.0}, "noise.snr", True),
        ({"peak": -8.0}, "loudness.true_peak", False),          # more headroom is fine
    ],
)
def test_regression_direction_per_feature(change, feature, regressed):
    _, by_feature = _comparisons(_result(**change), [_result()])
    assert (by_feature[feature]["status"] == "regressed") is regressed


def test_single_file_baseline_tolerates_small_changes():
    comparison, by_feature = _comparisons(
        _result(snr=29.0, lufs=-16.5, centroid=1560.0, f0=123.0), [_result()]
    )
    assert comparison["status"] == "ok"
    assert by_feature["spectral.centroid"]["tolerance"] == pytest.approx(150.0)


def test_regressions_rank_by_excess_in_tolerance_units_not_raw_units():
    # SNR 30 -> 18 dB is 3 tolerances (3 dB each) past the band; centroid 1500 -> 1900 Hz is 1.67 (150 Hz each).
    comparison, _ = _comparisons(_result(snr=18.0, centroid=1900.0), [_result()])
    ranked = [item["feature"] for item in comparison["largest_regressions"]]
    assert ranked[:2] == ["noise.snr", "spectral.centroid"]
    assert comparison["regression_score"] == pytest.approx(3.0 + 250 / 150, abs=0.01)


@pytest.mark.parametrize(("peak", "severity"), [(-0.5, "warn"), (0.5, "fail")])
def test_true_peak_blocks_ci_only_above_full_scale(peak, severity):
    result = _result(peak=peak)
    enrich_results([result], ci=True)
    label = {label["id"]: label for label in result.insights["defect_labels"]}["clipping_risk"]
    gate = {check["id"]: check for check in result.insights["ci_checks"]}["audio_quality_gate"]
    assert label["severity"] == severity
    assert gate["status"] == ("fail" if severity == "fail" else "pass")


def test_drift_uses_relative_tolerance_for_hz_features():
    steady, shifted = _result(centroid=1500.0), _result(path="b.wav", centroid=1560.0)
    enrich_results([steady, shifted], drift=True)
    assert shifted.insights["drift_monitor"]["status"] == "stable"
    moved = _result(path="c.wav", centroid=1800.0)
    enrich_results([steady, moved], drift=True)
    assert [change["feature"] for change in moved.insights["drift_monitor"]["changes"]] == ["spectral.centroid"]


def test_watch_style_re_enrichment_keeps_earlier_drift(tmp_path):
    state = tmp_path / "drift.json"
    first, second = _result("a.wav", snr=30.0), _result("b.wav", snr=20.0)
    enrich_results([first, second], drift=True, drift_state_path=state)
    before = [r.insights["drift_monitor"] for r in (first, second)]

    third = _result("c.wav", snr=10.0)
    enrich_results([first, second, third], drift=True, drift_state_path=state, preserve_drift=True)

    assert [r.insights["drift_monitor"] for r in (first, second)] == before
    change = third.insights["drift_monitor"]["changes"][0]
    assert (change["feature"], change["previous"], change["current"]) == ("noise.snr", 20.0, 10.0)

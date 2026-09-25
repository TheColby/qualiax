import math

import pytest
from scipy import stats

from qualiax.calibration import (
    CalibrationCase,
    calibrate_labels,
    cases_from_results,
    confidence_interval,
    consistency_report,
    variant_consistency,
)
from qualiax.models import FileResult, MetricResult


def _cases():
    return [
        CalibrationCase("a", {"noisy"}, {"noisy"}),
        CalibrationCase("b", set(), {"noisy"}),
        CalibrationCase("c", {"clip"}, {"clip", "noisy"}),
        CalibrationCase("d", {"clip", "loud"}, set()),
    ]


def test_calibrate_labels_per_label_counts_and_scores():
    report = calibrate_labels(_cases())

    noisy = report["labels"]["noisy"]
    assert (noisy["true_positive"], noisy["false_positive"], noisy["false_negative"]) == (1, 2, 0)
    assert noisy["precision"] == pytest.approx(1 / 3, abs=1e-6)
    assert noisy["recall"] == 1.0
    assert noisy["f1"] == 0.5
    assert noisy["support"] == 1

    clip = report["labels"]["clip"]
    assert (clip["precision"], clip["recall"], clip["support"]) == (1.0, 0.5, 2)
    assert clip["f1"] == pytest.approx(2 / 3, abs=1e-6)

    loud = report["labels"]["loud"]
    assert (loud["precision"], loud["recall"], loud["f1"], loud["support"]) == (0.0, 0.0, 0.0, 1)
    assert report["case_count"] == 4


def test_calibrate_labels_macro_and_micro_averages():
    report = calibrate_labels(_cases())

    assert report["macro_precision"] == pytest.approx((1 / 3 + 1 + 0) / 3, abs=1e-6)
    assert report["macro_recall"] == pytest.approx((1 + 0.5 + 0) / 3, abs=1e-6)
    assert report["macro_f1"] == pytest.approx((0.5 + 2 / 3 + 0) / 3, abs=1e-6)
    # Pooled: TP=2, FP=2, FN=2.
    assert report["micro_precision"] == 0.5
    assert report["micro_recall"] == 0.5
    assert report["micro_f1"] == 0.5


def test_calibrate_labels_can_restrict_to_a_taxonomy():
    report = calibrate_labels(_cases(), labels=["clip"])

    assert list(report["labels"]) == ["clip"]
    assert report["macro_f1"] == pytest.approx(2 / 3, abs=1e-6)
    assert report["micro_precision"] == 1.0


def test_calibrate_labels_empty_input():
    report = calibrate_labels([])
    assert report["labels"] == {}
    assert report["macro_f1"] == 0.0
    assert report["micro_f1"] == 0.0


def test_cases_from_results_pairs_ground_truth_and_skips_unannotated():
    annotated = FileResult(path="a.wav", insights={"defect_labels": [{"id": "noisy_floor"}, {"id": "clipping_risk"}]})
    clean = FileResult(path="b.wav", insights={"defect_labels": []})
    unannotated = FileResult(path="c.wav", insights={"defect_labels": [{"id": "noisy_floor"}]})
    segment = FileResult(
        path="d.wav [segment 1/2 0.00-1.00s]",
        source_file="d.wav",
        insights={"defect_labels": [{"id": "under_loud"}]},
    )

    cases = cases_from_results(
        [annotated, clean, unannotated, segment],
        {"a.wav": ["noisy_floor"], "b.wav": [], "d.wav": ["under_loud"]},
    )

    by_id = {case.case_id: case for case in cases}
    assert set(by_id) == {"a.wav", "b.wav", "d.wav [segment 1/2 0.00-1.00s]"}
    assert by_id["a.wav"].predicted_labels == {"noisy_floor", "clipping_risk"}
    assert by_id["a.wav"].expected_labels == {"noisy_floor"}
    report = calibrate_labels(cases)
    assert report["labels"]["clipping_risk"]["false_positive"] == 1
    assert report["labels"]["noisy_floor"]["false_positive"] == 0


def test_cases_from_results_reuses_generator_annotations_for_segments():
    segments = [
        FileResult(
            path=f"call.wav segment {index}",
            source_file="call.wav",
            insights={"defect_labels": [{"id": "noisy_floor"}]},
        )
        for index in range(2)
    ]
    cases = cases_from_results(segments, {"call.wav": iter(["noisy_floor"])})

    assert [case.expected_labels for case in cases] == [{"noisy_floor"}, {"noisy_floor"}]
    report = calibrate_labels(cases)
    assert report["labels"]["noisy_floor"]["false_positive"] == 0
    assert report["labels"]["noisy_floor"]["support"] == 2
    cases[0].expected_labels.clear()
    assert cases[1].expected_labels == {"noisy_floor"}


def test_confidence_interval_widens_with_confidence_level():
    values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    widths = [hi - lo for lo, hi in (confidence_interval(values, level) for level in (0.5, 0.8, 0.9, 0.95, 0.99))]

    assert widths == sorted(widths)
    assert len(set(widths)) == len(widths)


@pytest.mark.parametrize("level", [0.5, 0.9, 0.95, 0.99])
def test_confidence_interval_normal_matches_reference(level):
    values = [3.1, 2.9, 3.4, 3.0, 2.7, 3.3]
    mean = sum(values) / len(values)
    se = stats.tstd(values) / math.sqrt(len(values))
    z = stats.norm.ppf(0.5 + level / 2)

    lo, hi = confidence_interval(values, level)

    assert lo == pytest.approx(mean - z * se, abs=1e-5)
    assert hi == pytest.approx(mean + z * se, abs=1e-5)


def test_confidence_interval_t_method_matches_student_t_and_is_wider():
    values = [3.1, 2.9, 3.4, 3.0, 2.7]
    expected = stats.t.interval(0.95, df=len(values) - 1, loc=sum(values) / len(values), scale=stats.sem(values))

    lo, hi = confidence_interval(values, 0.95, method="t")
    normal_lo, normal_hi = confidence_interval(values, 0.95)

    assert (lo, hi) == pytest.approx(expected, abs=1e-5)
    assert hi - lo > normal_hi - normal_lo


def test_confidence_interval_input_validation():
    assert confidence_interval([2.0, float("nan")]) == (2.0, 2.0)
    with pytest.raises(ValueError):
        confidence_interval([])
    with pytest.raises(ValueError):
        confidence_interval([1.0, 2.0], 1.0)
    with pytest.raises(ValueError):
        confidence_interval([1.0, 2.0], method="bootstrap")


def test_consistency_report_spread_and_float_noise():
    assert consistency_report({"wav": 3.2, "mp3": 3.0}, tolerance=0.2)["consistent"] is True
    report = consistency_report({"wav": 3.2, "mp3": 2.9, "aac": 3.1}, tolerance=0.25)
    assert report["consistent"] is False
    assert report["spread"] == pytest.approx(0.3)
    assert report["tolerance"] == 0.25


def test_consistency_report_flags_failed_variants():
    report = consistency_report({"wav": 3.2, "mp3": float("nan"), "opus": None}, tolerance=1.0)

    assert report["consistent"] is False
    assert report["invalid_variants"] == ["mp3", "opus"]
    empty = consistency_report({"mp3": None}, tolerance=1.0)
    assert empty["consistent"] is False
    assert empty["spread"] is None
    assert empty["tolerance"] == 1.0


def test_consistency_report_against_reference_variant():
    report = consistency_report({"wav": 3.5, "mp3": 3.3, "aac": 3.0}, tolerance=0.25, reference="wav")

    assert report["deviations"]["mp3"] == {"deviation": -0.2, "within_tolerance": True}
    assert report["deviations"]["aac"]["within_tolerance"] is False
    assert report["consistent"] is False
    missing_reference = consistency_report({"mp3": 3.3}, tolerance=0.25, reference="wav")
    assert missing_reference["consistent"] is False
    assert "wav" in missing_reference["invalid_variants"]


def test_consistency_report_rejects_bad_tolerance():
    with pytest.raises(ValueError):
        consistency_report({"a": 1.0}, tolerance=-0.1)
    with pytest.raises(ValueError):
        consistency_report({"a": 1.0}, tolerance=float("nan"))


def _variant(snr, lufs=None):
    metrics = [MetricResult(name="Estimated SNR", value=snr, unit="dB")]
    if lufs is not None:
        metrics.append(MetricResult(name="Integrated Loudness (LUFS)", value=lufs, unit="LUFS"))
    return FileResult(path="clip.wav", metrics=metrics)


def test_variant_consistency_checks_each_metric_with_its_own_tolerance():
    results = {"wav": _variant(30.0, -16.0), "mp3": _variant(28.5, -16.2), "opus": _variant(29.0, -16.1)}

    report = variant_consistency(
        results,
        {"Estimated SNR": 1.0, "Integrated Loudness (LUFS)": 0.5},
        reference="wav",
    )

    assert report["consistent"] is False
    assert report["failing_metrics"] == ["Estimated SNR"]
    assert report["metrics"]["Integrated Loudness (LUFS)"]["consistent"] is True
    assert report["metrics"]["Estimated SNR"]["deviations"]["mp3"]["deviation"] == -1.5


def test_variant_consistency_treats_missing_metric_as_inconsistent():
    results = {"wav": _variant(30.0, -16.0), "mp3": _variant(30.0)}

    report = variant_consistency(results, {"Integrated Loudness (LUFS)": 1.0})

    assert report["consistent"] is False
    assert report["metrics"]["Integrated Loudness (LUFS)"]["invalid_variants"] == ["mp3"]
    with pytest.raises(ValueError):
        variant_consistency(results, {"Estimated SNR": 1.0}, reference="flac")

"""Calibration and consistency evaluation for insight labels and scores."""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist, mean, stdev
from typing import Iterable, Mapping

from .models import FileResult


@dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    expected_labels: set[str]
    predicted_labels: set[str]


def cases_from_results(
    results: Iterable[FileResult],
    expected: Mapping[str, Iterable[str]],
) -> list[CalibrationCase]:
    """Pair analyzed results with ground-truth labels.

    ``expected`` maps a result path (or its ``source_file``) to the labels a human
    assigned. Results without an entry are skipped rather than assumed clean, so a
    partially annotated benchmark does not inflate false-positive counts. Predicted
    labels are the ``id`` values of ``result.insights["defect_labels"]``.
    """
    cases: list[CalibrationCase] = []
    expected_labels: dict[str, set[str]] = {}
    for result in results:
        key = result.path if result.path in expected else result.source_file
        if key is None or key not in expected:
            continue
        if key not in expected_labels:
            expected_labels[key] = {str(label) for label in expected[key]}
        predicted = {
            str(label.get("id"))
            for label in result.insights.get("defect_labels", [])
            if isinstance(label, dict) and label.get("id") is not None
        }
        cases.append(CalibrationCase(result.path, set(expected_labels[key]), predicted))
    return cases


def calibrate_labels(cases: Iterable[CalibrationCase], *, labels: Iterable[str] | None = None) -> dict:
    """Compute per-label precision, recall, F1, and support plus macro and micro averages.

    ``labels`` restricts the evaluation to a fixed taxonomy; by default every label
    that appears in either the expected or predicted sets is evaluated. Labels with
    no support and no predictions score 0.0 (scikit-learn's ``zero_division=0``).
    """
    cases = list(cases)
    if labels is None:
        label_names = sorted({label for case in cases for label in case.expected_labels | case.predicted_labels})
    else:
        label_names = sorted(set(labels))
    metrics = {}
    total_tp = total_fp = total_fn = 0
    for label in label_names:
        true_positive = sum(label in case.expected_labels and label in case.predicted_labels for case in cases)
        false_positive = sum(label not in case.expected_labels and label in case.predicted_labels for case in cases)
        false_negative = sum(label in case.expected_labels and label not in case.predicted_labels for case in cases)
        total_tp += true_positive
        total_fp += false_positive
        total_fn += false_negative
        precision, recall, f1 = _prf(true_positive, false_positive, false_negative)
        metrics[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": true_positive + false_negative,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }
    micro_precision, micro_recall, micro_f1 = _prf(total_tp, total_fp, total_fn)
    return {
        "case_count": len(cases),
        "labels": metrics,
        "macro_precision": round(mean(item["precision"] for item in metrics.values()), 6) if metrics else 0.0,
        "macro_recall": round(mean(item["recall"] for item in metrics.values()), 6) if metrics else 0.0,
        "macro_f1": round(mean(item["f1"] for item in metrics.values()), 6) if metrics else 0.0,
        "micro_precision": round(micro_precision, 6),
        "micro_recall": round(micro_recall, 6),
        "micro_f1": round(micro_f1, 6),
    }


def confidence_interval(
    values: Iterable[float],
    confidence: float = 0.95,
    *,
    method: str = "normal",
) -> tuple[float, float]:
    """Return a two-sided confidence interval for the mean of a numeric sample.

    ``method="normal"`` (default) uses the normal approximation. ``method="t"``
    uses Student's t distribution, which is the better choice for the small
    sample sizes typical of benchmark runs. Non-finite values are ignored.
    """
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if method not in {"normal", "t"}:
        raise ValueError("method must be 'normal' or 't'")
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        raise ValueError("confidence_interval requires at least one finite value")
    if len(values) == 1:
        return values[0], values[0]
    if method == "t":
        from scipy.stats import t as student_t

        critical = float(student_t.ppf(0.5 + confidence / 2.0, df=len(values) - 1))
    else:
        critical = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    margin = critical * stdev(values) / math.sqrt(len(values))
    center = mean(values)
    return round(center - margin, 6), round(center + margin, 6)


def consistency_report(
    values_by_variant: Mapping[str, float | None],
    *,
    tolerance: float,
    reference: str | None = None,
) -> dict:
    """Assess whether codec/sample-rate variants of one score stay within ``tolerance``.

    Without ``reference`` the check is on the spread (max - min) across variants.
    With ``reference`` every variant is compared against that variant's value and
    each gets its own ``deviation`` and ``within_tolerance`` entry. Variants whose
    value is missing or non-finite are listed in ``invalid_variants`` and make the
    report inconsistent, because a failed variant cannot be shown to agree.
    """
    tolerance = float(tolerance)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite, non-negative number")
    finite: dict[str, float] = {}
    invalid: list[str] = []
    for name, value in values_by_variant.items():
        number = _finite_or_none(value)
        if number is None:
            invalid.append(str(name))
        else:
            finite[str(name)] = number
    report: dict = {
        "consistent": False,
        "spread": None,
        "tolerance": tolerance,
        "reference": reference,
        "variants": finite,
        "invalid_variants": sorted(invalid),
    }
    if not finite:
        return report
    spread = max(finite.values()) - min(finite.values())
    report["spread"] = round(spread, 6)
    if reference is None:
        report["consistent"] = spread <= tolerance + 1e-12 and not invalid
        return report
    if reference not in finite:
        report["invalid_variants"] = sorted(set(invalid) | {reference})
        return report
    base = finite[reference]
    deviations = {}
    for name, value in finite.items():
        deviation = round(value - base, 6)
        deviations[name] = {"deviation": deviation, "within_tolerance": abs(value - base) <= tolerance + 1e-12}
    report["deviations"] = deviations
    report["consistent"] = not invalid and all(item["within_tolerance"] for item in deviations.values())
    return report


def variant_consistency(
    results_by_variant: Mapping[str, FileResult],
    tolerances: Mapping[str, float],
    *,
    reference: str | None = None,
) -> dict:
    """Check that the same content analyzed through different codecs/variants agrees.

    ``results_by_variant`` maps a variant name (``"wav"``, ``"mp3-128k"``, ...) to the
    FileResult for that encoding. ``tolerances`` maps metric names to the maximum
    allowed disagreement. Each metric gets a :func:`consistency_report`; a metric
    that is missing from any variant is inconsistent.
    """
    if not results_by_variant:
        raise ValueError("variant_consistency requires at least one variant")
    if reference is not None and reference not in results_by_variant:
        raise ValueError(f"reference variant {reference!r} is not in results_by_variant")
    per_metric = {}
    for metric_name, tolerance in tolerances.items():
        values = {
            variant: _metric_value(result, metric_name)
            for variant, result in results_by_variant.items()
        }
        per_metric[metric_name] = consistency_report(values, tolerance=tolerance, reference=reference)
    failing = sorted(name for name, report in per_metric.items() if not report["consistent"])
    return {
        "consistent": not failing,
        "reference": reference,
        "variants": sorted(results_by_variant),
        "metrics": per_metric,
        "failing_metrics": failing,
    }


def _metric_value(result: FileResult, name: str) -> float | None:
    for metric in result.metrics:
        if metric.name == name:
            return _finite_or_none(metric.value)
    return None


def _finite_or_none(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _prf(true_positive: int, false_positive: int, false_negative: int) -> tuple[float, float, float]:
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)
    return precision, recall, f1


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0

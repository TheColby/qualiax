"""Calibration and consistency evaluation for insight labels and scores."""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import mean, stdev
from typing import Iterable


@dataclass(frozen=True)
class CalibrationCase:
    case_id: str
    expected_labels: set[str]
    predicted_labels: set[str]


def calibrate_labels(cases: Iterable[CalibrationCase]) -> dict:
    """Compute per-label precision, recall, F1, and support."""
    cases = list(cases)
    labels = sorted({label for case in cases for label in case.expected_labels | case.predicted_labels})
    metrics = {}
    for label in labels:
        true_positive = sum(label in case.expected_labels and label in case.predicted_labels for case in cases)
        false_positive = sum(label not in case.expected_labels and label in case.predicted_labels for case in cases)
        false_negative = sum(label in case.expected_labels and label not in case.predicted_labels for case in cases)
        precision = _ratio(true_positive, true_positive + false_positive)
        recall = _ratio(true_positive, true_positive + false_negative)
        f1 = _ratio(2 * precision * recall, precision + recall)
        metrics[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": true_positive + false_negative,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
        }
    macro_f1 = mean(item["f1"] for item in metrics.values()) if metrics else 0.0
    return {"case_count": len(cases), "labels": metrics, "macro_f1": round(macro_f1, 6)}


def confidence_interval(values: Iterable[float], confidence: float = 0.95) -> tuple[float, float]:
    """Return a normal-approximation confidence interval for a numeric sample."""
    values = [float(value) for value in values if math.isfinite(float(value))]
    if not values:
        raise ValueError("confidence_interval requires at least one finite value")
    if len(values) == 1:
        return values[0], values[0]
    z = 1.959963984540054 if confidence == 0.95 else _approximate_z(confidence)
    margin = z * stdev(values) / math.sqrt(len(values))
    center = mean(values)
    return round(center - margin, 6), round(center + margin, 6)


def consistency_report(values_by_variant: dict[str, float], *, tolerance: float) -> dict:
    """Assess whether codec/sample-rate variants remain within a score tolerance."""
    finite = {name: float(value) for name, value in values_by_variant.items() if math.isfinite(float(value))}
    if not finite:
        return {"consistent": False, "spread": None, "variants": {}}
    spread = max(finite.values()) - min(finite.values())
    return {
        "consistent": spread <= tolerance,
        "spread": round(spread, 6),
        "tolerance": tolerance,
        "variants": finite,
    }


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _approximate_z(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between 0 and 1")
    # Good enough for calibration reporting without adding a statistics dependency.
    return 1.0 + 2.0 * max(0.0, confidence - 0.68) / 0.32

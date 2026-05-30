"""
Aggregate scorecards for batches of qualiax results.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from html import escape
from typing import Any

import numpy as np

from .models import FileResult, MetricResult
from .version import OUTPUT_SCHEMA_VERSION, __version__


@dataclass(frozen=True)
class MetricOutlier:
    file: str
    value: float
    deviation_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "file": self.file,
            "value": self.value,
            "deviation_score": round(self.deviation_score, 3),
        }


@dataclass(frozen=True)
class MetricRollup:
    name: str
    group: str
    unit: str
    value_kind: str
    source_count: int
    measured_count: int
    missing_count: int
    warning_count: int
    confidence_counts: dict[str, int]
    in_range_count: int | None
    out_of_range_count: int | None
    min_value: float | None
    min_file: str | None
    max_value: float | None
    max_file: str | None
    mean: float | None
    median: float | None
    p05: float | None
    p95: float | None
    category_counts: dict[str, int] = field(default_factory=dict)
    dominant_category: str | None = None
    dominant_category_count: int | None = None
    outliers: tuple[MetricOutlier, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "unit": self.unit,
            "value_kind": self.value_kind,
            "source_count": self.source_count,
            "measured_count": self.measured_count,
            "missing_count": self.missing_count,
            "warning_count": self.warning_count,
            "confidence_counts": self.confidence_counts,
            "in_range_count": self.in_range_count,
            "out_of_range_count": self.out_of_range_count,
            "min_value": self.min_value,
            "min_file": self.min_file,
            "max_value": self.max_value,
            "max_file": self.max_file,
            "mean": self.mean,
            "median": self.median,
            "p05": self.p05,
            "p95": self.p95,
            "category_counts": self.category_counts,
            "dominant_category": self.dominant_category,
            "dominant_category_count": self.dominant_category_count,
            "outliers": [outlier.to_dict() for outlier in self.outliers],
        }


@dataclass(frozen=True)
class Scorecard:
    schema_version: str
    tool_version: str
    file_count: int
    status_counts: dict[str, int]
    confidence_notes: tuple[str, ...]
    metric_rollups: tuple[MetricRollup, ...] = field(default_factory=tuple)
    insight_summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tool_version": self.tool_version,
            "file_count": self.file_count,
            "status_counts": self.status_counts,
            "confidence_notes": list(self.confidence_notes),
            "metric_rollups": [rollup.to_dict() for rollup in self.metric_rollups],
            "insight_summary": self.insight_summary,
        }


def build_scorecard(results: list[FileResult]) -> Scorecard:
    grouped: dict[tuple[str, str], dict[str, list[tuple[FileResult, MetricResult]]]] = {}
    status_counts = {"ok": 0, "warn": 0, "partial": 0, "error": 0}
    confidence_notes: list[str] = []
    seen_notes = set()
    source_keys = set()

    for result in results:
        status_counts[_result_status(result)] += 1
        source_key = result.source_file or result.path
        source_keys.add(source_key)
        for note in result.confidence_notes:
            if note in seen_notes:
                continue
            confidence_notes.append(note)
            seen_notes.add(note)
        for metric in result.metrics:
            grouped.setdefault((metric.group, metric.name), {}).setdefault(source_key, []).append((result, metric))

    rollups = [
        _build_rollup(group, name, entries_by_source, total_sources=len(source_keys))
        for (group, name), entries_by_source in sorted(grouped.items())
    ]

    return Scorecard(
        schema_version=OUTPUT_SCHEMA_VERSION,
        tool_version=__version__,
        file_count=len(source_keys),
        status_counts=status_counts,
        confidence_notes=tuple(confidence_notes),
        metric_rollups=tuple(rollups),
        insight_summary=_build_insight_summary(results),
    )


def _build_insight_summary(results: list[FileResult]) -> dict[str, Any]:
    defect_label_counts: dict[str, int] = {}
    label_severity_counts: dict[str, int] = {}
    ci_status_counts: dict[str, int] = {}
    ci_check_counts: dict[str, int] = {}
    drift_status_counts: dict[str, int] = {}
    dataset_issue_counts: dict[str, int] = {}

    for result in results:
        insights = result.insights or {}
        for label in insights.get("defect_labels", []):
            label_id = str(label.get("id", "unknown"))
            severity = str(label.get("severity", "info"))
            defect_label_counts[label_id] = defect_label_counts.get(label_id, 0) + 1
            label_severity_counts[severity] = label_severity_counts.get(severity, 0) + 1
        for check in insights.get("ci_checks", []):
            status = str(check.get("status", "unknown"))
            check_id = str(check.get("id", "unknown"))
            ci_status_counts[status] = ci_status_counts.get(status, 0) + 1
            ci_check_counts[check_id] = ci_check_counts.get(check_id, 0) + 1
        drift = insights.get("drift_monitor", {})
        if drift:
            status = str(drift.get("status", "unknown"))
            drift_status_counts[status] = drift_status_counts.get(status, 0) + 1
        for issue in insights.get("dataset_audit", []):
            issue_id = str(issue.get("id", "unknown"))
            dataset_issue_counts[issue_id] = dataset_issue_counts.get(issue_id, 0) + 1

    return {
        "defect_label_counts": dict(sorted(defect_label_counts.items())),
        "label_severity_counts": dict(sorted(label_severity_counts.items())),
        "ci_status_counts": dict(sorted(ci_status_counts.items())),
        "ci_check_counts": dict(sorted(ci_check_counts.items())),
        "drift_status_counts": dict(sorted(drift_status_counts.items())),
        "dataset_issue_counts": dict(sorted(dataset_issue_counts.items())),
    }


def render_scorecard(scorecard: Scorecard, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(scorecard.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)
    if fmt == "html":
        return _render_html(scorecard)
    return _render_markdown(scorecard)


def _build_rollup(
    group: str,
    name: str,
    entries_by_source: dict[str, list[tuple[FileResult, MetricResult]]],
    *,
    total_sources: int,
) -> MetricRollup:
    numeric_values: list[tuple[str, float]] = []
    categorical_values: list[tuple[str, str]] = []
    warning_count = 0
    unit = ""
    confidence_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    in_range_count = 0
    out_of_range_count = 0
    saw_reference_range = False

    for source_key, entries in entries_by_source.items():
        if any(metric.warning for _, metric in entries):
            warning_count += 1
        for _, metric in entries:
            unit = unit or metric.unit
            if metric.confidence:
                confidence_counts[metric.confidence] = confidence_counts.get(metric.confidence, 0) + 1

        reference_range = next(
            (metric.reference_range for _, metric in entries if metric.reference_range is not None),
            None,
        )
        if reference_range is not None and len(reference_range) == 2:
            saw_reference_range = True

        source_numeric_values = [
            float(metric.value)
            for _, metric in entries
            if isinstance(metric.value, (int, float))
            and not isinstance(metric.value, bool)
            and math.isfinite(float(metric.value))
        ]
        source_categories = [
            str(metric.value)
            for _, metric in entries
            if metric.value is not None
            and (isinstance(metric.value, bool) or not isinstance(metric.value, (int, float)))
        ]
        if source_categories:
            category = _dominant_category(source_categories)
            categorical_values.append((source_key, category))
            category_counts[category] = category_counts.get(category, 0) + 1
        if not source_numeric_values:
            continue

        aggregate_value = float(np.mean(source_numeric_values))
        numeric_values.append((source_key, aggregate_value))

        if reference_range is not None and len(reference_range) == 2:
            lo, hi = float(reference_range[0]), float(reference_range[1])
            if lo <= aggregate_value <= hi:
                in_range_count += 1
            else:
                out_of_range_count += 1

    measured_count = len(numeric_values)
    categorical_count = len(categorical_values)
    source_measured_count = max(measured_count, categorical_count)
    missing_count = max(0, total_sources - source_measured_count)
    if not numeric_values:
        dominant_category = None
        dominant_category_count = None
        if category_counts:
            dominant_category, dominant_category_count = max(
                category_counts.items(),
                key=lambda item: (item[1], item[0]),
            )
        return MetricRollup(
            name=name,
            group=group,
            unit=unit,
            value_kind="categorical" if category_counts else "missing",
            source_count=total_sources,
            measured_count=source_measured_count,
            missing_count=max(0, total_sources - source_measured_count),
            warning_count=warning_count,
            confidence_counts=confidence_counts,
            in_range_count=0 if saw_reference_range else None,
            out_of_range_count=0 if saw_reference_range else None,
            min_value=None,
            min_file=None,
            max_value=None,
            max_file=None,
            mean=None,
            median=None,
            p05=None,
            p95=None,
            category_counts=category_counts,
            dominant_category=dominant_category,
            dominant_category_count=dominant_category_count,
        )

    files, values = zip(*numeric_values)
    arr = np.asarray(values, dtype=np.float64)
    min_idx = int(np.argmin(arr))
    max_idx = int(np.argmax(arr))
    dominant_category = None
    dominant_category_count = None
    if category_counts:
        dominant_category, dominant_category_count = max(
            category_counts.items(),
            key=lambda item: (item[1], item[0]),
        )
    return MetricRollup(
        name=name,
        group=group,
        unit=unit,
        value_kind="mixed" if category_counts else "numeric",
        source_count=total_sources,
        measured_count=source_measured_count,
        missing_count=missing_count,
        warning_count=warning_count,
        confidence_counts=confidence_counts,
        in_range_count=in_range_count if saw_reference_range else None,
        out_of_range_count=out_of_range_count if saw_reference_range else None,
        min_value=round(float(arr[min_idx]), 6),
        min_file=files[min_idx],
        max_value=round(float(arr[max_idx]), 6),
        max_file=files[max_idx],
        mean=round(float(np.mean(arr)), 6),
        median=round(float(np.median(arr)), 6),
        p05=round(float(np.percentile(arr, 5)), 6),
        p95=round(float(np.percentile(arr, 95)), 6),
        category_counts=category_counts,
        dominant_category=dominant_category,
        dominant_category_count=dominant_category_count,
        outliers=tuple(_detect_outliers(files, arr)),
    )


def _detect_outliers(files: tuple[str, ...], values: np.ndarray) -> list[MetricOutlier]:
    if len(values) < 4:
        return []
    median = float(np.median(values))
    abs_dev = np.abs(values - median)
    mad = float(np.median(abs_dev))
    if mad <= 1e-12:
        return []
    robust_scale = 1.4826 * mad
    scores = abs_dev / robust_scale
    indices = np.where(scores >= 3.0)[0]
    ranked = sorted(indices, key=lambda idx: float(scores[idx]), reverse=True)[:5]
    return [
        MetricOutlier(file=files[idx], value=float(values[idx]), deviation_score=float(scores[idx]))
        for idx in ranked
    ]


def _result_status(result: FileResult) -> str:
    if result.error:
        return "error"
    if any(metric.warning for metric in result.metrics):
        return "warn"
    if any(metric.value is None for metric in result.metrics):
        return "partial"
    return "ok"


def _render_markdown(scorecard: Scorecard) -> str:
    lines = [
        "# qualiax Scorecard",
        "",
        f"- Tool version: `{scorecard.tool_version}`",
        f"- Schema version: `{scorecard.schema_version}`",
        f"- Files: `{scorecard.file_count}`",
        (
            "- Status counts: "
            + ", ".join(f"`{name}={count}`" for name, count in scorecard.status_counts.items())
        ),
    ]
    if scorecard.confidence_notes:
        lines.extend(["", "## Confidence & Calibration", ""])
        lines.extend(f"- {note}" for note in scorecard.confidence_notes)
    if any(scorecard.insight_summary.values()):
        lines.extend(["", "## Insight Rollups", ""])
        for section, counts in scorecard.insight_summary.items():
            if not counts:
                continue
            lines.append(f"- {_md(section)}: " + ", ".join(f"`{_md(str(name))}={count}`" for name, count in counts.items()))
    lines.extend(["", "## Metric Rollups", ""])
    lines.append("| Group | Metric | Count | Warn | Kind | Confidence | Mean | Median | P05 | P95 | Top Values | Min | Max | Outliers |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for rollup in scorecard.metric_rollups:
        outliers = ", ".join(outlier.file for outlier in rollup.outliers) or ""
        confidence = _format_confidence_counts(rollup.confidence_counts)
        categories = _format_category_summary(rollup)
        lines.append(
            "| "
            + " | ".join(
                [
                    _md(rollup.group),
                    _md(rollup.name),
                    f"{rollup.measured_count}/{rollup.source_count}",
                    str(rollup.warning_count),
                    _md(rollup.value_kind),
                    _md(confidence),
                    _fmt_number(rollup.mean, rollup.unit),
                    _fmt_number(rollup.median, rollup.unit),
                    _fmt_number(rollup.p05, rollup.unit),
                    _fmt_number(rollup.p95, rollup.unit),
                    _md(categories),
                    _fmt_extreme(rollup.min_value, rollup.min_file, rollup.unit),
                    _fmt_extreme(rollup.max_value, rollup.max_file, rollup.unit),
                    _md(outliers),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _render_html(scorecard: Scorecard) -> str:
    rows = "\n".join(
        f"""
        <tr>
          <td>{escape(rollup.group)}</td>
          <td>{escape(rollup.name)}</td>
          <td>{rollup.measured_count}/{rollup.source_count}</td>
          <td>{rollup.warning_count}</td>
          <td>{escape(rollup.value_kind)}</td>
          <td>{escape(_format_confidence_counts(rollup.confidence_counts))}</td>
          <td>{escape(_fmt_number(rollup.mean, rollup.unit))}</td>
          <td>{escape(_fmt_number(rollup.median, rollup.unit))}</td>
          <td>{escape(_fmt_number(rollup.p05, rollup.unit))}</td>
          <td>{escape(_fmt_number(rollup.p95, rollup.unit))}</td>
          <td>{escape(_format_category_summary(rollup))}</td>
          <td>{escape(_fmt_extreme(rollup.min_value, rollup.min_file, rollup.unit))}</td>
          <td>{escape(_fmt_extreme(rollup.max_value, rollup.max_file, rollup.unit))}</td>
          <td>{escape(', '.join(outlier.file for outlier in rollup.outliers))}</td>
        </tr>"""
        for rollup in scorecard.metric_rollups
    )
    confidence = ""
    if scorecard.confidence_notes:
        confidence = (
            "<section><h2>Confidence &amp; Calibration</h2><ul>"
            + "".join(f"<li>{escape(note)}</li>" for note in scorecard.confidence_notes)
            + "</ul></section>"
        )
    insight_summary = ""
    if any(scorecard.insight_summary.values()):
        insight_summary = (
            "<section><h2>Insight Rollups</h2><ul>"
            + "".join(
                f"<li>{escape(section)}: {escape(', '.join(f'{name}={count}' for name, count in counts.items()))}</li>"
                for section, counts in scorecard.insight_summary.items()
                if counts
            )
            + "</ul></section>"
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>qualiax Scorecard</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 0; padding: 24px; background: #f7f4ee; color: #211d18; }}
    main {{ max-width: 1180px; margin: 0 auto; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border-bottom: 1px solid #e4d8ca; padding: 10px 12px; text-align: left; vertical-align: top; }}
    th {{ background: #efe5d8; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.08em; }}
    .chips {{ display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 18px; }}
    .chip {{ padding: 8px 12px; border-radius: 999px; background: white; border: 1px solid #d7c8b8; }}
  </style>
</head>
<body>
  <main>
    <h1>qualiax Scorecard</h1>
    <div class="chips">
      <div class="chip">tool {escape(scorecard.tool_version)}</div>
      <div class="chip">schema {escape(scorecard.schema_version)}</div>
      <div class="chip">{scorecard.file_count} files</div>
    </div>
    {confidence}
    {insight_summary}
    <table>
      <thead>
        <tr>
          <th>Group</th>
          <th>Metric</th>
          <th>Count</th>
          <th>Warn</th>
          <th>Kind</th>
          <th>Confidence</th>
          <th>Mean</th>
          <th>Median</th>
          <th>P05</th>
          <th>P95</th>
          <th>Top Values</th>
          <th>Min</th>
          <th>Max</th>
          <th>Outliers</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
  </main>
</body>
</html>"""


def _fmt_number(value: float | None, unit: str) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f} {unit}".strip()


def _fmt_extreme(value: float | None, file: str | None, unit: str) -> str:
    if value is None:
        return "N/A"
    if file:
        return f"{value:.3f} {unit} ({file})".strip()
    return _fmt_number(value, unit)


def _md(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


def _dominant_category(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _format_confidence_counts(confidence_counts: dict[str, int]) -> str:
    if not confidence_counts:
        return ""
    return ", ".join(
        f"{name}={count}"
        for name, count in sorted(confidence_counts.items())
    )


def _format_category_summary(rollup: MetricRollup) -> str:
    if not rollup.category_counts:
        return ""
    ranked = sorted(
        rollup.category_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[:3]
    return ", ".join(f"{value}={count}" for value, count in ranked)

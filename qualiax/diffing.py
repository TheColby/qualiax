"""
Compare two qualiax JSON reports and summarize regressions.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class DiffChange:
    file: str
    metric: str
    group: str
    before: Any
    after: Any
    unit: str = ""
    classification: str = "changed"
    note: str = ""


@dataclass(frozen=True)
class DiffSummary:
    before_report: str
    after_report: str
    files_added: int
    files_removed: int
    regressions: int
    improvements: int
    changes: int
    entries: list[DiffChange]

    def to_dict(self) -> dict[str, Any]:
        return {
            "before_report": self.before_report,
            "after_report": self.after_report,
            "summary": {
                "files_added": self.files_added,
                "files_removed": self.files_removed,
                "regressions": self.regressions,
                "improvements": self.improvements,
                "changes": self.changes,
            },
            "entries": [asdict(entry) for entry in self.entries],
        }


def diff_reports(before_path: Path, after_path: Path) -> DiffSummary:
    before_items = _load_report(before_path)
    after_items = _load_report(after_path)
    before_by_file = {item["file"]: item for item in before_items}
    after_by_file = {item["file"]: item for item in after_items}
    all_files = sorted(set(before_by_file) | set(after_by_file))

    entries: list[DiffChange] = []
    files_added = 0
    files_removed = 0
    regressions = 0
    improvements = 0

    for file_name in all_files:
        before_file = before_by_file.get(file_name)
        after_file = after_by_file.get(file_name)
        if before_file is None:
            files_added += 1
            entries.append(DiffChange(file=file_name, metric="file", group="meta", before=None, after="present", classification="added", note="New file added to report"))
            continue
        if after_file is None:
            files_removed += 1
            entries.append(DiffChange(file=file_name, metric="file", group="meta", before="present", after=None, classification="removed", note="File removed from report"))
            continue

        if before_file.get("error") != after_file.get("error"):
            classification = "regression" if after_file.get("error") else "improvement"
            if classification == "regression":
                regressions += 1
            else:
                improvements += 1
            entries.append(
                DiffChange(
                    file=file_name,
                    metric="error",
                    group="meta",
                    before=before_file.get("error"),
                    after=after_file.get("error"),
                    classification=classification,
                    note="File-level error state changed",
                )
            )

        before_metrics = _metrics_by_key(before_file)
        after_metrics = _metrics_by_key(after_file)
        all_metrics = sorted(set(before_metrics) | set(after_metrics))
        for key in all_metrics:
            before_metric = before_metrics.get(key)
            after_metric = after_metrics.get(key)
            change = _diff_metric(file_name, before_metric, after_metric)
            if change is None:
                continue
            entries.append(change)
            if change.classification == "regression":
                regressions += 1
            elif change.classification == "improvement":
                improvements += 1

    changes = len(entries)
    return DiffSummary(
        before_report=str(before_path),
        after_report=str(after_path),
        files_added=files_added,
        files_removed=files_removed,
        regressions=regressions,
        improvements=improvements,
        changes=changes,
        entries=entries,
    )


def render_diff(summary: DiffSummary, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(summary.to_dict(), indent=2, ensure_ascii=False, allow_nan=False)
    if fmt == "markdown":
        return _render_diff_markdown(summary)
    return _render_diff_pretty(summary)


def _load_report(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        items = [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
        if not all(isinstance(item, dict) for item in items):
            raise ValueError(f"Expected '{path}' to contain one JSON object per line.")
        return items
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    raise ValueError(f"Expected '{path}' to contain a JSON list of file results.")


def _metrics_by_key(item: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    metrics = item.get("metrics") or []
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for metric in metrics:
        group = str(metric.get("group") or "")
        name = str(metric.get("name") or "")
        out[(group, name)] = metric
    return out


def _diff_metric(
    file_name: str,
    before_metric: Optional[dict[str, Any]],
    after_metric: Optional[dict[str, Any]],
) -> Optional[DiffChange]:
    if before_metric is None and after_metric is None:
        return None
    metric = after_metric or before_metric or {}
    group = str(metric.get("group") or "")
    name = str(metric.get("name") or "")
    unit = str(metric.get("unit") or "")
    before_value = None if before_metric is None else before_metric.get("value")
    after_value = None if after_metric is None else after_metric.get("value")

    if before_metric is None:
        return DiffChange(file=file_name, metric=name, group=group, before=None, after=after_value, unit=unit, classification="added", note="Metric added")
    if after_metric is None:
        return DiffChange(file=file_name, metric=name, group=group, before=before_value, after=None, unit=unit, classification="removed", note="Metric removed")

    before_warning = before_metric.get("warning")
    after_warning = after_metric.get("warning")
    if _values_equal(before_value, after_value) and before_warning == after_warning:
        return None

    classification = _classify_change(before_metric, after_metric)
    note = _change_note(before_metric, after_metric, classification)
    return DiffChange(
        file=file_name,
        metric=name,
        group=group,
        before=before_value,
        after=after_value,
        unit=unit,
        classification=classification,
        note=note,
    )


def _classify_change(before_metric: dict[str, Any], after_metric: dict[str, Any]) -> str:
    before_warning = before_metric.get("warning")
    after_warning = after_metric.get("warning")
    before_value = before_metric.get("value")
    after_value = after_metric.get("value")

    if before_warning and not after_warning:
        return "improvement"
    if not before_warning and after_warning:
        return "regression"

    higher_is_better = after_metric.get("higher_is_better")
    if isinstance(before_value, (int, float)) and isinstance(after_value, (int, float)):
        if higher_is_better is True:
            return "improvement" if after_value > before_value else "regression"
        if higher_is_better is False:
            return "improvement" if after_value < before_value else "regression"
        reference_range = after_metric.get("reference_range") or before_metric.get("reference_range")
        if reference_range and len(reference_range) == 2:
            lo, hi = float(reference_range[0]), float(reference_range[1])
            before_in = lo <= float(before_value) <= hi
            after_in = lo <= float(after_value) <= hi
            if before_in and not after_in:
                return "regression"
            if not before_in and after_in:
                return "improvement"
    return "changed"


def _change_note(before_metric: dict[str, Any], after_metric: dict[str, Any], classification: str) -> str:
    if classification == "regression":
        return "Regression detected"
    if classification == "improvement":
        return "Improvement detected"
    if before_metric.get("warning") != after_metric.get("warning"):
        return "Warning state changed"
    return "Metric value changed"


def _values_equal(before: Any, after: Any) -> bool:
    if isinstance(before, (int, float)) and isinstance(after, (int, float)):
        if isinstance(before, float) and math.isnan(before) and isinstance(after, float) and math.isnan(after):
            return True
        return math.isclose(float(before), float(after), rel_tol=1e-9, abs_tol=1e-12)
    return before == after


def _render_diff_pretty(summary: DiffSummary) -> str:
    lines = [
        "qualiax Diff Report",
        f"Before: {summary.before_report}",
        f"After:  {summary.after_report}",
        "",
        (
            "Summary: "
            f"{summary.changes} change(s), "
            f"{summary.regressions} regression(s), "
            f"{summary.improvements} improvement(s), "
            f"{summary.files_added} file(s) added, "
            f"{summary.files_removed} file(s) removed"
        ),
    ]
    if not summary.entries:
        lines.append("")
        lines.append("No differences detected.")
        return "\n".join(lines)
    for entry in summary.entries:
        before = _fmt_value(entry.before, entry.unit)
        after = _fmt_value(entry.after, entry.unit)
        lines.extend(
            [
                "",
                f"[{entry.classification}] {entry.file} :: {entry.group}/{entry.metric}",
                f"  {before} -> {after}",
                f"  {entry.note}",
            ]
        )
    return "\n".join(lines)


def _render_diff_markdown(summary: DiffSummary) -> str:
    lines = [
        "# qualiax Diff Report",
        "",
        f"- Before: `{summary.before_report}`",
        f"- After: `{summary.after_report}`",
        f"- Changes: `{summary.changes}`",
        f"- Regressions: `{summary.regressions}`",
        f"- Improvements: `{summary.improvements}`",
        f"- Files added: `{summary.files_added}`",
        f"- Files removed: `{summary.files_removed}`",
        "",
        "| Classification | File | Group | Metric | Before | After | Note |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in summary.entries:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape(entry.classification),
                    _md_escape(entry.file),
                    _md_escape(entry.group),
                    _md_escape(entry.metric),
                    _md_escape(_fmt_value(entry.before, entry.unit)),
                    _md_escape(_fmt_value(entry.after, entry.unit)),
                    _md_escape(entry.note),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _fmt_value(value: Any, unit: str) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        text = f"{value:.4f}"
    else:
        text = str(value)
    return f"{text} {unit}".strip()


def _md_escape(value: Any) -> str:
    return str(value).replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")

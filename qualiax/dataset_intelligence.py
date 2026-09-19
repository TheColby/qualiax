"""Dataset-level duplicate, coverage, imbalance, and leakage analysis."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from .models import FileResult


def audit_dataset(
    results: Iterable[FileResult],
    *,
    expected_content_type: str | None = None,
    split_names: tuple[str, ...] = ("train", "validation", "test", "val"),
) -> dict:
    results = list(results)
    fingerprint_groups: dict[str, list[str]] = defaultdict(list)
    speaker_counts: Counter[str] = Counter()
    content_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    mismatch_risks = []

    for result in results:
        signature = result.insights.get("quality_fingerprint", {}).get("signature")
        if signature:
            fingerprint_groups[str(signature)].append(result.path)
        content_counts[result.content_type or "unknown"] += 1
        speaker = _metric_value(result, "Speaker Cluster") or _metric_value(result, "Speaker ID")
        if speaker is not None:
            speaker_counts[str(speaker)] += 1
        labels = [str(label.get("id")) for label in result.insights.get("defect_labels", [])]
        label_counts.update(labels)
        if expected_content_type and result.content_type not in {None, "unknown", expected_content_type}:
            mismatch_risks.append(
                {
                    "file": result.path,
                    "expected": expected_content_type,
                    "observed": result.content_type,
                }
            )

    duplicates = [paths for paths in fingerprint_groups.values() if len(paths) > 1]
    leakage = []
    for paths in duplicates:
        splits = {_split_for_path(path, split_names) for path in paths}
        splits.discard(None)
        if len(splits) > 1:
            leakage.append({"files": paths, "splits": sorted(splits)})

    remediation = _remediation_plan(duplicates, leakage, speaker_counts, mismatch_risks, label_counts)
    return {
        "file_count": len(results),
        "near_duplicate_groups": duplicates,
        "split_leakage": leakage,
        "speaker_distribution": dict(sorted(speaker_counts.items())),
        "content_distribution": dict(sorted(content_counts.items())),
        "defect_distribution": dict(sorted(label_counts.items())),
        "label_mismatch_risks": mismatch_risks,
        "coverage": {
            "content_types": len(content_counts),
            "speakers": len(speaker_counts),
            "fingerprints": len(fingerprint_groups),
        },
        "remediation_plan": remediation,
    }


def _metric_value(result: FileResult, name: str):
    return next((metric.value for metric in result.metrics if metric.name == name), None)


def _split_for_path(path: str, split_names: tuple[str, ...]) -> str | None:
    parts = {part.lower() for part in Path(path).parts}
    for name in split_names:
        if name.lower() in parts:
            return "validation" if name.lower() == "val" else name.lower()
    return None


def _remediation_plan(duplicates, leakage, speakers, mismatches, labels) -> list[dict]:
    plan = []
    if duplicates:
        plan.append({"priority": 1, "action": "deduplicate", "items": len(duplicates)})
    if leakage:
        plan.append({"priority": 1, "action": "remove_split_leakage", "items": len(leakage)})
    if speakers and max(speakers.values()) > max(1, sum(speakers.values()) * 0.6):
        plan.append({"priority": 2, "action": "rebalance_speakers", "items": len(speakers)})
    if mismatches:
        plan.append({"priority": 2, "action": "review_content_labels", "items": len(mismatches)})
    if labels:
        plan.append({"priority": 3, "action": "repair_or_remove_quality_outliers", "items": sum(labels.values())})
    return sorted(plan, key=lambda item: (item["priority"], item["action"]))

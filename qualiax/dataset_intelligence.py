"""Dataset-level duplicate, coverage, imbalance, and leakage analysis."""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import PurePath
from typing import Callable, Iterable, Mapping, Optional, Union

from .models import FileResult

SpeakerSource = Union[Mapping[str, str], Callable[[FileResult], Optional[str]]]

_SPEAKER_METRICS = ("Speaker Cluster", "Speaker ID")


def audit_dataset(
    results: Iterable[FileResult],
    *,
    expected_content_type: str | None = None,
    split_names: tuple[str, ...] = ("train", "validation", "test", "val"),
    splits: Mapping[str, str] | None = None,
    speakers: SpeakerSource | None = None,
    declared_labels: Mapping[str, str] | None = None,
    imbalance_ratio: float = 0.6,
) -> dict:
    """Audit an analyzed dataset for duplicates, split leakage, imbalance, and label risks.

    Near-duplicates are results that share a ``quality_fingerprint`` signature, so
    the results must come from an ``insights=True`` run (a warning is emitted
    otherwise). Segments of the same source file are not reported as duplicates of
    each other.

    ``splits`` maps a path to its split name; otherwise the split is inferred from
    the directory component nearest the file that matches ``split_names``
    (``val`` is reported as ``validation``). ``speakers`` maps a path to a speaker
    id, or is a callable returning one; qualiax does not identify speakers itself,
    so without it the speaker checks only use a ``Speaker Cluster``/``Speaker ID``
    metric supplied by a plugin. ``declared_labels`` maps a path to the content type
    the dataset manifest claims (``speech``, ``music``, ...), which is compared with
    the detected content type.
    """
    results = list(results)
    fingerprint_groups: dict[str, list[FileResult]] = defaultdict(list)
    speaker_counts: Counter[str] = Counter()
    content_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    speaker_files: dict[str, list[str]] = defaultdict(list)
    split_of: dict[str, str | None] = {}
    labeled_files: list[str] = []
    failed_files: list[str] = []
    mismatch_risks: list[dict] = []

    for result in results:
        if result.error:
            failed_files.append(result.path)
            continue
        signature = result.insights.get("quality_fingerprint", {}).get("signature")
        if signature:
            fingerprint_groups[str(signature)].append(result)
        content_counts[result.content_type or "unknown"] += 1

        split = _split_for_result(result, splits, split_names)
        split_of[result.path] = split
        split_counts[split or "unassigned"] += 1

        speaker = _speaker_for_result(result, speakers)
        if speaker is not None:
            speaker_counts[speaker] += 1
            speaker_files[speaker].append(result.path)
            if split is not None:
                speaker_splits[speaker].add(split)

        labels = [
            str(label.get("id"))
            for label in result.insights.get("defect_labels", [])
            if isinstance(label, dict) and label.get("id") is not None
        ]
        if labels:
            labeled_files.append(result.path)
        label_counts.update(labels)

        observed = result.content_type
        if expected_content_type and observed not in {None, "unknown", expected_content_type}:
            mismatch_risks.append(
                {
                    "file": result.path,
                    "reason": "unexpected_content_type",
                    "expected": expected_content_type,
                    "observed": observed,
                }
            )
        declared = _lookup(declared_labels, result)
        if declared is not None and observed not in {None, "unknown"} and str(declared) != observed:
            mismatch_risks.append(
                {
                    "file": result.path,
                    "reason": "declared_label_mismatch",
                    "expected": str(declared),
                    "observed": observed,
                }
            )

    duplicate_results = [
        grouped
        for grouped in fingerprint_groups.values()
        if len(grouped) > 1 and len({_source_key(item) for item in grouped}) > 1
    ]
    duplicates = [[item.path for item in grouped] for grouped in duplicate_results]

    leakage: list[dict] = []
    for paths in duplicates:
        found = {split_of.get(path) for path in paths}
        found.discard(None)
        if len(found) > 1:
            leakage.append({"type": "duplicate", "files": paths, "splits": sorted(found)})
    for speaker, found in sorted(speaker_splits.items()):
        if len(found) > 1:
            leakage.append(
                {"type": "speaker", "speaker": speaker, "files": speaker_files[speaker], "splits": sorted(found)}
            )

    if declared_labels:
        for grouped in duplicate_results:
            declared = {str(_lookup(declared_labels, item)) for item in grouped if _lookup(declared_labels, item) is not None}
            if len(declared) > 1:
                mismatch_risks.append(
                    {
                        "files": [item.path for item in grouped],
                        "reason": "conflicting_duplicate_labels",
                        "declared": sorted(declared),
                    }
                )

    warnings: list[str] = []
    analyzed = len(results) - len(failed_files)
    if analyzed and not fingerprint_groups:
        warnings.append(
            "No quality fingerprints found; run analysis with insights=True to enable near-duplicate and leakage checks."
        )
    if analyzed and not speaker_counts:
        warnings.append("No speaker ids available; pass speakers=... to enable speaker balance and leakage checks.")

    remediation = _remediation_plan(
        duplicates=duplicates,
        leakage=leakage,
        speakers=speaker_counts,
        speaker_files=speaker_files,
        mismatches=mismatch_risks,
        labels=label_counts,
        labeled_files=labeled_files,
        failed_files=failed_files,
        imbalance_ratio=imbalance_ratio,
    )
    return {
        "file_count": len(results),
        "near_duplicate_groups": duplicates,
        "split_leakage": leakage,
        "speaker_distribution": dict(sorted(speaker_counts.items())),
        "content_distribution": dict(sorted(content_counts.items())),
        "defect_distribution": dict(sorted(label_counts.items())),
        "split_distribution": dict(sorted(split_counts.items())),
        "label_mismatch_risks": mismatch_risks,
        "failed_files": failed_files,
        "coverage": {
            "content_types": len(content_counts),
            "speakers": len(speaker_counts),
            "fingerprints": len(fingerprint_groups),
        },
        "warnings": warnings,
        "remediation_plan": remediation,
    }


def _metric_value(result: FileResult, name: str):
    return next((metric.value for metric in result.metrics if metric.name == name), None)


def _lookup(mapping: Mapping[str, str] | None, result: FileResult):
    if not mapping:
        return None
    if result.path in mapping:
        return mapping[result.path]
    if result.source_file and result.source_file in mapping:
        return mapping[result.source_file]
    return None


def _source_key(result: FileResult) -> str:
    return result.source_file or result.path


def _speaker_for_result(result: FileResult, speakers: SpeakerSource | None) -> str | None:
    if callable(speakers):
        speaker = speakers(result)
    else:
        speaker = _lookup(speakers, result)
    if speaker is None:
        for name in _SPEAKER_METRICS:
            speaker = _metric_value(result, name)
            if speaker is not None:
                break
    if speaker is None or str(speaker).strip() == "":
        return None
    return str(speaker)


def _split_for_result(
    result: FileResult,
    splits: Mapping[str, str] | None,
    split_names: tuple[str, ...],
) -> str | None:
    explicit = _lookup(splits, result)
    if explicit is not None:
        return _canonical_split(str(explicit))
    return _split_for_path(_source_key(result), split_names)


def _split_for_path(path: str, split_names: tuple[str, ...]) -> str | None:
    wanted = {name.lower() for name in split_names}
    # Walk from the file towards the root so /home/test/data/train/a.wav is "train".
    for part in reversed(PurePath(path).parts[:-1]):
        if part.lower() in wanted:
            return _canonical_split(part)
    return None


def _canonical_split(name: str) -> str:
    lowered = name.strip().lower()
    return "validation" if lowered == "val" else lowered


def _remediation_plan(
    *,
    duplicates: list[list[str]],
    leakage: list[dict],
    speakers: Counter[str],
    speaker_files: Mapping[str, list[str]],
    mismatches: list[dict],
    labels: Counter[str],
    labeled_files: list[str],
    failed_files: list[str],
    imbalance_ratio: float,
) -> list[dict]:
    plan = []
    if duplicates:
        # Keep the first file of each group (sorted for determinism) and drop the rest.
        drop = [path for group in duplicates for path in sorted(group)[1:]]
        plan.append({"priority": 1, "action": "deduplicate", "items": len(duplicates), "files": drop})
    if leakage:
        files = sorted({path for item in leakage for path in item["files"]})
        plan.append({"priority": 1, "action": "remove_split_leakage", "items": len(leakage), "files": files})
    if failed_files:
        plan.append(
            {"priority": 1, "action": "fix_or_remove_unreadable", "items": len(failed_files), "files": list(failed_files)}
        )
    total_speakers = sum(speakers.values())
    if speakers and max(speakers.values()) > max(1, total_speakers * imbalance_ratio):
        dominant, _count = speakers.most_common(1)[0]
        plan.append(
            {
                "priority": 2,
                "action": "rebalance_speakers",
                "items": len(speakers),
                "dominant_speaker": dominant,
                "files": list(speaker_files.get(dominant, [])),
            }
        )
    if mismatches:
        files = sorted(
            {item["file"] for item in mismatches if "file" in item}
            | {path for item in mismatches for path in item.get("files", [])}
        )
        plan.append({"priority": 2, "action": "review_content_labels", "items": len(mismatches), "files": files})
    if labels:
        plan.append(
            {
                "priority": 3,
                "action": "repair_or_remove_quality_outliers",
                "items": sum(labels.values()),
                "files": list(labeled_files),
            }
        )
    return sorted(plan, key=lambda item: (item["priority"], item["action"]))

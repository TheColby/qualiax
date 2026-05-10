"""
Public Python API for qualiax.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Iterable, Sequence, Union

from .analyzer import AudioAnalyzer
from .discovery import collect_requested_files
from .metrics import available_metric_groups
from .models import FileResult
from .presets import get_preset
from .rules import apply_threshold_rules

PathLike = Union[str, Path]
AnalysisResult = list[FileResult]


def analyze(
    paths: PathLike | Sequence[PathLike],
    *,
    reference: PathLike | None = None,
    metrics: str | Iterable[str] | None = None,
    preset: str | None = None,
    verbose: bool = False,
    workers: int = 1,
    segment_seconds: float | None = None,
    strict: bool = True,
    include_demographics: bool = False,
) -> AnalysisResult:
    """Analyze one or more paths and always return a list of FileResult objects."""
    files = collect_requested_files(_normalize_paths(paths))
    if not files:
        raise FileNotFoundError("No supported audio files found.")

    selected_preset = get_preset(preset) if preset is not None else None

    analyzer = AudioAnalyzer(
        metric_groups=_normalize_metric_groups(
            metrics,
            selected_preset.metric_groups if selected_preset is not None else None,
        ),
        reference=Path(reference) if reference is not None else None,
        verbose=verbose,
        workers=workers,
        segment_seconds=segment_seconds,
        strict=strict,
        include_demographics=include_demographics,
    )
    results = analyzer.analyze_all(files)
    if selected_preset is not None:
        apply_threshold_rules(results, list(selected_preset.rules))
    return results


async def analyze_async(
    paths: PathLike | Sequence[PathLike],
    *,
    reference: PathLike | None = None,
    metrics: str | Iterable[str] | None = None,
    preset: str | None = None,
    verbose: bool = False,
    workers: int = 1,
    segment_seconds: float | None = None,
    strict: bool = True,
    include_demographics: bool = False,
) -> AnalysisResult:
    """Asynchronously analyze one or more paths and return a list of FileResult objects."""
    files = collect_requested_files(_normalize_paths(paths))
    if not files:
        raise FileNotFoundError("No supported audio files found.")

    selected_preset = get_preset(preset) if preset is not None else None
    analyzer = AudioAnalyzer(
        metric_groups=_normalize_metric_groups(
            metrics,
            selected_preset.metric_groups if selected_preset is not None else None,
        ),
        reference=Path(reference) if reference is not None else None,
        verbose=verbose,
        workers=workers,
        segment_seconds=segment_seconds,
        strict=strict,
        include_demographics=include_demographics,
    )
    results = await analyzer.analyze_all_async(files)
    if selected_preset is not None:
        apply_threshold_rules(results, list(selected_preset.rules))
    return results


def analyze_one(
    path: PathLike,
    *,
    reference: PathLike | None = None,
    metrics: str | Iterable[str] | None = None,
    preset: str | None = None,
    verbose: bool = False,
    workers: int = 1,
    segment_seconds: float | None = None,
    strict: bool = True,
    include_demographics: bool = False,
) -> FileResult:
    """Analyze exactly one requested path and always return a single FileResult."""
    results = analyze(
        path,
        reference=reference,
        metrics=metrics,
        preset=preset,
        verbose=verbose,
        workers=workers,
        segment_seconds=segment_seconds,
        strict=strict,
        include_demographics=include_demographics,
    )
    if len(results) != 1:
        raise ValueError("analyze_one() expected exactly one result.")
    return results[0]


def analyze_many(
    paths: Sequence[PathLike],
    *,
    reference: PathLike | None = None,
    metrics: str | Iterable[str] | None = None,
    preset: str | None = None,
    verbose: bool = False,
    workers: int = 1,
    segment_seconds: float | None = None,
    strict: bool = True,
    include_demographics: bool = False,
) -> list[FileResult]:
    """Analyze one or more paths and always return a list of FileResult objects."""
    return analyze(
        paths,
        reference=reference,
        metrics=metrics,
        preset=preset,
        verbose=verbose,
        workers=workers,
        segment_seconds=segment_seconds,
        strict=strict,
        include_demographics=include_demographics,
    )


def _normalize_paths(paths: PathLike | Sequence[PathLike]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _normalize_metric_groups(
    metrics: str | Iterable[str] | None,
    preset_groups: tuple[str, ...] | None = None,
) -> set[str]:
    if metrics is None:
        if preset_groups:
            return set(preset_groups)
        return {"all"}
    if isinstance(metrics, str):
        groups = {group.strip().lower() for group in metrics.split(",") if group.strip()}
    else:
        groups = {str(group).strip().lower() for group in metrics if str(group).strip()}
    if not groups:
        raise ValueError("Provide at least one metric group.")
    valid = set(available_metric_groups()) | {"all"}
    unknown = groups - valid
    if unknown:
        raise ValueError(
            f"Unknown metric groups: {', '.join(sorted(unknown))}. "
            f"Valid groups: {', '.join(sorted(valid))}"
        )
    return groups

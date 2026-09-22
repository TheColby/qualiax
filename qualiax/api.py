"""
Public Python API for qualiax.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Union

from .analyzer import AudioAnalyzer
from .discovery import collect_requested_files
from .insights import enrich_results
from .metrics import available_metric_groups
from .models import FileResult
from .performance import AnalysisCache, analysis_cache_config, analyze_with_cache, cache_lookup, cache_store
from .plugins import PluginManager
from .presets import get_preset
from .rules import apply_threshold_rules

PathLike = Union[str, Path]
AnalysisResult = list[FileResult]


def _as_cache(cache: AnalysisCache | PathLike | None) -> AnalysisCache | None:
    if cache is None or isinstance(cache, AnalysisCache):
        return cache
    return AnalysisCache(cache)


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
    insights: bool = False,
    baseline: PathLike | None = None,
    ci: bool = False,
    drift: bool = False,
    drift_state: PathLike | None = None,
    fingerprint_sensitivity: str = "balanced",
    fingerprint_weights: dict[str, float] | None = None,
    insight_rules: PathLike | None = None,
    cache: AnalysisCache | PathLike | None = None,
    plugins: PluginManager | None = None,
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
    cache_dir = _as_cache(cache)
    if cache_dir is None:
        results = analyzer.analyze_all(files)
    else:
        results = analyze_with_cache(files, analyzer.analyze_all, cache_dir, analysis_cache_config(analyzer))
    if selected_preset is not None:
        apply_threshold_rules(results, list(selected_preset.rules))
    if insights:
        enrich_results(
            results,
            baseline_path=baseline,
            ci=ci,
            drift=drift,
            drift_state_path=drift_state,
            fingerprint_sensitivity=fingerprint_sensitivity,
            fingerprint_weights=fingerprint_weights,
            preset=preset,
            insight_rules_path=insight_rules,
            plugins=plugins,
        )
    elif plugins is not None:
        plugins.apply_label_providers(results)
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
    insights: bool = False,
    baseline: PathLike | None = None,
    ci: bool = False,
    drift: bool = False,
    drift_state: PathLike | None = None,
    fingerprint_sensitivity: str = "balanced",
    fingerprint_weights: dict[str, float] | None = None,
    insight_rules: PathLike | None = None,
    cache: AnalysisCache | PathLike | None = None,
    plugins: PluginManager | None = None,
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
    cache_dir = _as_cache(cache)
    if cache_dir is None:
        results = await analyzer.analyze_all_async(files)
    else:
        config = analysis_cache_config(analyzer)
        hits, misses = cache_lookup(files, cache_dir, config)
        fresh = await analyzer.analyze_all_async(misses) if misses else []
        results = cache_store(files, hits, misses, fresh, cache_dir, config)
    if selected_preset is not None:
        apply_threshold_rules(results, list(selected_preset.rules))
    if insights:
        enrich_results(
            results,
            baseline_path=baseline,
            ci=ci,
            drift=drift,
            drift_state_path=drift_state,
            fingerprint_sensitivity=fingerprint_sensitivity,
            fingerprint_weights=fingerprint_weights,
            preset=preset,
            insight_rules_path=insight_rules,
            plugins=plugins,
        )
    elif plugins is not None:
        plugins.apply_label_providers(results)
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
    insights: bool = False,
    baseline: PathLike | None = None,
    ci: bool = False,
    drift: bool = False,
    drift_state: PathLike | None = None,
    fingerprint_sensitivity: str = "balanced",
    fingerprint_weights: dict[str, float] | None = None,
    insight_rules: PathLike | None = None,
    cache: AnalysisCache | PathLike | None = None,
    plugins: PluginManager | None = None,
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
        insights=insights,
        baseline=baseline,
        ci=ci,
        drift=drift,
        drift_state=drift_state,
        fingerprint_sensitivity=fingerprint_sensitivity,
        fingerprint_weights=fingerprint_weights,
        insight_rules=insight_rules,
        cache=cache,
        plugins=plugins,
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
    insights: bool = False,
    baseline: PathLike | None = None,
    ci: bool = False,
    drift: bool = False,
    drift_state: PathLike | None = None,
    fingerprint_sensitivity: str = "balanced",
    fingerprint_weights: dict[str, float] | None = None,
    insight_rules: PathLike | None = None,
    cache: AnalysisCache | PathLike | None = None,
    plugins: PluginManager | None = None,
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
        insights=insights,
        baseline=baseline,
        ci=ci,
        drift=drift,
        drift_state=drift_state,
        fingerprint_sensitivity=fingerprint_sensitivity,
        fingerprint_weights=fingerprint_weights,
        insight_rules=insight_rules,
        cache=cache,
        plugins=plugins,
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

"""Caching, incremental analysis, chunking, and execution adapters."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Protocol, Sequence, TypeVar

from .version import __version__

T = TypeVar("T")
R = TypeVar("R")


def analysis_cache_config(analyzer) -> dict:
    """Everything about an ``AudioAnalyzer`` that changes its output, for cache keys."""
    from .metrics import available_metric_groups

    groups = set(analyzer.metric_groups)
    if "all" in groups:
        groups = set(available_metric_groups())
    reference = None
    if analyzer.reference is not None:
        reference = {
            "path": str(Path(analyzer.reference).expanduser().resolve(strict=False)),
            "signature": list(source_signature(analyzer.reference)),
        }
    return {
        "metric_groups": sorted(groups),
        "reference": reference,
        "segment_seconds": analyzer.segment_seconds,
        "include_demographics": analyzer.include_demographics,
    }


def analyze_with_cache(
    files: Sequence[Path],
    analyze: Callable[[list[Path]], list],
    cache: AnalysisCache | None,
    config: Mapping,
) -> list:
    """Serve ``files`` from ``cache`` where possible and ``analyze`` the rest.

    Results come back in ``files`` order. Fresh results are cached per source file
    unless analysis reported an error, so a transient failure is retried next time.
    """
    if cache is None:
        return analyze(list(files))
    hits, misses = cache_lookup(files, cache, config)
    return cache_store(files, hits, misses, analyze(misses) if misses else [], cache, config)


def cache_lookup(files: Sequence[Path], cache: AnalysisCache, config: Mapping) -> tuple[dict[str, list], list[Path]]:
    """Split ``files`` into cached results (keyed by ``str(path)``) and paths still to analyze."""
    from .models import FileResult

    hits: dict[str, list] = {}
    misses: list[Path] = []
    for path in files:
        payload = cache.get(path, config=config)
        items = payload.get("results") if payload else None
        if isinstance(items, list):
            hits[str(path)] = [FileResult.from_dict(item) for item in items]
        else:
            misses.append(path)
    return hits, misses


def cache_store(
    files: Sequence[Path],
    hits: dict[str, list],
    misses: Sequence[Path],
    fresh: list,
    cache: AnalysisCache,
    config: Mapping,
) -> list:
    """Cache ``fresh`` results for ``misses`` and merge everything back into ``files`` order."""
    miss_keys = {str(path) for path in misses}
    produced: dict[str, list] = {key: [] for key in miss_keys}
    unmatched = []
    for result in fresh:
        key = result.source_file or result.path
        (produced[key] if key in miss_keys else unmatched).append(result)
    merged = dict(hits)
    for path in misses:
        results = produced[str(path)]
        if results and all(_cacheable(result) for result in results):
            cache.put(path, {"results": [result.to_dict() for result in results]}, config=config)
        merged[str(path)] = results
    return [result for path in files for result in merged.get(str(path), [])] + unmatched


def _cacheable(result) -> bool:
    return result.error is None and all(health.status != "error" for health in result.group_health)


class AnalysisCache:
    """Small file-backed cache invalidated by source size and modification time.

    Entries are also keyed by an optional ``config`` mapping (metric groups,
    reference file, segment length, ...) so results computed with different
    analysis options never collide, and are ignored when they were written by a
    different qualiax version.
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, source: str | Path, *, config: Mapping | None = None) -> dict | None:
        path = Path(source)
        cache_path = self._cache_path(path, config)
        if not cache_path.exists():
            return None
        try:
            record = json.loads(cache_path.read_text(encoding="utf-8"))
            if record.get("signature") != list(source_signature(path)):
                return None
            if record.get("tool_version") != __version__:
                return None
            if record.get("config_hash") != _config_hash(config):
                return None
            payload = record.get("payload")
            return payload if isinstance(payload, dict) else None
        except (OSError, ValueError, AttributeError):
            return None

    def put(self, source: str | Path, payload: Mapping, *, config: Mapping | None = None) -> Path:
        path = Path(source)
        cache_path = self._cache_path(path, config)
        record = {
            "source": str(path.expanduser().resolve(strict=False)),
            "signature": list(source_signature(path)),
            "tool_version": __version__,
            "config_hash": _config_hash(config),
            "payload": dict(payload),
        }
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        temporary.replace(cache_path)
        return cache_path

    def invalidate(self, source: str | Path, *, config: Mapping | None = None) -> bool:
        path = self._cache_path(Path(source), config)
        if not path.exists():
            return False
        path.unlink()
        return True

    def clear(self) -> int:
        """Delete every cache entry in this directory; returns the number removed.

        Only files named like cache entries (64 hex characters + ``.json``) are
        touched, so pointing the cache at a shared folder cannot delete reports.
        """
        removed = 0
        for entry in self.directory.glob("*.json"):
            if len(entry.stem) == 64 and all(char in "0123456789abcdef" for char in entry.stem):
                entry.unlink()
                removed += 1
        return removed

    def _cache_path(self, source: Path, config: Mapping | None = None) -> Path:
        identity = str(source.expanduser().resolve(strict=False))
        config_hash = _config_hash(config)
        if config_hash is not None:
            identity = f"{identity}\0{config_hash}"
        key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        return self.directory / f"{key}.json"


def source_signature(source: str | Path) -> tuple[int, int]:
    stat = Path(source).stat()
    return stat.st_mtime_ns, stat.st_size


def snapshot_signatures(sources: Iterable[str | Path]) -> dict[str, list[int]]:
    """Record current signatures (JSON-serializable) for a later :func:`incremental_sources` call.

    Missing files are left out, so they count as changed if they reappear.
    """
    snapshot: dict[str, list[int]] = {}
    for source in sources:
        try:
            snapshot[str(Path(source))] = list(source_signature(source))
        except OSError:
            continue
    return snapshot


def incremental_sources(
    sources: Iterable[str | Path], previous: Mapping[str, Sequence[int]]
) -> list[Path]:
    """Return the sources that are new or changed since ``previous`` was snapshotted."""
    changed = []
    for source in sources:
        path = Path(source)
        expected = tuple(previous.get(str(path), ()))
        try:
            current = source_signature(path)
        except OSError:
            changed.append(path)
            continue
        if current != expected:
            changed.append(path)
    return changed


def iter_audio_chunks(samples: Sequence[T] | Iterable[T], chunk_size: int) -> Iterator:
    """Yield consecutive chunks of at most ``chunk_size`` items.

    NumPy arrays yield views (no copy, chunked along the first axis). Other
    sequences yield lists. Plain iterables such as generators are consumed
    lazily, so memory stays bounded by ``chunk_size``.
    """
    if isinstance(chunk_size, bool) or int(chunk_size) != chunk_size or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    chunk_size = int(chunk_size)
    if hasattr(samples, "__array_interface__"):
        for offset in range(0, len(samples), chunk_size):
            yield samples[offset : offset + chunk_size]
        return
    if isinstance(samples, Sequence):
        for offset in range(0, len(samples), chunk_size):
            yield list(samples[offset : offset + chunk_size])
        return
    iterator = iter(samples)
    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            return
        yield chunk


def benchmark_budget(
    baseline: Mapping[str, float],
    current: Mapping[str, float],
    *,
    max_regression: float = 0.1,
    higher_is_better: Iterable[str] = (),
    allow_missing: bool = False,
) -> dict:
    """Check current performance numbers against a baseline with a relative budget.

    By default lower values are better (runtimes, memory). Names listed in
    ``higher_is_better`` (throughput, files/s) regress when they drop. A metric
    in the baseline but absent from ``current`` fails the budget unless
    ``allow_missing=True``, so a benchmark that silently stops reporting cannot pass.
    """
    higher = set(higher_is_better)
    metrics = []
    passed = True
    for name in sorted(baseline.keys() & current.keys()):
        old = float(baseline[name])
        new = float(current[name])
        change = (old - new) if name in higher else (new - old)
        if old == 0:
            regression = 0.0 if change <= 0 else math.inf
        else:
            regression = change / abs(old)
        within_budget = regression <= max_regression
        passed = passed and within_budget
        metrics.append(
            {
                "metric": name,
                "baseline": old,
                "current": new,
                "regression": regression,
                "within_budget": within_budget,
                "direction": "higher_is_better" if name in higher else "lower_is_better",
            }
        )
    missing = sorted(baseline.keys() - current.keys())
    if missing and not allow_missing:
        passed = False
    return {"passed": passed, "max_regression": max_regression, "metrics": metrics, "missing": missing}


class DistributedAdapter(Protocol):
    def map(self, function: Callable[[T], R], items: Iterable[T]) -> list[R]: ...


class LocalExecutorAdapter:
    """Executor-compatible adapter that can be swapped for a remote backend."""

    def __init__(self, *, max_workers: int | None = None, executor: Executor | None = None):
        self.max_workers = max_workers
        self.executor = executor

    def map(self, function: Callable[[T], R], items: Iterable[T]) -> list[R]:
        if self.executor is not None:
            return list(self.executor.map(function, items))
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            return list(executor.map(function, items))


def _config_hash(config: Mapping | None) -> str | None:
    if not config:
        return None
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]

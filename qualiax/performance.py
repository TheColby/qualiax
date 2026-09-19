"""Caching, incremental analysis, chunking, and execution adapters."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import Executor, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Protocol, Sequence, TypeVar


T = TypeVar("T")
R = TypeVar("R")


class AnalysisCache:
    """Small file-backed cache invalidated by source size and modification time."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def get(self, source: str | Path) -> dict | None:
        path = Path(source)
        cache_path = self._cache_path(path)
        if not cache_path.exists():
            return None
        try:
            record = json.loads(cache_path.read_text(encoding="utf-8"))
            if record.get("signature") != list(source_signature(path)):
                return None
            payload = record.get("payload")
            return payload if isinstance(payload, dict) else None
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def put(self, source: str | Path, payload: Mapping) -> Path:
        path = Path(source)
        cache_path = self._cache_path(path)
        record = {
            "source": str(path.expanduser().resolve(strict=False)),
            "signature": list(source_signature(path)),
            "payload": dict(payload),
        }
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")
        temporary.replace(cache_path)
        return cache_path

    def invalidate(self, source: str | Path) -> bool:
        path = self._cache_path(Path(source))
        if not path.exists():
            return False
        path.unlink()
        return True

    def _cache_path(self, source: Path) -> Path:
        key = hashlib.sha256(str(source.expanduser().resolve(strict=False)).encode("utf-8")).hexdigest()
        return self.directory / f"{key}.json"


def source_signature(source: str | Path) -> tuple[int, int]:
    stat = Path(source).stat()
    return stat.st_mtime_ns, stat.st_size


def incremental_sources(
    sources: Iterable[str | Path], previous: Mapping[str, Sequence[int]]
) -> list[Path]:
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


def iter_audio_chunks(samples: Sequence[T], chunk_size: int) -> Iterator[list[T]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    for offset in range(0, len(samples), chunk_size):
        yield list(samples[offset : offset + chunk_size])


def benchmark_budget(
    baseline: Mapping[str, float],
    current: Mapping[str, float],
    *,
    max_regression: float = 0.1,
) -> dict:
    metrics = []
    passed = True
    for name in sorted(baseline.keys() & current.keys()):
        old = float(baseline[name])
        new = float(current[name])
        regression = 0.0 if old == 0 and new == 0 else float("inf") if old == 0 else (new - old) / abs(old)
        within_budget = regression <= max_regression
        passed = passed and within_budget
        metrics.append(
            {
                "metric": name,
                "baseline": old,
                "current": new,
                "regression": regression,
                "within_budget": within_budget,
            }
        )
    return {"passed": passed, "max_regression": max_regression, "metrics": metrics}


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

import json
import math
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from qualiax.performance import (
    AnalysisCache,
    LocalExecutorAdapter,
    benchmark_budget,
    incremental_sources,
    iter_audio_chunks,
    snapshot_signatures,
    source_signature,
)


def _touch(path, data=b"abc", mtime_ns=None):
    path.write_bytes(data)
    if mtime_ns is not None:
        os.utime(path, ns=(mtime_ns, mtime_ns))
    return path


def test_cache_hit_and_signature_invalidation(tmp_path):
    source = _touch(tmp_path / "a.wav", mtime_ns=1_000_000_000)
    cache = AnalysisCache(tmp_path / "cache")
    cache.put(source, {"score": 1})

    assert cache.get(source) == {"score": 1}
    _touch(source, b"xyz", mtime_ns=2_000_000_000)  # same size, newer mtime
    assert cache.get(source) is None
    cache.put(source, {"score": 2})
    _touch(source, b"longer", mtime_ns=2_000_000_000)  # same mtime, new size
    assert cache.get(source) is None


def test_cache_entries_are_keyed_by_analysis_config(tmp_path):
    source = _touch(tmp_path / "a.wav")
    cache = AnalysisCache(tmp_path / "cache")
    basic = {"metrics": ["basic"], "segment_seconds": None}
    full = {"metrics": ["all"], "segment_seconds": None}

    cache.put(source, {"groups": "basic"}, config=basic)

    assert cache.get(source, config=full) is None
    assert cache.get(source) is None
    assert cache.get(source, config={"segment_seconds": None, "metrics": ["basic"]}) == {"groups": "basic"}
    cache.put(source, {"groups": "all"}, config=full)
    assert cache.get(source, config=basic) == {"groups": "basic"}
    assert cache.invalidate(source, config=basic) is True
    assert cache.get(source, config=basic) is None
    assert cache.get(source, config=full) == {"groups": "all"}


def test_cache_ignores_entries_from_another_qualiax_version(tmp_path, monkeypatch):
    source = _touch(tmp_path / "a.wav")
    cache = AnalysisCache(tmp_path / "cache")
    cache.put(source, {"score": 1})

    monkeypatch.setattr("qualiax.performance.__version__", "9.9.9")

    assert cache.get(source) is None


def test_cache_survives_corrupt_entries_and_clear_only_removes_its_files(tmp_path):
    source = _touch(tmp_path / "a.wav")
    cache_dir = tmp_path / "shared"
    cache = AnalysisCache(cache_dir)
    entry = cache.put(source, {"score": 1})
    report = cache_dir / "report.json"
    report.write_text("[]")

    entry.write_text("not json")
    assert cache.get(source) is None
    entry.write_text(json.dumps(["unexpected"]))
    assert cache.get(source) is None

    cache.put(source, {"score": 1})
    assert cache.clear() == 1
    assert report.exists()
    assert cache.get(source) is None
    assert cache.invalidate(source) is False


def test_snapshot_and_incremental_selection(tmp_path):
    a = _touch(tmp_path / "a.wav", mtime_ns=1_000_000_000)
    b = _touch(tmp_path / "b.wav", mtime_ns=1_000_000_000)
    snapshot = json.loads(json.dumps(snapshot_signatures([a, b, tmp_path / "gone.wav"])))

    assert set(snapshot) == {str(a), str(b)}
    assert snapshot[str(a)] == list(source_signature(a))
    assert incremental_sources([a, b], snapshot) == []

    _touch(b, b"changed", mtime_ns=3_000_000_000)
    c = _touch(tmp_path / "c.wav")
    assert incremental_sources([a, b, c, tmp_path / "gone.wav"], snapshot) == [b, c, tmp_path / "gone.wav"]


def test_iter_audio_chunks_uses_numpy_views():
    audio = np.arange(10, dtype=np.float32)

    chunks = list(iter_audio_chunks(audio, 4))

    assert [chunk.tolist() for chunk in chunks] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert all(isinstance(chunk, np.ndarray) and chunk.dtype == np.float32 for chunk in chunks)
    assert all(np.shares_memory(chunk, audio) for chunk in chunks)
    stereo = np.zeros((9, 2))
    assert [chunk.shape for chunk in iter_audio_chunks(stereo, 4)] == [(4, 2), (4, 2), (1, 2)]


def test_iter_audio_chunks_bounds_memory_for_streams():
    pulled = []

    def stream():
        for value in range(1_000_000):
            pulled.append(value)
            yield value

    iterator = iter_audio_chunks(stream(), 3)
    first = next(iterator)

    assert first == [0, 1, 2]
    assert len(pulled) == 3
    assert list(iter_audio_chunks((1, 2, 3), 2)) == [[1, 2], [3]]


@pytest.mark.parametrize("size", [0, -1, 2.5, True])
def test_iter_audio_chunks_rejects_bad_sizes(size):
    with pytest.raises(ValueError):
        list(iter_audio_chunks([1, 2, 3], size))


def test_benchmark_budget_directions_and_missing_metrics():
    baseline = {"runtime_s": 1.0, "files_per_s": 100.0, "peak_mb": 200.0}

    ok = benchmark_budget(baseline, {"runtime_s": 1.05, "files_per_s": 95.0, "peak_mb": 150.0}, higher_is_better=["files_per_s"])
    slow = benchmark_budget(baseline, {"runtime_s": 1.3, "files_per_s": 100.0, "peak_mb": 200.0}, higher_is_better=["files_per_s"])
    fewer = benchmark_budget(baseline, {"runtime_s": 1.0, "files_per_s": 80.0, "peak_mb": 200.0}, higher_is_better=["files_per_s"])
    missing = benchmark_budget(baseline, {"runtime_s": 1.0})

    assert ok["passed"] is True
    rows = {row["metric"]: row for row in ok["metrics"]}
    assert rows["files_per_s"]["regression"] == pytest.approx(0.05)
    assert rows["peak_mb"]["regression"] == pytest.approx(-0.25)
    assert slow["passed"] is False
    assert fewer["passed"] is False
    assert {row["metric"]: row["within_budget"] for row in fewer["metrics"]}["files_per_s"] is False
    assert missing["passed"] is False
    assert missing["missing"] == ["files_per_s", "peak_mb"]
    assert benchmark_budget(baseline, {"runtime_s": 1.0}, allow_missing=True)["passed"] is True


def test_benchmark_budget_zero_baseline():
    assert benchmark_budget({"errors": 0}, {"errors": 0})["passed"] is True
    grew = benchmark_budget({"errors": 0}, {"errors": 2})
    assert grew["passed"] is False
    assert math.isinf(grew["metrics"][0]["regression"])


def test_local_executor_adapter_uses_supplied_executor():
    calls = []

    class RecordingExecutor(ThreadPoolExecutor):
        def map(self, fn, *iterables, **kwargs):
            calls.append(fn)
            return super().map(fn, *iterables, **kwargs)

    with RecordingExecutor(max_workers=2) as executor:
        assert LocalExecutorAdapter(executor=executor).map(abs, [-1, -2, 3]) == [1, 2, 3]
    assert calls == [abs]
    assert LocalExecutorAdapter(max_workers=2).map(str, [1, 2]) == ["1", "2"]

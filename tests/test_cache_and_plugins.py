"""Analysis cache wiring (API + CLI) and plugin labels composing with --insights."""
from __future__ import annotations

import asyncio
import json
import os
import wave
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import qualiax.plugins as plugins_module
from qualiax import analyze, analyze_async
from qualiax.analyzer import AudioAnalyzer
from qualiax.cli import main
from qualiax.insights import enrich_results
from qualiax.metrics import METRIC_GROUPS, unregister_metric_group
from qualiax.models import FileResult, MetricResult
from qualiax.plugins import PluginManager
from qualiax.validation import validate_report_payload


def _tone(path: Path, freq: float = 220.0, seconds: float = 1.5, sr: int = 16_000) -> Path:
    t = np.arange(int(seconds * sr)) / sr
    audio = 0.3 * np.sin(2 * np.pi * freq * t) * (0.5 * (1 - np.cos(2 * np.pi * 2 * t)))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())
    return path


@pytest.fixture
def analyzed_paths(monkeypatch):
    """Record which paths AudioAnalyzer actually analyzes."""
    seen: list[list[str]] = []
    real = AudioAnalyzer.analyze_all

    def spy(self, paths, on_progress=None):
        seen.append([str(p) for p in paths])
        return real(self, paths, on_progress=on_progress)

    monkeypatch.setattr(AudioAnalyzer, "analyze_all", spy)
    return seen


# ── FileResult round trip ────────────────────────────────────────────────────

@pytest.mark.parametrize("segment_seconds", [None, 0.5])
def test_file_result_round_trips_through_dict(tmp_path, segment_seconds):
    results = analyze(_tone(tmp_path / "a.wav"), metrics="basic,loudness,noise", segment_seconds=segment_seconds)
    for result in results:
        as_json = json.loads(json.dumps(result.to_dict()))
        assert json.loads(json.dumps(FileResult.from_dict(as_json).to_dict())) == as_json


# ── cache ────────────────────────────────────────────────────────────────────

def test_analyze_serves_unchanged_files_from_cache(tmp_path, analyzed_paths):
    a, b = _tone(tmp_path / "a.wav", 220), _tone(tmp_path / "b.wav", 330)
    cache = tmp_path / "cache"

    first = analyze([a, b], metrics="basic,loudness", cache=cache)
    second = analyze([a, b], metrics="basic,loudness", cache=cache)

    assert analyzed_paths == [[str(a), str(b)]]
    assert [r.to_dict() for r in second] == [r.to_dict() for r in first]


def test_cache_misses_on_changed_file_or_options_and_keeps_order(tmp_path, analyzed_paths):
    a, b, c = (_tone(tmp_path / f"{n}.wav", f) for n, f in (("a", 220), ("b", 330), ("c", 440)))
    cache = tmp_path / "cache"
    analyze([a, b, c], metrics="basic", cache=cache)

    _tone(b, 550, seconds=2.0)
    os.utime(b, ns=(b.stat().st_atime_ns, b.stat().st_mtime_ns + 5_000_000_000))
    results = analyze([a, b, c], metrics="basic", cache=cache)
    analyze([a], metrics="basic,loudness", cache=cache)

    assert analyzed_paths[1:] == [[str(b)], [str(a)]]
    assert [Path(r.path).name for r in results] == ["a.wav", "b.wav", "c.wav"]
    assert results[1].duration_s == pytest.approx(2.0, abs=0.01)


def test_segmented_results_are_cached_per_source(tmp_path, analyzed_paths):
    a = _tone(tmp_path / "a.wav", seconds=1.5)
    first = analyze(a, metrics="basic", segment_seconds=0.5, cache=tmp_path / "cache")
    second = analyze(a, metrics="basic", segment_seconds=0.5, cache=tmp_path / "cache")
    assert len(analyzed_paths) == 1
    assert [r.segment_index for r in second] == [r.segment_index for r in first] == [1, 2, 3]


def test_failed_analysis_is_not_cached(tmp_path, monkeypatch, analyzed_paths):
    def broken(audio, sr, ref_audio=None, ref_sr=None):
        raise ValueError("transient")

    monkeypatch.setitem(METRIC_GROUPS, "temporal", broken)
    a = _tone(tmp_path / "a.wav")
    analyze(a, metrics="basic,temporal", strict=False, cache=tmp_path / "cache")
    analyze(a, metrics="basic,temporal", strict=False, cache=tmp_path / "cache")
    assert len(analyzed_paths) == 2


def test_analyze_async_uses_the_cache(tmp_path, monkeypatch):
    a = _tone(tmp_path / "a.wav")
    first = asyncio.run(analyze_async(a, metrics="basic", cache=tmp_path / "cache"))

    async def fail(self, paths):
        raise AssertionError(f"should have been cached: {paths}")

    monkeypatch.setattr(AudioAnalyzer, "analyze_all_async", fail)
    second = asyncio.run(analyze_async(a, metrics="basic", cache=tmp_path / "cache"))
    assert [r.to_dict() for r in second] == [r.to_dict() for r in first]


def test_cli_cache_reuses_results_and_rejects_watch(tmp_path, analyzed_paths):
    a = _tone(tmp_path / "a.wav")
    args = [str(a), "--metrics", "basic", "--cache", str(tmp_path / "cache"), "--silent", "--force"]
    runner = CliRunner()
    first = runner.invoke(main, args + ["--output", str(tmp_path / "one.json")])
    second = runner.invoke(main, args + ["--output", str(tmp_path / "two.json")])

    assert first.exit_code == second.exit_code == 0, second.output
    assert len(analyzed_paths) == 1
    assert json.loads((tmp_path / "one.json").read_text()) == json.loads((tmp_path / "two.json").read_text())

    watch = runner.invoke(main, [str(tmp_path), "--watch", "--cache", str(tmp_path / "cache")])
    assert watch.exit_code == 1
    assert "--cache is not supported with --watch" in watch.output


# ── plugins + insights ───────────────────────────────────────────────────────

def _manager(severity="fail", confidence=None) -> PluginManager:
    manager = PluginManager()

    def hum(result):
        label = {"id": "mains_hum", "severity": severity, "evidence": "50 Hz peak"}
        if confidence is not None:
            label["confidence"] = confidence
        return [label]

    manager.register_label_provider("hum", hum)
    return manager


def _result(path="x.wav") -> FileResult:
    return FileResult(
        path=path,
        duration_s=2.0,
        sample_rate=16_000,
        channels=1,
        content_type="speech",
        metrics=[MetricResult("Estimated SNR", 35.0, "dB", group="noise")],
    )


def test_plugin_labels_feed_ci_gate_and_triage():
    result = _result()
    enrich_results([result], ci=True, plugins=_manager(severity="fail"))

    labels = {label["id"]: label for label in result.insights["defect_labels"]}
    assert labels["mains_hum"]["source"] == "plugin:hum"
    gate = {check["id"]: check for check in result.insights["ci_checks"]}["audio_quality_gate"]
    assert gate["status"] == "fail"
    assert "mains_hum" in result.insights["triage"]["reasons"]
    assert result.insights["triage"]["score"] > 0
    assert validate_report_payload([result.to_dict()]) == []


def test_plugin_labels_survive_re_enrichment_without_duplicates():
    manager = _manager(severity="warn", confidence=0.7)
    applied_first = _result()
    manager.apply_label_providers([applied_first])
    enrich_results([applied_first])
    assert [label["id"] for label in applied_first.insights["defect_labels"]] == ["mains_hum"]

    rerun = _result()
    enrich_results([rerun], plugins=manager)
    enrich_results([rerun], plugins=manager)
    assert [label["id"] for label in rerun.insights["defect_labels"]] == ["mains_hum"]


def test_analyze_applies_plugins_with_or_without_insights(tmp_path):
    a = _tone(tmp_path / "a.wav")
    plain = analyze(a, metrics="basic", plugins=_manager(severity="warn"))[0]
    enriched = analyze(a, metrics="basic", insights=True, ci=True, plugins=_manager(severity="warn"))[0]

    assert [label["id"] for label in plain.insights["defect_labels"]] == ["mains_hum"]
    assert "version" not in plain.insights
    assert "mains_hum" in [label["id"] for label in enriched.insights["defect_labels"]]
    assert validate_report_payload([plain.to_dict(), enriched.to_dict()]) == []


class _EntryPoint:
    def __init__(self, name, plugin=None, error=None):
        self.name, self.value = name, f"fake:{name}"
        self._plugin, self._error = plugin, error

    def load(self):
        if self._error:
            raise self._error
        return self._plugin


def test_cli_plugins_flag_discovers_labels_and_metric_groups(tmp_path, monkeypatch):
    def register(manager):
        from qualiax import register_metric_group

        register_metric_group(
            "hum_check",
            lambda audio, sr, ref_audio=None, ref_sr=None: [MetricResult("Hum Level", -60.0, "dB", group="hum_check")],
            overwrite=True,
        )
        manager.register_label_provider("hum", lambda result: [{"id": "mains_hum", "severity": "warn"}])

    monkeypatch.setattr(
        plugins_module,
        "_entry_points",
        lambda group: [_EntryPoint("hum", register), _EntryPoint("broken", error=ImportError("missing dep"))],
    )
    try:
        out = tmp_path / "r.json"
        result = CliRunner().invoke(
            main, [str(_tone(tmp_path / "a.wav")), "--plugins", "--metrics", "basic,hum_check", "--insights",
                   "--output", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert "Plugin 'broken' was skipped (load): missing dep" in result.output
        item = json.loads(out.read_text())[0]
        assert "Hum Level" in {metric["name"] for metric in item["metrics"]}
        assert "mains_hum" in [label["id"] for label in item["insights"]["defect_labels"]]
    finally:
        unregister_metric_group("hum_check")


def test_cli_rejects_unknown_groups_without_plugins(tmp_path):
    result = CliRunner().invoke(main, [str(_tone(tmp_path / "a.wav")), "--metrics", "hum_check"])
    assert result.exit_code == 1
    assert "Unknown metric groups: hum_check" in result.output

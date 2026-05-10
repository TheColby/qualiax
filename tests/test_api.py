import asyncio
import wave
from pathlib import Path

import numpy as np

from qualiax import (
    FileResult,
    MetricResult,
    analyze,
    analyze_async,
    analyze_many,
    analyze_one,
    available_metric_groups,
    available_presets,
    build_scorecard,
    get_preset,
    register_metric_group,
    unregister_metric_group,
)


def _write_wav(path: Path, sr: int = 16_000, duration_s: float = 0.25) -> None:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    pcm = np.clip(audio * 32767, -32768, 32767).astype("<i2")

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def test_analyze_returns_typed_file_result_for_single_file(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_wav(wav_path)

    results = analyze(wav_path, metrics=["basic"])

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0].path.endswith("tone.wav")
    assert any(metric.group == "basic" for metric in results[0].metrics)


def test_analyze_async_returns_result_for_single_file(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_wav(wav_path)

    results = asyncio.run(analyze_async(wav_path, metrics="basic"))

    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0].sample_rate == 16_000


def test_analyze_one_and_many_provide_stable_return_shapes(tmp_path):
    first = tmp_path / "tone_a.wav"
    second = tmp_path / "tone_b.wav"
    _write_wav(first)
    _write_wav(second)

    one = analyze_one(first, metrics=["basic"])
    many = analyze_many([first, second], metrics=["basic"])

    assert isinstance(one, FileResult)
    assert isinstance(many, list)
    assert len(many) == 2


def test_analyze_returns_list_for_single_file(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_wav(wav_path)

    results = analyze(wav_path, metrics=["basic"])

    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], FileResult)


def test_analyze_rejects_unsupported_direct_input(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("not audio")

    try:
        analyze(notes)
    except ValueError as exc:
        assert "Unsupported file type" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported input")


def test_custom_metric_group_can_be_registered_and_used(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_wav(wav_path)

    def compute_custom(audio, sr, ref_audio=None, ref_sr=None):
        return [MetricResult(name="Custom Score", value=7, group="custom")]

    register_metric_group("custom", compute_custom)
    try:
        assert "custom" in available_metric_groups()
        results = analyze(wav_path, metrics=["custom"])
        assert isinstance(results, list)
        assert results[0].metrics[0].name == "Custom Score"
    finally:
        unregister_metric_group("custom")


def test_analyze_rejects_unknown_metric_group(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_wav(wav_path)

    try:
        analyze(wav_path, metrics=["speach"])
    except ValueError as exc:
        assert "Unknown metric groups: speach" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown metric group")


def test_available_presets_exposes_builtin_profiles():
    presets = available_presets()

    assert "podcast" in presets
    assert "call-center-qa" in presets
    assert "speech-enhancement" in presets
    assert "music-mastering" in presets


def test_analyze_applies_preset_metric_groups_and_rules(monkeypatch, tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_wav(wav_path)
    captured = {}

    class PresetAnalyzer:
        def __init__(self, *args, **kwargs):
            captured["metric_groups"] = kwargs["metric_groups"]

        def analyze_all(self, paths):
            return [
                FileResult(
                    path=str(paths[0]),
                    metrics=[
                        MetricResult(
                            name="Integrated Loudness (LUFS)",
                            value=-13.0,
                            unit="LUFS",
                            group="loudness",
                        ),
                        MetricResult(
                            name="True Peak",
                            value=-0.5,
                            unit="dBTP",
                            group="loudness",
                        ),
                    ],
                )
            ]

    monkeypatch.setattr("qualiax.api.AudioAnalyzer", PresetAnalyzer)

    results = analyze(wav_path, preset="podcast")

    assert isinstance(results, list)
    result = results[0]
    assert captured["metric_groups"] == set(get_preset("podcast").metric_groups)
    assert result.notes == ["threshold rules: 2 violation(s)"]
    assert result.metrics[0].warning is not None
    assert result.metrics[1].warning is not None


def test_analyze_rejects_unknown_preset(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_wav(wav_path)

    try:
        analyze(wav_path, preset="broadcastz")
    except ValueError as exc:
        assert "Unknown preset: broadcastz" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown preset")


def test_analyze_surfaces_confidence_notes_for_proxy_metrics(tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_wav(wav_path, duration_s=0.5)

    results = analyze(wav_path, metrics=["perceptual"])
    scorecard = build_scorecard(results)
    result = results[0]

    assert isinstance(result, FileResult)
    assert result.confidence_notes
    assert scorecard.confidence_notes


def test_analyze_forwards_strict_and_include_demographics(monkeypatch, tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_wav(wav_path)
    captured = {}

    class CapturingAnalyzer:
        def __init__(self, *args, **kwargs):
            captured["strict"] = kwargs["strict"]
            captured["include_demographics"] = kwargs["include_demographics"]

        def analyze_all(self, paths):
            return [FileResult(path=str(paths[0]))]

    monkeypatch.setattr("qualiax.api.AudioAnalyzer", CapturingAnalyzer)

    analyze(wav_path, strict=True, include_demographics=True)

    assert captured["strict"] is True
    assert captured["include_demographics"] is True


def test_analyze_async_uses_async_analyzer_path(monkeypatch, tmp_path):
    wav_path = tmp_path / "tone.wav"
    _write_wav(wav_path)

    class AsyncAnalyzer:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_all(self, paths):
            raise AssertionError("sync analyze_all should not be used")

        async def analyze_all_async(self, paths):
            return [FileResult(path=str(paths[0]), sample_rate=16_000)]

    monkeypatch.setattr("qualiax.api.AudioAnalyzer", AsyncAnalyzer)

    results = asyncio.run(analyze_async(wav_path, metrics="basic"))

    assert len(results) == 1
    assert results[0].sample_rate == 16_000

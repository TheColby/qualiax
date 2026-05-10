import time
import wave
from pathlib import Path

import numpy as np

from qualiax import analyzer as analyzer_module
from qualiax.analyzer import AudioAnalyzer, AudioLoader
from qualiax.models import MetricResult
from qualiax.speech_detector import SpeechDetectionResult


def _write_pcm_wav(path: Path, sample_width: int, frames: bytes, sr: int = 8_000) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(sample_width)
        wf.setframerate(sr)
        wf.writeframes(frames)


def test_wav_stdlib_fallback_decodes_unsigned_8bit_pcm(tmp_path):
    wav_path = tmp_path / "unsigned8.wav"
    _write_pcm_wav(wav_path, 1, bytes([0, 128, 255]))

    audio, sr = AudioLoader._load_wav_stdlib(wav_path)

    assert sr == 8_000
    assert audio.shape == (3,)
    np.testing.assert_allclose(audio, np.array([-1.0, 0.0, 127 / 128], dtype=np.float32), atol=1e-6)


def test_wav_stdlib_fallback_decodes_24bit_pcm(tmp_path):
    wav_path = tmp_path / "pcm24.wav"
    samples = [-8388608, 0, 8388607]
    frames = b"".join(value.to_bytes(3, byteorder="little", signed=True) for value in samples)
    _write_pcm_wav(wav_path, 3, frames)

    audio, sr = AudioLoader._load_wav_stdlib(wav_path)

    assert sr == 8_000
    assert audio.shape == (3,)
    np.testing.assert_allclose(
        audio,
        np.array([-1.0, 0.0, 8388607 / 8388608], dtype=np.float32),
        atol=1e-6,
    )


def test_reference_load_failure_surfaces_as_file_error(monkeypatch, tmp_path):
    audio_path = tmp_path / "input.wav"
    ref_path = tmp_path / "reference.wav"
    audio_path.write_bytes(b"audio")
    ref_path.write_bytes(b"reference")

    def fake_load(path: Path):
        if path == ref_path:
            raise RuntimeError("unsupported reference format")
        return np.zeros(16_000, dtype=np.float32), 16_000

    monkeypatch.setattr(AudioLoader, "load", staticmethod(fake_load))

    analyzer = AudioAnalyzer(metric_groups={"basic"}, reference=ref_path)
    result = analyzer.analyze_file(audio_path)

    assert result.error is not None
    assert "unsupported reference format" in result.error
    assert result.diagnostics[0].code == "reference_load_failed"


def test_group_warnings_are_recorded_in_notes(monkeypatch, tmp_path):
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"audio")

    def fake_load(path: Path):
        return np.zeros(16_000, dtype=np.float32), 16_000

    def warning_group(audio, sr, ref_audio=None, ref_sr=None):
        import warnings

        warnings.warn("approximate result", RuntimeWarning)
        return [MetricResult(name="Dummy Metric", value=1.0, group="dummy")]

    monkeypatch.setattr(AudioLoader, "load", staticmethod(fake_load))
    monkeypatch.setitem(analyzer_module.METRIC_GROUPS, "dummy", warning_group)

    analyzer = AudioAnalyzer(metric_groups={"dummy"})
    result = analyzer.analyze_file(audio_path)

    assert result.error is None
    assert result.notes == ["dummy: approximate result"]
    assert result.diagnostics[0].code == "metric_group_warning"
    assert result.group_health[0].diagnostic_count == 1
    assert result.provenance is not None


def test_reference_is_loaded_once_with_parallel_workers(monkeypatch, tmp_path):
    ref_path = tmp_path / "reference.wav"
    ref_path.write_bytes(b"reference")
    audio_paths = []
    for idx in range(4):
        path = tmp_path / f"input_{idx}.wav"
        path.write_bytes(b"audio")
        audio_paths.append(path)

    ref_loads = {"count": 0}

    def fake_load(path: Path):
        if path == ref_path:
            ref_loads["count"] += 1
            time.sleep(0.05)
        return np.zeros(16_000, dtype=np.float32), 16_000

    monkeypatch.setattr(AudioLoader, "load", staticmethod(fake_load))
    monkeypatch.setattr(
        analyzer_module,
        "detect_speech",
        lambda audio, sr: SpeechDetectionResult(
            is_speech=False,
            confidence=0.05,
            threshold=0.45,
            speech_fraction=0.0,
            content_type="noise",
        ),
    )

    analyzer = AudioAnalyzer(metric_groups={"basic"}, reference=ref_path, workers=4)
    results = analyzer.analyze_all(audio_paths)

    assert len(results) == 4
    assert ref_loads["count"] == 1


def test_non_speech_content_skips_speech_groups_and_filters_speech_perceptual(monkeypatch, tmp_path):
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"audio")

    def fake_load(path: Path):
        return np.zeros(16_000, dtype=np.float32), 16_000

    def fake_speech(audio, sr, ref_audio=None, ref_sr=None):
        return [MetricResult(name="MFCC-1 Mean", value=0.1, group="speech")]

    def fake_perceptual(audio, sr, ref_audio=None, ref_sr=None):
        return [
            MetricResult(name="PESQ (ITU-T P.862)", value=2.0, group="perceptual"),
            MetricResult(name="SI-SDR", value=10.0, unit="dB", group="perceptual"),
        ]

    monkeypatch.setattr(AudioLoader, "load", staticmethod(fake_load))
    monkeypatch.setattr(
        analyzer_module,
        "detect_speech",
        lambda audio, sr: SpeechDetectionResult(
            is_speech=False,
            confidence=0.12,
            threshold=0.45,
            speech_fraction=0.0,
            content_type="noise",
        ),
    )
    monkeypatch.setitem(analyzer_module.METRIC_GROUPS, "speech", fake_speech)
    monkeypatch.setitem(analyzer_module.METRIC_GROUPS, "perceptual", fake_perceptual)

    analyzer = AudioAnalyzer(metric_groups={"speech", "perceptual"})
    result = analyzer.analyze_file(audio_path)

    assert result.content_type == "noise"
    assert result.speech_confidence == 0.12
    assert any("speech: skipped" in note for note in result.notes)
    assert any("perceptual: omitted speech-specific metrics" in note for note in result.notes)
    assert [metric.name for metric in result.metrics] == ["SI-SDR"]


def test_mixed_content_with_substantial_speech_keeps_speech_metrics(monkeypatch, tmp_path):
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"audio")

    def fake_load(path: Path):
        return np.zeros(16_000, dtype=np.float32), 16_000

    def fake_speech(audio, sr, ref_audio=None, ref_sr=None):
        return [MetricResult(name="MFCC-1 Mean", value=0.1, group="speech")]

    def fake_perceptual(audio, sr, ref_audio=None, ref_sr=None):
        return [
            MetricResult(name="PESQ (ITU-T P.862)", value=2.0, group="perceptual"),
            MetricResult(name="SI-SDR", value=10.0, unit="dB", group="perceptual"),
        ]

    monkeypatch.setattr(AudioLoader, "load", staticmethod(fake_load))
    monkeypatch.setattr(
        analyzer_module,
        "detect_speech",
        lambda audio, sr: SpeechDetectionResult(
            is_speech=False,
            confidence=0.38,
            threshold=0.45,
            speech_fraction=0.55,
            content_type="mixed",
        ),
    )
    monkeypatch.setitem(analyzer_module.METRIC_GROUPS, "speech", fake_speech)
    monkeypatch.setitem(analyzer_module.METRIC_GROUPS, "perceptual", fake_perceptual)

    analyzer = AudioAnalyzer(metric_groups={"speech", "perceptual"})
    result = analyzer.analyze_file(audio_path)

    assert result.content_type == "mixed"
    assert any("retaining speech-oriented metrics" in note for note in result.notes)
    assert not any("speech: skipped" in note for note in result.notes)
    assert [metric.name for metric in result.metrics] == [
        "MFCC-1 Mean",
        "PESQ (ITU-T P.862)",
        "SI-SDR",
    ]


def test_segment_mode_splits_audio_and_carries_segment_metadata(monkeypatch, tmp_path):
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"audio")

    def fake_load(path: Path):
        return np.arange(32_000, dtype=np.float32), 16_000

    def fake_basic(audio, sr, ref_audio=None, ref_sr=None):
        return [MetricResult(name="Samples", value=len(audio), group="basic")]

    monkeypatch.setattr(AudioLoader, "load", staticmethod(fake_load))
    monkeypatch.setattr(
        analyzer_module,
        "detect_speech",
        lambda audio, sr: SpeechDetectionResult(
            is_speech=True,
            confidence=0.91,
            threshold=0.45,
            speech_fraction=1.0,
            content_type="speech",
        ),
    )
    monkeypatch.setitem(analyzer_module.METRIC_GROUPS, "basic", fake_basic)

    analyzer = AudioAnalyzer(metric_groups={"basic"}, segment_seconds=1.0)
    results = analyzer.analyze_all([audio_path])

    assert len(results) == 2
    assert results[0].source_file == str(audio_path)
    assert results[0].segment_index == 1
    assert results[0].total_segments == 2
    assert results[0].segment_start_s == 0.0
    assert results[0].segment_end_s == 1.0
    assert results[0].metrics[0].value == 16_000
    assert "[segment 1/2 0.00-1.00s]" in results[0].path


def test_speaker_demographics_are_opt_in(monkeypatch, tmp_path):
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"audio")
    calls = []

    def fake_load(path: Path):
        return np.zeros(16_000, dtype=np.float32), 16_000

    def fake_compute_speaker(audio, sr, ref_audio=None, ref_sr=None, *, include_demographics=False):
        calls.append(include_demographics)
        return [MetricResult(name="CPP", value=5.0, group="speaker")]

    monkeypatch.setattr(AudioLoader, "load", staticmethod(fake_load))
    monkeypatch.setattr(
        analyzer_module,
        "detect_speech",
        lambda audio, sr: SpeechDetectionResult(
            is_speech=True,
            confidence=0.9,
            threshold=0.45,
            speech_fraction=1.0,
            content_type="speech",
        ),
    )
    monkeypatch.setattr("qualiax.metrics_speaker.compute_speaker", fake_compute_speaker)

    default_result = AudioAnalyzer(metric_groups={"speaker"}).analyze_file(audio_path)
    opted_in_result = AudioAnalyzer(metric_groups={"speaker"}, include_demographics=True).analyze_file(audio_path)

    assert default_result.metrics[0].name == "CPP"
    assert opted_in_result.metrics[0].name == "CPP"
    assert calls == [False, True]


def test_strict_mode_raises_on_group_failure(monkeypatch, tmp_path):
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"audio")

    def fake_load(path: Path):
        return np.zeros(16_000, dtype=np.float32), 16_000

    def exploding_group(audio, sr, ref_audio=None, ref_sr=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(AudioLoader, "load", staticmethod(fake_load))
    monkeypatch.setattr(
        analyzer_module,
        "detect_speech",
        lambda audio, sr: SpeechDetectionResult(
            is_speech=True,
            confidence=0.9,
            threshold=0.45,
            speech_fraction=1.0,
            content_type="speech",
        ),
    )
    monkeypatch.setitem(analyzer_module.METRIC_GROUPS, "dummy", exploding_group)

    analyzer = AudioAnalyzer(metric_groups={"dummy"}, strict=True)

    try:
        analyzer.analyze_file(audio_path)
    except RuntimeError as exc:
        assert "dummy group failed" in str(exc)
    else:
        raise AssertionError("Expected strict analyzer to raise on group failure")

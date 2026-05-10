import numpy as np

from qualiax.metrics import compute_basic, compute_speech


def _metric_lookup(results):
    return {metric.name: metric.value for metric in results}


def test_golden_benchmark_silence_fixture():
    sr = 16_000
    audio = np.zeros(sr, dtype=np.float32)

    values = _metric_lookup(compute_basic(audio, sr))

    assert values["Peak Amplitude"] == 0.0
    assert values["Silence Ratio"] == 100.0
    assert values["Clipping Detected"] == 0
    assert values["Zero Crossing Rate"] == 0.0


def test_golden_benchmark_sine_fixture():
    sr = 16_000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    basic = _metric_lookup(compute_basic(audio, sr))
    speech = _metric_lookup(compute_speech(audio, sr))

    assert 0.19 <= basic["Peak Amplitude"] <= 0.21
    assert -17.5 <= basic["RMS Level"] <= -16.5
    assert speech["F1-Region Energy (300–1000 Hz)"] >= 0.0
    assert speech["Active Speech Level (ASL)"] is not None


def test_golden_benchmark_clipped_fixture():
    sr = 16_000
    t = np.linspace(0, 1, sr, endpoint=False)
    clipped = np.clip(1.4 * np.sin(2 * np.pi * 440 * t), -1.0, 1.0).astype(np.float32)

    values = _metric_lookup(compute_basic(clipped, sr))

    assert values["Clipping Detected"] == 1
    assert values["Peak Amplitude"] == 1.0

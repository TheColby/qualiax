"""Shared fixtures for the qualiax test suite.

Synthetic test signals are generated in-test with numpy so every expected value
can be derived from first principles (no binary fixtures).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_WAV = REPO_ROOT / "sample.wav"


class Signals:
    """Deterministic synthetic signal generators."""

    @staticmethod
    def time(duration_s: float, sr: int) -> np.ndarray:
        return np.arange(int(round(duration_s * sr))) / sr

    @staticmethod
    def sine(freq: float, duration_s: float, sr: int = 16_000, amplitude: float = 1.0, phase: float = 0.0) -> np.ndarray:
        t = Signals.time(duration_s, sr)
        return amplitude * np.sin(2 * np.pi * freq * t + phase)

    @staticmethod
    def harmonic(f0: float, duration_s: float, sr: int = 16_000, n_harmonics: int = 12, amplitude: float = 0.3) -> np.ndarray:
        """Sawtooth-like harmonic complex with an exact (continuous-phase) F0."""
        t = Signals.time(duration_s, sr)
        y = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, n_harmonics + 1) if k * f0 < sr / 2)
        return amplitude * y / np.max(np.abs(y))

    @staticmethod
    def perturbed_harmonic(
        f0: float,
        duration_s: float,
        sr: int = 16_000,
        *,
        period_jitter: float = 0.0,
        amplitude_shimmer: float = 0.0,
        n_harmonics: int = 10,
        seed: int = 1,
    ) -> np.ndarray:
        """Harmonic voice-like signal with per-cycle period / amplitude perturbation.

        Cycle i lasts T0 * (1 + period_jitter * e_i) seconds and is scaled by
        (1 + amplitude_shimmer * a_i), with e_i, a_i ~ N(0, 1). Cycles may have
        fractional lengths: the waveform is evaluated from a piecewise-linear
        phase, so an unperturbed signal is exactly periodic.
        """
        rng = np.random.default_rng(seed)
        n = int(round(duration_s * sr))
        t = np.arange(n) / sr
        periods, gains, total = [], [], 0.0
        while total < duration_s + 1.0 / f0:
            period = (1.0 / f0) * (1.0 + period_jitter * rng.standard_normal())
            periods.append(period)
            gains.append(1.0 + amplitude_shimmer * rng.standard_normal())
            total += period
        edges = np.concatenate([[0.0], np.cumsum(periods)])
        idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, len(periods) - 1)
        cycle_phase = (t - edges[idx]) / np.asarray(periods)[idx]
        wave = sum(np.sin(2 * np.pi * k * cycle_phase) / k for k in range(1, n_harmonics + 1))
        y = np.asarray(gains)[idx] * wave
        return 0.3 * y / np.max(np.abs(y))

    @staticmethod
    def white_noise(duration_s: float, sr: int = 16_000, std: float = 0.1, seed: int = 0) -> np.ndarray:
        return std * np.random.default_rng(seed).standard_normal(int(round(duration_s * sr)))

    @staticmethod
    def power_law_noise(exponent: float, duration_s: float, sr: int = 16_000, std: float = 0.1, seed: int = 0) -> np.ndarray:
        """Noise whose power spectral density is proportional to 1 / f**exponent.

        exponent = 0 is white, 1 is pink (-3.01 dB/octave), 2 is brown (-6.02 dB/octave).
        """
        n = int(round(duration_s * sr))
        rng = np.random.default_rng(seed)
        spectrum = np.fft.rfft(rng.standard_normal(n))
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        freqs[0] = freqs[1]
        y = np.fft.irfft(spectrum / freqs ** (exponent / 2.0), n)
        return std * y / np.std(y)

    @staticmethod
    def am_tone(carrier: float, rate: float, duration_s: float, sr: int = 16_000, amplitude: float = 0.3) -> np.ndarray:
        """Tone with a raised-cosine envelope at ``rate`` Hz (one 'syllable' per cycle)."""
        t = Signals.time(duration_s, sr)
        envelope = 0.5 * (1.0 - np.cos(2 * np.pi * rate * t))
        return amplitude * envelope * np.sin(2 * np.pi * carrier * t)


@pytest.fixture
def sig() -> type[Signals]:
    return Signals


@pytest.fixture
def values():
    """Map a list of MetricResult objects to {name: value}."""
    def _values(results):
        return {metric.name: metric.value for metric in results}
    return _values


@pytest.fixture
def metrics_by_name():
    def _lookup(results):
        return {metric.name: metric for metric in results}
    return _lookup


@pytest.fixture(scope="session")
def sample_speech():
    """The bundled 2.3 s, 16 kHz real-speech recording."""
    from qualiax.analyzer import AudioLoader

    if not SAMPLE_WAV.exists():
        pytest.skip("sample.wav is not available")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        audio, sr = AudioLoader.load(SAMPLE_WAV)
    return audio, sr


@pytest.fixture
def quiet_warnings():
    """Silence expected metric warnings inside a test body."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


# The developer's real model cache, captured before any test redirects it.
from qualiax import assets as _assets  # noqa: E402

_REAL_MODEL_DIR = _assets.model_dir()


@pytest.fixture(scope="session")
def _empty_model_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("no-models")


@pytest.fixture(autouse=True)
def _no_models_by_default(monkeypatch, _empty_model_dir):
    """Keep results independent of whatever models happen to be downloaded locally."""
    monkeypatch.setenv("QUALIAX_MODEL_DIR", str(_empty_model_dir))


@pytest.fixture
def official_dnsmos(monkeypatch):
    """Use the real, checksum-verified DNSMOS models; skip when they aren't downloaded."""
    pytest.importorskip("onnxruntime")
    if any(_assets.asset_status(name, _REAL_MODEL_DIR) != "ok" for name in ("dnsmos-p835", "dnsmos-p808")):
        pytest.skip("official DNSMOS models not downloaded (run `qualiax models download`)")
    monkeypatch.setenv("QUALIAX_MODEL_DIR", str(_REAL_MODEL_DIR))
    return _REAL_MODEL_DIR

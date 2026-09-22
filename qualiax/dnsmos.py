"""Microsoft DNSMOS P.835 and P.808 scoring with the official ONNX models.

A numpy port of ``DNSMOS/dnsmos_local.py`` from microsoft/DNS-Challenge (which pins
librosa 0.8.1): 9.01 s windows with a 1 s hop over 16 kHz audio, clips shorter
than one window tiled until they fill it, the published non-personalized
calibration polynomials for P.835, and librosa-0.8.1-equivalent log-mel features
for P.808. Scores are means over windows.

The reference script resamples with librosa/resampy, whose filters differ between
versions, so scores for input that isn't 16 kHz are only reproducible up to the
resampler. :func:`resample_to_16k` uses a fixed Kaiser windowed-sinc design with
resampy's output-length rule; on the VoiceBank-DEMAND test set (48 kHz) it agrees
with resampy 0.4.3 ``kaiser_best`` to a median 0.02 MOS (max 0.09).
"""
from __future__ import annotations

from functools import lru_cache
from math import gcd
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

SAMPLING_RATE = 16_000
INPUT_LENGTH = 9.01

# numpy.poly1d coefficients (highest power first) from dnsmos_local.py.
_P835_POLYFIT = {
    "SIG": (-0.08397278, 1.22083953, 0.0052439),
    "BAK": (-0.13166888, 1.60915514, -0.39604546),
    "OVRL": (-0.06766283, 1.11546468, 0.04602535),
}
_MEL_N_FFT = 321
_MEL_HOP = 160
_MEL_BANDS = 120


class DnsmosScorer:
    def __init__(self, p835_model: str | Path, p808_model: str | Path | None = None):
        import onnxruntime as ort

        providers = ["CPUExecutionProvider"]
        self._p835 = ort.InferenceSession(str(p835_model), providers=providers)
        self._p808 = ort.InferenceSession(str(p808_model), providers=providers) if p808_model else None

    def score(self, audio_16k: NDArray) -> dict[str, float] | None:
        """Mean SIG/BAK/OVRL (and P808_MOS when that model is loaded) over 9.01 s windows."""
        audio = np.asarray(audio_16k, dtype=np.float64)
        if audio.size == 0 or not np.any(audio):
            return None
        fs = SAMPLING_RATE
        len_samples = int(INPUT_LENGTH * fs)
        while len(audio) < len_samples:
            audio = np.append(audio, audio)
        num_hops = int(np.floor(len(audio) / fs) - INPUT_LENGTH) + 1

        raw, p808 = [], []
        for idx in range(num_hops):
            segment = audio[int(idx * fs): int((idx + INPUT_LENGTH) * fs)]
            if len(segment) < len_samples:
                continue
            raw.append(self._p835.run(None, {"input_1": segment.astype(np.float32)[np.newaxis, :]})[0][0])
            if self._p808 is not None:
                features = log_mel_features(segment[:-_MEL_HOP]).astype(np.float32)[np.newaxis, :, :]
                p808.append(self._p808.run(None, {"input_1": features})[0][0][0])
        if not raw:
            return None
        raw_scores = np.asarray(raw, dtype=np.float64)
        scores = {
            name: float(np.mean(np.polyval(coefficients, raw_scores[:, column])))
            for column, (name, coefficients) in enumerate(_P835_POLYFIT.items())
        }
        if p808:
            scores["P808_MOS"] = float(np.mean(p808))
        scores["windows"] = len(raw)
        return scores


def resample_to_16k(audio: NDArray, sr: int) -> NDArray:
    audio = np.asarray(audio, dtype=np.float64)
    if sr == SAMPLING_RATE:
        return audio
    from scipy.signal import resample_poly

    g = gcd(int(sr), SAMPLING_RATE)
    up, down = SAMPLING_RATE // g, int(sr) // g
    resampled = resample_poly(audio, up, down, window=_kaiser_sinc(up, down))
    return resampled[: int(len(audio) * SAMPLING_RATE / sr)]


@lru_cache(maxsize=8)
def _kaiser_sinc(up: int, down: int) -> NDArray:
    from scipy.signal import firwin

    rate = max(up, down)
    return firwin(2 * 64 * rate + 1, 0.9475937167399596 / rate, window=("kaiser", 14.769656459379492))


def log_mel_features(audio: NDArray) -> NDArray:
    """``(librosa.power_to_db(melspectrogram(...), ref=np.max) + 40) / 40``, transposed,
    with librosa 0.8.1 defaults: centered frames, reflect padding, periodic Hann window,
    power 2, Slaney mel filters, 80 dB dynamic range."""
    y = np.pad(np.asarray(audio, dtype=np.float64), _MEL_N_FFT // 2, mode="reflect")
    n_frames = 1 + (len(y) - _MEL_N_FFT) // _MEL_HOP
    starts = _MEL_HOP * np.arange(n_frames)
    frames = y[starts[:, None] + np.arange(_MEL_N_FFT)[None, :]] * _periodic_hann(_MEL_N_FFT)
    power = np.abs(np.fft.rfft(frames, n=_MEL_N_FFT, axis=1)) ** 2
    mel = power @ _slaney_mel_basis(SAMPLING_RATE, _MEL_N_FFT, _MEL_BANDS).T
    amin = 1e-10
    log_mel = 10.0 * np.log10(np.maximum(amin, mel)) - 10.0 * np.log10(max(amin, float(mel.max())))
    log_mel = np.maximum(log_mel, log_mel.max() - 80.0)
    return (log_mel + 40.0) / 40.0


def _periodic_hann(n: int) -> NDArray:
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


@lru_cache(maxsize=4)
def _slaney_mel_basis(sr: int, n_fft: int, n_mels: int) -> NDArray:
    # librosa.filters.mel(htk=False, norm="slaney", dtype=float32); like librosa, the
    # FFT bin centres are spaced over [0, sr/2] even for an odd n_fft.
    fft_freqs = np.linspace(0.0, sr / 2.0, 1 + n_fft // 2)
    mel_points = _mel_to_hz(np.linspace(_hz_to_mel(0.0), _hz_to_mel(sr / 2.0), n_mels + 2))
    widths = np.diff(mel_points)
    ramps = mel_points[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / widths[:-1, None]
    upper = ramps[2:] / widths[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    weights *= (2.0 / (mel_points[2:] - mel_points[:-2]))[:, None]
    return weights.astype(np.float32)


_F_SP = 200.0 / 3
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def _hz_to_mel(frequencies):
    f = np.asarray(frequencies, dtype=np.float64)
    mels = f / _F_SP
    log_region = f >= _MIN_LOG_HZ
    return np.where(log_region, _MIN_LOG_MEL + np.log(np.maximum(f, _MIN_LOG_HZ) / _MIN_LOG_HZ) / _LOGSTEP, mels)


def _mel_to_hz(mels):
    m = np.asarray(mels, dtype=np.float64)
    freqs = _F_SP * m
    log_region = m >= _MIN_LOG_MEL
    return np.where(log_region, _MIN_LOG_HZ * np.exp(_LOGSTEP * (m - _MIN_LOG_MEL)), freqs)

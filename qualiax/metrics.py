"""
Metric computation modules.

Each metric group is a function that takes (audio, sr, ref_audio, ref_sr)
and returns a list of MetricResult objects.

All groups are registered in METRIC_GROUPS dict at the bottom.
"""
from __future__ import annotations

import functools
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from .models import MetricResult
from . import gpu as _gpu  # hardware-accelerated kernels (Metal / CUDA / CPU)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _warn_metric_failure(name: str, exc: Exception) -> None:
    warnings.warn(f"Metric '{name}' failed: {exc}")


def _with_trust(
    metric: MetricResult,
    *,
    confidence: str,
    calibration_note: str | None = None,
) -> MetricResult:
    if confidence == "proxy":
        confidence, calibration_note = _measured_proxy_trust(metric.name, calibration_note)
    metric.confidence = confidence
    metric.calibration_note = calibration_note
    return metric


# A proxy keeps the "proxy" label only when the lower 95% bound of its Pearson r
# against the official model is at least this; otherwise it is "heuristic".
PROXY_MIN_CORRELATION = 0.7

# Proxies whose agreement with the official models is measured by
# benchmarks/proxy_benchmark.py; editing any of them makes the packaged
# calibration stale (tests/test_proxy_calibration.py checks this).
_BENCHMARKED_PROXY_FUNCTIONS = (
    "_compute_dnsmos_proxy",
    "_compute_pseudo_mos",
    "_compute_p563_proxy",
    "_compute_learned_mos",
)


@functools.lru_cache(maxsize=1)
def _proxy_calibration() -> dict:
    from importlib.resources import files

    try:
        return json.loads(files("qualiax").joinpath("data/proxy_calibration.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _measured_proxy_trust(name: str, fallback_note: str | None) -> tuple[str, str | None]:
    calibration = _proxy_calibration()
    evidence = calibration.get("metrics", {}).get(name)
    if not evidence:
        return "proxy", fallback_note
    low, high = evidence["pearson_ci95"]
    note = (
        f"Measured against {evidence['reference']} on the {calibration['corpus']} "
        f"({calibration['clips']} clips): Pearson r {evidence['pearson']:.2f} (95% CI {low:.2f}–{high:.2f}), "
        f"mean absolute error {evidence['mae']:.2f}, bias {evidence['bias']:+.2f}. See docs/proxy-benchmark.md."
    )
    return ("proxy" if low >= PROXY_MIN_CORRELATION else "heuristic"), note


def proxy_code_fingerprint() -> str:
    """SHA-256 of the benchmarked proxies' source, recorded with each calibration run."""
    import hashlib
    import inspect

    module = sys.modules[__name__]
    source = "\n".join(inspect.getsource(getattr(module, name)) for name in _BENCHMARKED_PROXY_FUNCTIONS)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _db(x: float) -> float:
    """Convert linear amplitude to dBFS."""
    if x <= 0:
        return -math.inf
    return 20 * math.log10(x)


def _power_db(x: float) -> float:
    if x <= 0:
        return -math.inf
    return 10 * math.log10(x)


def _to_mono(audio: NDArray) -> NDArray:
    if audio.ndim == 2:
        return audio.mean(axis=0)
    return audio


def _rms(audio: NDArray) -> float:
    return float(np.sqrt(np.mean(audio ** 2)))


def _eps(dtype=np.float64):
    return np.finfo(dtype).eps


# ─────────────────────────────────────────────────────────────────────────────
# GROUP: basic
# ─────────────────────────────────────────────────────────────────────────────

def compute_basic(audio: NDArray, sr: int, ref_audio=None, ref_sr=None) -> list[MetricResult]:
    mono = _to_mono(audio)
    duration = len(mono) / sr
    n_channels = 1 if audio.ndim == 1 else audio.shape[0]

    # Peak and clipping are facts about the stored samples, so they are taken
    # over every channel: a downmix hides a clipped or out-of-phase channel.
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    clipped = bool(np.any(np.abs(audio) >= 0.999))
    peak_db = _db(peak)
    rms_val = _rms(mono)
    rms_db = _db(rms_val)
    # Crest factor / dynamic range are undefined (0/0) for digital silence.
    crest_db = _db(peak / rms_val) if rms_val > 0 and peak > 0 else None
    dynamic_range = (peak_db - rms_db) if rms_val > 0 and peak > 0 else None
    dc_offset = float(np.mean(mono))
    silence_ratio = float(np.mean(np.abs(mono) < 0.001))

    results = [
        MetricResult("Duration", duration, "s", "Total duration of the audio file", "basic"),
        MetricResult("Sample Rate", sr, "Hz", "Audio sample rate", "basic"),
        MetricResult("Channels", n_channels, "", "Number of audio channels", "basic"),
        MetricResult("Peak Amplitude", peak, "", "Maximum absolute sample value (0–1 scale)", "basic",
                     higher_is_better=False,
                     reference_range=(0.0, 0.99)),
        MetricResult("Peak Level", peak_db, "dBFS", "Peak level in dBFS (0 dBFS = full scale)", "basic",
                     higher_is_better=False,
                     reference_range=(-3.0, 0.0),
                     warning="Clipping risk" if peak >= 0.999 else (
                         "Very low peak level" if peak_db < -18 else None)),
        MetricResult("RMS Level", rms_db, "dBFS", "Root-mean-square energy level in dBFS", "basic",
                     reference_range=(-20.0, -9.0)),
        MetricResult("Crest Factor", crest_db, "dB",
                     "Peak-to-RMS ratio. High values indicate highly dynamic or sparse signal.", "basic"),
        MetricResult("DC Offset", dc_offset, "",
                     "Mean sample value. Non-zero indicates a DC component.", "basic",
                     higher_is_better=False,
                     warning="Significant DC offset detected" if abs(dc_offset) > 0.01 else None),
        MetricResult("Silence Ratio", silence_ratio * 100, "%",
                     "Percentage of samples below -60 dB (near silence)", "basic",
                     warning="More than 50% silence" if silence_ratio > 0.5 else None),
        MetricResult("Dynamic Range (simple)", dynamic_range, "dB",
                     "Difference between peak and RMS level", "basic"),
        MetricResult("Clipping Detected",
                     int(clipped),
                     "", "1 if any samples appear clipped (>=0.999 full scale)", "basic",
                     higher_is_better=False,
                     warning="Clipping detected!" if clipped else None),
        MetricResult("Zero Crossing Rate", float(np.mean(np.abs(np.diff(np.sign(mono))) > 0)), "crossings/sample",
                     "Rate of sign changes (correlated with pitch/noisiness)", "basic"),
    ]

    if audio.ndim == 2 and audio.shape[0] >= 2:
        results.extend(_multi_channel_metrics(audio, sr))

    return results


def _multi_channel_metrics(audio: NDArray, sr: int) -> list[MetricResult]:
    """Stereo and inter-channel metrics for multi-channel content."""
    results: list[MetricResult] = []
    channel_count = audio.shape[0]
    for idx in range(channel_count):
        rms_db = _db(_rms(audio[idx]))
        peak_db = _db(float(np.max(np.abs(audio[idx]))))
        results.append(MetricResult(
            f"Channel {idx + 1} RMS Level", rms_db, "dBFS",
            f"Per-channel RMS level for channel {idx + 1}", "basic",
        ))
        results.append(MetricResult(
            f"Channel {idx + 1} Peak Level", peak_db, "dBFS",
            f"Per-channel peak level for channel {idx + 1}", "basic",
        ))

    left = audio[0] - float(np.mean(audio[0]))
    right = audio[1] - float(np.mean(audio[1]))
    left_rms = _rms(left)
    right_rms = _rms(right)
    if left_rms > 0 and right_rms > 0:
        ild = 20 * math.log10((left_rms + _eps()) / (right_rms + _eps()))
    else:
        ild = 0.0

    if np.std(left) > 0 and np.std(right) > 0:
        phase_corr = float(np.corrcoef(left, right)[0, 1])
    else:
        phase_corr = 1.0
    mid = (left + right) * 0.5
    side = (left - right) * 0.5
    stereo_width = float(np.clip(_rms(side) / (_rms(mid) + _rms(side) + _eps()), 0.0, 1.0))

    max_lag = max(1, int(0.0015 * sr))
    lags = range(-max_lag, max_lag + 1)
    corr_values = []
    for lag in lags:
        if lag >= 0:
            a = left[lag:]
            b = right[: len(left) - lag]
        else:
            a = left[: len(left) + lag]
            b = right[-lag:]
        corr_values.append(float(np.dot(a, b)))
    lag = int(np.argmax(corr_values) - max_lag)
    itd_ms = lag / sr * 1000.0

    results.extend(
        [
            MetricResult(
                "Stereo Phase Correlation", phase_corr, "",
                "Correlation between left and right channels (-1 = inverted, +1 = mono-compatible).",
                "basic",
                reference_range=(-0.2, 1.0),
                warning="Negative phase correlation may collapse in mono" if phase_corr < 0 else None,
            ),
            MetricResult(
                "Stereo Width", stereo_width, "",
                "Side-to-mid energy ratio mapped to 0–1. Higher values indicate a wider stereo image.",
                "basic",
                reference_range=(0.1, 0.8),
            ),
            MetricResult(
                "Interaural Level Difference (ILD)", ild, "dB",
                "Level difference between the first two channels.",
                "basic",
                higher_is_better=False,
                warning="Large left/right level imbalance" if abs(ild) > 6.0 else None,
            ),
            MetricResult(
                "Interaural Time Difference (ITD)", itd_ms, "ms",
                "Peak cross-correlation lag between the first two channels.",
                "basic",
                higher_is_better=False,
            ),
        ]
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# GROUP: loudness  (ITU-R BS.1770 / EBU R128)
# ─────────────────────────────────────────────────────────────────────────────

def _k_weighting_filter(audio: NDArray, sr: int) -> NDArray:
    """Apply BS.1770 K-weighting. GPU-accelerated (Metal/CUDA/CPU)."""
    return _gpu.k_weighting_filter(audio, sr)


_ABSOLUTE_GATE_LUFS = -70.0
_MOMENTARY_BLOCK_S = 0.4
_SHORT_TERM_BLOCK_S = 3.0
_LOUDNESS_HOP_S = 0.1


def _as_channels(audio: NDArray) -> NDArray:
    """Return audio as (channels, samples)."""
    return audio[None, :] if audio.ndim == 1 else audio


def _k_weighted_channels(audio: NDArray, sr: int) -> NDArray:
    return np.stack([_k_weighting_filter(ch, sr) for ch in _as_channels(audio)])


def _block_energies(filtered: NDArray, sr: int, block_s: float, hop_s: float = _LOUDNESS_HOP_S) -> NDArray:
    """Channel-summed mean-square energy z of each gating block (BS.1770-4 eq. 3-4).

    Channels are summed with unit weights, which is exact for mono, stereo and
    L/R/C content; surround weights (1.41 for Ls/Rs, LFE excluded) are not
    applied because the channel layout is unknown.
    """
    block = int(round(block_s * sr))
    hop = int(round(hop_s * sr))
    n = filtered.shape[-1]
    if block <= 0 or hop <= 0 or n < block:
        return np.zeros(0)
    starts = range(0, n - block + 1, hop)
    return np.array([
        float(np.sum(np.mean(filtered[:, s:s + block] ** 2, axis=1)))
        for s in starts
    ])


def _energy_to_lufs(z) -> float:
    return -0.691 + 10 * math.log10(z)


def _gated_integrated_lufs(z: NDArray) -> Optional[float]:
    """BS.1770-4 two-stage gating over momentary block energies."""
    z = z[z > 0]
    if z.size == 0:
        return None
    loud = np.array([_energy_to_lufs(v) for v in z])
    z_abs = z[loud >= _ABSOLUTE_GATE_LUFS]
    if z_abs.size == 0:
        return None
    relative_gate = _energy_to_lufs(float(np.mean(z_abs))) - 10.0
    z_rel = z_abs[np.array([_energy_to_lufs(v) for v in z_abs]) >= relative_gate]
    if z_rel.size == 0:
        return None
    # Average in the energy domain; -0.691 is applied exactly once.
    return _energy_to_lufs(float(np.mean(z_rel)))


def _gated_loudness_range(z: NDArray) -> Optional[float]:
    """EBU Tech 3342 LRA from short-term (3 s) block energies."""
    z = z[z > 0]
    if z.size < 2:
        return None
    loud = np.array([_energy_to_lufs(v) for v in z])
    abs_gated = loud[loud >= _ABSOLUTE_GATE_LUFS]
    if abs_gated.size < 2:
        return None
    relative_gate = _energy_to_lufs(float(np.mean(z[loud >= _ABSOLUTE_GATE_LUFS]))) - 20.0
    rel_gated = abs_gated[abs_gated >= relative_gate]
    if rel_gated.size < 2:
        return None
    return float(np.percentile(rel_gated, 95) - np.percentile(rel_gated, 10))


def _integrated_loudness_lufs(audio: NDArray, sr: int) -> Optional[float]:
    """Integrated loudness in LUFS per BS.1770-4, or None when undefined."""
    filtered = _k_weighted_channels(audio, sr)
    return _gated_integrated_lufs(_block_energies(filtered, sr, _MOMENTARY_BLOCK_S))


def _loudness_range(audio: NDArray, sr: int) -> Optional[float]:
    """EBU R128 Loudness Range (LRA) in LU, or None when undefined."""
    filtered = _k_weighted_channels(audio, sr)
    return _gated_loudness_range(_block_energies(filtered, sr, _SHORT_TERM_BLOCK_S))


def compute_loudness(audio: NDArray, sr: int, ref_audio=None, ref_sr=None) -> list[MetricResult]:
    channels = _as_channels(audio)
    duration = channels.shape[-1] / sr if sr else 0.0

    lufs = lra = max_short_term = None
    lufs_note = lra_note = short_term_note = None
    try:
        filtered = _k_weighted_channels(audio, sr)
        z_momentary = _block_energies(filtered, sr, _MOMENTARY_BLOCK_S)
        z_short = _block_energies(filtered, sr, _SHORT_TERM_BLOCK_S)
        lufs = _gated_integrated_lufs(z_momentary)
        lra = _gated_loudness_range(z_short)
        positive_short = z_short[z_short > 0]
        if positive_short.size:
            max_short_term = _energy_to_lufs(float(np.max(positive_short)))

        # Explain every unavailable value the same way (value None + warning).
        if lufs is None:
            lufs_note = (
                f"Unavailable: requires at least {_MOMENTARY_BLOCK_S:g} s of audio (got {duration:.2f} s)"
                if z_momentary.size == 0
                else f"Unavailable: signal never exceeds the {_ABSOLUTE_GATE_LUFS:g} LUFS absolute gate"
            )
        if max_short_term is None:
            short_term_note = (
                f"Unavailable: requires at least {_SHORT_TERM_BLOCK_S:g} s of audio (got {duration:.2f} s)"
                if z_short.size == 0
                else "Unavailable: signal is digitally silent"
            )
        if lra is None:
            if z_short.size == 0:
                lra_note = f"Unavailable: requires at least {_SHORT_TERM_BLOCK_S:g} s of audio (got {duration:.2f} s)"
            elif z_short.size < 2:
                lra_note = (
                    f"Unavailable: requires at least two {_SHORT_TERM_BLOCK_S:g} s short-term windows "
                    f"(got {duration:.2f} s)"
                )
            else:
                lra_note = f"Unavailable: too few short-term windows above the {_ABSOLUTE_GATE_LUFS:g} LUFS gate"
    except Exception as e:
        _warn_metric_failure("Loudness", e)
        lufs_note = lra_note = short_term_note = f"Unavailable: loudness computation failed ({e})"

    # True peak (4x band-limited oversampling), maximum over channels (BS.1770-4 Annex 2)
    try:
        true_peak_dbtp = max(_gpu.true_peak(ch, sr, oversample=4) for ch in channels)
    except Exception as e:
        _warn_metric_failure("True Peak", e)
        true_peak_dbtp = _db(float(np.max(np.abs(channels)))) if channels.size else None

    if lufs is not None:
        lufs_warning = ("Too loud for streaming" if lufs > -14 else
                        "Too quiet for streaming" if lufs < -18 else None)
    else:
        lufs_warning = lufs_note

    results = [
        MetricResult("Integrated Loudness (LUFS)", lufs, "LUFS",
                     "ITU-R BS.1770 / EBU R128 integrated loudness (gated)", "loudness",
                     reference_range=(-16.0, -14.0),
                     warning=lufs_warning),
        MetricResult("Loudness Range (LRA)", lra, "LU",
                     "EBU R128 loudness range: dynamic variation of the program", "loudness",
                     reference_range=(5.0, 15.0),
                     warning=lra_note),
        MetricResult("Max Short-Term Loudness", max_short_term, "LUFS",
                     "Maximum 3-second sliding window loudness", "loudness",
                     warning=short_term_note),
        MetricResult("True Peak", true_peak_dbtp, "dBTP",
                     "Inter-sample peak level (4x oversampled). Streaming limit: -1 dBTP", "loudness",
                     higher_is_better=False,
                     warning=("Exceeds -1 dBTP streaming limit"
                              if true_peak_dbtp is not None and true_peak_dbtp > -1 else None)),
    ]

    # ── Streaming loudness target advisor ────────────────────────────────────
    results.extend(_loudness_targets(lufs))

    return results


# Platform loudness targets (integrated LUFS, tolerance in LU)
_LOUDNESS_TARGETS: list[tuple[str, float, float]] = [
    ("Spotify",          -14.0, 1.0),
    ("Apple Music",      -16.0, 1.0),
    ("YouTube",          -14.0, 2.0),
    ("Amazon Music",     -14.0, 1.0),
    ("Tidal",            -14.0, 1.0),
    ("Podcast (AES)",    -16.0, 1.0),
    ("EBU R128",         -23.0, 1.0),
    ("ATSC A/85",        -24.0, 2.0),
]


def _loudness_targets(lufs: Optional[float]) -> list[MetricResult]:
    """Report delta from each platform's integrated loudness target."""
    results = []
    for platform, target, tol in _LOUDNESS_TARGETS:
        if lufs is None:
            results.append(MetricResult(
                f"Loudness Delta vs {platform}", None, "LU",
                f"Target: {target} LUFS ±{tol} LU — requires valid LUFS reading",
                "loudness",
            ))
            continue
        delta = round(lufs - target, 2)
        in_range = abs(delta) <= tol
        results.append(MetricResult(
            f"Loudness Delta vs {platform}", delta, "LU",
            f"Target: {target} LUFS ±{tol} LU. Positive = too loud, negative = too quiet.",
            "loudness",
            higher_is_better=None,
            reference_range=(-tol, tol),
            warning=None if in_range else (
                f"{abs(delta):.1f} LU {'above' if delta > 0 else 'below'} {platform} target"
            ),
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# GROUP: spectral
# ─────────────────────────────────────────────────────────────────────────────

def _stft(mono: NDArray, sr: int, n_fft=2048):
    """GPU-accelerated STFT (Metal/CUDA/CPU)."""
    f, magnitude = _gpu.stft(mono, sr, n_fft=n_fft, hop_length=n_fft // 2)
    t = np.arange(magnitude.shape[1]) * (n_fft // 2) / sr
    return f, t, magnitude


def compute_spectral(audio: NDArray, sr: int, ref_audio=None, ref_sr=None) -> list[MetricResult]:
    mono = _to_mono(audio)
    n_fft = 2048

    # Still need STFT for band energies + skewness + effective bandwidth
    try:
        f, t, mag = _stft(mono, sr, n_fft)
    except Exception as e:
        return [MetricResult("Spectral analysis", None, "", f"Failed: {e}", "spectral")]

    # ── Single-pass spectral descriptors (shared with the gpu module) ────────
    try:
        _sf = _gpu.spectral_features(mono, sr, n_fft=n_fft)
    except Exception as e:
        warnings.warn(f"Metric 'Spectral features' fell back to the STFT magnitude path: {e}")
        _sf = _gpu._spectral_features_from_mag(f, mag)
    centroid = _sf["centroid"]
    bandwidth = _sf["bandwidth"]
    rolloff = _sf["rolloff"]
    flatness_db = _sf["flatness_db"]
    flux = _sf["flux"]
    hfc_gpu = _sf["hfc"]

    power = mag ** 2
    total_power = power.sum(axis=0) + _eps()
    freqs = f

    # Spectral skewness (always computed from STFT; requires centroid)
    norm_power = power / (total_power[None, :] + _eps())
    mean_f = np.sum(freqs[:, None] * norm_power, axis=0)
    std_f = np.sqrt(np.sum((freqs[:, None] - mean_f[None, :]) ** 2 * norm_power, axis=0) + _eps())
    skew = float(np.mean(np.sum((freqs[:, None] - mean_f[None, :]) ** 3 * norm_power / (std_f[None, :] ** 3 + _eps()), axis=0)))

    # Band energy ratios
    def band_energy(flo, fhi):
        idx = np.where((freqs >= flo) & (freqs < fhi))[0]
        if len(idx) == 0:
            return 0.0
        return float(np.mean(power[idx, :].sum(axis=0) / total_power))

    sub_bass = band_energy(20, 80)
    bass = band_energy(80, 300)
    low_mid = band_energy(300, 2000)
    presence = band_energy(2000, 6000)
    air = band_energy(6000, 20000)

    # Estimated fundamental frequency (autocorrelation method)
    f0 = _estimate_f0(mono, sr)

    # Spectral entropy
    norm_p = power / (total_power[None, :] + _eps())
    entropy = float(np.mean(-np.sum(norm_p * np.log2(norm_p + _eps()), axis=0)))

    # Bandwidth (effective Hz occupied by significant energy)
    threshold = np.max(power) * 0.001
    sig_mask = power > threshold
    if sig_mask.any():
        active_freqs = freqs[np.any(sig_mask, axis=1)]
        effective_bw = float(active_freqs[-1] - active_freqs[0]) if len(active_freqs) > 1 else 0.0
    else:
        effective_bw = 0.0

    return [
        MetricResult("Spectral Centroid", centroid, "Hz",
                     "Weighted mean frequency (brightness indicator)", "spectral"),
        MetricResult("Spectral Bandwidth", bandwidth, "Hz",
                     "Weighted standard deviation around centroid", "spectral"),
        MetricResult("Spectral Rolloff (95%)", rolloff, "Hz",
                     "Frequency below which 95% of spectral energy is concentrated", "spectral"),
        MetricResult("Spectral Flatness", flatness_db, "dB",
                     "Ratio of geometric to arithmetic mean of spectrum (0 dB = white noise, low = tonal)", "spectral"),
        MetricResult("Spectral Flux", flux, "",
                     "Mean frame-to-frame spectral change (higher = more dynamic/noisy)", "spectral"),
        MetricResult("Spectral Skewness", skew, "",
                     "Asymmetry of the spectral distribution", "spectral"),
        MetricResult("Spectral Entropy", entropy, "bits",
                     "Randomness/uniformity of spectral energy (higher = more noise-like)", "spectral"),
        MetricResult("Effective Bandwidth", effective_bw, "Hz",
                     "Frequency range containing significant energy (>-30 dB from peak)", "spectral"),
        MetricResult("Estimated Fundamental (F0)", f0, "Hz",
                     "Estimated pitch / fundamental frequency via autocorrelation", "spectral"),
        MetricResult("Band Energy: Sub-Bass (20–80 Hz)", sub_bass * 100, "%",
                     "Proportion of energy in sub-bass band", "spectral"),
        MetricResult("Band Energy: Bass (80–300 Hz)", bass * 100, "%",
                     "Proportion of energy in bass band", "spectral"),
        MetricResult("Band Energy: Low-Mid (300–2000 Hz)", low_mid * 100, "%",
                     "Proportion of energy in low-mid band (speech fundamentals)", "spectral"),
        MetricResult("Band Energy: Presence (2–6 kHz)", presence * 100, "%",
                     "Proportion of energy in presence/intelligibility band", "spectral"),
        MetricResult("Band Energy: Air (6–20 kHz)", air * 100, "%",
                     "Proportion of energy in high-frequency 'air' band", "spectral"),
        MetricResult("High-Frequency Content (HFC)", hfc_gpu, "",
                     "Frequency-weighted energy sum (sensitive to high-frequency transients)", "spectral"),
    ]


def _estimate_f0(mono: NDArray, sr: int, fmin=60, fmax=600) -> Optional[float]:
    """Pitch estimate via CREPE when available, otherwise autocorrelation."""
    try:
        from .metrics_prosody import compute_f0_track_with_backend

        f0_arr, _, _ = compute_f0_track_with_backend(mono, sr, f0_min=fmin, f0_max=fmax, voiced_thresh=0.30)
        pitches = f0_arr[f0_arr > 0]
        if len(pitches) > 0:
            return float(np.median(pitches))
        return None
    except Exception as e:
        _warn_metric_failure("Estimated Fundamental (F0)", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GROUP: temporal
# ─────────────────────────────────────────────────────────────────────────────

def compute_temporal(audio: NDArray, sr: int, ref_audio=None, ref_sr=None) -> list[MetricResult]:
    mono = _to_mono(audio)

    # Energy envelope — GPU-accelerated batch computation
    energies = _gpu.batch_energy(mono, sr, frame_len_ms=25.0, hop_ms=10.0)
    if len(energies) == 0:
        return []
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)
    energy_db = np.where(energies > 0, 10 * np.log10(energies + _eps()), -120.0)

    # Speech activity (VAD-like: energy-based)
    threshold = np.percentile(energy_db, 30)
    active = energy_db > max(threshold, -50)
    speech_ratio = float(np.mean(active))

    # Attack time (frames to reach 90% of peak energy)
    peak_e = np.max(energies)
    attack_frames = np.argmax(energies >= 0.9 * peak_e)
    attack_time = attack_frames * hop / sr

    # Temporal centroid
    t = np.arange(len(energies)) * hop / sr
    if energies.sum() > 0:
        temporal_centroid = float(np.sum(t * energies) / energies.sum())
    else:
        temporal_centroid = 0.0

    # Zero crossing rate statistics
    zcr_frames = np.array([
        float(np.mean(np.abs(np.diff(np.sign(mono[i:i + frame_len]))) > 0))
        for i in range(0, len(mono) - frame_len, hop)
    ])

    # Pause detection
    pauses = []
    in_pause = False
    pause_start = 0
    for i, a in enumerate(active):
        if not a and not in_pause:
            in_pause = True
            pause_start = i
        elif a and in_pause:
            pause_len = (i - pause_start) * hop / sr
            if pause_len > 0.1:  # >100ms = real pause
                pauses.append(pause_len)
            in_pause = False

    return [
        MetricResult("Speech/Activity Ratio", speech_ratio * 100, "%",
                     "Fraction of time with active speech/sound (energy-based VAD)", "temporal",
                     reference_range=(30, 80)),
        MetricResult("Attack Time", attack_time * 1000, "ms",
                     "Time to reach 90% of peak energy from start", "temporal"),
        MetricResult("Temporal Centroid", temporal_centroid, "s",
                     "Energy-weighted center of mass in time", "temporal"),
        MetricResult("Num Pauses (>100ms)", len(pauses), "",
                     "Count of silent pauses longer than 100ms", "temporal"),
        MetricResult("Mean Pause Duration", float(np.mean(pauses)) if pauses else 0.0, "s",
                     "Average duration of detected pauses", "temporal"),
        MetricResult("Max Pause Duration", float(np.max(pauses)) if pauses else 0.0, "s",
                     "Length of the longest detected pause", "temporal"),
        MetricResult("Energy Variance", float(np.var(energy_db)), "dB²",
                     "Variance of frame energy — high = dynamic, low = monotone", "temporal"),
        MetricResult("ZCR Mean", float(np.mean(zcr_frames)), "crossings/sample",
                     "Mean zero-crossing rate across frames", "temporal"),
        MetricResult("ZCR Variance", float(np.var(zcr_frames)), "",
                     "Variance of zero-crossing rate (higher = more varied texture)", "temporal"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# GROUP: noise
# ─────────────────────────────────────────────────────────────────────────────

# Energy-ratio SNRs are capped here. Reached when the quietest frames are
# digital silence (or 100+ dB down), i.e. the measurement is floor-limited.
_SNR_CEILING_DB = 100.0
_SNR_FLOOR_LIMITED_NOTE = (
    "Floor-limited: the noise floor is digital silence (or >= 100 dB down), "
    "so the SNR is reported at the 100 dB ceiling; the true value is at least this."
)


def _floor_limited_snr_db(signal_power: float, noise_power: float) -> float:
    """Power-ratio SNR in dB, capped at the ceiling.

    A digital-silence noise floor is the best case, so it maps to the ceiling;
    only a silent *signal* yields 0 dB.
    """
    if signal_power <= 0:
        return 0.0
    if noise_power <= 0:
        return _SNR_CEILING_DB
    return min(10 * math.log10(signal_power / noise_power), _SNR_CEILING_DB)


def compute_noise(audio: NDArray, sr: int, ref_audio=None, ref_sr=None) -> list[MetricResult]:
    mono = _to_mono(audio)

    # GPU-accelerated batch frame energy
    energies = _gpu.batch_energy(mono, sr, frame_len_ms=25.0, hop_ms=10.0)
    if len(energies) == 0:
        return []

    # Noise floor estimation: median of lowest 10th percentile frames
    noise_threshold = np.percentile(energies, 10)
    noise_frames_e = energies[energies <= noise_threshold]
    noise_floor_e = float(np.mean(noise_frames_e)) if len(noise_frames_e) > 0 else 0.0
    # A digitally silent floor has no finite level: report None, never NaN.
    noise_floor_db = _power_db(noise_floor_e) if noise_floor_e > 0 else None

    # Signal floor: energy of speech-active frames (top 50%)
    sig_threshold = np.percentile(energies, 50)
    sig_frames_e = energies[energies >= sig_threshold]
    signal_e = float(np.mean(sig_frames_e)) if len(sig_frames_e) > 0 else 0.0

    # SNR estimate. A digital-silence floor is the best case, not "unknown":
    # report the SNR ceiling and flag the value as floor-limited.
    snr = None
    snr_note = None
    floor_note = None
    if signal_e > 0:
        snr = _floor_limited_snr_db(signal_e, noise_floor_e)
        if snr >= _SNR_CEILING_DB:
            snr_note = _SNR_FLOOR_LIMITED_NOTE
    if noise_floor_e <= 0:
        floor_note = "No measurable noise floor: the quietest frames are digital silence."

    # Spectral noise estimation (via STFT on low-energy frames)
    spectral_snr = None
    spectral_note = None
    try:
        from scipy.signal import stft as scipy_stft
        f, t_ax, Zxx = scipy_stft(mono, fs=sr, nperseg=512, noverlap=384)
        mag = np.abs(Zxx)
        frame_power = mag.mean(axis=0)
        noise_mask = frame_power <= np.percentile(frame_power, 15)
        if noise_mask.any() and (~noise_mask).any():
            noise_spec = mag[:, noise_mask].mean(axis=1)
            sig_spec = mag[:, ~noise_mask].mean(axis=1)
            sig_p = float(np.mean(sig_spec ** 2))
            noise_p = float(np.mean(noise_spec ** 2))
            if sig_p > 0:
                spectral_snr = _floor_limited_snr_db(sig_p, noise_p)
                if spectral_snr >= _SNR_CEILING_DB:
                    spectral_note = _SNR_FLOOR_LIMITED_NOTE
    except Exception as e:
        warnings.warn(f"Metric 'Spectral SNR' failed: {e}")

    # Harmonic-to-Noise Ratio (HNR) via autocorrelation
    hnr = _compute_hnr(mono, sr)

    # Clipping indicator (consecutive identical max samples)
    peak_val = float(np.max(np.abs(mono)))
    clipped_count = int(np.sum(np.abs(mono) >= 0.999 * peak_val)) if peak_val > 0.5 else 0

    # Dropout detection (sudden near-zero segments in otherwise active audio)
    dropouts = _detect_dropouts(mono, sr)

    return [
        MetricResult("Estimated Noise Floor", noise_floor_db, "dBFS",
                     "Energy of quietest 10% of frames (noise floor estimate)", "noise",
                     higher_is_better=False,
                     calibration_note=floor_note),
        MetricResult("Estimated SNR", snr, "dB",
                     "Signal-to-noise ratio: active speech vs noise floor (energy-based)", "noise",
                     higher_is_better=True,
                     reference_range=(20.0, 40.0),
                     warning="Poor SNR for intelligible speech" if snr is not None and snr < 15 else None,
                     calibration_note=snr_note),
        MetricResult("Spectral SNR", spectral_snr, "dB",
                     "SNR estimated from spectral power of active vs quiet frames", "noise",
                     higher_is_better=True,
                     calibration_note=spectral_note),
        MetricResult("Harmonic-to-Noise Ratio (HNR)", hnr, "dB",
                     "Ratio of harmonic (periodic) energy to noise. High = cleaner voiced speech.", "noise",
                     higher_is_better=True,
                     reference_range=(15.0, 30.0),
                     warning="Rough/breathy voice quality" if hnr is not None and hnr < 10 else None),
        MetricResult("Near-Clipped Samples", clipped_count, "samples",
                     "Number of samples at or near full scale (potential clipping)", "noise",
                     higher_is_better=False,
                     warning="Possible clipping artifact" if clipped_count > 5 else None),
        MetricResult("Detected Dropouts", dropouts, "",
                     "Count of sudden near-silence drops in otherwise active audio (possible glitches)", "noise",
                     higher_is_better=False,
                     warning="Possible glitches/dropouts" if dropouts > 0 else None),
    ]


def _normalized_acf(frame: NDArray) -> NDArray:
    """Autocorrelation of a Hann-windowed frame corrected for the window (Boersma 1993).

    Dividing by the window's own autocorrelation removes the lag taper of the
    biased estimate, so a perfectly periodic frame has r(T0) ~= 1 at any pitch.
    Returns lags 0..len(frame)//2 (beyond that the window correction is unstable).
    """
    n = len(frame)
    win = np.hanning(n)
    xw = (frame - np.mean(frame)) * win
    n_fft = 2 * n
    fx = np.fft.rfft(xw, n=n_fft)
    acf = np.fft.irfft(fx * np.conj(fx), n=n_fft)[: n // 2 + 1]
    fw = np.fft.rfft(win, n=n_fft)
    acf_w = np.fft.irfft(fw * np.conj(fw), n=n_fft)[: n // 2 + 1]
    if acf[0] <= 0 or acf_w[0] <= 0:
        return np.zeros(n // 2 + 1)
    return (acf / acf[0]) / np.maximum(acf_w / acf_w[0], 1e-12)


def _compute_hnr(mono: NDArray, sr: int, fmin=75, fmax=500) -> Optional[float]:
    """Estimate HNR via the window-corrected autocorrelation method (Boersma 1993).

    HNR = 10*log10(r / (1 - r)) with r the normalized autocorrelation peak in the
    pitch range; the median over analysed frames is reported.
    """
    try:
        # Three periods of the lowest pitch per frame (40 ms at 75 Hz).
        frame_len = max(int(0.04 * sr), int(math.ceil(3.0 * sr / fmin)))
        hop = int(0.01 * sr)
        min_lag = max(1, int(sr / fmax))
        max_lag = int(sr / fmin)
        hnrs = []
        for start in range(0, len(mono) - frame_len, hop):
            frame = mono[start:start + frame_len]
            if np.max(np.abs(frame - frame.mean())) < 0.005:
                continue
            acf = _normalized_acf(frame)
            if max_lag + 1 >= len(acf):
                continue
            k = min_lag + int(np.argmax(acf[min_lag:max_lag + 1]))
            r_max = float(acf[k])
            # The period is rarely an integer number of samples: refine the peak
            # height with a parabola through the three samples around it.
            y0, y1, y2 = float(acf[k - 1]), r_max, float(acf[k + 1])
            curvature = y0 - 2.0 * y1 + y2
            if curvature < 0:
                r_max = y1 - (y0 - y2) ** 2 / (8.0 * curvature)
            # Numerical ceiling: a perfectly periodic frame is capped at ~60 dB.
            r_max = min(r_max, 1.0 - 1e-6)
            if r_max > 0.0:
                hnrs.append(10 * math.log10(r_max / (1.0 - r_max)))
        return float(np.median(hnrs)) if hnrs else None
    except Exception as e:
        _warn_metric_failure("Harmonic-to-Noise Ratio (HNR)", e)
        return None


_DROPOUT_BLOCK_S = 0.001        # envelope resolution
_DROPOUT_SILENCE_DB = -60.0     # "near silence": this far below the active level
_DROPOUT_ACTIVE_DB = -30.0      # level right before the gap must be within this of active
_DROPOUT_MIN_GAP_S = 0.003      # ignore sub-3 ms dips
_DROPOUT_PRE_S = 0.003          # abruptness window before the gap


def _detect_dropouts(mono: NDArray, sr: int) -> int:
    """Count abrupt mid-signal drops to digital silence or near-silence.

    A dropout is a run of >= 3 ms whose 1 ms RMS envelope sits at least 60 dB
    below the active level (95th percentile of the envelope) - which includes
    exact digital zeros - AND that is entered abruptly: within the 3 ms before
    the run the level is still within 30 dB of the active level. Runs touching
    the start or end of the file are leading/trailing silence, not dropouts.
    Natural pauses decay gradually to a noise floor and are not counted.
    Each gap counts once, however long it is.
    """
    block = max(1, int(round(_DROPOUT_BLOCK_S * sr)))
    n_blocks = len(mono) // block
    if n_blocks < 3:
        return 0
    x = np.asarray(mono[: n_blocks * block], dtype=np.float64).reshape(n_blocks, block)
    env = np.sqrt(np.mean(x ** 2, axis=1))
    active_level = float(np.percentile(env, 95))
    if active_level <= 0:
        return 0
    silent = env <= active_level * 10 ** (_DROPOUT_SILENCE_DB / 20)
    min_gap = max(1, int(round(_DROPOUT_MIN_GAP_S / _DROPOUT_BLOCK_S)))
    pre = max(1, int(round(_DROPOUT_PRE_S / _DROPOUT_BLOCK_S)))
    active_floor = active_level * 10 ** (_DROPOUT_ACTIVE_DB / 20)

    count = 0
    i = 0
    while i < n_blocks:
        if not silent[i]:
            i += 1
            continue
        j = i
        while j < n_blocks and silent[j]:
            j += 1
        # run of silent blocks is [i, j)
        mid_signal = i > 0 and j < n_blocks
        if mid_signal and (j - i) >= min_gap:
            if float(np.max(env[max(0, i - pre):i])) >= active_floor:
                count += 1
        i = j
    return count


# ─────────────────────────────────────────────────────────────────────────────
# GROUP: speech
# ─────────────────────────────────────────────────────────────────────────────

def compute_speech(audio: NDArray, sr: int, ref_audio=None, ref_sr=None) -> list[MetricResult]:
    mono = _to_mono(audio)

    # MFCCs (via DCT of log mel filterbank)
    mfccs = _gpu.mfcc(mono, sr, n_mfcc=13)

    results = []

    if mfccs is not None:
        # MFCC statistics
        mfcc_mean = mfccs.mean(axis=1)
        mfcc_std = mfccs.std(axis=1)
        results += [
            MetricResult(f"MFCC-{i+1} Mean", float(mfcc_mean[i]), "",
                         f"Mean of MFCC coefficient {i+1}", "speech")
            for i in range(len(mfcc_mean))
        ]
        results += [
            MetricResult(f"MFCC-{i+1} Std", float(mfcc_std[i]), "",
                         f"Std dev of MFCC coefficient {i+1}", "speech")
            for i in range(len(mfcc_std))
        ]

    # Formant-inspired band ratios (approximate F1/F2 regions)
    try:
        from scipy.signal import stft as scipy_stft
        f, _, Zxx = scipy_stft(mono, fs=sr, nperseg=1024, noverlap=768)
        power = np.abs(Zxx) ** 2
        total = power.sum(axis=0) + _eps()

        def band_ratio(flo, fhi):
            idx = (f >= flo) & (f < fhi)
            if not idx.any():
                return 0.0
            return float(np.mean(power[idx, :].sum(axis=0) / total))

        f1_band = band_ratio(300, 1000)
        f2_band = band_ratio(1000, 2500)
        f3_band = band_ratio(2500, 3500)
        results += [
            MetricResult("F1-Region Energy (300–1000 Hz)", f1_band * 100, "%",
                         "Energy in first formant region (vowel quality)", "speech"),
            MetricResult("F2-Region Energy (1–2.5 kHz)", f2_band * 100, "%",
                         "Energy in second formant region", "speech"),
            MetricResult("F3-Region Energy (2.5–3.5 kHz)", f3_band * 100, "%",
                         "Energy in third formant region (voice timbre)", "speech"),
        ]
    except Exception as e:
        warnings.warn(f"Metric 'Speech formant-region energy' failed: {e}")

    # Voiced/unvoiced ratio
    vu_ratio = _voiced_unvoiced_ratio(mono, sr)
    if vu_ratio is not None:
        results.append(MetricResult(
            "Voiced/Unvoiced Ratio", vu_ratio, "",
            "Ratio of voiced to unvoiced frames (correlated with speaking style)", "speech"
        ))

    # Speaking rate estimate (syllable nuclei via energy peaks)
    rate = _estimate_speaking_rate(mono, sr)
    if rate is not None:
        results.append(MetricResult(
            "Estimated Speaking Rate", rate, "syllables/s",
            "Rough syllable rate estimated via energy envelope peaks", "speech"
        ))

    asl = _active_speech_level(mono, sr)
    if asl is not None:
        results.append(MetricResult(
            "Active Speech Level (ASL)", asl, "dBFS",
            "P.56-inspired active speech RMS level over speech-active frames", "speech",
            warning="Low active speech level" if asl < -30 else (
                "High active speech level" if asl > -14 else None
            ),
        ))

    # Pause statistics (already computed in temporal; add speech-specific reading)
    results.append(MetricResult(
        "Speech-Band SNR (300 Hz – 3.4 kHz)", _speech_band_snr(mono, sr), "dB",
        "SNR restricted to the telephone/speech band (classic intelligibility range)", "speech",
        higher_is_better=True,
        reference_range=(15.0, 35.0),
    ))

    return results


def _compute_mfccs(mono: NDArray, sr: int, n_mfcc=13, n_mels=40, n_fft=512) -> Optional[NDArray]:
    try:
        from scipy.fft import dct
        from scipy.signal import stft as scipy_stft

        f, _, Zxx = scipy_stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
        power = np.abs(Zxx) ** 2

        # Mel filterbank
        fmin, fmax = 80.0, min(sr / 2.0, 8000.0)
        mel_min = 2595 * np.log10(1 + fmin / 700)
        mel_max = 2595 * np.log10(1 + fmax / 700)
        mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((n_fft / 2 + 1) * hz_points / (sr / 2)).astype(int)
        bin_points = np.clip(bin_points, 0, power.shape[0] - 1)

        filterbank = np.zeros((n_mels, power.shape[0]))
        for m in range(1, n_mels + 1):
            f_m_minus = bin_points[m - 1]
            f_m = bin_points[m]
            f_m_plus = bin_points[m + 1]
            for k in range(f_m_minus, f_m + 1):
                if f_m > f_m_minus:
                    filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
            for k in range(f_m, f_m_plus + 1):
                if f_m_plus > f_m:
                    filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

        mel_power = filterbank @ power
        log_mel = np.log(mel_power + _eps())
        mfccs = dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]
        return mfccs
    except Exception as e:
        _warn_metric_failure("MFCC (cepstral distance)", e)
        return None


def _active_speech_level(mono: NDArray, sr: int) -> Optional[float]:
    """Approximate active speech level (ASL) in dBFS using speech-active frames."""
    try:
        frame_len = int(0.030 * sr)
        hop = int(0.010 * sr)
        if len(mono) < frame_len:
            return None

        frame_rms = np.array([
            float(np.sqrt(np.mean(mono[start:start + frame_len] ** 2)))
            for start in range(0, len(mono) - frame_len + 1, hop)
        ])
        if len(frame_rms) == 0:
            return None

        frame_db = np.array([_db(v) if v > 0 else -math.inf for v in frame_rms])
        finite = frame_db[np.isfinite(frame_db)]
        if len(finite) == 0:
            return None

        # P.56-inspired activity gate: keep frames above a robust low-percentile floor.
        threshold_db = max(float(np.percentile(finite, 30)), -50.0)
        active = frame_db > threshold_db
        if not np.any(active):
            return None

        # Apply a short hangover so brief speech dips stay part of the active region.
        hangover = max(1, int(round(0.20 / (hop / sr))))
        active = np.convolve(active.astype(np.int32), np.ones(hangover, dtype=np.int32), mode="same") > 0

        active_rms = float(np.sqrt(np.mean(frame_rms[active] ** 2)))
        return _db(active_rms)
    except Exception as e:
        _warn_metric_failure("Active Speech Level (ASL)", e)
        return None


def _voiced_unvoiced_ratio(mono: NDArray, sr: int) -> Optional[float]:
    """Estimate voiced fraction via ZCR + energy combination."""
    try:
        frame_len = int(0.025 * sr)
        hop = int(0.010 * sr)
        voiced, total = 0, 0
        for start in range(0, len(mono) - frame_len, hop):
            frame = mono[start:start + frame_len]
            e = float(np.mean(frame ** 2))
            zcr = float(np.mean(np.abs(np.diff(np.sign(frame))) > 0))
            if e > 0.0001:
                total += 1
                if zcr < 0.15:  # low ZCR = likely voiced
                    voiced += 1
        return voiced / total if total > 0 else None
    except Exception as e:
        _warn_metric_failure("Voiced/Unvoiced Ratio", e)
        return None


def _estimate_speaking_rate(mono: NDArray, sr: int) -> Optional[float]:
    """Syllable nucleus detection via smooth energy envelope peaks."""
    try:
        from scipy.signal import find_peaks, medfilt
        frame_len = int(0.025 * sr)
        hop = int(0.005 * sr)
        energies = np.array([
            float(np.mean(mono[i:i + frame_len] ** 2))
            for i in range(0, len(mono) - frame_len, hop)
        ])
        if len(energies) < 3:
            return None
        smooth = medfilt(energies, kernel_size=min(21, len(energies) | 1))
        # Median filtering flattens every maximum into a plateau, so a strict
        # "greater than both neighbours" test finds no peaks at all; find_peaks
        # reports one peak per plateau.
        threshold = np.percentile(smooth, 60)
        peaks, _ = find_peaks(smooth)
        peaks = [p for p in peaks if smooth[p] > threshold]
        duration = len(mono) / sr
        if duration > 0.5 and len(peaks) > 1:
            return len(peaks) / duration
        return None
    except Exception as e:
        _warn_metric_failure("Estimated Speaking Rate", e)
        return None


def _speech_band_snr(mono: NDArray, sr: int, flo=300, fhi=3400) -> Optional[float]:
    try:
        from scipy.signal import stft as scipy_stft
        f, _, Zxx = scipy_stft(mono, fs=sr, nperseg=512, noverlap=384)
        mag = np.abs(Zxx)
        speech_idx = (f >= flo) & (f <= fhi)
        speech_mag = mag[speech_idx, :]
        non_speech_mag = mag[~speech_idx, :]
        if speech_mag.size == 0 or non_speech_mag.size == 0:
            return None
        sp = float(np.mean(speech_mag ** 2))
        np_ = float(np.mean(non_speech_mag ** 2))
        if np_ <= 0:
            return None
        return float(10 * math.log10(sp / np_))
    except Exception as e:
        _warn_metric_failure("Speech-Band SNR", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GROUP: perceptual  (intrusive: needs reference; non-intrusive otherwise)
# ─────────────────────────────────────────────────────────────────────────────

_DNSMOS_MODEL_NOTE = (
    "Official Microsoft DNSMOS weights (microsoft/DNS-Challenge, CC BY 4.0) verified by SHA-256; "
    "scores match the reference dnsmos_local.py for 16 kHz input."
)


@functools.lru_cache(maxsize=4)
def _dnsmos_scorer(p835_path: str, p808_path: str | None):
    from .dnsmos import DnsmosScorer

    return DnsmosScorer(p835_path, p808_path)


def _compute_dnsmos_model(mono: NDArray, sr: int) -> list[MetricResult] | None:
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return None
    from . import assets
    from .dnsmos import resample_to_16k

    status = assets.asset_status("dnsmos-p835")
    if status == "missing":
        return None
    if status == "checksum_mismatch":
        warnings.warn(
            f"DNSMOS model at {assets.asset_path('dnsmos-p835')} does not match its pinned SHA-256; "
            "using the proxy. Run `qualiax models download --force` to replace it."
        )
        return None
    p808 = assets.verified_asset_path("dnsmos-p808")
    try:
        scores = _dnsmos_scorer(str(assets.asset_path("dnsmos-p835")), str(p808) if p808 else None).score(
            resample_to_16k(mono, sr)
        )
    except Exception as e:
        warnings.warn(f"DNSMOS model inference failed: {e}; using proxy")
        return None
    if scores is None:
        return None

    def model_metric(name, value, description):
        return _with_trust(
            MetricResult(name, round(value, 3), "MOS [1–5]", description, "perceptual",
                         higher_is_better=True, reference_range=(3.5, 5.0)),
            confidence="model",
            calibration_note=_DNSMOS_MODEL_NOTE,
        )

    results = [
        model_metric("DNSMOS P.835 SIG", scores["SIG"], "Speech signal quality (official DNSMOS P.835 model)"),
        model_metric("DNSMOS P.835 BAK", scores["BAK"], "Background noise quality (official DNSMOS P.835 model)"),
        model_metric("DNSMOS P.835 OVRL", scores["OVRL"], "Overall quality (official DNSMOS P.835 model)"),
    ]
    if "P808_MOS" in scores:
        results.append(model_metric("DNSMOS P.808 MOS", scores["P808_MOS"], "Overall MOS (official DNSMOS P.808 model)"))
    return results


def _compute_dnsmos_p835(mono: NDArray, sr: int) -> list[MetricResult]:
    """
    Microsoft DNSMOS P.835 (and P.808) with the official ONNX models, else a proxy.

    The models are used when ``onnxruntime`` is installed and ``qualiax models
    download`` has fetched files that match their pinned SHA-256 checksums.

    Proxy approach otherwise:
    - SIG  (signal quality): spectral SNR + spectral tilt consistency
    - BAK  (background noise): noise floor stationarity + spectral flatness
    - OVRL (overall):          weighted combination (SIG×0.46 + BAK×0.23 + 0.93)
    """
    model_results = _compute_dnsmos_model(mono, sr)
    if model_results is not None:
        return model_results
    return _compute_dnsmos_proxy(mono, sr)


def _compute_dnsmos_proxy(mono: NDArray, sr: int) -> list[MetricResult]:
    try:
        from scipy.signal import stft as scipy_stft

        if sr != 16000:
            m16 = _resample(mono, sr, 16000)
            sr16 = 16000
        else:
            m16, sr16 = mono, sr

        f, _, Zxx = scipy_stft(m16, fs=sr16, nperseg=512, noverlap=384)
        mag   = np.abs(Zxx)
        power = mag ** 2

        # ── SIG proxy: spectral SNR + harmonic structure ─────────────────────
        frame_p    = power.mean(axis=0)
        noise_floor = np.percentile(frame_p, 10)
        sig_p      = np.percentile(frame_p, 90)
        snr_db     = _floor_limited_snr_db(sig_p, noise_floor)

        # Spectral tilt consistency: speech-like tilt ≈ −6 dB/oct
        mean_p = power.mean(axis=1) + _eps()
        valid  = (f > 100) & (f < sr16 / 2.5)
        if valid.sum() > 5:
            slope = float(np.polyfit(np.log10(f[valid] + _eps()),
                                     np.log10(mean_p[valid]), 1)[0])
            tilt_score = float(np.clip(1.0 - abs(slope + 1.2) / 2.5, 0.0, 1.0))
        else:
            tilt_score = 0.5

        sig_score = float(np.clip(1.5 + snr_db / 12.0 * 2.5 + tilt_score * 0.5, 1.0, 5.0))

        # ── BAK proxy: noise stationarity + high spectral flatness indicates noise ──
        # Stationarity: low frame-to-frame variance of spectral shape = stable bg
        spectral_var = float(np.mean(np.std(mag, axis=1)))
        geo  = float(np.exp(np.mean(np.log(mean_p))))
        arith = float(np.mean(mean_p)) + _eps()
        flatness = geo / arith  # high flatness = noise-like

        # Lower flatness → cleaner background (less noise bleed)
        bak_score = float(np.clip(4.5 - flatness * 10 - spectral_var * 5, 1.0, 5.0))

        # ── OVRL proxy: P.835-inspired combination ────────────────────────────
        ovrl_score = float(np.clip(sig_score * 0.46 + bak_score * 0.23 + 0.31 * 3.0,
                                   1.0, 5.0))

        proxy_warn = "Proxy (install `onnxruntime` and run `qualiax models download` for the official DNSMOS model)"
        return [
            _with_trust(
                MetricResult("DNSMOS P.835 SIG (proxy)",  round(sig_score,  2), "MOS [1–5]",
                             "Signal quality proxy (SNR + spectral tilt). " + proxy_warn,
                             "perceptual", higher_is_better=True, reference_range=(3.5, 5.0),
                             warning=proxy_warn),
                confidence="proxy",
                calibration_note="Proxy DNSMOS values are directional estimates, not drop-in replacements for the official model.",
            ),
            _with_trust(
                MetricResult("DNSMOS P.835 BAK (proxy)",  round(bak_score,  2), "MOS [1–5]",
                             "Background quality proxy (noise stationarity). " + proxy_warn,
                             "perceptual", higher_is_better=True, reference_range=(3.5, 5.0),
                             warning=proxy_warn),
                confidence="proxy",
                calibration_note="Proxy DNSMOS values are directional estimates, not drop-in replacements for the official model.",
            ),
            _with_trust(
                MetricResult("DNSMOS P.835 OVRL (proxy)", round(ovrl_score, 2), "MOS [1–5]",
                             "Overall quality proxy. " + proxy_warn,
                             "perceptual", higher_is_better=True, reference_range=(3.5, 5.0),
                             warning=proxy_warn),
                confidence="proxy",
                calibration_note="Proxy DNSMOS values are directional estimates, not drop-in replacements for the official model.",
            ),
        ]
    except Exception as e:
        warnings.warn(f"DNSMOS proxy failed: {e}")
        return [MetricResult("DNSMOS P.835", None, "", f"Failed: {e}", "perceptual")]


def _compute_aecmos(mono: NDArray, sr: int) -> MetricResult:
    """
    Echo-aware quality proxy (AECMOS-inspired).

    Microsoft's AECMOS model needs the far-end reference, microphone, and processed
    signals, so it can't run on a single recording; this is a signal-domain proxy:
    - Estimate echo tail via autocorrelation at 20–300 ms lags
    - Map the strongest normalized tail correlation to a MOS-like [1–5] score
    """
    # ── Echo proxy ───────────────────────────────────────────────────────────
    try:
        # Normalized autocorrelation: echo appears as peaks at 20–300 ms lags
        lag_min = int(0.020 * sr)  # 20 ms
        lag_max = min(int(0.300 * sr), len(mono) // 2)

        norm_mono = mono - mono.mean()
        energy    = float(np.dot(norm_mono, norm_mono)) + _eps()
        if lag_max <= lag_min:
            raise ValueError("Signal too short for AECMOS proxy")

        # Check a few lag windows
        n_fft_acf = 2 * len(norm_mono)
        fft_m = np.fft.rfft(norm_mono, n=n_fft_acf)
        acf   = np.fft.irfft(fft_m * np.conj(fft_m))[:len(norm_mono)]
        acf  /= (acf[0] + _eps())

        echo_strength = float(np.max(np.abs(acf[lag_min:lag_max])))

        # Low echo_strength → no echo → high score
        # echo_strength > 0.3 → noticeable echo
        aecmos_score = float(np.clip(5.0 - echo_strength * 8.0, 1.0, 5.0))

        proxy_warn = "Proxy (Microsoft AECMOS needs far-end and microphone signals, so it cannot run on one recording)"
        return _with_trust(
            MetricResult(
                "AECMOS (proxy)", round(aecmos_score, 2), "MOS [1–5]",
                "Echo-aware quality proxy via autocorrelation tail. " + proxy_warn,
                "perceptual",
                higher_is_better=True,
                reference_range=(3.5, 5.0),
            ),
            confidence="heuristic",
            calibration_note=(
                "Not benchmarked: Microsoft's AECMOS needs far-end, microphone, and processed signals, so no "
                "single-recording reference exists. Speech periodicity also raises the autocorrelation tail: on "
                "echo-free VoiceBank-DEMAND speech it averages 3.5 (10th percentile 2.6), so only much lower "
                "scores suggest echo."
            ),
        )
    except Exception as e:
        warnings.warn(f"AECMOS proxy failed: {e}")
        return MetricResult("AECMOS", None, "", f"Failed: {e}", "perceptual")


def compute_perceptual(audio: NDArray, sr: int, ref_audio=None, ref_sr=None) -> list[MetricResult]:
    results = []

    # ── Non-intrusive ───────────────────────────────────────────────────────

    mono = _to_mono(audio)

    # DNSMOS P.835 (ONNX or proxy)
    results.extend(_compute_dnsmos_p835(mono, sr))

    # AECMOS echo-aware quality
    results.append(_compute_aecmos(mono, sr))

    # Heuristic MOS proxy (kept for backward compatibility / cross-reference)
    results.append(_compute_pseudo_mos(mono, sr))
    results.extend(_compute_learned_mos(mono, sr))
    results.extend(_compute_codec_artifact_metrics(mono, sr))

    # P.563 proxy (mono, phone-band SNR based)
    results.append(_compute_p563_proxy(mono, sr))

    # ── Intrusive (requires reference) ──────────────────────────────────────
    if ref_audio is None:
        results.append(MetricResult(
            "PESQ (ITU-T P.862)", None, "",
            "Perceptual Evaluation of Speech Quality — requires --reference", "perceptual",
            warning="Provide --reference for PESQ/STOI/SI-SDR metrics"
        ))
        results.append(MetricResult(
            "STOI (Short-Time Objective Intelligibility)", None, "",
            "Speech intelligibility score [0–1] — requires --reference", "perceptual",
        ))
        results.append(MetricResult(
            "SI-SDR (Scale-Invariant SDR)", None, "",
            "Scale-invariant signal-to-distortion ratio — requires --reference", "perceptual",
        ))
        results.append(MetricResult(
            "Log-Spectral Distance", None, "",
            "Spectral distortion vs reference — requires --reference", "perceptual",
        ))
        return results

    # Align and resample reference to match
    ref_mono = _to_mono(ref_audio)
    if ref_sr != sr:
        ref_mono = _resample(ref_mono, ref_sr, sr)

    # Align lengths
    n = min(len(mono), len(ref_mono))
    y = mono[:n]
    r = ref_mono[:n]

    # PESQ (via library if available, else proxy)
    results.append(_compute_pesq(y, r, sr))

    # STOI
    results.append(_compute_stoi(y, r, sr))

    # SI-SDR
    si_sdr = _compute_si_sdr(y, r)
    results.append(MetricResult(
        "SI-SDR", si_sdr, "dB",
        "Scale-invariant signal-to-distortion ratio (higher = less distortion)", "perceptual",
        higher_is_better=True,
        reference_range=(15.0, 30.0),
    ))

    # SDR
    sdr = _compute_sdr(y, r)
    results.append(MetricResult(
        "SDR (Signal-to-Distortion Ratio)", sdr, "dB",
        "Classical BSS eval signal-to-distortion ratio", "perceptual",
        higher_is_better=True,
    ))

    # Log-spectral distance
    lsd = _log_spectral_distance(y, r, sr)
    results.append(MetricResult(
        "Log-Spectral Distance", lsd, "dB",
        "Mean log-spectral distance between test and reference (lower = more similar)", "perceptual",
        higher_is_better=False,
        reference_range=(0.0, 4.0),
    ))

    # SSIM-inspired spectral correlation
    sc = _spectral_correlation(y, r, sr)
    results.append(MetricResult(
        "Spectral Correlation", sc, "",
        "Mean cosine similarity between test and reference spectral frames [0–1]", "perceptual",
        higher_is_better=True,
        reference_range=(0.9, 1.0),
    ))

    # Cepstral Distance
    cd = _cepstral_distance(y, r, sr)
    results.append(MetricResult(
        "Cepstral Distance", cd, "dB",
        "Distance in cepstral domain (lower = more similar timbre to reference)", "perceptual",
        higher_is_better=False,
    ))

    return results


def _resample(audio: NDArray, orig_sr: int, target_sr: int) -> NDArray:
    """GPU-accelerated resampling (Metal/CUDA/CPU)."""
    return _gpu.resample(audio, orig_sr, target_sr)


def _compute_pseudo_mos(mono: NDArray, sr: int) -> MetricResult:
    """Approximate non-intrusive MOS (P.835-inspired proxy)."""
    try:
        from scipy.signal import stft as scipy_stft
        f, _, Zxx = scipy_stft(mono, fs=sr, nperseg=512, noverlap=384)
        mag = np.abs(Zxx)
        power = mag ** 2

        # Estimate SNR
        frame_p = power.mean(axis=0)
        noise_p = np.percentile(frame_p, 15)
        sig_p = np.percentile(frame_p, 85)
        snr = _floor_limited_snr_db(sig_p, noise_p)

        # Spectral flatness (lower = more speech-like)
        geo = np.exp(np.mean(np.log(power.mean(axis=1) + _eps())))
        arith = np.mean(power.mean(axis=1)) + _eps()
        flatness_db = _power_db(geo / arith)

        # Map to MOS-like score [1–5]
        # Heuristic: snr 0→5 = MOS ~1.5→4.5; penalize flat spectrum
        snr_score = np.clip(1.0 + snr / 10.0, 1.0, 4.5)
        flat_penalty = max(0.0, (flatness_db + 10) / 20.0)  # penalty if very flat
        mos = float(np.clip(snr_score - flat_penalty * 0.5, 1.0, 5.0))

        return _with_trust(
            MetricResult(
                "Estimated MOS (non-intrusive proxy)", round(mos, 2), "MOS [1–5]",
                "Heuristic MOS estimate (SNR + spectral shape). For model-based MOS run `qualiax models download`.",
                "perceptual",
                higher_is_better=True,
                reference_range=(3.5, 5.0),
                warning="Proxy only — the official DNSMOS models give model-based MOS",
            ),
            confidence="proxy",
            calibration_note="Estimated MOS is a heuristic blend and should be used for ranking, not certification.",
        )
    except Exception as e:
        return MetricResult("Estimated MOS (proxy)", None, "", f"Failed: {e}", "perceptual")


def _compute_learned_mos(mono: NDArray, sr: int) -> list[MetricResult]:
    """Learned-MOS stand-ins blended from the pseudo-MOS and P.563 proxies.

    No official UTMOS or SHEET ONNX weights exist to pin, so these stay proxies.
    """
    mos_metric = _compute_pseudo_mos(mono, sr)
    p563_metric = _compute_p563_proxy(mono, sr)
    mos_value = float(mos_metric.value) if mos_metric.value is not None else 3.0
    p563_value = float(p563_metric.value) if p563_metric.value is not None else 3.0
    utmos_proxy = float(np.clip(0.75 * mos_value + 0.25 * (p563_value / 4.5 * 5.0), 1.0, 5.0))
    sheet_proxy = float(np.clip(0.60 * mos_value + 0.40 * (p563_value / 4.5 * 5.0), 1.0, 5.0))
    proxy_warn = "Proxy blended from the pseudo-MOS and P.563 proxies; not the UTMOS or SHEET models"
    return [
        _with_trust(
            MetricResult(
                "UTMOS (proxy)", round(utmos_proxy, 2), "MOS [1–5]",
                "Learned-MOS proxy blended from existing perceptual quality cues. " + proxy_warn,
                "perceptual",
                higher_is_better=True,
                reference_range=(3.5, 5.0),
                warning=proxy_warn,
            ),
            confidence="proxy",
            calibration_note="Proxy learned-MOS values approximate the trend of missing model-backed scores.",
        ),
        _with_trust(
            MetricResult(
                "SHEET MOS (proxy)", round(sheet_proxy, 2), "MOS [1–5]",
                "Speech naturalness proxy blended from existing perceptual quality cues. " + proxy_warn,
                "perceptual",
                higher_is_better=True,
                reference_range=(3.5, 5.0),
                warning=proxy_warn,
            ),
            confidence="proxy",
            calibration_note="Proxy learned-MOS values approximate the trend of missing model-backed scores.",
        ),
    ]


def _compute_codec_artifact_metrics(mono: NDArray, sr: int) -> list[MetricResult]:
    """Heuristic codec-artifact metrics for compressed or bandwidth-limited audio."""
    try:
        from scipy.signal import stft as scipy_stft

        f, _, Zxx = scipy_stft(mono, fs=sr, nperseg=1024, noverlap=768)
        mag = np.abs(Zxx) + _eps()
        power = mag ** 2
        mean_power = power.mean(axis=1)
        mean_db = 10.0 * np.log10(mean_power + _eps())
        peak_db = float(np.max(mean_db))

        active_bins = np.where(mean_db >= peak_db - 45.0)[0]
        cutoff_hz = float(f[active_bins[-1]]) if len(active_bins) else 0.0

        smooth_kernel = np.ones(9, dtype=np.float64) / 9.0
        smooth_db = np.convolve(mean_db, smooth_kernel, mode="same")
        band_mask = (f >= 1000) & (f <= min(sr / 2.0, 8000.0))
        hole_ratio = float(np.mean((smooth_db[band_mask] - mean_db[band_mask]) > 12.0) * 100.0) if np.any(band_mask) else 0.0

        frame_energy = power.sum(axis=0)
        if len(frame_energy) >= 6:
            diff = np.diff(frame_energy)
            onset_idx = np.where(diff > np.percentile(diff, 90))[0] + 1
            pre_echo_samples = []
            for idx in onset_idx:
                pre = float(np.mean(frame_energy[max(0, idx - 2):idx])) if idx > 0 else 0.0
                post = float(np.mean(frame_energy[idx:min(len(frame_energy), idx + 2)]))
                if post > 0:
                    pre_echo_samples.append(pre / (post + _eps()))
            pre_echo_risk = float(np.clip(np.mean(pre_echo_samples), 0.0, 1.0)) if pre_echo_samples else 0.0
        else:
            pre_echo_risk = 0.0

        expected_cutoff = min(sr / 2.0, 16000.0)
        cutoff_penalty = float(np.clip((expected_cutoff - cutoff_hz) / max(expected_cutoff, 1.0), 0.0, 1.0))
        artifact_risk = float(
            np.clip(100.0 * (0.45 * cutoff_penalty + 0.35 * (hole_ratio / 100.0) + 0.20 * pre_echo_risk), 0.0, 100.0)
        )

        return [
            MetricResult(
                "Estimated Codec Bandwidth Cutoff", round(cutoff_hz, 1), "Hz",
                "Estimated upper bandwidth before codec-style low-pass attenuation dominates.", "perceptual",
                higher_is_better=True,
            ),
            MetricResult(
                "Spectral Hole Ratio", round(hole_ratio, 2), "%",
                "Percentage of mid/high-band bins with deep notches relative to the smoothed spectrum.", "perceptual",
                higher_is_better=False,
            ),
            MetricResult(
                "Pre-echo Risk", round(pre_echo_risk, 3), "",
                "Transient smear heuristic. Higher values suggest codec pre-echo or ringing.", "perceptual",
                higher_is_better=False,
            ),
            MetricResult(
                "Codec Artifact Risk", round(artifact_risk, 2), "%",
                "Combined heuristic risk from bandwidth loss, spectral holes, and pre-echo cues.", "perceptual",
                higher_is_better=False,
                warning="Elevated codec artifact risk" if artifact_risk >= 40.0 else None,
            ),
        ]
    except Exception as e:
        warnings.warn(f"Codec artifact metrics failed: {e}")
        return [MetricResult("Codec Artifact Risk", None, "", f"Failed: {e}", "perceptual")]


def _compute_p563_proxy(mono: NDArray, sr: int) -> MetricResult:
    """Improved P.563-inspired non-intrusive narrowband quality proxy."""
    try:
        from scipy.signal import stft as scipy_stft

        # Resample to 8 kHz if needed (P.563 is narrowband)
        if sr != 8000:
            mono = _resample(mono, sr, 8000)
            sr = 8000

        f, _, Zxx = scipy_stft(mono, fs=sr, nperseg=256, noverlap=192)
        mag = np.abs(Zxx)
        power = mag ** 2

        # Activity detection
        frame_e = power.sum(axis=0)
        active = frame_e > np.percentile(frame_e, 20)
        if active.sum() == 0:
            return MetricResult("P.563 Proxy Score", None, "", "No active frames", "perceptual")

        # Frequency balance. The speech band starts at 80 Hz because a male F0 and its
        # first harmonic sit below 300 Hz; only energy under the lowest F0 is rumble.
        speech_idx = (f >= 80) & (f <= 3400)
        rumble_idx = f < 80
        hiss_idx = f > 3400
        speech_energy = float(power[speech_idx, :].sum()) if speech_idx.any() else 0.0
        rumble_energy = float(power[rumble_idx, :].sum()) if rumble_idx.any() else 0.0
        hiss_energy = float(power[hiss_idx, :].sum()) if hiss_idx.any() else 0.0
        total_energy = float(power.sum()) + _eps()
        speech_ratio = speech_energy / total_energy
        rumble_ratio = rumble_energy / total_energy
        hiss_ratio = hiss_energy / total_energy

        # Spectral flatness over active frames: noisier / more artifacted signals are flatter.
        active_power = power[:, active] + _eps()
        frame_flatness = np.exp(np.mean(np.log(active_power), axis=0)) / np.mean(active_power, axis=0)
        flatness = float(np.mean(frame_flatness))

        # Simple temporal artifact cues.
        discontinuities = int(np.sum(np.abs(np.diff(mono)) > 0.5))
        duration_s = max(len(mono) / sr, _eps())
        discontinuity_rate = discontinuities / duration_s
        clipping_ratio = float(np.mean(np.abs(mono) >= 0.98))
        silence_ratio = 1.0 - float(np.mean(active))

        # Map each cue to normalized quality / penalty terms.
        speech_score = float(np.clip((speech_ratio - 0.35) / 0.45, 0.0, 1.0))
        activity_score = float(np.clip(1.0 - max(0.0, silence_ratio - 0.15) / 0.55, 0.0, 1.0))
        flatness_penalty = float(np.clip(max(0.0, flatness - 0.22) / 0.38, 0.0, 1.0))
        rumble_penalty = float(np.clip(max(0.0, rumble_ratio - 0.08) / 0.22, 0.0, 1.0))
        hiss_penalty = float(np.clip(max(0.0, hiss_ratio - 0.10) / 0.25, 0.0, 1.0))
        discontinuity_penalty = float(np.clip(discontinuity_rate / 12.0, 0.0, 1.0))
        clipping_penalty = float(np.clip(clipping_ratio * 30.0, 0.0, 1.0))

        score = (
            1.0
            + 2.4 * speech_score
            + 0.8 * activity_score
            - 0.9 * flatness_penalty
            - 0.6 * rumble_penalty
            - 0.8 * hiss_penalty
            - 0.8 * discontinuity_penalty
            - 1.1 * clipping_penalty
        )
        score = float(np.clip(score, 1.0, 4.5))

        return _with_trust(
            MetricResult(
                "P.563 Proxy (NB Quality Estimate)", round(score, 2), "MOS [1–4.5]",
                "Improved narrowband non-intrusive quality proxy using spectral balance, "
                "activity continuity, artifact rate, and clipping cues", "perceptual",
                higher_is_better=True,
                reference_range=(3.0, 4.5),
                warning="Proxy only — not an ITU-T P.563 implementation",
            ),
            confidence="proxy",
            calibration_note="P.563 proxy is tuned for narrowband speech and should not be interpreted as a true ITU score.",
        )
    except Exception as e:
        return MetricResult("P.563 Proxy", None, "", f"Failed: {e}", "perceptual")


def _compute_pesq(y: NDArray, r: NDArray, sr: int) -> MetricResult:
    """Compute PESQ via `pesq` library if available, else proxy."""
    try:
        from pesq import pesq as pesq_fn, PesqError
        mode = "wb" if sr >= 16000 else "nb"
        target_sr = 16000 if mode == "wb" else 8000
        if sr != target_sr:
            y = _resample(y, sr, target_sr)
            r = _resample(r, sr, target_sr)
            sr = target_sr
        score = float(pesq_fn(sr, r, y, mode))
        return _with_trust(
            MetricResult(
                "PESQ (ITU-T P.862)", round(score, 3), "MOS-LQO",
                f"Perceptual Evaluation of Speech Quality [{mode.upper()} mode] (1.0–4.5)", "perceptual",
                higher_is_better=True,
                reference_range=(3.0, 4.5),
            ),
            confidence="measured",
        )
    except ImportError:
        # Fallback proxy
        lsd = _log_spectral_distance(y, r, sr)
        if lsd is not None:
            mos_proxy = float(np.clip(4.5 - lsd / 5.0, 1.0, 4.5))
            return _with_trust(
                MetricResult(
                    "PESQ proxy (install `pesq` for true score)", round(mos_proxy, 3), "MOS-LQO (approx)",
                    "Approximate PESQ via log-spectral distance mapping", "perceptual",
                    higher_is_better=True,
                    warning="Install `pip install pesq` for accurate PESQ score",
                ),
                confidence="proxy",
                calibration_note="PESQ proxy uses log-spectral distance and is not a standards-compliant replacement.",
            )
        return MetricResult("PESQ", None, "", "Install `pip install pesq` for PESQ", "perceptual")
    except Exception as e:
        return MetricResult("PESQ", None, "", f"PESQ failed: {e}", "perceptual")


def _compute_stoi(y: NDArray, r: NDArray, sr: int) -> MetricResult:
    """Compute STOI via `pystoi` library if available, else proxy."""
    try:
        from pystoi import stoi as stoi_fn
        score = float(stoi_fn(r, y, sr, extended=False))
        return _with_trust(
            MetricResult(
                "STOI (Short-Time Objective Intelligibility)", round(score, 4), "[0–1]",
                "Speech intelligibility score (1 = perfectly intelligible)", "perceptual",
                higher_is_better=True,
                reference_range=(0.7, 1.0),
                warning="Low intelligibility" if score < 0.6 else None,
            ),
            confidence="measured",
        )
    except ImportError:
        # Spectral correlation proxy for intelligibility
        sc = _spectral_correlation(y, r, sr)
        if sc is not None:
            return _with_trust(
                MetricResult(
                    "STOI proxy (install `pystoi` for true score)", round(sc, 4), "[0–1] (approx)",
                    "Approximate intelligibility via spectral correlation", "perceptual",
                    higher_is_better=True,
                    warning="Install `pip install pystoi` for accurate STOI",
                ),
                confidence="proxy",
                calibration_note="STOI proxy uses spectral correlation and is only suitable for rough relative comparison.",
            )
        return MetricResult("STOI", None, "", "Install `pip install pystoi` for STOI", "perceptual")
    except Exception as e:
        return MetricResult("STOI", None, "", f"STOI failed: {e}", "perceptual")


def _compute_si_sdr(y: NDArray, r: NDArray) -> Optional[float]:
    """Scale-invariant SDR."""
    try:
        r = r - r.mean()
        y = y - y.mean()
        alpha = float(np.dot(y, r) / (np.dot(r, r) + _eps()))
        target = alpha * r
        noise = y - target
        si_sdr = 10 * math.log10(
            (np.dot(target, target) + _eps()) / (np.dot(noise, noise) + _eps())
        )
        return float(si_sdr)
    except Exception as e:
        _warn_metric_failure("SI-SDR", e)
        return None


def _compute_sdr(y: NDArray, r: NDArray) -> Optional[float]:
    try:
        noise = y - r
        sdr = 10 * math.log10(
            (np.dot(r, r) + _eps()) / (np.dot(noise, noise) + _eps())
        )
        return float(sdr)
    except Exception as e:
        _warn_metric_failure("SDR (Signal-to-Distortion Ratio)", e)
        return None


def _log_spectral_distance(y: NDArray, r: NDArray, sr: int) -> Optional[float]:
    try:
        from scipy.signal import stft as scipy_stft
        n_fft = 512
        _, _, Zy = scipy_stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
        _, _, Zr = scipy_stft(r, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
        n = min(Zy.shape[1], Zr.shape[1])
        py = np.abs(Zy[:, :n]) ** 2 + _eps()
        pr = np.abs(Zr[:, :n]) ** 2 + _eps()
        lsd = float(np.mean(np.sqrt(np.mean((10 * np.log10(py / pr)) ** 2, axis=0))))
        return lsd
    except Exception as e:
        _warn_metric_failure("Log-Spectral Distance", e)
        return None


def _spectral_correlation(y: NDArray, r: NDArray, sr: int) -> Optional[float]:
    try:
        from scipy.signal import stft as scipy_stft
        n_fft = 512
        _, _, Zy = scipy_stft(y, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
        _, _, Zr = scipy_stft(r, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
        n = min(Zy.shape[1], Zr.shape[1])
        my = np.abs(Zy[:, :n])
        mr = np.abs(Zr[:, :n])
        norms_y = np.linalg.norm(my, axis=0) + _eps()
        norms_r = np.linalg.norm(mr, axis=0) + _eps()
        corr = float(np.mean(np.sum(my * mr, axis=0) / (norms_y * norms_r)))
        return corr
    except Exception as e:
        _warn_metric_failure("Spectral Correlation", e)
        return None


def _cepstral_distance(y: NDArray, r: NDArray, sr: int, n_ceps=13) -> Optional[float]:
    try:
        my = _compute_mfccs(y, sr, n_mfcc=n_ceps)
        mr = _compute_mfccs(r, sr, n_mfcc=n_ceps)
        if my is None or mr is None:
            return None
        n = min(my.shape[1], mr.shape[1])
        diff = my[:, :n] - mr[:, :n]
        cd = float(np.mean(np.sqrt(np.sum(diff ** 2, axis=0))))
        return cd
    except Exception as e:
        _warn_metric_failure("Cepstral Distance", e)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

from .metrics_prosody       import compute_prosody
from .metrics_psychoacoustic import compute_psychoacoustic
from .metrics_speaker       import compute_speaker

MetricGroupFn = Callable[[NDArray, int, Optional[NDArray], Optional[int]], list[MetricResult]]


METRIC_GROUPS: dict[str, MetricGroupFn] = {
    "basic":          compute_basic,
    "loudness":       compute_loudness,
    "spectral":       compute_spectral,
    "temporal":       compute_temporal,
    "noise":          compute_noise,
    "speech":         compute_speech,
    "perceptual":     compute_perceptual,
    "prosody":        compute_prosody,
    "psychoacoustic": compute_psychoacoustic,
    "speaker":        compute_speaker,
}

_BUILTIN_METRIC_GROUPS = set(METRIC_GROUPS)


def available_metric_groups() -> list[str]:
    """Return the currently registered metric group names."""
    return list(METRIC_GROUPS.keys())


def register_metric_group(name: str, fn: MetricGroupFn, *, overwrite: bool = False) -> None:
    """Register a custom metric group for the analyzer and public API."""
    normalized = name.strip().lower()
    if not normalized:
        raise ValueError("Metric group name must be non-empty.")
    if normalized == "all":
        raise ValueError("'all' is reserved and cannot be registered as a metric group.")
    if normalized in METRIC_GROUPS and not overwrite:
        raise ValueError(f"Metric group '{normalized}' is already registered.")
    METRIC_GROUPS[normalized] = fn


def unregister_metric_group(name: str, *, allow_builtin: bool = False) -> None:
    """Unregister a custom metric group."""
    normalized = name.strip().lower()
    if normalized in _BUILTIN_METRIC_GROUPS and not allow_builtin:
        raise ValueError(f"Metric group '{normalized}' is built in and cannot be removed.")
    if normalized not in METRIC_GROUPS:
        raise KeyError(normalized)
    del METRIC_GROUPS[normalized]

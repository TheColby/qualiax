"""
Prosodic metrics: F0 trajectory, jitter, shimmer, tremor, speaking rate.

All metrics use only numpy/scipy — no librosa required.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks

from .models import MetricResult

# ─── F0 tracking ─────────────────────────────────────────────────────────────

def _frame_signal(audio: NDArray, frame_len: int, hop_len: int) -> NDArray:
    """Slice audio into overlapping frames via stride tricks."""
    n_frames = max(0, (len(audio) - frame_len) // hop_len)
    if n_frames == 0:
        return np.empty((0, frame_len))
    idx = (np.arange(frame_len)[None, :] +
           hop_len * np.arange(n_frames)[:, None])
    return audio[idx]


def compute_f0_track(
    audio: NDArray,
    sr: int,
    f0_min: float = 60.0,
    f0_max: float = 500.0,
    voiced_thresh: float = 0.40,
) -> tuple[NDArray, NDArray]:
    """
    Per-frame F0 and voicing strength via normalized autocorrelation.

    Returns
    -------
    f0_arr : ndarray, shape (n_frames,)
        F0 in Hz for each frame; 0.0 for unvoiced frames.
    strength_arr : ndarray, shape (n_frames,)
        Normalized autocorrelation peak height (0–1).
    """
    frame_len = int(0.025 * sr)
    hop_len   = int(0.010 * sr)
    tau_min   = max(1, int(sr / f0_max))
    tau_max   = int(sr / f0_min)

    if len(audio) < frame_len:
        return np.array([]), np.array([])

    frames = _frame_signal(audio, frame_len, hop_len)
    win    = np.hanning(frame_len)

    f0_out  = np.zeros(len(frames))
    str_out = np.zeros(len(frames))

    for i, frame in enumerate(frames):
        windowed = frame * win
        # Normalized autocorrelation via FFT
        n_pad = 2 * frame_len
        fft   = np.fft.rfft(windowed, n=n_pad)
        acf   = np.fft.irfft(fft * np.conj(fft))[:frame_len]
        if acf[0] < 1e-10:
            continue
        acf /= acf[0]

        if tau_max >= len(acf):
            continue

        region = acf[tau_min : tau_max + 1]
        if len(region) == 0:
            continue

        peaks, props = find_peaks(region, height=voiced_thresh)
        if len(peaks) == 0:
            continue

        best      = peaks[np.argmax(props["peak_heights"])]
        strength  = float(region[best])
        str_out[i] = strength
        f0_out[i]  = sr / (tau_min + best)

    return f0_out, str_out


# ─── Jitter & Shimmer ─────────────────────────────────────────────────────────

def _jitter_local(f0_arr: NDArray) -> Optional[float]:
    """Local jitter (relative period perturbation) from voiced F0 sequence."""
    voiced = f0_arr[f0_arr > 0]
    if len(voiced) < 4:
        return None
    periods = 1.0 / voiced
    diffs   = np.abs(np.diff(periods))
    return float(np.mean(diffs) / (np.mean(periods) + 1e-14))


def _shimmer_local(audio: NDArray, sr: int, f0_arr: NDArray) -> Optional[float]:
    """Local shimmer (relative amplitude perturbation) on voiced frames."""
    frame_len = int(0.025 * sr)
    hop_len   = int(0.010 * sr)
    voiced_idx = np.where(f0_arr > 0)[0]
    if len(voiced_idx) < 4:
        return None

    rms_vals = []
    for i in voiced_idx:
        start = i * hop_len
        end   = start + frame_len
        if end > len(audio):
            break
        seg = audio[start:end]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        if rms > 1e-6:
            rms_vals.append(rms)

    if len(rms_vals) < 4:
        return None
    arr   = np.array(rms_vals)
    diffs = np.abs(np.diff(arr))
    return float(np.mean(diffs) / (np.mean(arr) + 1e-14))


# ─── Tremor ───────────────────────────────────────────────────────────────────

def _tremor(voiced_f0: NDArray, hop_s: float) -> tuple[Optional[float], Optional[float]]:
    """
    Detect low-frequency F0 modulation (tremor / vibrato) via FFT of F0 track.

    Returns (dominant_rate_hz, power).  Returns (None, None) if insufficient data.
    """
    if len(voiced_f0) < 20:
        return None, None

    centered  = voiced_f0 - np.mean(voiced_f0)
    spectrum  = np.abs(np.fft.rfft(centered * np.hanning(len(centered)))) ** 2
    freqs     = np.fft.rfftfreq(len(centered), d=hop_s)

    mask = (freqs >= 2.0) & (freqs <= 15.0)
    if not np.any(mask):
        return None, None

    best = np.argmax(spectrum[mask])
    return float(freqs[mask][best]), float(spectrum[mask][best])


# ─── Speaking rate ────────────────────────────────────────────────────────────

def _speaking_rate(audio: NDArray, sr: int) -> Optional[float]:
    """
    Estimate syllable rate (syllables/second) via energy envelope peaks.

    Based on Mermelstein (1975): syllable nuclei correspond to local energy maxima
    separated by at least ~80 ms.
    """
    if len(audio) < sr * 0.3:
        return None

    frame_len = int(0.010 * sr)
    hop_len   = int(0.005 * sr)
    n_frames  = max(0, (len(audio) - frame_len) // hop_len)
    if n_frames < 10:
        return None

    energy = np.array([
        np.sqrt(np.mean(audio[i * hop_len : i * hop_len + frame_len] ** 2))
        for i in range(n_frames)
    ])

    # Smooth with median filter
    from scipy.signal import medfilt
    smoothed    = medfilt(energy, kernel_size=min(21, len(energy) | 1))
    min_dist    = max(1, int(0.08 / (hop_len / sr)))   # 80 ms between syllables
    threshold   = np.percentile(smoothed, 55)

    peaks, _ = find_peaks(smoothed, height=threshold, distance=min_dist)
    duration = n_frames * hop_len / sr
    if duration < 0.3 or len(peaks) < 2:
        return None

    return len(peaks) / duration


# ─── Main entry point ─────────────────────────────────────────────────────────

def compute_prosody(
    audio: NDArray, sr: int, ref_audio=None, ref_sr=None
) -> list[MetricResult]:
    """Full prosodic analysis: F0 statistics, jitter, shimmer, tremor, rate."""
    try:
        from ._mono import to_mono
    except ImportError:
        def to_mono(a):
            return a.mean(axis=0) if a.ndim == 2 else a

    mono   = audio.mean(axis=0) if audio.ndim == 2 else audio
    f0_arr, strength_arr = compute_f0_track(mono, sr)

    if len(f0_arr) == 0:
        return []

    n_total  = len(f0_arr)
    voiced_mask = f0_arr > 0
    voiced_f0   = f0_arr[voiced_mask]
    n_voiced    = len(voiced_f0)
    hop_s       = 0.010  # 10 ms hop

    results: list[MetricResult] = []

    # ── Voiced frame ratio ──────────────────────────────────────────────────
    voiced_ratio = n_voiced / n_total if n_total > 0 else 0.0
    results.append(MetricResult(
        "Voiced Frame Ratio", round(voiced_ratio * 100, 2), "%",
        "Fraction of frames detected as voiced speech", "prosody",
        higher_is_better=None,
    ))

    if n_voiced < 4:
        # Still try speaking rate even without voiced frames
        rate = _speaking_rate(mono, sr)
        if rate is not None:
            results.append(MetricResult(
                "Estimated Speech Rate", round(rate, 2), "syll/s",
                "Syllable rate estimate via energy peaks (3–7 syll/s = typical speech)", "prosody",
            ))
        return results

    # ── F0 statistics ───────────────────────────────────────────────────────
    f0_mean  = float(np.mean(voiced_f0))
    f0_std   = float(np.std(voiced_f0))
    f0_min_  = float(np.min(voiced_f0))
    f0_max_  = float(np.max(voiced_f0))
    f0_range = f0_max_ - f0_min_

    results += [
        MetricResult("F0 Mean", round(f0_mean, 2), "Hz",
                     "Mean fundamental frequency over voiced frames", "prosody"),
        MetricResult("F0 Std", round(f0_std, 2), "Hz",
                     "Standard deviation of F0 — proxy for pitch expressiveness", "prosody"),
        MetricResult("F0 Min", round(f0_min_, 2), "Hz",
                     "Minimum observed F0 (voiced frames)", "prosody"),
        MetricResult("F0 Max", round(f0_max_, 2), "Hz",
                     "Maximum observed F0 (voiced frames)", "prosody"),
        MetricResult("F0 Range", round(f0_range, 2), "Hz",
                     "Peak-to-peak F0 range — wider = more expressive intonation", "prosody"),
    ]

    # Pitch variability (coefficient of variation)
    if f0_mean > 0:
        pitch_cv = f0_std / f0_mean * 100
        results.append(MetricResult(
            "Pitch Variability (CV)", round(pitch_cv, 2), "%",
            "F0 coefficient of variation. > 20% = expressive, < 5% = monotone", "prosody",
            reference_range=(10.0, 40.0),
        ))

    # F0 slope via linear regression over voiced frame times
    voiced_times = np.where(voiced_mask)[0] * hop_s
    if len(voiced_times) >= 3:
        slope = float(np.polyfit(voiced_times, voiced_f0, 1)[0])
        results.append(MetricResult(
            "F0 Slope", round(slope, 3), "Hz/s",
            "Linear F0 trend over time. Positive = rising, negative = falling", "prosody",
        ))

    # ── Jitter ──────────────────────────────────────────────────────────────
    jitter = _jitter_local(f0_arr)
    if jitter is not None:
        jitter_pct = jitter * 100
        results.append(MetricResult(
            "Jitter (Local)", round(jitter_pct, 4), "%",
            "Cycle-to-cycle F0 period perturbation. Normal: < 1.0%. Higher = roughness/dysphonia",
            "prosody",
            higher_is_better=False,
            reference_range=(0.0, 1.0),
            warning="Elevated jitter — may indicate vocal roughness" if jitter_pct > 1.0 else None,
        ))

    # ── Shimmer ─────────────────────────────────────────────────────────────
    shimmer = _shimmer_local(mono, sr, f0_arr)
    if shimmer is not None:
        shimmer_pct = shimmer * 100
        results.append(MetricResult(
            "Shimmer (Local)", round(shimmer_pct, 4), "%",
            "Cycle-to-cycle amplitude perturbation. Normal: < 3.0%. Higher = breathiness/hoarseness",
            "prosody",
            higher_is_better=False,
            reference_range=(0.0, 3.0),
            warning="Elevated shimmer — may indicate breathiness" if shimmer_pct > 3.0 else None,
        ))

    # ── Tremor ──────────────────────────────────────────────────────────────
    if n_voiced >= 20:
        t_rate, t_depth = _tremor(voiced_f0, hop_s)
        if t_rate is not None:
            results += [
                MetricResult(
                    "Tremor Rate", round(t_rate, 2), "Hz",
                    "Dominant low-frequency F0 modulation rate. "
                    "4–7 Hz may indicate pathological tremor; 5–8 Hz = vibrato", "prosody",
                    warning="Possible vocal tremor (4–7 Hz modulation)" if 4.0 <= t_rate <= 7.0 else None,
                ),
                MetricResult(
                    "Tremor Depth", round(t_depth, 4), "Hz² (power)",
                    "Power spectral density of the dominant F0 modulation component", "prosody",
                ),
            ]

    # ── Speaking rate ────────────────────────────────────────────────────────
    rate = _speaking_rate(mono, sr)
    if rate is not None:
        results.append(MetricResult(
            "Estimated Speech Rate", round(rate, 2), "syll/s",
            "Syllable rate via energy envelope peaks. "
            "Typical conversational speech: 3–7 syll/s", "prosody",
            reference_range=(3.0, 7.0),
        ))

    return results

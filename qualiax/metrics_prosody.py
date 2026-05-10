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
from scipy.signal import find_peaks, resample_poly

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


def _compute_crepe_f0_track(
    audio: NDArray,
    sr: int,
    f0_min: float,
    f0_max: float,
    voiced_thresh: float,
) -> Optional[tuple[NDArray, NDArray]]:
    """Try optional CREPE neural pitch tracking before falling back."""
    try:
        import crepe
    except ImportError:
        return None
    try:
        target_sr = 16000
        work = resample_poly(audio, target_sr, sr) if sr != target_sr else audio
        work = np.asarray(work, dtype=np.float32)
        if len(work) < int(0.05 * target_sr):
            return None
        times, freq, conf, _ = crepe.predict(
            work,
            target_sr,
            viterbi=True,
            step_size=10,
            model_capacity="tiny",
            center=True,
            verbose=0,
        )
        freq = np.asarray(freq, dtype=np.float64).reshape(-1)
        conf = np.asarray(conf, dtype=np.float64).reshape(-1)
        valid = (conf >= voiced_thresh) & (freq >= f0_min) & (freq <= f0_max)
        f0 = np.where(valid, freq, 0.0)
        strength = np.where(valid, conf, 0.0)
        return f0, strength
    except Exception as e:
        warnings.warn(f"CREPE F0 failed: {e}; falling back to autocorrelation")
        return None


def compute_f0_track_with_backend(
    audio: NDArray,
    sr: int,
    f0_min: float = 60.0,
    f0_max: float = 500.0,
    voiced_thresh: float = 0.40,
) -> tuple[NDArray, NDArray, str]:
    """
    Per-frame F0 and voicing strength via CREPE when available, otherwise
    normalized autocorrelation.

    Returns
    -------
    f0_arr : ndarray, shape (n_frames,)
        F0 in Hz for each frame; 0.0 for unvoiced frames.
    strength_arr : ndarray, shape (n_frames,)
        Normalized autocorrelation peak height (0–1).
    backend : str
        Pitch estimator backend used for the track.
    """
    crepe_result = _compute_crepe_f0_track(audio, sr, f0_min, f0_max, voiced_thresh)
    if crepe_result is not None:
        f0_arr, strength_arr = crepe_result
        return f0_arr, strength_arr, "crepe"

    frame_len = int(0.025 * sr)
    hop_len   = int(0.010 * sr)
    tau_min   = max(1, int(sr / f0_max))
    tau_max   = int(sr / f0_min)

    if len(audio) < frame_len:
        return np.array([]), np.array([]), "autocorrelation"

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

    return f0_out, str_out, "autocorrelation"


def compute_f0_track(
    audio: NDArray,
    sr: int,
    f0_min: float = 60.0,
    f0_max: float = 500.0,
    voiced_thresh: float = 0.40,
) -> tuple[NDArray, NDArray]:
    f0_arr, strength_arr, _ = compute_f0_track_with_backend(
        audio,
        sr,
        f0_min=f0_min,
        f0_max=f0_max,
        voiced_thresh=voiced_thresh,
    )
    return f0_arr, strength_arr


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


# ─── Extended Jitter measures ─────────────────────────────────────────────────

def _jitter_rap(f0_arr: NDArray) -> Optional[float]:
    """Relative Average Perturbation (3-period smoothing window)."""
    voiced = f0_arr[f0_arr > 0]
    if len(voiced) < 5:
        return None
    periods = 1.0 / voiced
    t_mean = float(np.mean(periods))
    running_avg = (periods[:-2] + periods[1:-1] + periods[2:]) / 3.0
    return float(np.mean(np.abs(periods[1:-1] - running_avg)) / (t_mean + 1e-14))


def _jitter_ppq5(f0_arr: NDArray) -> Optional[float]:
    """5-point Period Perturbation Quotient."""
    voiced = f0_arr[f0_arr > 0]
    if len(voiced) < 7:
        return None
    periods = 1.0 / voiced
    t_mean = float(np.mean(periods))
    diffs = [abs(periods[i] - np.mean(periods[i - 2:i + 3]))
             for i in range(2, len(periods) - 2)]
    return float(np.mean(diffs) / (t_mean + 1e-14)) if diffs else None


def _jitter_ddp(f0_arr: NDArray) -> Optional[float]:
    """DDP: mean absolute second difference of periods (= 2× RAP for symmetric windows)."""
    voiced = f0_arr[f0_arr > 0]
    if len(voiced) < 5:
        return None
    periods = 1.0 / voiced
    t_mean = float(np.mean(periods))
    d2 = np.abs(np.diff(np.diff(periods)))
    return float(np.mean(d2) / (t_mean + 1e-14))


# ─── Extended Shimmer measures ────────────────────────────────────────────────

def _get_voiced_rms(audio: NDArray, sr: int, f0_arr: NDArray) -> Optional[NDArray]:
    """Per-voiced-frame RMS amplitude array."""
    frame_len = int(0.025 * sr)
    hop_len   = int(0.010 * sr)
    voiced_idx = np.where(f0_arr > 0)[0]
    rms_vals = []
    for i in voiced_idx:
        start = i * hop_len
        end   = start + frame_len
        if end > len(audio):
            break
        rms = float(np.sqrt(np.mean(audio[start:end] ** 2)))
        if rms > 1e-6:
            rms_vals.append(rms)
    return np.array(rms_vals) if len(rms_vals) >= 5 else None


def _shimmer_apq3(audio: NDArray, sr: int, f0_arr: NDArray) -> Optional[float]:
    """3-point Amplitude Perturbation Quotient."""
    arr = _get_voiced_rms(audio, sr, f0_arr)
    if arr is None or len(arr) < 5:
        return None
    a_mean = float(np.mean(arr))
    running_avg = (arr[:-2] + arr[1:-1] + arr[2:]) / 3.0
    return float(np.mean(np.abs(arr[1:-1] - running_avg)) / (a_mean + 1e-14))


def _shimmer_apq5(audio: NDArray, sr: int, f0_arr: NDArray) -> Optional[float]:
    """5-point Amplitude Perturbation Quotient."""
    arr = _get_voiced_rms(audio, sr, f0_arr)
    if arr is None or len(arr) < 7:
        return None
    a_mean = float(np.mean(arr))
    diffs = [abs(arr[i] - np.mean(arr[i - 2:i + 3]))
             for i in range(2, len(arr) - 2)]
    return float(np.mean(diffs) / (a_mean + 1e-14)) if diffs else None


def _shimmer_dda(audio: NDArray, sr: int, f0_arr: NDArray) -> Optional[float]:
    """DDA shimmer = mean absolute second difference of amplitudes."""
    arr = _get_voiced_rms(audio, sr, f0_arr)
    if arr is None or len(arr) < 5:
        return None
    a_mean = float(np.mean(arr))
    d2 = np.abs(np.diff(np.diff(arr)))
    return float(np.mean(d2) / (a_mean + 1e-14))


# ─── Noise-to-Harmonics Ratio ─────────────────────────────────────────────────

def _nhr(audio: NDArray, sr: int, f0_arr: NDArray) -> Optional[float]:
    """
    Noise-to-Harmonics Ratio.

    Harmonic power is estimated as the sum of STFT power within ±2 bins of
    each integer multiple of the mean F0 (up to Nyquist).  Everything else
    is treated as noise.  NHR = noise_power / harmonic_power.
    """
    try:
        from scipy.signal import stft as scipy_stft
        voiced = f0_arr[f0_arr > 0]
        if len(voiced) < 4:
            return None
        f0_mean = float(np.mean(voiced))

        n_fft = 1024
        _, _, Zxx = scipy_stft(audio, fs=sr, nperseg=n_fft,
                               noverlap=n_fft * 3 // 4)
        mean_power = (np.abs(Zxx) ** 2).mean(axis=1)   # (n_fft//2+1,)
        total_power = float(mean_power.sum()) + 1e-14

        bin_hz     = sr / n_fft
        max_harm   = int((sr / 2.0) / f0_mean)
        harmonic_p = 0.0
        for h in range(1, min(max_harm + 1, 20)):
            c = int(round(h * f0_mean / bin_hz))
            lo = max(0, c - 2)
            hi = min(len(mean_power), c + 3)
            harmonic_p += float(mean_power[lo:hi].sum())

        harmonic_p = min(harmonic_p, total_power)
        noise_p    = total_power - harmonic_p
        return float(noise_p / (harmonic_p + 1e-14))
    except Exception as e:
        warnings.warn(f"Metric 'Noise-to-Harmonics Ratio (prosody)' failed: {e}")
        return None


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
    f0_arr, strength_arr, f0_backend = compute_f0_track_with_backend(mono, sr)

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
    results.append(MetricResult(
        "F0 Estimator Backend", f0_backend, "",
        "Pitch tracker backend used for this file (`crepe` when available, otherwise autocorrelation).",
        "prosody",
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

    rap = _jitter_rap(f0_arr)
    if rap is not None:
        results.append(MetricResult(
            "Jitter (RAP)", round(rap * 100, 4), "%",
            "Relative Average Perturbation: 3-period smoothed jitter. Normal: < 0.68%",
            "prosody",
            higher_is_better=False,
            reference_range=(0.0, 0.68),
            warning="Elevated RAP jitter" if rap * 100 > 0.68 else None,
        ))

    ppq5 = _jitter_ppq5(f0_arr)
    if ppq5 is not None:
        results.append(MetricResult(
            "Jitter (PPQ5)", round(ppq5 * 100, 4), "%",
            "5-point Period Perturbation Quotient. Normal: < 0.84%",
            "prosody",
            higher_is_better=False,
            reference_range=(0.0, 0.84),
            warning="Elevated PPQ5 jitter" if ppq5 * 100 > 0.84 else None,
        ))

    ddp = _jitter_ddp(f0_arr)
    if ddp is not None:
        results.append(MetricResult(
            "Jitter (DDP)", round(ddp * 100, 4), "%",
            "Mean absolute second difference of periods (≈ 3× RAP). Normal: < 2.0%",
            "prosody",
            higher_is_better=False,
            reference_range=(0.0, 2.0),
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

    apq3 = _shimmer_apq3(mono, sr, f0_arr)
    if apq3 is not None:
        results.append(MetricResult(
            "Shimmer (APQ3)", round(apq3 * 100, 4), "%",
            "3-point Amplitude Perturbation Quotient. Normal: < 3.07%",
            "prosody",
            higher_is_better=False,
            reference_range=(0.0, 3.07),
            warning="Elevated APQ3 shimmer" if apq3 * 100 > 3.07 else None,
        ))

    apq5 = _shimmer_apq5(mono, sr, f0_arr)
    if apq5 is not None:
        results.append(MetricResult(
            "Shimmer (APQ5)", round(apq5 * 100, 4), "%",
            "5-point Amplitude Perturbation Quotient. Normal: < 4.23%",
            "prosody",
            higher_is_better=False,
            reference_range=(0.0, 4.23),
        ))

    dda = _shimmer_dda(mono, sr, f0_arr)
    if dda is not None:
        results.append(MetricResult(
            "Shimmer (DDA)", round(dda * 100, 4), "%",
            "Mean absolute second difference of amplitudes (≈ 3× APQ3). Normal: < 9.0%",
            "prosody",
            higher_is_better=False,
            reference_range=(0.0, 9.0),
        ))

    # ── NHR ─────────────────────────────────────────────────────────────────
    nhr = _nhr(mono, sr, f0_arr)
    if nhr is not None:
        results.append(MetricResult(
            "NHR (Noise-to-Harmonics Ratio)", round(nhr, 6), "",
            "Ratio of non-harmonic to harmonic energy. Normal voice: < 0.19. "
            "Higher values indicate breathiness or noise.",
            "prosody",
            higher_is_better=False,
            reference_range=(0.0, 0.19),
            warning="Elevated NHR — possible breathiness or noise" if nhr > 0.19 else None,
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

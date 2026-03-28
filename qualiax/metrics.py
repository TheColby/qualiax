"""
Metric computation modules.

Each metric group is a function that takes (audio, sr, ref_audio, ref_sr)
and returns a list of MetricResult objects.

All groups are registered in METRIC_GROUPS dict at the bottom.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .models import MetricResult
from . import gpu as _gpu  # hardware-accelerated kernels (Metal / CUDA / CPU)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe(fn, *args, name="", **kwargs):
    """Run fn; return None on any exception."""
    try:
        result = fn(*args, **kwargs)
        return result
    except Exception as e:
        warnings.warn(f"Metric '{name}' failed: {e}")
        return None


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

    peak = float(np.max(np.abs(mono)))
    peak_db = _db(peak)
    rms_val = _rms(mono)
    rms_db = _db(rms_val)
    crest_factor = (peak / (rms_val + _eps())) if rms_val > 0 else 0.0
    crest_db = _db(crest_factor)
    dc_offset = float(np.mean(mono))
    silence_ratio = float(np.mean(np.abs(mono) < 0.001))

    return [
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
        MetricResult("Dynamic Range (simple)", peak_db - rms_db, "dB",
                     "Difference between peak and RMS level", "basic"),
        MetricResult("Clipping Detected",
                     int(np.any(np.abs(mono) >= 0.999)),
                     "", "1 if any samples appear clipped (>=0.999 full scale)", "basic",
                     higher_is_better=False,
                     warning="Clipping detected!" if np.any(np.abs(mono) >= 0.999) else None),
        MetricResult("Zero Crossing Rate", float(np.mean(np.abs(np.diff(np.sign(mono))) > 0)), "crossings/sample",
                     "Rate of sign changes (correlated with pitch/noisiness)", "basic"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# GROUP: loudness  (ITU-R BS.1770 / EBU R128)
# ─────────────────────────────────────────────────────────────────────────────

def _k_weighting_filter(audio: NDArray, sr: int) -> NDArray:
    """Apply BS.1770 K-weighting. GPU-accelerated (Metal/CUDA/CPU)."""
    return _gpu.k_weighting_filter(audio, sr)


def _integrated_loudness_lufs(audio: NDArray, sr: int) -> float:
    """Compute integrated loudness in LUFS per BS.1770-4."""
    mono = _to_mono(audio)
    filtered = _k_weighting_filter(mono, sr)

    # 400ms blocks with 75% overlap
    block_size = int(0.4 * sr)
    hop = int(0.1 * sr)
    if len(filtered) < block_size:
        return float("nan")

    block_loudnesses = []
    for start in range(0, len(filtered) - block_size + 1, hop):
        block = filtered[start:start + block_size]
        mean_sq = float(np.mean(block ** 2))
        if mean_sq > 0:
            block_loudnesses.append(-0.691 + 10 * math.log10(mean_sq))

    if not block_loudnesses:
        return float("nan")

    # Absolute gate: -70 LUFS
    gated = [l for l in block_loudnesses if l >= -70.0]
    if not gated:
        return float("nan")

    # Relative gate: -10 from ungated mean
    mean_ungated = -0.691 + 10 * math.log10(np.mean([10 ** (l / 10) for l in gated]))
    threshold = mean_ungated - 10.0
    gated2 = [l for l in gated if l >= threshold]
    if not gated2:
        return float("nan")

    return -0.691 + 10 * math.log10(np.mean([10 ** (l / 10) for l in gated2]))


def _loudness_range(audio: NDArray, sr: int) -> float:
    """EBU R128 Loudness Range (LRA)."""
    mono = _to_mono(audio)
    filtered = _k_weighting_filter(mono, sr)

    block_size = int(3.0 * sr)
    hop = int(0.1 * sr)
    if len(filtered) < block_size:
        return float("nan")

    block_loudnesses = []
    for start in range(0, len(filtered) - block_size + 1, hop):
        block = filtered[start:start + block_size]
        mean_sq = float(np.mean(block ** 2))
        if mean_sq > 0:
            block_loudnesses.append(-0.691 + 10 * math.log10(mean_sq))

    if len(block_loudnesses) < 2:
        return float("nan")

    # Absolute gate
    gated = [l for l in block_loudnesses if l >= -70.0]
    if not gated:
        return float("nan")

    # Relative gate
    mean_g = -0.691 + 10 * math.log10(np.mean([10 ** (l / 10) for l in gated]))
    gated2 = sorted([l for l in gated if l >= mean_g - 20.0])

    if len(gated2) < 2:
        return float("nan")

    lo = np.percentile(gated2, 10)
    hi = np.percentile(gated2, 95)
    return float(hi - lo)


def compute_loudness(audio: NDArray, sr: int, ref_audio=None, ref_sr=None) -> list[MetricResult]:
    mono = _to_mono(audio)

    lufs = _safe(_integrated_loudness_lufs, audio, sr, name="LUFS")
    lra = _safe(_loudness_range, audio, sr, name="LRA")

    # Short-term loudness (3s window max)
    max_short_term = None
    try:
        filtered = _k_weighting_filter(mono, sr)
        block = int(3 * sr)
        if len(filtered) >= block:
            vals = []
            for i in range(0, len(filtered) - block + 1, int(0.1 * sr)):
                sq = float(np.mean(filtered[i:i + block] ** 2))
                if sq > 0:
                    vals.append(-0.691 + 10 * math.log10(sq))
            max_short_term = max(vals) if vals else None
    except Exception:
        pass

    # True peak (oversample 4x via upsampling) — GPU-accelerated
    try:
        true_peak_dbtp = _gpu.true_peak(mono, sr, oversample=4)
    except Exception:
        true_peak_dbtp = _db(float(np.max(np.abs(mono))))

    results = [
        MetricResult("Integrated Loudness (LUFS)", lufs, "LUFS",
                     "ITU-R BS.1770 / EBU R128 integrated loudness (gated)", "loudness",
                     reference_range=(-16.0, -14.0),
                     warning=("Too loud for streaming" if lufs and lufs > -14 else
                              "Too quiet for streaming" if lufs and lufs < -18 else None)),
        MetricResult("Loudness Range (LRA)", lra, "LU",
                     "EBU R128 loudness range: dynamic variation of the program", "loudness",
                     reference_range=(5.0, 15.0)),
        MetricResult("Max Short-Term Loudness", max_short_term, "LUFS",
                     "Maximum 3-second sliding window loudness", "loudness"),
        MetricResult("True Peak", true_peak_dbtp, "dBTP",
                     "Inter-sample peak level (4x oversampled). Streaming limit: -1 dBTP", "loudness",
                     higher_is_better=False,
                     warning="Exceeds -1 dBTP streaming limit" if true_peak_dbtp and true_peak_dbtp > -1 else None),
    ]
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

    # ── GPU single-pass spectral feature extraction ──────────────────────────
    try:
        _sf = _gpu.spectral_features(mono, sr, n_fft=n_fft)
        centroid   = _sf["centroid"]
        bandwidth  = _sf["bandwidth"]
        rolloff    = _sf["rolloff"]
        flatness_db = _sf["flatness_db"]
        flux       = _sf["flux"]
        entropy    = _sf["entropy"]
        hfc_gpu    = _sf["hfc"]
        gpu_ok = True
    except Exception as e:
        gpu_ok = False

    # Still need STFT for band energies + skewness + effective bandwidth
    try:
        f, t, mag = _stft(mono, sr, n_fft)
    except Exception as e:
        return [MetricResult("Spectral analysis", None, "", f"Failed: {e}", "spectral")]

    power = mag ** 2
    total_power = power.sum(axis=0) + _eps()
    freqs = f

    if not gpu_ok:
        centroid = float(np.mean(np.sum(freqs[:, None] * power, axis=0) / total_power))
        diff = (freqs[:, None] - centroid) ** 2
        bandwidth = float(np.mean(np.sqrt(np.sum(diff * power, axis=0) / total_power)))
        cumsum = np.cumsum(power, axis=0)
        rolloff_idx = np.argmax(cumsum >= 0.95 * total_power[None, :], axis=0)
        rolloff = float(np.mean(freqs[np.clip(rolloff_idx, 0, len(freqs) - 1)]))
        geo_mean = np.exp(np.mean(np.log(power + _eps()), axis=0))
        arith_mean = np.mean(power, axis=0) + _eps()
        flatness = float(np.mean(geo_mean / arith_mean))
        flatness_db = _power_db(flatness)
        diff_mag = np.diff(mag, axis=1)
        flux = float(np.mean(np.sqrt(np.sum(diff_mag ** 2, axis=0))))
        norm_p = power / (total_power[None, :] + _eps())
        entropy = float(np.mean(-np.sum(norm_p * np.log2(norm_p + _eps()), axis=0)))
        hfc_gpu = float(np.mean(np.sum(freqs[:, None] * power, axis=0)))

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
    """Simple autocorrelation-based F0 estimate (mean over frames)."""
    try:
        frame_len = int(0.04 * sr)  # 40ms frames
        hop = int(0.01 * sr)
        min_lag = int(sr / fmax)
        max_lag = int(sr / fmin)

        pitches = []
        for start in range(0, len(mono) - frame_len, hop):
            frame = mono[start:start + frame_len]
            frame -= frame.mean()
            if np.max(np.abs(frame)) < 0.01:
                continue
            # Autocorrelation via FFT
            n = 2 * frame_len
            fft = np.fft.rfft(frame, n=n)
            acf = np.fft.irfft(fft * np.conj(fft))
            acf = acf[:frame_len]
            if max_lag >= len(acf):
                continue
            region = acf[min_lag:max_lag]
            if len(region) == 0:
                continue
            lag = np.argmax(region) + min_lag
            if acf[0] > 0 and acf[lag] / acf[0] > 0.3:
                pitches.append(sr / lag)

        if pitches:
            return float(np.median(pitches))
        return None
    except Exception:
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
    zcr_frames = np.array([float(np.mean(np.abs(np.diff(np.sign(f))) > 0)) for f in frames])

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

def compute_noise(audio: NDArray, sr: int, ref_audio=None, ref_sr=None) -> list[MetricResult]:
    mono = _to_mono(audio)

    # GPU-accelerated batch frame energy
    energies = _gpu.batch_energy(mono, sr, frame_len_ms=25.0, hop_ms=10.0)
    if len(energies) == 0:
        return []
    frame_len = int(0.025 * sr)
    hop = int(0.010 * sr)

    # Noise floor estimation: median of lowest 10th percentile frames
    noise_threshold = np.percentile(energies, 10)
    noise_frames_e = energies[energies <= noise_threshold]
    noise_floor_e = float(np.mean(noise_frames_e)) if len(noise_frames_e) > 0 else 0.0
    noise_floor_db = _power_db(noise_floor_e) if noise_floor_e > 0 else float("nan")

    # Signal floor: energy of speech-active frames (top 50%)
    sig_threshold = np.percentile(energies, 50)
    sig_frames_e = energies[energies >= sig_threshold]
    signal_e = float(np.mean(sig_frames_e)) if len(sig_frames_e) > 0 else 0.0

    # SNR estimate
    snr = float("nan")
    if noise_floor_e > 0 and signal_e > 0:
        snr = 10 * math.log10(signal_e / noise_floor_e)

    # Spectral noise estimation (via STFT on low-energy frames)
    spectral_snr = float("nan")
    try:
        from scipy.signal import stft as scipy_stft
        f, t_ax, Zxx = scipy_stft(mono, fs=sr, nperseg=512, noverlap=384)
        mag = np.abs(Zxx)
        frame_power = mag.mean(axis=0)
        noise_mask = frame_power <= np.percentile(frame_power, 15)
        if noise_mask.any() and (~noise_mask).any():
            noise_spec = mag[:, noise_mask].mean(axis=1)
            sig_spec = mag[:, ~noise_mask].mean(axis=1)
            spectral_snr = float(10 * np.log10(
                (np.mean(sig_spec ** 2) + _eps()) / (np.mean(noise_spec ** 2) + _eps())
            ))
    except Exception:
        pass

    # Harmonic-to-Noise Ratio (HNR) via autocorrelation
    hnr = _compute_hnr(mono, sr)

    # Clipping indicator (consecutive identical max samples)
    peak_val = float(np.max(np.abs(mono)))
    clipped_count = int(np.sum(np.abs(mono) >= 0.999 * peak_val)) if peak_val > 0.5 else 0

    # Dropout detection (sudden near-zero segments in otherwise active audio)
    dropouts = _detect_dropouts(energies, hop, sr)

    return [
        MetricResult("Estimated Noise Floor", noise_floor_db, "dBFS",
                     "Energy of quietest 10% of frames (noise floor estimate)", "noise",
                     higher_is_better=False),
        MetricResult("Estimated SNR", snr, "dB",
                     "Signal-to-noise ratio: active speech vs noise floor (energy-based)", "noise",
                     higher_is_better=True,
                     reference_range=(20.0, 40.0),
                     warning="Poor SNR for intelligible speech" if snr < 15 else None),
        MetricResult("Spectral SNR", spectral_snr, "dB",
                     "SNR estimated from spectral power of active vs quiet frames", "noise",
                     higher_is_better=True),
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


def _compute_hnr(mono: NDArray, sr: int, fmin=75, fmax=500) -> Optional[float]:
    """Estimate HNR via autocorrelation method (mean over voiced frames)."""
    try:
        frame_len = int(0.04 * sr)
        hop = int(0.01 * sr)
        min_lag = int(sr / fmax)
        max_lag = int(sr / fmin)
        hnrs = []
        for start in range(0, len(mono) - frame_len, hop):
            frame = mono[start:start + frame_len]
            frame = frame - frame.mean()
            if np.max(np.abs(frame)) < 0.005:
                continue
            n = 2 * frame_len
            fft = np.fft.rfft(frame, n=n)
            acf = np.fft.irfft(fft * np.conj(fft))[:frame_len]
            acf /= (acf[0] + _eps())
            if max_lag >= len(acf):
                continue
            r_max = np.max(acf[min_lag:max_lag])
            if r_max >= 1.0:
                continue
            if r_max > 0.0:
                hnrs.append(10 * math.log10(r_max / (1.0 - r_max + _eps())))
        return float(np.median(hnrs)) if hnrs else None
    except Exception:
        return None


def _detect_dropouts(energies: NDArray, hop: int, sr: int) -> int:
    """Count frames that drop >30 dB below surrounding context."""
    count = 0
    win = 10  # frames context
    for i in range(win, len(energies) - win):
        context = np.concatenate([energies[i - win:i], energies[i + 1:i + win + 1]])
        if context.mean() > 0 and energies[i] > 0:
            drop = 10 * math.log10(context.mean() / (energies[i] + _eps()))
            if drop > 30:
                count += 1
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
        f, _, mag = scipy_stft(mono, fs=sr, nperseg=1024, noverlap=768)
        power = mag ** 2
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
    except Exception:
        pass

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
    except Exception:
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
    except Exception:
        return None


def _estimate_speaking_rate(mono: NDArray, sr: int) -> Optional[float]:
    """Syllable nucleus detection via smooth energy envelope peaks."""
    try:
        from scipy.signal import medfilt
        frame_len = int(0.025 * sr)
        hop = int(0.005 * sr)
        energies = np.array([
            float(np.mean(mono[i:i + frame_len] ** 2))
            for i in range(0, len(mono) - frame_len, hop)
        ])
        smooth = medfilt(energies, kernel_size=min(21, len(energies) | 1))
        # Find peaks in energy
        threshold = np.percentile(smooth, 60)
        peaks = []
        for i in range(1, len(smooth) - 1):
            if smooth[i] > smooth[i - 1] and smooth[i] > smooth[i + 1] and smooth[i] > threshold:
                peaks.append(i)
        duration = len(mono) / sr
        if duration > 0.5 and len(peaks) > 1:
            return len(peaks) / duration
        return None
    except Exception:
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
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# GROUP: perceptual  (intrusive: needs reference; non-intrusive otherwise)
# ─────────────────────────────────────────────────────────────────────────────

def compute_perceptual(audio: NDArray, sr: int, ref_audio=None, ref_sr=None) -> list[MetricResult]:
    results = []

    # ── Non-intrusive ───────────────────────────────────────────────────────

    # DNSMOS-like (approximate via SNR + spectral measure)
    # Real DNSMOS requires ONNX model; we compute a proxy
    mono = _to_mono(audio)
    results.append(_compute_pseudo_mos(mono, sr))

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
        snr = (10 * math.log10(sig_p / (noise_p + _eps()))) if noise_p > 0 else 0.0

        # Spectral flatness (lower = more speech-like)
        geo = np.exp(np.mean(np.log(power.mean(axis=1) + _eps())))
        arith = np.mean(power.mean(axis=1)) + _eps()
        flatness_db = _power_db(geo / arith)

        # Map to MOS-like score [1–5]
        # Heuristic: snr 0→5 = MOS ~1.5→4.5; penalize flat spectrum
        snr_score = np.clip(1.0 + snr / 10.0, 1.0, 4.5)
        flat_penalty = max(0.0, (flatness_db + 10) / 20.0)  # penalty if very flat
        mos = float(np.clip(snr_score - flat_penalty * 0.5, 1.0, 5.0))

        return MetricResult(
            "Estimated MOS (non-intrusive proxy)", round(mos, 2), "MOS [1–5]",
            "Heuristic MOS estimate (SNR + spectral shape). For accurate MOS install `dnsmos`.", "perceptual",
            higher_is_better=True,
            reference_range=(3.5, 5.0),
            warning="Note: proxy only — use DNSMOS ONNX model for accurate MOS" if True else None,
        )
    except Exception as e:
        return MetricResult("Estimated MOS (proxy)", None, "", f"Failed: {e}", "perceptual")


def _compute_p563_proxy(mono: NDArray, sr: int) -> MetricResult:
    """Simplified P.563-inspired non-intrusive quality metric."""
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

        # Waveform discontinuities (proxy for circuit noise)
        discontinuities = int(np.sum(np.abs(np.diff(mono)) > 0.5))

        # Unnatural silence ratio
        silence_ratio = 1.0 - float(np.mean(active))

        # Frequency imbalance: should have most energy 300-3400 Hz
        speech_idx = (f >= 300) & (f <= 3400)
        speech_energy = float(power[speech_idx, :].sum()) if speech_idx.any() else 0.0
        total_energy = float(power.sum()) + _eps()
        speech_ratio = speech_energy / total_energy

        # Heuristic score
        score = 4.0 * speech_ratio - 0.5 * silence_ratio - discontinuities * 0.001
        score = float(np.clip(score, 1.0, 4.5))

        return MetricResult(
            "P.563 Proxy (NB Quality Estimate)", round(score, 2), "MOS [1–4.5]",
            "Narrowband non-intrusive quality proxy (P.563-inspired heuristic)", "perceptual",
            higher_is_better=True,
            reference_range=(3.0, 4.5),
            warning="Proxy only — install `itu-p563` for true P.563",
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
        return MetricResult(
            "PESQ (ITU-T P.862)", round(score, 3), "MOS-LQO",
            f"Perceptual Evaluation of Speech Quality [{mode.upper()} mode] (1.0–4.5)", "perceptual",
            higher_is_better=True,
            reference_range=(3.0, 4.5),
        )
    except ImportError:
        # Fallback proxy
        lsd = _log_spectral_distance(y, r, sr)
        if lsd is not None:
            mos_proxy = float(np.clip(4.5 - lsd / 5.0, 1.0, 4.5))
            return MetricResult(
                "PESQ proxy (install `pesq` for true score)", round(mos_proxy, 3), "MOS-LQO (approx)",
                "Approximate PESQ via log-spectral distance mapping", "perceptual",
                higher_is_better=True,
                warning="Install `pip install pesq` for accurate PESQ score",
            )
        return MetricResult("PESQ", None, "", "Install `pip install pesq` for PESQ", "perceptual")
    except Exception as e:
        return MetricResult("PESQ", None, "", f"PESQ failed: {e}", "perceptual")


def _compute_stoi(y: NDArray, r: NDArray, sr: int) -> MetricResult:
    """Compute STOI via `pystoi` library if available, else proxy."""
    try:
        from pystoi import stoi as stoi_fn
        score = float(stoi_fn(r, y, sr, extended=False))
        return MetricResult(
            "STOI (Short-Time Objective Intelligibility)", round(score, 4), "[0–1]",
            "Speech intelligibility score (1 = perfectly intelligible)", "perceptual",
            higher_is_better=True,
            reference_range=(0.7, 1.0),
            warning="Low intelligibility" if score < 0.6 else None,
        )
    except ImportError:
        # Spectral correlation proxy for intelligibility
        sc = _spectral_correlation(y, r, sr)
        if sc is not None:
            return MetricResult(
                "STOI proxy (install `pystoi` for true score)", round(sc, 4), "[0–1] (approx)",
                "Approximate intelligibility via spectral correlation", "perceptual",
                higher_is_better=True,
                warning="Install `pip install pystoi` for accurate STOI",
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
    except Exception:
        return None


def _compute_sdr(y: NDArray, r: NDArray) -> Optional[float]:
    try:
        noise = y - r
        sdr = 10 * math.log10(
            (np.dot(r, r) + _eps()) / (np.dot(noise, noise) + _eps())
        )
        return float(sdr)
    except Exception:
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
    except Exception:
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
    except Exception:
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
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

METRIC_GROUPS: dict[str, callable] = {
    "basic": compute_basic,
    "loudness": compute_loudness,
    "spectral": compute_spectral,
    "temporal": compute_temporal,
    "noise": compute_noise,
    "speech": compute_speech,
    "perceptual": compute_perceptual,
}

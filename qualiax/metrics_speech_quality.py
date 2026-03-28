"""
Speech-specific quality and intelligibility metrics.

These metrics are ONLY computed when speech presence is detected.
They complement the existing `speech` and `perceptual` groups with
deeper voice quality, room acoustics, and intelligibility analysis.
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .models import MetricResult

GROUP = "speech_quality"

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _eps() -> float:
    return 1e-10


def _to_mono(audio: NDArray) -> NDArray:
    return audio.mean(axis=0) if audio.ndim == 2 else audio


def _db(x: float) -> float:
    return 20 * math.log10(max(x, _eps()))


def _pdb(x: float) -> float:
    return 10 * math.log10(max(x, _eps()))


def _resample(audio: NDArray, orig_sr: int, target_sr: int) -> NDArray:
    if orig_sr == target_sr:
        return audio
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(orig_sr, target_sr)
    return resample_poly(audio.astype(np.float64), target_sr // g, orig_sr // g).astype(np.float32)


def _voiced_frames(mono: NDArray, sr: int, energy_thresh=0.0001, zcr_thresh=0.15):
    """Yield (frame, pitch_period_samples) for voiced frames only."""
    frame_len = int(0.025 * sr)
    hop = int(0.005 * sr)
    min_lag = int(sr / 400)
    max_lag = int(sr / 60)
    for i in range(0, len(mono) - frame_len, hop):
        frame = mono[i:i + frame_len]
        e = float(np.mean(frame ** 2))
        zcr = float(np.mean(np.abs(np.diff(np.sign(frame))) > 0))
        if e < energy_thresh or zcr >= zcr_thresh:
            continue
        # Autocorrelation to get pitch period
        frame_z = frame - frame.mean()
        n = 2 * frame_len
        fft = np.fft.rfft(frame_z, n=n)
        acf = np.fft.irfft(fft * np.conj(fft))[:frame_len]
        if acf[0] <= 0:
            continue
        acf /= (acf[0] + _eps())
        if max_lag >= len(acf):
            continue
        region = acf[min_lag:max_lag]
        pk = int(np.argmax(region)) + min_lag
        if acf[pk] > 0.3:
            yield frame, pk


# ─────────────────────────────────────────────────────────────────────────────
# Jitter  (fundamental period variability — perturbation measure)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_jitter(mono: NDArray, sr: int) -> dict[str, Optional[float]]:
    """
    Compute jitter metrics from voiced frame pitch periods.

    Jitter measures cycle-to-cycle variability in pitch period.
    Elevated jitter indicates hoarseness, vocal pathology, or background noise.

    Returns:
        jitter_local   - mean absolute diff of adjacent periods / mean period (%)
        jitter_rap     - relative average perturbation (3-pt smoothing)
        jitter_ppq5    - 5-point period perturbation quotient
        jitter_ddp     - difference of differences (= 3 × RAP)
    """
    periods = [pk for _, pk in _voiced_frames(mono, sr)]

    if len(periods) < 10:
        return {k: None for k in ("jitter_local", "jitter_rap", "jitter_ppq5", "jitter_ddp")}

    periods = np.array(periods, dtype=np.float64)
    T_mean = float(np.mean(periods))

    # Local jitter
    diffs = np.abs(np.diff(periods))
    jitter_local = float(np.mean(diffs) / T_mean * 100.0)

    # RAP: 3-point smoothing
    rap_vals = []
    for i in range(1, len(periods) - 1):
        smooth = np.mean(periods[i - 1:i + 2])
        rap_vals.append(abs(periods[i] - smooth))
    jitter_rap = float(np.mean(rap_vals) / T_mean * 100.0) if rap_vals else None

    # PPQ5: 5-point perturbation quotient
    ppq5_vals = []
    for i in range(2, len(periods) - 2):
        smooth = np.mean(periods[i - 2:i + 3])
        ppq5_vals.append(abs(periods[i] - smooth))
    jitter_ppq5 = float(np.mean(ppq5_vals) / T_mean * 100.0) if ppq5_vals else None

    # DDP = 3 × RAP
    jitter_ddp = jitter_rap * 3.0 if jitter_rap is not None else None

    return {
        "jitter_local": jitter_local,
        "jitter_rap": jitter_rap,
        "jitter_ppq5": jitter_ppq5,
        "jitter_ddp": jitter_ddp,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shimmer  (amplitude perturbation)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_shimmer(mono: NDArray, sr: int) -> dict[str, Optional[float]]:
    """
    Compute shimmer metrics.

    Shimmer measures cycle-to-cycle amplitude variability.
    Elevated shimmer indicates breathiness, roughness, or noise.
    """
    frame_len = int(0.025 * sr)
    hop = int(0.005 * sr)
    min_lag = int(sr / 400)
    max_lag = int(sr / 60)

    amplitudes = []
    for i in range(0, len(mono) - frame_len, hop):
        frame = mono[i:i + frame_len]
        e = float(np.mean(frame ** 2))
        zcr = float(np.mean(np.abs(np.diff(np.sign(frame))) > 0))
        if e < 0.0001 or zcr >= 0.15:
            continue
        # Peak-to-peak amplitude of the frame
        amplitudes.append(float(np.max(np.abs(frame))))

    if len(amplitudes) < 10:
        return {k: None for k in ("shimmer_local", "shimmer_apq3", "shimmer_apq5", "shimmer_dda")}

    amps = np.array(amplitudes, dtype=np.float64)
    A_mean = float(np.mean(amps))

    # Local shimmer (dB)
    diffs_db = np.abs(20 * np.log10((amps[1:] + _eps()) / (amps[:-1] + _eps())))
    shimmer_local_db = float(np.mean(diffs_db))

    # Local shimmer (%)
    shimmer_local_pct = float(np.mean(np.abs(np.diff(amps))) / A_mean * 100.0)

    # APQ3
    apq3_vals = []
    for i in range(1, len(amps) - 1):
        smooth = np.mean(amps[i - 1:i + 2])
        apq3_vals.append(abs(amps[i] - smooth))
    shimmer_apq3 = float(np.mean(apq3_vals) / A_mean * 100.0) if apq3_vals else None

    # APQ5
    apq5_vals = []
    for i in range(2, len(amps) - 2):
        smooth = np.mean(amps[i - 2:i + 3])
        apq5_vals.append(abs(amps[i] - smooth))
    shimmer_apq5 = float(np.mean(apq5_vals) / A_mean * 100.0) if apq5_vals else None

    # DDA = 3 × APQ3
    shimmer_dda = shimmer_apq3 * 3.0 if shimmer_apq3 is not None else None

    return {
        "shimmer_local_pct": shimmer_local_pct,
        "shimmer_local_db": shimmer_local_db,
        "shimmer_apq3": shimmer_apq3,
        "shimmer_apq5": shimmer_apq5,
        "shimmer_dda": shimmer_dda,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Noise-to-Harmonics Ratio (NHR) — complement to HNR
# ─────────────────────────────────────────────────────────────────────────────

def _compute_nhr(mono: NDArray, sr: int) -> Optional[float]:
    """NHR = 1 / (HNR linear) — fraction of noise energy to total."""
    try:
        frame_len = int(0.04 * sr)
        hop = int(0.01 * sr)
        min_lag = int(sr / 400)
        max_lag = int(sr / 60)
        nhrs = []
        for i in range(0, len(mono) - frame_len, hop):
            frame = mono[i:i + frame_len]
            frame = frame - frame.mean()
            if np.max(np.abs(frame)) < 0.005:
                continue
            n = 2 * frame_len
            fft = np.fft.rfft(frame, n=n)
            acf = np.fft.irfft(fft * np.conj(fft))[:frame_len]
            if acf[0] <= 0:
                continue
            acf /= acf[0]
            if max_lag >= len(acf):
                continue
            r_max = float(np.max(acf[min_lag:max_lag]))
            r_max = max(0.0, min(r_max, 0.9999))
            if r_max > 0:
                nhrs.append((1 - r_max) / (r_max + _eps()))
        return float(np.median(nhrs)) if nhrs else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Speech Transmission Index (STI) proxy
# ─────────────────────────────────────────────────────────────────────────────

def _compute_sti_proxy(mono: NDArray, sr: int) -> Optional[float]:
    """
    STI proxy via Modulation Transfer Function (MTF).
    
    True STI requires a test signal (STIPA) and measurement at receiver.
    This proxy estimates signal modulation preservation in octave bands,
    normalized to a 0–1 scale where 1 = perfect transmission.
    
    Based on IEC 60268-16 methodology adapted for single-channel analysis.
    """
    try:
        from scipy.signal import stft as scipy_stft, butter, filtfilt
        from scipy.fft import fft

        # Octave-band center frequencies relevant to speech intelligibility
        octave_centers = [125, 250, 500, 1000, 2000, 4000]

        # Modulation frequencies used in STI (IEC 60268-16)
        mod_freqs = [0.63, 0.80, 1.00, 1.25, 1.60, 2.00, 2.50, 3.15, 4.00, 5.00, 6.30, 8.00, 10.0, 12.5]

        # Frame envelope per octave band
        mtf_vals = []
        for fc in octave_centers:
            if fc >= sr / 2.5:
                continue
            # Bandpass filter
            lo = fc / math.sqrt(2)
            hi = min(fc * math.sqrt(2), sr / 2.5)
            if lo < 1 or hi >= sr / 2:
                continue
            try:
                b, a = butter(2, [lo / (sr / 2), hi / (sr / 2)], btype="band")
                band_signal = filtfilt(b, a, mono.astype(np.float64))
            except Exception:
                continue

            # Envelope via Hilbert-like: squared + smooth
            env = band_signal ** 2
            # Smooth to get amplitude envelope
            win = max(1, int(0.02 * sr))
            from scipy.signal import medfilt
            env_smooth = medfilt(env, kernel_size=min(win | 1, len(env) | 1))
            env_smooth = np.sqrt(np.maximum(env_smooth, 0))

            if env_smooth.sum() < _eps():
                continue

            # MTF: modulation depth at each modulation frequency
            n = len(env_smooth)
            env_fft = np.fft.rfft(env_smooth - env_smooth.mean(), n=n)
            env_freqs = np.fft.rfftfreq(n, d=1.0 / sr)
            env_power = np.abs(env_fft) ** 2

            E0 = float(np.mean(env_smooth ** 2)) + _eps()

            for fm in mod_freqs:
                if fm >= sr / 2:
                    continue
                # Find closest bin
                idx = int(np.argmin(np.abs(env_freqs - fm)))
                m = float(env_power[idx]) / (E0 * n ** 2 + _eps())
                m_clipped = float(np.clip(m * 2000, 0.0, 1.0))
                mtf_vals.append(m_clipped)

        if not mtf_vals:
            return None

        # STI from MTF: STIPA-like aggregation
        # Transmission Index (TI) per MTF value
        tis = [max(-15.0, min(15.0, 10 * math.log10(m / (1 - m + _eps()) + _eps())))
               for m in mtf_vals if m > 0 and m < 1]
        if not tis:
            return None

        sti = float(np.clip((np.mean(tis) + 15) / 30, 0.0, 1.0))
        return sti
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Articulation Index / Speech Intelligibility Index (SII) proxy
# Based on ANSI S3.5-1997, simplified
# ─────────────────────────────────────────────────────────────────────────────

def _compute_sii_proxy(mono: NDArray, sr: int) -> Optional[float]:
    """
    Speech Intelligibility Index proxy (ANSI S3.5 simplified).
    
    The SII is computed in one-third octave bands weighted by their
    importance to speech intelligibility.
    
    Returns value in [0, 1] where:
        1.0 = excellent intelligibility (no noise)
        0.7 = good
        0.5 = fair (about 70% word recognition)
        <0.4 = poor
    """
    try:
        from scipy.signal import stft as scipy_stft, butter, filtfilt

        # 1/3-octave band centers important for speech (ANSI S3.5 Table 3)
        # with corresponding importance weights
        bands = [
            # (center_hz, weight)
            (200,  0.0083), (250, 0.0095), (315, 0.0150), (400, 0.0289),
            (500,  0.0440), (630, 0.0578), (800, 0.0653), (1000, 0.0711),
            (1250, 0.0691), (1600, 0.0781), (2000, 0.0709), (2500, 0.0660),
            (3150, 0.0521), (4000, 0.0371), (5000, 0.0301), (6300, 0.0195),
            (8000, 0.0124),
        ]

        # Estimate per-band SNR using quiet vs active frames
        frame_len = int(0.025 * sr)
        hop = int(0.010 * sr)

        sii_sum = 0.0
        weight_sum = 0.0

        for fc, w in bands:
            lo = fc / (10 ** 0.05)  # 1/3 oct lower edge
            hi = fc * (10 ** 0.05)  # 1/3 oct upper edge
            if hi >= sr / 2.1 or lo < 10:
                continue

            try:
                b, a = butter(2, [lo / (sr / 2), hi / (sr / 2)], btype="band")
                band = filtfilt(b, a, mono.astype(np.float64))
            except Exception:
                continue

            # Frame energies
            frame_energies = np.array([
                float(np.mean(band[i:i + frame_len] ** 2))
                for i in range(0, len(band) - frame_len, hop)
            ])
            if len(frame_energies) < 4:
                continue

            # Noise: 10th percentile frames
            noise_e = float(np.percentile(frame_energies, 10)) + _eps()
            # Signal: 80th percentile frames
            signal_e = float(np.percentile(frame_energies, 80)) + _eps()

            # Band SNR (dB)
            snr_db = 10 * math.log10(signal_e / noise_e)
            # Clip to [-15, +15] per ANSI S3.5
            snr_clipped = max(-15.0, min(15.0, snr_db))
            # Band audibility function: (SNR + 15) / 30
            band_ai = (snr_clipped + 15.0) / 30.0

            sii_sum += w * band_ai
            weight_sum += w

        if weight_sum <= 0:
            return None

        sii = float(np.clip(sii_sum / weight_sum, 0.0, 1.0))
        return sii
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Clarity Index C50 and C80 (room acoustics)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_clarity(mono: NDArray, sr: int) -> dict[str, Optional[float]]:
    """
    Clarity indices C50 and C80 (dB).
    
    C50 = 10 log10(E_early / E_late), where early = 0–50ms, late = 50ms+.
    C80 = same but 0–80ms vs 80ms+.
    
    These are usually computed from an impulse response; here we compute them
    directly from the speech signal as an approximation.
    
    C50 > 0 dB → more direct speech than reverberation (good for intelligibility).
    C50 < 0 dB → reverberation-dominated.
    """
    try:
        early50 = int(0.050 * sr)
        early80 = int(0.080 * sr)

        if len(mono) <= early80:
            return {"C50": None, "C80": None}

        # Use squared signal as proxy for energy
        sq = mono ** 2

        # Window-based: compute over 200ms sliding windows
        win = int(0.200 * sr)
        hop = int(0.050 * sr)

        c50s, c80s = [], []
        for start in range(0, len(sq) - win, hop):
            seg = sq[start:start + win]
            e_early50 = float(seg[:early50].sum()) + _eps()
            e_early80 = float(seg[:early80].sum()) + _eps()
            e_late50  = float(seg[early50:].sum()) + _eps()
            e_late80  = float(seg[early80:].sum()) + _eps()
            c50s.append(10 * math.log10(e_early50 / e_late50))
            c80s.append(10 * math.log10(e_early80 / e_late80))

        return {
            "C50": float(np.mean(c50s)) if c50s else None,
            "C80": float(np.mean(c80s)) if c80s else None,
        }
    except Exception:
        return {"C50": None, "C80": None}


# ─────────────────────────────────────────────────────────────────────────────
# SRMR — Speech-to-Reverberation Modulation energy Ratio (proxy)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_srmr_proxy(mono: NDArray, sr: int) -> Optional[float]:
    """
    SRMR proxy: ratio of modulation energy in speech-rate bands vs
    reverb-dominated low-rate bands.
    
    True SRMR (Falk et al.) requires gammatone filterbank; this proxy uses
    octave-band envelopes and checks modulation rate distribution.
    
    Returns: SRMR proxy value (higher = less reverberant / cleaner speech).
    Reference: anechoic speech ~4–8, reverberant room ~2–4.
    """
    try:
        from scipy.signal import butter, filtfilt, medfilt

        octave_centers = [500, 1000, 2000, 4000]
        # Speech modulation: 2–16 Hz. Reverberation adds energy at 0.5–2 Hz.
        speech_mod_lo, speech_mod_hi = 2.0, 16.0
        reverb_mod_lo, reverb_mod_hi = 0.5, 2.0

        speech_energy_total = 0.0
        reverb_energy_total = 0.0

        for fc in octave_centers:
            if fc >= sr / 2.5:
                continue
            lo = fc / math.sqrt(2)
            hi = min(fc * math.sqrt(2), sr / 2.5)
            if lo < 1 or hi >= sr / 2:
                continue
            try:
                b, a = butter(2, [lo / (sr / 2), hi / (sr / 2)], btype="band")
                band = filtfilt(b, a, mono.astype(np.float64))
            except Exception:
                continue

            # Envelope
            env = np.abs(band)
            win = max(1, int(0.01 * sr) | 1)
            env_smooth = medfilt(env, kernel_size=min(win, len(env) | 1))

            n = len(env_smooth)
            if n < 64:
                continue

            env_fft = np.fft.rfft(env_smooth - env_smooth.mean(), n=n)
            freqs = np.fft.rfftfreq(n, d=1.0 / sr)
            psd = np.abs(env_fft) ** 2

            spm = (freqs >= speech_mod_lo) & (freqs <= speech_mod_hi)
            rvb = (freqs >= reverb_mod_lo) & (freqs < speech_mod_lo)

            if spm.any():
                speech_energy_total += float(psd[spm].sum())
            if rvb.any():
                reverb_energy_total += float(psd[rvb].sum())

        if reverb_energy_total < _eps():
            return None

        srmr = speech_energy_total / reverb_energy_total
        return float(srmr)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Background noise characterization (stationary vs non-stationary)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_noise_type(mono: NDArray, sr: int) -> dict[str, object]:
    """
    Characterize background noise type.
    
    Returns:
        noise_stationarity: 0 = highly non-stationary, 1 = perfectly stationary
        noise_type: "stationary" | "non-stationary" | "impulsive" | "low"
        impulse_count: number of detected impulsive noise events
    """
    try:
        frame_len = int(0.025 * sr)
        hop = int(0.010 * sr)
        frame_energies = np.array([
            float(np.mean(mono[i:i + frame_len] ** 2))
            for i in range(0, len(mono) - frame_len, hop)
        ])

        if len(frame_energies) < 10:
            return {"noise_stationarity": None, "noise_type": "unknown", "impulse_count": 0}

        # Noise frames: below 20th percentile energy
        noise_threshold = float(np.percentile(frame_energies, 20))
        noise_frames = frame_energies[frame_energies <= noise_threshold * 2]

        if len(noise_frames) < 5:
            return {"noise_stationarity": 1.0, "noise_type": "low", "impulse_count": 0}

        # Stationarity = 1 - (std / mean) of noise frames, normalized
        noise_mean = float(np.mean(noise_frames)) + _eps()
        noise_std = float(np.std(noise_frames))
        cv = noise_std / noise_mean  # coefficient of variation
        stationarity = float(np.clip(1.0 - cv / 2.0, 0.0, 1.0))

        # Impulse detection: frames >> mean by 20 dB
        mean_e = float(np.mean(frame_energies))
        impulse_mask = frame_energies > mean_e * 100  # 20 dB above mean
        impulse_count = int(np.sum(impulse_mask))

        if impulse_count > 3:
            noise_type = "impulsive"
        elif stationarity > 0.7:
            noise_type = "stationary"
        else:
            noise_type = "non-stationary"

        return {
            "noise_stationarity": stationarity,
            "noise_type": noise_type,
            "impulse_count": impulse_count,
        }
    except Exception:
        return {"noise_stationarity": None, "noise_type": "unknown", "impulse_count": 0}


# ─────────────────────────────────────────────────────────────────────────────
# Vocal effort / spectral tilt (Ltas-based)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_vocal_effort(mono: NDArray, sr: int) -> Optional[str]:
    """
    Estimate vocal effort level from long-term average spectrum (LTAS) tilt.
    
    At normal effort: spectral tilt ~-6 dB/octave.
    Raised voice: flatter tilt (less negative slope).
    Whisper: steeper tilt.
    """
    try:
        from scipy.signal import stft as scipy_stft
        f, _, Zxx = scipy_stft(mono, fs=sr, nperseg=2048, noverlap=1536)
        power_mean = np.abs(Zxx).mean(axis=1) ** 2

        valid = (f > 150) & (f < min(5000, sr / 2.1))
        if valid.sum() < 5:
            return None

        log_f = np.log10(f[valid] + _eps())
        log_p = np.log10(power_mean[valid] + _eps())
        slope = float(np.polyfit(log_f, log_p, 1)[0])

        # Slope in dB/octave ≈ slope_loglog × 3 × 10 × log10(2) / log10(10)
        # Approximate: slope × 10 (in log10 space per decade of frequency)
        if slope < -4.0:
            return "whisper/soft"
        elif slope < -2.5:
            return "normal"
        elif slope < -1.0:
            return "raised"
        else:
            return "loud/shouting"
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# DNSMOS-inspired 3-component proxy (OVRL, SIG, BAK)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_dnsmos_proxy(mono: NDArray, sr: int) -> dict[str, Optional[float]]:
    """
    Proxy for DNSMOS P.835 three-component MOS scores.
    
    True DNSMOS uses Microsoft's ONNX model. This proxy uses acoustic
    features to approximate the three components:
        SIG  = signal quality (speech clarity, P.808 SIG)
        BAK  = background noise quality (P.808 BAK)
        OVRL = overall quality (P.808 OVRL)
    
    Scale: 1.0 – 5.0 (higher = better).
    """
    try:
        from scipy.signal import stft as scipy_stft

        # Work at 16 kHz for speech analysis
        if sr != 16000:
            mono = _resample(mono, sr, 16000)
            sr = 16000

        f, _, Zxx = scipy_stft(mono, fs=sr, nperseg=512, noverlap=384)
        mag = np.abs(Zxx)
        power = mag ** 2

        frame_energy = power.sum(axis=0)
        active = frame_energy > float(np.percentile(frame_energy, 20))

        if active.sum() == 0:
            return {"DNSMOS_SIG": 1.0, "DNSMOS_BAK": 1.0, "DNSMOS_OVRL": 1.0}

        # Noise floor (quiet frames)
        noise_e = float(np.mean(frame_energy[~active])) + _eps()
        signal_e = float(np.mean(frame_energy[active])) + _eps()
        snr_db = 10 * math.log10(signal_e / noise_e)

        # Spectral flatness of noise (flat = white noise, low score for BAK)
        if (~active).any():
            noise_spec = power[:, ~active].mean(axis=1) + _eps()
            geo = math.exp(float(np.mean(np.log(noise_spec))))
            arith = float(np.mean(noise_spec))
            flatness = geo / (arith + _eps())
            bak_flatness_bonus = float(np.clip(1.0 - flatness * 2, 0.0, 1.0))
        else:
            bak_flatness_bonus = 0.5

        # Spectral centroid of speech (300-3400 Hz speech band content)
        speech_mask = (f >= 300) & (f <= 3400)
        if speech_mask.any() and active.any():
            speech_ratio = float(np.mean(
                power[speech_mask, :][:, active].sum(axis=0) /
                (power[:, active].sum(axis=0) + _eps())
            ))
        else:
            speech_ratio = 0.5

        # --- SIG (signal quality) ---
        # High SNR + speech band energy → good SIG
        snr_score = float(np.clip(snr_db / 30.0, 0.0, 1.0))
        sig_raw = snr_score * 0.6 + speech_ratio * 0.4
        dnsmos_sig = float(np.clip(1.0 + sig_raw * 4.0, 1.0, 5.0))

        # --- BAK (background noise) ---
        # Flat noise is worse. Low noise floor is better.
        noise_db = 10 * math.log10(noise_e + _eps())
        noise_level_score = float(np.clip(1.0 + (noise_db + 60) / 30, 0.0, 1.0))
        bak_raw = noise_level_score * 0.5 + bak_flatness_bonus * 0.5
        dnsmos_bak = float(np.clip(1.0 + bak_raw * 4.0, 1.0, 5.0))

        # --- OVRL (overall) ---
        dnsmos_ovrl = float(np.clip(0.5 * dnsmos_sig + 0.5 * dnsmos_bak, 1.0, 5.0))

        return {
            "DNSMOS_SIG": round(dnsmos_sig, 2),
            "DNSMOS_BAK": round(dnsmos_bak, 2),
            "DNSMOS_OVRL": round(dnsmos_ovrl, 2),
        }
    except Exception as e:
        return {"DNSMOS_SIG": None, "DNSMOS_BAK": None, "DNSMOS_OVRL": None}


# ─────────────────────────────────────────────────────────────────────────────
# Articulation Rate (refined speaking rate from inter-stress intervals)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_articulation_rate(mono: NDArray, sr: int) -> dict[str, Optional[float]]:
    """
    Articulation rate: speaking rate excluding pauses.
    Uses energy envelope to detect syllable nuclei and compute rate only
    during speech-active intervals.
    """
    try:
        from scipy.signal import medfilt

        frame_len = int(0.025 * sr)
        hop = int(0.005 * sr)
        energies = np.array([
            float(np.mean(mono[i:i + frame_len] ** 2))
            for i in range(0, len(mono) - frame_len, hop)
        ])
        smooth = medfilt(energies, kernel_size=min(31, len(energies) | 1))

        # Active frames
        threshold = float(np.percentile(smooth, 40))
        active = smooth > threshold

        # Active duration
        active_duration_s = float(active.sum()) * hop / sr

        if active_duration_s < 0.3:
            return {"articulation_rate": None, "speech_time_fraction": None}

        # Syllable nuclei in active portions only
        peaks = []
        for i in range(1, len(smooth) - 1):
            if (active[i] and
                    smooth[i] > smooth[i - 1] and
                    smooth[i] > smooth[i + 1] and
                    smooth[i] > threshold * 1.5):
                peaks.append(i)

        total_duration = len(mono) / sr
        speech_fraction = active_duration_s / total_duration

        if len(peaks) < 2:
            return {
                "articulation_rate": None,
                "speech_time_fraction": speech_fraction,
            }

        # Rate only during active speech
        articulation_rate = len(peaks) / active_duration_s

        return {
            "articulation_rate": articulation_rate,
            "speech_time_fraction": speech_fraction,
        }
    except Exception:
        return {"articulation_rate": None, "speech_time_fraction": None}


# ─────────────────────────────────────────────────────────────────────────────
# Main compute function
# ─────────────────────────────────────────────────────────────────────────────

def compute_speech_quality(
    audio: NDArray,
    sr: int,
    ref_audio=None,
    ref_sr=None,
) -> list[MetricResult]:
    """
    Compute all speech-conditional quality & intelligibility metrics.
    Called only when speech is detected.
    """
    mono = _to_mono(audio)
    results: list[MetricResult] = []

    # ── Voice quality perturbation measures ──────────────────────────────────
    jitter = _compute_jitter(mono, sr)
    shimmer = _compute_shimmer(mono, sr)
    nhr = _compute_nhr(mono, sr)

    j_local = jitter.get("jitter_local")
    j_rap   = jitter.get("jitter_rap")
    j_ppq5  = jitter.get("jitter_ppq5")
    j_ddp   = jitter.get("jitter_ddp")
    s_local_pct = shimmer.get("shimmer_local_pct")
    s_local_db  = shimmer.get("shimmer_local_db")
    s_apq3  = shimmer.get("shimmer_apq3")
    s_apq5  = shimmer.get("shimmer_apq5")
    s_dda   = shimmer.get("shimmer_dda")

    results += [
        MetricResult(
            "Jitter (Local)", j_local, "%",
            "Cycle-to-cycle pitch period variability. Normal voice: <1.04%", GROUP,
            higher_is_better=False,
            reference_range=(0.0, 1.04),
            warning="Elevated jitter — may indicate hoarseness or noise" if j_local and j_local > 1.04 else None,
        ),
        MetricResult(
            "Jitter (RAP)", j_rap, "%",
            "Relative average perturbation (3-point smoothed). Normal: <0.68%", GROUP,
            higher_is_better=False,
            reference_range=(0.0, 0.68),
            warning="Elevated RAP" if j_rap and j_rap > 0.68 else None,
        ),
        MetricResult(
            "Jitter (PPQ5)", j_ppq5, "%",
            "5-point period perturbation quotient. Normal: <0.84%", GROUP,
            higher_is_better=False,
            reference_range=(0.0, 0.84),
        ),
        MetricResult(
            "Jitter (DDP)", j_ddp, "%",
            "Difference of differences of periods (= 3×RAP). Normal: <2.04%", GROUP,
            higher_is_better=False,
            reference_range=(0.0, 2.04),
        ),
        MetricResult(
            "Shimmer (Local)", s_local_pct, "%",
            "Cycle-to-cycle amplitude variability. Normal voice: <3.81%", GROUP,
            higher_is_better=False,
            reference_range=(0.0, 3.81),
            warning="Elevated shimmer — may indicate breathiness or noise" if s_local_pct and s_local_pct > 3.81 else None,
        ),
        MetricResult(
            "Shimmer (Local, dB)", s_local_db, "dB",
            "Cycle-to-cycle amplitude variability in dB. Normal: <0.35 dB", GROUP,
            higher_is_better=False,
            reference_range=(0.0, 0.35),
        ),
        MetricResult(
            "Shimmer (APQ3)", s_apq3, "%",
            "3-point amplitude perturbation quotient. Normal: <3.07%", GROUP,
            higher_is_better=False,
            reference_range=(0.0, 3.07),
        ),
        MetricResult(
            "Shimmer (APQ5)", s_apq5, "%",
            "5-point amplitude perturbation quotient. Normal: <4.23%", GROUP,
            higher_is_better=False,
            reference_range=(0.0, 4.23),
        ),
        MetricResult(
            "Shimmer (DDA)", s_dda, "%",
            "Difference of differences of amplitudes (= 3×APQ3). Normal: <9.22%", GROUP,
            higher_is_better=False,
            reference_range=(0.0, 9.22),
        ),
        MetricResult(
            "Noise-to-Harmonics Ratio (NHR)", nhr, "",
            "Fraction of noise energy to harmonic energy. Lower is cleaner. Normal: <0.19", GROUP,
            higher_is_better=False,
            reference_range=(0.0, 0.19),
            warning="High NHR — rough or breathy voice" if nhr and nhr > 0.19 else None,
        ),
    ]

    # ── Intelligibility estimates ─────────────────────────────────────────────
    sti = _compute_sti_proxy(mono, sr)
    sii = _compute_sii_proxy(mono, sr)

    def _sti_label(v):
        if v is None: return None
        if v >= 0.75: return None
        if v >= 0.60: return "Good — minor intelligibility impact"
        if v >= 0.45: return "Fair — noticeable intelligibility loss"
        return "Poor — significant intelligibility loss"

    def _sii_label(v):
        if v is None: return None
        if v >= 0.75: return None
        if v >= 0.50: return "Fair intelligibility (~70% word recognition)"
        return "Poor intelligibility"

    results += [
        MetricResult(
            "STI Proxy (Speech Transmission Index)", sti, "[0–1]",
            "Modulation transfer function based intelligibility estimate. "
            "Excellent:≥0.75, Good:0.60–0.75, Fair:0.45–0.60, Poor:<0.45. "
            "Proxy only — use STIPA measurement for certified results.", GROUP,
            higher_is_better=True,
            reference_range=(0.60, 1.0),
            warning=_sti_label(sti),
        ),
        MetricResult(
            "SII Proxy (Speech Intelligibility Index)", sii, "[0–1]",
            "ANSI S3.5-inspired per-band SNR weighted intelligibility score. "
            ">0.75 = excellent, 0.50–0.75 = good, <0.50 = fair/poor.", GROUP,
            higher_is_better=True,
            reference_range=(0.75, 1.0),
            warning=_sii_label(sii),
        ),
    ]

    # ── Room acoustics ────────────────────────────────────────────────────────
    clarity = _compute_clarity(mono, sr)
    c50 = clarity.get("C50")
    c80 = clarity.get("C80")

    results += [
        MetricResult(
            "Clarity Index C50", c50, "dB",
            "Early (0–50ms) to late energy ratio. >0 dB = direct sound dominant (good for speech).", GROUP,
            higher_is_better=True,
            reference_range=(0.0, 20.0),
            warning="Possible reverb/room effect on speech" if c50 is not None and c50 < 0 else None,
        ),
        MetricResult(
            "Clarity Index C80", c80, "dB",
            "Early (0–80ms) to late energy ratio. Wider window than C50.", GROUP,
            higher_is_better=True,
        ),
    ]

    # ── SRMR ─────────────────────────────────────────────────────────────────
    srmr = _compute_srmr_proxy(mono, sr)
    results.append(MetricResult(
        "SRMR Proxy (Speech-to-Reverberation Modulation energy Ratio)", srmr, "",
        "Ratio of syllabic (2–16 Hz) to reverberant (<2 Hz) envelope modulation energy. "
        "Higher = less reverberant. Anechoic speech: ~4–8, reverberant room: ~2–4.", GROUP,
        higher_is_better=True,
        reference_range=(4.0, 20.0),
        warning="Possible reverberation / room noise" if srmr is not None and srmr < 3.0 else None,
    ))

    # ── DNSMOS proxy (3-component) ────────────────────────────────────────────
    dnsmos = _compute_dnsmos_proxy(mono, sr)
    results += [
        MetricResult(
            "DNSMOS SIG (Signal Quality Proxy)", dnsmos.get("DNSMOS_SIG"), "MOS [1–5]",
            "P.835 SIG proxy: perceived speech signal quality. 5 = excellent.", GROUP,
            higher_is_better=True,
            reference_range=(3.5, 5.0),
            warning="Note: proxy — install Microsoft DNSMOS ONNX for accurate scores",
        ),
        MetricResult(
            "DNSMOS BAK (Background Noise Proxy)", dnsmos.get("DNSMOS_BAK"), "MOS [1–5]",
            "P.835 BAK proxy: perceived background noise intrusiveness. 5 = not noticeable.", GROUP,
            higher_is_better=True,
            reference_range=(3.5, 5.0),
        ),
        MetricResult(
            "DNSMOS OVRL (Overall Quality Proxy)", dnsmos.get("DNSMOS_OVRL"), "MOS [1–5]",
            "P.835 OVRL proxy: overall perceived quality. 5 = excellent.", GROUP,
            higher_is_better=True,
            reference_range=(3.5, 5.0),
        ),
    ]

    # ── Articulation rate ─────────────────────────────────────────────────────
    art = _compute_articulation_rate(mono, sr)
    art_rate = art.get("articulation_rate")
    speech_frac = art.get("speech_time_fraction")

    results += [
        MetricResult(
            "Articulation Rate", art_rate, "syllables/s",
            "Speaking rate during active speech only (pauses excluded). Typical: 3.5–7 syl/s.", GROUP,
            reference_range=(3.5, 7.0),
            warning="Unusually fast — may affect intelligibility" if art_rate and art_rate > 8.0 else (
                "Unusually slow" if art_rate and art_rate < 2.0 else None),
        ),
        MetricResult(
            "Speech Time Fraction", speech_frac * 100 if speech_frac is not None else None, "%",
            "Fraction of total duration containing active speech (pauses excluded).", GROUP,
            reference_range=(30.0, 85.0),
        ),
    ]

    # ── Vocal effort ─────────────────────────────────────────────────────────
    effort = _compute_vocal_effort(mono, sr)
    results.append(MetricResult(
        "Vocal Effort Estimate", effort, "",
        "Estimated vocal effort from long-term average spectral tilt. "
        "Whisper/soft → normal → raised → loud/shouting.", GROUP,
    ))

    # ── Noise characterization ────────────────────────────────────────────────
    noise_info = _compute_noise_type(mono, sr)
    results += [
        MetricResult(
            "Background Noise Type", noise_info.get("noise_type"), "",
            "Classified as stationary, non-stationary, or impulsive based on "
            "noise-frame energy variance. Stationary noise is more predictable/filterable.", GROUP,
        ),
        MetricResult(
            "Noise Stationarity", noise_info.get("noise_stationarity"), "[0–1]",
            "Stationarity of noise floor. 1 = perfectly stationary (e.g. AWGN), "
            "0 = highly variable (traffic, crowd). Higher → easier to denoise.", GROUP,
            higher_is_better=True,
        ),
        MetricResult(
            "Impulsive Noise Events", noise_info.get("impulse_count"), "events",
            "Count of frames with impulsive noise (>20 dB above mean). "
            "Clicks, pops, interference.", GROUP,
            higher_is_better=False,
            warning="Impulsive noise detected — clicks/pops present"
            if (noise_info.get("impulse_count") or 0) > 2 else None,
        ),
    ]

    return results

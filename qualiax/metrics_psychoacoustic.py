"""
Psychoacoustic metrics: roughness, sensory dissonance, sharpness, tonality.

References
----------
- Sethares (1993) "Local consonance and the relationship between timbre and scale"
- Vassilakis (2001/2003) "Perceptual and Physical Properties of Amplitude Fluctuation"
- Zwicker & Fastl (1990) "Psychoacoustics: Facts and Models"
- Von Bismarck (1974) "Sharpness as an attribute of the timbre of steady sounds"
"""
from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray
from scipy.signal import find_peaks, stft as scipy_stft

from .models import MetricResult


# ─── Bark / critical-band utilities ──────────────────────────────────────────

def _hz_to_bark(f: NDArray) -> NDArray:
    """Traunmüller (1990) Bark formula."""
    return (26.81 * f / (1960.0 + f)) - 0.53


def _critical_bandwidth(f: float) -> float:
    """Zwicker critical bandwidth at frequency f (Hz)."""
    return 25.0 + 75.0 * (1.0 + 1.4 * (f / 1000.0) ** 2) ** 0.69


# ─── Shared STFT helper ───────────────────────────────────────────────────────

def _mean_spectrum(audio: NDArray, sr: int, n_fft: int = 2048) -> tuple[NDArray, NDArray]:
    """Return (freqs, mean_magnitude) averaged over frames."""
    hop  = n_fft // 4
    win  = np.hanning(n_fft)
    mags = []
    for i in range(0, len(audio) - n_fft, hop):
        mags.append(np.abs(np.fft.rfft(audio[i:i + n_fft] * win)))
    if not mags:
        return np.array([]), np.array([])
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return freqs, np.mean(np.stack(mags), axis=0)


def _top_partials(freqs: NDArray, mag: NDArray, n: int = 40) -> tuple[NDArray, NDArray]:
    """Extract up to n prominent spectral peaks (frequency, normalized amplitude)."""
    if mag.max() < 1e-12:
        return np.array([]), np.array([])
    peaks, _ = find_peaks(mag, height=mag.max() * 0.015, distance=2)
    if len(peaks) == 0:
        return np.array([]), np.array([])
    peaks  = peaks[np.argsort(mag[peaks])[-n:]]
    f_p    = freqs[peaks]
    a_p    = mag[peaks] / (mag.max() + 1e-14)
    return f_p, a_p


# ─── Roughness (Vassilakis 2001) ─────────────────────────────────────────────

def compute_roughness(audio: NDArray, sr: int) -> list[MetricResult]:
    """
    Perceptual roughness from amplitude modulation between spectral partial pairs.

    Based on Vassilakis (2001) eq. 6.1 — considers both amplitude and frequency
    proximity of partial pairs relative to the critical bandwidth.
    """
    n_use = min(len(audio), int(2.5 * sr))
    freqs, mag = _mean_spectrum(audio[:n_use], sr, n_fft=2048)
    if len(mag) == 0:
        return []

    f_p, a_p = _top_partials(freqs, mag, n=50)
    if len(f_p) < 2:
        return [MetricResult("Roughness", 0.0, "asper (rel.)",
                             "Perceptual roughness (AM in critical bands)", "psychoacoustic")]

    roughness = 0.0
    for i in range(len(f_p)):
        for j in range(i + 1, len(f_p)):
            f1, f2 = f_p[i], f_p[j]
            a1, a2 = a_p[i], a_p[j]
            if f1 <= 0:
                continue
            # Critical bandwidth at lower partial
            cbw = _critical_bandwidth(f1)
            x   = abs(f2 - f1) / (cbw + 1e-14)

            # Roughness curve (peaks at x ≈ 0.25, Plomp & Levelt 1965)
            if x > 0:
                r_curve = x ** 2 * math.exp(-(x / 0.25) ** 2)
            else:
                continue

            # Vassilakis amplitude weight: geometric and arithmetic mean ratio
            amp_wt = (a1 * a2) ** 0.1 * (
                2 * a1 * a2 / (a1 ** 2 + a2 ** 2 + 1e-14)
            ) ** 3.11

            roughness += amp_wt * r_curve

    return [MetricResult(
        "Roughness", round(float(roughness), 4), "asper (rel.)",
        "Vassilakis (2001) perceptual roughness. High = grating/harsh quality. "
        "Low = smooth/pure tones.", "psychoacoustic",
        higher_is_better=False,
    )]


# ─── Sensory Dissonance (Sethares 1993) ──────────────────────────────────────

def compute_dissonance(audio: NDArray, sr: int) -> list[MetricResult]:
    """
    Sethares (1993) sensory dissonance model based on spectral partial pairs.

    Dissonance arises when partials beat in the critical band (CBW ≈ minor 2nd).
    Musical consonance is the absence of dissonance.
    """
    n_use = min(len(audio), int(2.5 * sr))
    freqs, mag = _mean_spectrum(audio[:n_use], sr, n_fft=2048)
    if len(mag) == 0:
        return []

    f_p, a_p = _top_partials(freqs, mag, n=30)
    if len(f_p) < 2:
        return [MetricResult("Sensory Dissonance", 0.0, "(rel.)",
                             "Sethares (1993) sensory dissonance model", "psychoacoustic")]

    # Sethares parameters (fitted to Plomp & Levelt data)
    b1, b2 = 3.5, 5.75
    diss = 0.0
    for i in range(len(f_p)):
        for j in range(i + 1, len(f_p)):
            f1 = min(f_p[i], f_p[j])
            f2 = max(f_p[i], f_p[j])
            a1, a2 = a_p[i], a_p[j]
            if f1 <= 0:
                continue
            # Scale factor depends on frequency of lower partial
            s    = 0.24 / (0.0207 * f1 + 18.96)
            diff = abs(f2 - f1)
            diss += a1 * a2 * (math.exp(-b1 * s * diff) - math.exp(-b2 * s * diff))

    return [MetricResult(
        "Sensory Dissonance", round(max(0.0, float(diss)), 4), "(rel.)",
        "Sethares (1993) dissonance model. High = beating/clashing partials. "
        "Low = consonant/harmonic spectrum.", "psychoacoustic",
        higher_is_better=False,
    )]


# ─── Spectral Sharpness (Zwicker & Fastl 1990) ───────────────────────────────

def compute_sharpness(audio: NDArray, sr: int) -> list[MetricResult]:
    """
    Zwicker & Fastl (1990) spectral sharpness in acum.

    Sharpness measures the proportion of high-frequency energy relative to total
    loudness. 1 acum is defined as a narrow-band noise at 1 kHz, 60 dB SPL.
    Typical speech: 1–3 acum. Higher = brighter / more sibilance.
    """
    n_fft = 4096
    win   = np.hanning(n_fft)
    mag_sq = np.zeros(n_fft // 2 + 1)
    n = 0
    for i in range(0, len(audio) - n_fft, n_fft // 2):
        mag_sq += np.abs(np.fft.rfft(audio[i:i + n_fft] * win)) ** 2
        n += 1
    if n == 0:
        return []
    mag_sq /= n

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    # Bark-band specific loudness (simplified)
    bark_edges = np.arange(0, 24.5, 0.5)
    bark_f     = _hz_to_bark(freqs[1:])
    mag_sq_    = mag_sq[1:]

    N_specific = np.zeros(len(bark_edges) - 1)
    z_centers  = (bark_edges[:-1] + bark_edges[1:]) / 2

    for b in range(len(bark_edges) - 1):
        mask = (bark_f >= bark_edges[b]) & (bark_f < bark_edges[b + 1])
        if np.any(mask):
            N_specific[b] = np.sqrt(np.mean(mag_sq_[mask]))

    N_total = N_specific.sum() + 1e-14

    # Zwicker & Fastl sharpness weighting g(z): 1 up to 16 Bark, then
    # 0.066*exp(0.171 z), which is continuous (~1.02) at the 16 Bark knee.
    # (With the knee at 15 Bark the weight dropped to 0.86 and sharpness fell
    # as frequency rose between ~2.7 and 3.2 kHz.)
    g = np.where(z_centers <= 16.0, 1.0, 0.066 * np.exp(0.171 * z_centers))

    sharpness = 0.11 * float(np.sum(N_specific * g * z_centers) / N_total)

    return [MetricResult(
        "Sharpness", round(sharpness, 4), "acum",
        "Zwicker & Fastl spectral sharpness. "
        "1 acum = narrow-band noise at 1 kHz. Higher = brighter / more sibilance. "
        "Typical speech: 1–3 acum.", "psychoacoustic",
    )]


# ─── Tonality ─────────────────────────────────────────────────────────────────

def compute_tonality(audio: NDArray, sr: int) -> list[MetricResult]:
    """
    Tonal vs. noise-like character via spectral flatness measure (SFM).

    SFM = geometric mean / arithmetic mean of power spectrum.
    0 = pure sine, 1 = flat (white) spectrum.  Tonality = max(0, 1 – SFM).
    """
    n_fft = 4096
    win   = np.hanning(n_fft)
    sfm_vals = []

    for i in range(0, len(audio) - n_fft, n_fft // 2):
        frame = audio[i:i + n_fft]
        power = np.abs(np.fft.rfft(frame * win)) ** 2
        power = power[power > 1e-14]
        if len(power) < 4:
            continue
        geo   = math.exp(float(np.mean(np.log(power))))
        arith = float(np.mean(power))
        sfm_vals.append(geo / (arith + 1e-14))

    if not sfm_vals:
        return []

    sfm      = float(np.mean(sfm_vals))
    tonality = max(0.0, 1.0 - sfm)

    return [
        MetricResult(
            "Spectral Flatness (SFM)", round(sfm, 4), "0–1",
            "Ratio of geometric to arithmetic mean of power spectrum. "
            "0 = pure tone, 1 = white noise.", "psychoacoustic",
        ),
        MetricResult(
            "Tonality", round(tonality, 4), "0–1",
            "Tonal character derived from SFM. 1 = pure tone, 0 = noise-like. "
            "Voiced speech typically 0.3–0.7.", "psychoacoustic",
        ),
    ]


# ─── Harmonic Strength (salience of harmonic series) ─────────────────────────

def compute_harmonic_strength(audio: NDArray, sr: int) -> list[MetricResult]:
    """
    Harmonicity: how well the spectrum fits a harmonic series.

    Computed as the ratio of energy at detected harmonic overtone positions
    to total spectral energy.  1.0 = perfectly harmonic. 0 = inharmonic noise.
    """
    n_use = min(len(audio), int(2.0 * sr))
    freqs, mag = _mean_spectrum(audio[:n_use], sr, n_fft=4096)
    if len(mag) == 0 or mag.max() < 1e-12:
        return []

    # Find candidate F0 in 60–600 Hz via the dominant peak in cepstrum
    power     = mag ** 2
    log_power = np.log(power + 1e-14)
    cepstrum  = np.abs(np.fft.irfft(log_power))
    q_min     = max(1, int(sr / 600))
    q_max     = int(sr / 60)
    if q_max >= len(cepstrum):
        return []

    best_q  = q_min + int(np.argmax(cepstrum[q_min:q_max]))
    f0_est  = sr / best_q if best_q > 0 else 0.0

    if f0_est < 50 or f0_est > 700:
        return []

    # Score: how much energy lies within ±3% of each harmonic multiple
    freq_res = sr / (2 * (len(freqs) - 1)) * 1  # Hz per bin
    total_e  = float(np.sum(power[1:]))  # skip DC
    harmonic_e = 0.0
    n_harmonics = 0

    for k in range(1, 20):
        fk = k * f0_est
        if fk > freqs[-1]:
            break
        tol   = max(freq_res * 3, fk * 0.03)  # ±3% of frequency
        mask  = (freqs >= fk - tol) & (freqs <= fk + tol)
        if np.any(mask):
            harmonic_e += float(np.sum(power[mask]))
            n_harmonics += 1

    if total_e < 1e-14 or n_harmonics == 0:
        return []

    harmonic_ratio = min(1.0, harmonic_e / total_e)

    return [
        MetricResult(
            "F0 (cepstral estimate)", round(f0_est, 2), "Hz",
            "Dominant periodicity via cepstrum (mean-spectrum). "
            "Cross-check against per-frame F0 in prosody group.", "psychoacoustic",
        ),
        MetricResult(
            "Harmonicity", round(harmonic_ratio, 4), "0–1",
            "Fraction of spectral energy at harmonic multiples of F0. "
            "1.0 = purely harmonic (e.g. vowel). 0 = noise/inharmonic.", "psychoacoustic",
            higher_is_better=True,
        ),
    ]


# ─── Entry point ─────────────────────────────────────────────────────────────

def compute_psychoacoustic(
    audio: NDArray, sr: int, ref_audio=None, ref_sr=None
) -> list[MetricResult]:
    """Compute all psychoacoustic metrics."""
    mono = audio.mean(axis=0) if audio.ndim == 2 else audio
    results: list[MetricResult] = []

    for fn in (
        compute_roughness,
        compute_dissonance,
        compute_sharpness,
        compute_tonality,
        compute_harmonic_strength,
    ):
        try:
            results.extend(fn(mono, sr))
        except Exception as exc:
            import warnings
            warnings.warn(f"{fn.__name__} failed: {exc}")

    return results

"""
Speaker-characteristic metrics: formant tracking (F1–F4), spectral tilt,
voice quality (CPP, breathiness, creakiness), gender estimation, age estimation.

References
----------
- Hillenbrand et al. (1994) "Acoustic characteristics of American English vowels"
- Linville (2001) "Vocal aging"
- Xue & Deliyski (2001) "Effects of aging on selected acoustic voice parameters"
- Traunmüller & Eriksson (1995) "The frequency range of the voice fundamental"
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import toeplitz, solve
from scipy.signal import find_peaks

from .models import MetricResult


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _to_mono(audio: NDArray) -> NDArray:
    return audio.mean(axis=0) if audio.ndim == 2 else audio


def _frame_signal(audio: NDArray, frame_len: int, hop_len: int) -> NDArray:
    n_frames = max(0, (len(audio) - frame_len) // hop_len)
    if n_frames == 0:
        return np.empty((0, frame_len))
    idx = (np.arange(frame_len)[None, :] +
           hop_len * np.arange(n_frames)[:, None])
    return audio[idx]


# ─── LPC formant tracking ─────────────────────────────────────────────────────

def _autocorrelation(frame: NDArray, order: int) -> NDArray:
    """Biased autocorrelation of frame up to lag `order`."""
    n = len(frame)
    r = np.array([
        float(np.dot(frame[:n - k], frame[k:])) / n
        for k in range(order + 1)
    ])
    return r


def _lpc_coeffs(frame: NDArray, order: int) -> Optional[NDArray]:
    """
    LPC coefficients via Yule-Walker / Levinson-Durbin using scipy.

    Returns coefficients a[1..order] (not a[0]=1.0).
    """
    try:
        r = _autocorrelation(frame, order)
        if abs(r[0]) < 1e-14:
            return None
        R   = toeplitz(r[:order])
        rhs = -r[1:order + 1]
        a   = solve(R, rhs, assume_a='sym')
        return a
    except Exception:
        return None


def _formants_from_lpc(frame: NDArray, sr: int, order: int = 12) -> list[float]:
    """
    Extract formant frequencies from a single frame via LPC root analysis.

    Returns list of formant frequencies (Hz), sorted ascending, with
    bandwidth < 600 Hz and 50 < f < 5500 Hz.
    """
    # Pre-emphasis (reduces spectral tilt, improves LPC stability)
    frame = np.append(frame[0], np.diff(frame))
    frame *= np.hanning(len(frame))

    a = _lpc_coeffs(frame, order)
    if a is None:
        return []

    # All-pole polynomial: 1 + a[0]*z^-1 + ... + a[p-1]*z^-p
    poly   = np.concatenate([[1.0], a])
    roots  = np.roots(poly)

    formants = []
    for root in roots:
        if np.imag(root) < 0:          # keep only upper half of unit circle
            continue
        angle = float(np.angle(root))
        freq  = angle * sr / (2 * math.pi)
        bw    = -math.log(abs(root) + 1e-14) * sr / math.pi
        if 50 < freq < 5500 and 0 < bw < 600:
            formants.append(freq)

    return sorted(formants)


def compute_formants(audio: NDArray, sr: int) -> list[MetricResult]:
    """
    Track F1–F4 formant frequencies via LPC (order 12) over voiced frames.

    Returns median F1–F4 across voiced frames.  F1 ≈ vowel openness,
    F2 ≈ backness, F3 ≈ voice timbre/quality.
    """
    frame_len = int(0.025 * sr)
    hop_len   = int(0.010 * sr)
    mono      = _to_mono(audio)

    if len(mono) < frame_len:
        return []

    frames  = _frame_signal(mono, frame_len, hop_len)
    buckets: list[list[float]] = [[] for _ in range(4)]

    for frame in frames:
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms < 5e-4:            # skip silent frames
            continue
        fmts = _formants_from_lpc(frame, sr)
        for k, f in enumerate(fmts[:4]):
            buckets[k].append(f)

    labels    = ["F1", "F2", "F3", "F4"]
    min_count = 5
    results   = []

    for k, label in enumerate(labels):
        if len(buckets[k]) >= min_count:
            median_f = float(np.median(buckets[k]))
            std_f    = float(np.std(buckets[k]))
            results.append(MetricResult(
                f"{label} (Formant Frequency)",
                round(median_f, 1), "Hz",
                f"Median {label} over voiced frames (LPC order 12, std={std_f:.0f} Hz). "
                f"F1: vowel height, F2: vowel backness, F3: voice timbre/quality.",
                "speaker",
            ))

    return results


# ─── Spectral tilt ────────────────────────────────────────────────────────────

def compute_spectral_tilt(audio: NDArray, sr: int) -> tuple[list[MetricResult], float]:
    """
    Linear spectral tilt in dB/octave (power spectrum, 100 Hz–Nyquist).

    Typical speech: −6 to −12 dB/oct.
    Steep negative tilt (< −15) → low-pass / bass-heavy.
    Shallow tilt (> −4) → bright / noisy.
    """
    n_fft = 4096
    win   = np.hanning(n_fft)
    mag_sq = np.zeros(n_fft // 2 + 1)
    n = 0
    for i in range(0, len(audio) - n_fft, n_fft // 2):
        mag_sq += np.abs(np.fft.rfft(audio[i:i + n_fft] * win)) ** 2
        n += 1
    if n == 0:
        return [], 0.0
    mag_sq /= n

    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    mask  = (freqs >= 100) & (freqs <= sr / 2)
    f, p  = freqs[mask], mag_sq[mask]
    if len(f) < 4 or p.sum() < 1e-14:
        return [], 0.0

    log_f  = np.log2(f + 1e-14)
    log_p  = 10 * np.log10(p + 1e-14)
    slope, _ = np.polyfit(log_f, log_p, 1)
    tilt     = float(slope)

    return [MetricResult(
        "Spectral Tilt", round(tilt, 2), "dB/oct",
        "Linear slope of power spectrum in dB/octave (100 Hz–Nyquist). "
        "Typical speech: −6 to −12 dB/oct. Steeper = more bass. Shallower = brighter.",
        "speaker",
    )], tilt


# ─── Cepstral Peak Prominence (CPP) ──────────────────────────────────────────

def compute_cpp(audio: NDArray, sr: int) -> tuple[list[MetricResult], Optional[float]]:
    """
    Cepstral Peak Prominence (CPP) — Hillenbrand et al. (1994).

    CPP = height of cepstral peak above a regression baseline.
    High CPP → clear, modal voice.  Low CPP → breathy / dysphonic.
    Typical voiced speech: CPP > 5 dB.  Breathy: CPP < 3 dB.
    """
    n_fft  = 1024
    hop    = n_fft // 4
    q_min  = max(1, sr // 500)   # quefrency range for F0
    q_max  = min(n_fft // 4, sr // 50)
    cpp_vals = []
    mono   = _to_mono(audio)

    for i in range(0, len(mono) - n_fft, hop):
        frame = mono[i:i + n_fft] * np.hanning(n_fft)
        if np.sqrt(np.mean(frame ** 2)) < 1e-4:
            continue

        power = np.abs(np.fft.rfft(frame)) ** 2
        power = np.maximum(power, 1e-14)
        cepstrum = np.abs(np.fft.irfft(np.log(power)))

        if q_max >= len(cepstrum) // 2 or q_max <= q_min:
            continue

        region  = cepstrum[q_min:q_max]
        q_idx   = np.arange(q_min, q_max)
        peak_i  = int(np.argmax(region))
        peak_v  = float(region[peak_i])

        if len(q_idx) > 3:
            bl = float(np.polyval(np.polyfit(q_idx, region, 1), q_idx[peak_i]))
            cpp_vals.append(peak_v - bl)

    if not cpp_vals:
        return [], None

    cpp_mean = float(np.mean(cpp_vals))
    results  = [MetricResult(
        "Cepstral Peak Prominence (CPP)", round(cpp_mean, 4), "dB",
        "Hillenbrand (1994) CPP. Higher = clearer modal voice. "
        "< 3 dB may indicate breathiness or dysphonia.", "speaker",
        higher_is_better=True,
        reference_range=(5.0, 25.0),
        warning="Low CPP — possible breathy or dysphonic voice" if cpp_mean < 3.0 else None,
    )]
    return results, cpp_mean


# ─── Creak (vocal fry) detection ─────────────────────────────────────────────

_CREAK_F0_HZ = 80.0
_CREAK_MIN_F0_HZ = 20.0
_CREAK_MAX_F0_HZ = 500.0
_CREAK_VOICING_THRESHOLD = 0.5


def _frame_dominant_f0(segment: NDArray, win_len: int, sr: int, tau_min: int, tau_max: int) -> Optional[float]:
    """Dominant F0 of ``segment[:win_len]`` via the normalized cross-correlation (NCCF).

    The NCCF compares a fixed window with a lagged copy of equal length, so it
    does not taper with lag (unlike the biased autocorrelation). Every multiple
    of the period correlates almost equally well, so the *shortest* lag whose
    peak is within 90% of the best one is taken as the period.
    """
    ref = segment[:win_len]
    ref_energy = float(np.dot(ref, ref))
    if ref_energy <= 0:
        return None
    num = np.correlate(segment[: win_len + tau_max], ref, mode="valid")  # lags 0..tau_max
    sq = np.concatenate([[0.0], np.cumsum(segment[: win_len + tau_max] ** 2)])
    lag_energy = sq[win_len:win_len + tau_max + 1] - sq[: tau_max + 1]
    nccf = num / np.sqrt(ref_energy * np.maximum(lag_energy, 1e-20))
    region = nccf[tau_min:tau_max + 1]
    peaks, props = find_peaks(region, height=_CREAK_VOICING_THRESHOLD)
    if len(peaks) == 0:
        return None
    heights = props["peak_heights"]
    best = int(peaks[np.argmax(heights >= 0.9 * np.max(heights))])
    return sr / float(tau_min + best)


def _count_creaky_frames(mono: NDArray, sr: int) -> tuple[int, int]:
    """Return (creaky_frames, active_frames): active frames whose F0 is below 80 Hz."""
    win_len = int(0.025 * sr)
    hop_len = int(0.010 * sr)
    tau_min = max(1, int(sr / _CREAK_MAX_F0_HZ))
    tau_max = int(sr / _CREAK_MIN_F0_HZ)
    span = win_len + tau_max
    creak_count = 0
    active_count = 0
    for start in range(0, len(mono) - span + 1, hop_len):
        segment = np.asarray(mono[start:start + span], dtype=np.float64)
        if float(np.sqrt(np.mean(segment[:win_len] ** 2))) < 1e-3:
            continue
        active_count += 1
        f0 = _frame_dominant_f0(segment, win_len, sr, tau_min, tau_max)
        if f0 is not None and f0 < _CREAK_F0_HZ:
            creak_count += 1
    return creak_count, active_count


# ─── Voice quality ────────────────────────────────────────────────────────────

def compute_voice_quality(audio: NDArray, sr: int) -> list[MetricResult]:
    """
    Breathiness index (from CPP), creakiness / vocal fry ratio.
    """
    mono    = _to_mono(audio)
    results = []

    # CPP-based breathiness
    cpp_results, cpp_val = compute_cpp(mono, sr)
    results.extend(cpp_results)

    if cpp_val is not None:
        # Breathiness: 0 (clear) – 1 (highly breathy)
        breathiness = max(0.0, min(1.0, 1.0 - cpp_val / 20.0))
        results.append(MetricResult(
            "Breathiness Index", round(breathiness, 4), "0–1",
            "0 = modal/clear voice, 1 = highly breathy. Derived from CPP.", "speaker",
            higher_is_better=False,
            warning="Elevated breathiness detected" if breathiness > 0.5 else None,
        ))

    # Creakiness / vocal fry: F0 < 80 Hz in otherwise active frames
    creak_count, active_count = _count_creaky_frames(mono, sr)

    if active_count > 0:
        creak_ratio = creak_count / active_count
        results.append(MetricResult(
            "Creakiness (Vocal Fry) Ratio", round(creak_ratio * 100, 2), "%",
            "Fraction of active frames with F0 < 80 Hz (vocal fry / creak register). "
            "Common in glottalization or pathological creak.", "speaker",
            higher_is_better=False,
        ))

    return results


# ─── Gender estimation ────────────────────────────────────────────────────────

def _estimate_gender(
    f0_mean: float, voiced_ratio_pct: float
) -> list[MetricResult]:
    """
    Heuristic gender estimate from F0 mean frequency.

    Empirical distributions (Traunmüller & Eriksson 1995):
      Adult male:   mean ~120 Hz, typical range 85–180 Hz
      Adult female: mean ~210 Hz, typical range 160–300 Hz
      Overlap zone: 145–180 Hz (ambiguous)
      Children:     F0 > 250 Hz
    """
    if f0_mean <= 0 or voiced_ratio_pct < 8.0:
        label = "indeterminate (insufficient voiced speech)"
        conf  = 0.0
    elif f0_mean > 260:
        label = "child / high soprano"
        conf  = min(1.0, (f0_mean - 260) / 60.0)
    elif f0_mean > 180:
        label = "female"
        conf  = min(1.0, (f0_mean - 180) / 50.0)
    elif f0_mean < 145:
        label = "male"
        conf  = min(1.0, (145.0 - f0_mean) / 60.0)
    else:
        label = "ambiguous (145–180 Hz overlap zone)"
        conf  = 0.0

    return [
        MetricResult(
            "Estimated Gender", label, "",
            "Heuristic from F0 mean. 145–180 Hz is a biological overlap zone; "
            "many voices are ambiguous. Not a classification model.", "speaker",
            confidence="heuristic",
            calibration_note="Gender estimate is inferred from pitch statistics and should never be treated as identity ground truth.",
        ),
        MetricResult(
            "Gender Confidence", round(conf, 2), "0–1",
            "Distance from F0 overlap zone as a proxy for confidence in gender estimate.", "speaker",
            confidence="heuristic",
            calibration_note="Confidence only reflects distance from the heuristic overlap zone, not actual classifier accuracy.",
        ),
    ]


# ─── Age estimation ───────────────────────────────────────────────────────────

def _estimate_age(
    f0_mean: float,
    f0_std: float,
    jitter_pct: Optional[float],
    shimmer_pct: Optional[float],
    hnr_db: Optional[float],
    spectral_tilt: Optional[float],
) -> list[MetricResult]:
    """
    Very rough age-range estimate from acoustic features.

    Key age-related vocal changes (Linville 2001; Xue & Deliyski 2001):
    - Jitter and shimmer increase with age (especially 60+)
    - HNR decreases with age
    - Spectral tilt steepens
    - F0 range narrows
    - Children: F0 > 250 Hz

    Returns a broad range (± 15 years) as this is heuristic only.
    """
    if f0_mean > 250:
        return [MetricResult(
            "Estimated Age Range", "child (< 12 years est.)", "",
            "Heuristic age estimate. F0 > 250 Hz is consistent with pre-adolescent voice.",
            "speaker",
            confidence="heuristic",
            calibration_note="Age range is inferred from acoustic correlates and is not a biometric or medical assessment.",
        )]

    score   = 0
    evidence = []

    if jitter_pct is not None:
        if jitter_pct > 2.0:
            score += 3; evidence.append("high jitter")
        elif jitter_pct > 1.0:
            score += 1; evidence.append("mildly elevated jitter")
        elif jitter_pct < 0.4:
            score -= 1

    if shimmer_pct is not None:
        if shimmer_pct > 6.0:
            score += 3; evidence.append("high shimmer")
        elif shimmer_pct > 3.0:
            score += 1; evidence.append("mildly elevated shimmer")
        elif shimmer_pct < 1.5:
            score -= 1

    if hnr_db is not None:
        if hnr_db < 8:
            score += 2; evidence.append("low HNR")
        elif hnr_db > 22:
            score -= 1

    if spectral_tilt is not None:
        if spectral_tilt < -14:
            score += 1; evidence.append("steep spectral tilt")
        elif spectral_tilt > -4:
            score -= 1

    if f0_std is not None:
        if f0_std < 8:
            score += 1; evidence.append("low pitch variability")

    if score <= -2:
        age_range = "estimated 15–30 years"
    elif score <= 0:
        age_range = "estimated 25–45 years"
    elif score <= 2:
        age_range = "estimated 35–55 years"
    elif score <= 4:
        age_range = "estimated 50–70 years"
    else:
        age_range = "estimated 60+ years"

    note = f"Acoustic evidence: {', '.join(evidence)}" if evidence else "baseline features"

    return [MetricResult(
        "Estimated Age Range", age_range, "",
        f"Heuristic acoustic age estimate (± ~15 years). {note}. "
        "Not a medical or biometric assessment.", "speaker",
        confidence="heuristic",
        calibration_note="Age range is inferred from acoustic correlates and is not a biometric or medical assessment.",
    )]


# ─── Entry point ─────────────────────────────────────────────────────────────

def compute_speaker(
    audio: NDArray,
    sr: int,
    ref_audio=None,
    ref_sr=None,
    *,
    include_demographics: bool = False,
) -> list[MetricResult]:
    """
    Full speaker characteristic analysis:
    formants, spectral tilt, CPP, breathiness, creakiness, gender, age.
    """
    from .metrics_prosody import compute_f0_track, _jitter_local, _shimmer_local

    mono    = _to_mono(audio)
    results = []

    # ── Formants ────────────────────────────────────────────────────────────
    try:
        results.extend(compute_formants(mono, sr))
    except Exception as e:
        warnings.warn(f"Formant tracking failed: {e}")

    # ── Spectral tilt ────────────────────────────────────────────────────────
    tilt = None
    try:
        tilt_results, tilt = compute_spectral_tilt(mono, sr)
        results.extend(tilt_results)
    except Exception as e:
        warnings.warn(f"Spectral tilt failed: {e}")

    # ── Voice quality (CPP, breathiness, creakiness) ─────────────────────────
    try:
        results.extend(compute_voice_quality(mono, sr))
    except Exception as e:
        warnings.warn(f"Voice quality failed: {e}")

    # ── F0 stats (re-use prosody tracker) ────────────────────────────────────
    f0_mean     = 0.0
    f0_std      = 0.0
    voiced_pct  = 0.0
    jitter_pct  = None
    shimmer_pct = None
    hnr_db      = None

    try:
        f0_arr, _ = compute_f0_track(mono, sr)
        if len(f0_arr) > 0:
            voiced = f0_arr[f0_arr > 0]
            if len(voiced) >= 4:
                f0_mean    = float(np.mean(voiced))
                f0_std     = float(np.std(voiced))
                voiced_pct = len(voiced) / len(f0_arr) * 100

                j = _jitter_local(f0_arr)
                if j is not None:
                    jitter_pct = j * 100

                s = _shimmer_local(mono, sr, f0_arr)
                if s is not None:
                    shimmer_pct = s * 100
    except Exception as e:
        warnings.warn(f"F0 extraction in speaker module failed: {e}")

    # HNR from noise group (recompute locally if needed)
    try:
        from .metrics import _compute_hnr
        hnr_db = _compute_hnr(mono, sr)
    except Exception as e:
        warnings.warn(f"Speaker HNR fallback failed: {e}")

    if include_demographics:
        # ── Gender ───────────────────────────────────────────────────────────
        results.extend(_estimate_gender(f0_mean, voiced_pct))

        # ── Age ──────────────────────────────────────────────────────────────
        results.extend(_estimate_age(f0_mean, f0_std, jitter_pct, shimmer_pct, hnr_db, tilt))

    return results

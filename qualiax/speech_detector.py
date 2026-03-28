"""
Speech presence detection.

Uses multiple independent acoustic cues to determine whether an audio file
contains speech (as opposed to music, ambient noise, silence, etc.).
Each cue votes with a weight; the combined score is a confidence in [0, 1].
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray


# ─────────────────────────────────────────────────────────────────────────────
# Result type
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SpeechDetectionResult:
    is_speech: bool                    # Hard decision (confidence >= threshold)
    confidence: float                  # 0.0 – 1.0
    threshold: float                   # Decision threshold used
    cues: dict[str, float] = field(default_factory=dict)    # Individual evidence scores
    f0_hz: Optional[float] = None      # Detected pitch if speech (None otherwise)
    speech_fraction: float = 0.0       # Fraction of frames classified as speech
    content_type: str = "unknown"      # "speech" | "music" | "noise" | "silence" | "mixed" | "unknown"

    def summary(self) -> str:
        pct = f"{self.confidence * 100:.1f}%"
        return (
            f"{self.content_type.upper()} "
            f"(speech confidence: {pct}, "
            f"active speech: {self.speech_fraction * 100:.1f}%)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _to_mono(audio: NDArray) -> NDArray:
    return audio.mean(axis=0) if audio.ndim == 2 else audio


def _eps() -> float:
    return 1e-10


def _frames(mono: NDArray, sr: int, win_ms=25, hop_ms=10):
    """Yield overlapping frames."""
    win = int(win_ms * sr / 1000)
    hop = int(hop_ms * sr / 1000)
    for i in range(0, len(mono) - win, hop):
        yield mono[i:i + win]


# ─────────────────────────────────────────────────────────────────────────────
# Individual cues  (each returns a score in [0, 1])
# ─────────────────────────────────────────────────────────────────────────────

def _cue_speech_band_energy(mono: NDArray, sr: int) -> float:
    """
    Speech energy concentration cue.
    Real speech concentrates 60-85 % of its energy in 300–3400 Hz.
    Music and wideband noise do not.
    Returns higher score when energy is concentrated in speech band.
    """
    try:
        from scipy.signal import stft as scipy_stft
        f, _, Zxx = scipy_stft(mono, fs=sr, nperseg=512, noverlap=384)
        power = np.abs(Zxx) ** 2 + _eps()
        total = power.sum(axis=0) + _eps()
        speech_mask = (f >= 300) & (f <= 3400)
        if not speech_mask.any():
            return 0.0
        speech_ratio = float(np.mean(power[speech_mask, :].sum(axis=0) / total))
        # Speech: 0.6-0.85. Music: 0.3-0.6. Noise: ~0.5 (uniform). Silence: undefined.
        # Score peaks around 0.72.
        return float(np.clip(1.0 - abs(speech_ratio - 0.72) / 0.4, 0.0, 1.0))
    except Exception:
        return 0.5


def _cue_f0_in_speech_range(mono: NDArray, sr: int) -> tuple[float, Optional[float]]:
    """
    Fundamental frequency cue.
    Human speech F0: 80–300 Hz (male 80–180 Hz, female 160–300 Hz).
    Returns (score, detected_f0_or_None).
    """
    try:
        frame_len = int(0.04 * sr)
        hop = int(0.01 * sr)
        min_lag = max(1, int(sr / 500))
        max_lag = int(sr / 60)

        pitches = []
        for start in range(0, len(mono) - frame_len, hop):
            frame = mono[start:start + frame_len]
            frame = frame - frame.mean()
            if np.max(np.abs(frame)) < 0.005:
                continue
            n = 2 * frame_len
            fft = np.fft.rfft(frame, n=n)
            acf = np.fft.irfft(fft * np.conj(fft))[:frame_len]
            if acf[0] <= 0:
                continue
            acf /= (acf[0] + _eps())
            if max_lag >= len(acf):
                continue
            region = acf[min_lag:max_lag]
            peak_idx = int(np.argmax(region))
            r_peak = float(region[peak_idx])
            lag = peak_idx + min_lag
            if r_peak > 0.3:
                pitches.append(sr / lag)

        if not pitches:
            return 0.0, None

        f0 = float(np.median(pitches))
        pitch_fraction = len(pitches) / max(1, (len(mono) - frame_len) // hop)

        # Score: high if F0 in speech range AND enough voiced frames
        in_range = 80 <= f0 <= 320
        score = float(np.clip(pitch_fraction * 2.0, 0.0, 1.0)) if in_range else 0.0
        # Partial credit for near-range
        if not in_range and (50 <= f0 <= 500):
            score = 0.25

        return score, f0 if in_range else None
    except Exception:
        return 0.5, None


def _cue_zcr_bimodality(mono: NDArray, sr: int) -> float:
    """
    ZCR bimodality cue.
    Speech alternates between voiced segments (low ZCR ~0.02–0.08) and
    unvoiced/fricative segments (high ZCR ~0.15–0.40).
    The histogram of ZCR across frames should be bimodal.
    Music tends to have more uniform or unimodal ZCR.
    White noise has uniformly high ZCR.
    """
    try:
        zcrs = []
        for frame in _frames(mono, sr):
            zcr = float(np.mean(np.abs(np.diff(np.sign(frame))) > 0))
            zcrs.append(zcr)

        if len(zcrs) < 10:
            return 0.5

        zcrs = np.array(zcrs)
        # Fraction of frames in the voiced ZCR range (0.01–0.10)
        voiced_frac = float(np.mean((zcrs > 0.01) & (zcrs < 0.10)))
        # Fraction of frames in unvoiced ZCR range (0.15–0.45)
        unvoiced_frac = float(np.mean((zcrs > 0.15) & (zcrs < 0.45)))

        # Both fractions should be substantial for speech
        bimodal_score = min(voiced_frac * 2.0, 1.0) * 0.6 + min(unvoiced_frac * 3.0, 1.0) * 0.4
        return float(np.clip(bimodal_score, 0.0, 1.0))
    except Exception:
        return 0.5


def _cue_amplitude_modulation(mono: NDArray, sr: int) -> float:
    """
    Syllabic modulation cue.
    Speech energy envelope is modulated at 2–12 Hz (syllabic rate).
    Music also has some modulation, but at different depths/rates.
    Noise is barely modulated.
    """
    try:
        # Energy envelope via short-time RMS
        frame_len = int(0.025 * sr)
        hop = int(0.005 * sr)
        envelope = np.array([
            float(np.sqrt(np.mean(mono[i:i + frame_len] ** 2)))
            for i in range(0, len(mono) - frame_len, hop)
        ])

        if len(envelope) < 20:
            return 0.5

        # Power spectral density of envelope
        env_sr = sr / hop  # "sample rate" of the envelope signal
        n = len(envelope)
        fft_env = np.fft.rfft(envelope - envelope.mean(), n=n)
        freqs = np.fft.rfftfreq(n, d=1.0 / env_sr)
        psd = np.abs(fft_env) ** 2

        # Energy in syllabic band (2–12 Hz) vs total envelope energy
        syl_mask = (freqs >= 2.0) & (freqs <= 12.0)
        total_mask = freqs >= 0.5
        if not syl_mask.any() or not total_mask.any():
            return 0.5

        syl_energy = float(psd[syl_mask].sum())
        total_energy = float(psd[total_mask].sum()) + _eps()
        syl_ratio = syl_energy / total_energy

        # Speech: syl_ratio typically 0.25–0.65
        # Noise: low (<0.15), Music: variable
        return float(np.clip((syl_ratio - 0.1) / 0.4, 0.0, 1.0))
    except Exception:
        return 0.5


def _cue_spectral_tilt(mono: NDArray, sr: int) -> float:
    """
    Spectral tilt cue.
    Speech has a characteristic -6 dB/octave tilt (more energy in low freqs).
    White noise is flat. Music varies widely.
    """
    try:
        from scipy.signal import stft as scipy_stft
        f, _, Zxx = scipy_stft(mono, fs=sr, nperseg=1024, noverlap=768)
        power_mean = np.abs(Zxx).mean(axis=1) ** 2

        # Fit linear regression to log-log power
        valid = (f > 100) & (f < sr / 2.5)
        if valid.sum() < 5:
            return 0.5

        log_f = np.log10(f[valid] + _eps())
        log_p = np.log10(power_mean[valid] + _eps())
        slope = float(np.polyfit(log_f, log_p, 1)[0])

        # Speech slope: typically -2.0 to -0.5 (in log-log)
        # Flat noise: ~0, White noise: ~0, Music: varies
        if -3.5 <= slope <= -0.3:
            # Score peaks around -1.2 (typical voiced speech)
            return float(np.clip(1.0 - abs(slope + 1.2) / 2.0, 0.0, 1.0))
        return 0.1
    except Exception:
        return 0.5


def _cue_voicing_continuity(mono: NDArray, sr: int) -> float:
    """
    Voicing continuity cue.
    Speech has sustained voiced segments (>80ms) interspersed with pauses.
    Checks for the presence of continuous low-ZCR, medium-energy runs.
    """
    try:
        frame_len = int(0.025 * sr)
        hop = int(0.010 * sr)
        frames_list = list(_frames(mono, sr, win_ms=25, hop_ms=10))
        if len(frames_list) < 5:
            return 0.3

        is_voiced = []
        for frame in frames_list:
            e = float(np.mean(frame ** 2))
            zcr = float(np.mean(np.abs(np.diff(np.sign(frame))) > 0))
            is_voiced.append(e > 0.0001 and zcr < 0.12)

        # Find longest run of voiced frames
        max_run = 0
        run = 0
        for v in is_voiced:
            if v:
                run += 1
                max_run = max(max_run, run)
            else:
                run = 0

        # Voiced run duration in ms
        run_ms = max_run * 10  # 10ms hop
        # Speech: typically has runs of 100–500ms. Music: also has runs. Noise: short runs.
        score = float(np.clip(run_ms / 300.0, 0.0, 1.0))
        return score
    except Exception:
        return 0.5


def _cue_music_discriminator(mono: NDArray, sr: int) -> float:
    """
    Music discrimination cue (returns LOWER score if audio looks like music).
    Uses spectral flux regularity and sub-bass presence.
    Music tends to have high sub-bass energy and rhythmically regular flux.
    Returns 1.0 = definitely NOT music, 0.0 = definitely IS music.
    """
    try:
        from scipy.signal import stft as scipy_stft
        f, _, Zxx = scipy_stft(mono, fs=sr, nperseg=1024, noverlap=768)
        mag = np.abs(Zxx)
        power = mag ** 2

        total = power.sum(axis=0) + _eps()
        # Sub-bass ratio (music often has strong kick/bass drum below 80 Hz)
        sub_mask = f < 80
        sub_ratio = float(np.mean(power[sub_mask, :].sum(axis=0) / total)) if sub_mask.any() else 0.0

        # High-frequency content ratio (music has more HF than speech)
        hf_mask = f > 6000
        hf_ratio = float(np.mean(power[hf_mask, :].sum(axis=0) / total)) if hf_mask.any() else 0.0

        # Spectral flux (music is often more spectrally dynamic)
        flux = float(np.mean(np.abs(np.diff(mag, axis=1))))

        # Music indicators: high sub-bass AND high HF AND high flux
        music_score = (
            min(sub_ratio / 0.05, 1.0) * 0.3 +
            min(hf_ratio / 0.15, 1.0) * 0.4 +
            min(flux / 0.05, 1.0) * 0.3
        )
        # Return "NOT music" score = inverse
        return float(np.clip(1.0 - music_score * 0.7, 0.0, 1.0))
    except Exception:
        return 0.7


def _cue_silence_check(mono: NDArray, sr: int) -> float:
    """Returns 0.0 if file is mostly silence (can't be speech), 1.0 otherwise."""
    rms = float(np.sqrt(np.mean(mono ** 2)))
    if rms < 0.001:
        return 0.0  # Nearly silent
    # Fraction of near-silent frames
    frame_rms = np.array([
        float(np.sqrt(np.mean(mono[i:i + int(0.025 * sr)] ** 2)))
        for i in range(0, len(mono) - int(0.025 * sr), int(0.010 * sr))
    ])
    active_frac = float(np.mean(frame_rms > 0.005))
    return float(np.clip(active_frac * 2.0, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Speech fraction (energy-based VAD)
# ─────────────────────────────────────────────────────────────────────────────

def _speech_fraction(mono: NDArray, sr: int) -> float:
    """Fraction of frames that appear to contain voiced speech."""
    try:
        voiced = []
        for frame in _frames(mono, sr):
            e = float(np.mean(frame ** 2))
            zcr = float(np.mean(np.abs(np.diff(np.sign(frame))) > 0))
            voiced.append(e > 0.0001 and zcr < 0.15)
        return float(np.mean(voiced)) if voiced else 0.0
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main detector
# ─────────────────────────────────────────────────────────────────────────────

SPEECH_THRESHOLD = 0.45  # Confidence threshold for hard speech decision


def detect_speech(
    audio: NDArray,
    sr: int,
    threshold: float = SPEECH_THRESHOLD,
) -> SpeechDetectionResult:
    """
    Run multi-cue speech detection.

    Returns a SpeechDetectionResult with:
        - is_speech: bool
        - confidence: float [0, 1]
        - cues: dict of individual evidence scores
        - f0_hz: detected fundamental frequency (or None)
        - speech_fraction: fraction of frames with voiced speech
        - content_type: "speech" | "music" | "noise" | "silence" | "mixed"
    """
    mono = _to_mono(audio)

    # Silence guard
    silence_score = _cue_silence_check(mono, sr)
    if silence_score < 0.1:
        return SpeechDetectionResult(
            is_speech=False,
            confidence=0.0,
            threshold=threshold,
            cues={"silence": silence_score},
            speech_fraction=0.0,
            content_type="silence",
        )

    # Run all cues (suppress warnings internally)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        c_band  = _cue_speech_band_energy(mono, sr)
        c_f0, f0_hz = _cue_f0_in_speech_range(mono, sr)
        c_zcr   = _cue_zcr_bimodality(mono, sr)
        c_mod   = _cue_amplitude_modulation(mono, sr)
        c_tilt  = _cue_spectral_tilt(mono, sr)
        c_voice = _cue_voicing_continuity(mono, sr)
        c_nomus = _cue_music_discriminator(mono, sr)
        spf     = _speech_fraction(mono, sr)

    cues = {
        "speech_band_energy": c_band,
        "f0_in_speech_range": c_f0,
        "zcr_bimodality":     c_zcr,
        "syllabic_modulation": c_mod,
        "spectral_tilt":      c_tilt,
        "voicing_continuity": c_voice,
        "not_music":          c_nomus,
    }

    # Weighted combination (weights tuned for real-world performance)
    weights = {
        "speech_band_energy": 1.5,
        "f0_in_speech_range": 2.5,   # Strong evidence
        "zcr_bimodality":     1.5,
        "syllabic_modulation": 1.5,
        "spectral_tilt":      1.0,
        "voicing_continuity": 2.0,   # Strong evidence
        "not_music":          1.0,
    }
    total_w = sum(weights.values())
    confidence = sum(cues[k] * weights[k] for k in cues) / total_w
    confidence = float(np.clip(confidence, 0.0, 1.0))

    is_speech = confidence >= threshold

    # Content type heuristic
    if confidence >= 0.65:
        content_type = "speech"
    elif confidence >= threshold:
        content_type = "speech"  # borderline speech
    elif c_nomus < 0.4 and confidence < 0.35:
        content_type = "music"
    elif silence_score < 0.4:
        content_type = "silence"
    elif confidence < 0.25:
        content_type = "noise"
    else:
        content_type = "mixed"

    return SpeechDetectionResult(
        is_speech=is_speech,
        confidence=confidence,
        threshold=threshold,
        cues=cues,
        f0_hz=f0_hz,
        speech_fraction=spf,
        content_type=content_type,
    )

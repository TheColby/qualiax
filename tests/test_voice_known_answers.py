"""Known-answer tests for prosody, psychoacoustic and speaker metrics and for
the speech-presence detector, using synthetic signals with known F0,
perturbation, spectral slope and formants.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.signal import lfilter

from qualiax import metrics_prosody as P
from qualiax import metrics_psychoacoustic as PA
from qualiax import metrics_speaker as S
from qualiax.speech_detector import detect_speech

SR = 16_000


def _prosody(x, sr=SR):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {m.name: m.value for m in P.compute_prosody(x, sr)}


# ─────────────────────────────────────────────────────────────────────────────
# F0 tracking
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("f0", [100.0, 120.0, 150.0, 220.0, 300.0])
def test_f0_of_harmonic_tone(sig, f0):
    v = _prosody(sig.harmonic(f0, 2.0))

    # A 100 Hz voice was never voiced before: the tapered autocorrelation
    # peak stayed under the 0.40 voicing threshold.
    assert v["Voiced Frame Ratio"] > 95.0
    assert v["F0 Mean"] == pytest.approx(f0, rel=0.005)
    assert v["F0 Std"] < 0.5


@pytest.mark.parametrize("f0", [120.0, 220.0])
def test_f0_of_pure_sine(sig, f0):
    # The lag taper used to pull the peak to shorter lags (120 Hz read as 125 Hz).
    assert _prosody(sig.sine(f0, 2.0, amplitude=0.5))["F0 Mean"] == pytest.approx(f0, abs=1.0)


def test_white_noise_is_unvoiced(sig):
    v = _prosody(sig.white_noise(2.0, std=0.1))
    assert v["Voiced Frame Ratio"] < 5.0


def test_f0_slope_of_linear_glide(sig):
    # Instantaneous F0 rises 150 -> 250 Hz over 2 s: slope 50 Hz/s.
    t = sig.time(2.0, SR)
    phase = 2 * np.pi * np.cumsum(150 + 50 * t) / SR
    x = 0.3 * sum(np.sin(k * phase) / k for k in range(1, 8))
    v = _prosody(x)
    assert v["F0 Slope"] == pytest.approx(50.0, abs=2.0)
    assert v["F0 Mean"] == pytest.approx(200.0, abs=3.0)


def test_vibrato_rate_depth_and_range(sig):
    # F0 = 200 + 10 sin(2 pi 5 t): tremor rate 5 Hz, F0 std 10/sqrt(2), range 20 Hz.
    t = sig.time(3.0, SR)
    phase = 2 * np.pi * np.cumsum(200 + 10 * np.sin(2 * np.pi * 5 * t)) / SR
    x = 0.3 * sum(np.sin(k * phase) / k for k in range(1, 8))
    v = _prosody(x)

    assert v["Tremor Rate"] == pytest.approx(5.0, abs=0.35)  # FFT resolution ~0.33 Hz
    assert v["F0 Std"] == pytest.approx(10 / math.sqrt(2), abs=0.5)
    assert v["F0 Range"] == pytest.approx(20.0, abs=2.0)
    assert v["F0 Mean"] == pytest.approx(200.0, abs=1.0)


def test_prosody_speech_rate_counts_syllables(sig):
    assert _prosody(sig.am_tone(200, 4.0, 3.0))["Estimated Speech Rate"] == pytest.approx(4.0, abs=0.3)


# ─────────────────────────────────────────────────────────────────────────────
# jitter / shimmer / NHR
# ─────────────────────────────────────────────────────────────────────────────

def test_periodic_voice_has_near_zero_jitter_and_shimmer(sig):
    for f0 in (120.0, 150.0, 220.0):
        v = _prosody(sig.perturbed_harmonic(f0, 3.0))
        assert v["Jitter (Local)"] < 0.05, f0
        assert v["Jitter (RAP)"] < 0.05, f0
        # Fixed 25 ms RMS windows held a fractional number of periods and read
        # 2-5 % shimmer on a perfectly steady voice (above the 3 % "normal" limit).
        assert v["Shimmer (Local)"] < 0.05, f0
        assert v["Shimmer (APQ3)"] < 0.05, f0


def test_jitter_increases_monotonically_with_period_perturbation(sig):
    levels = [0.0, 0.01, 0.02, 0.04]
    jitters = [_prosody(sig.perturbed_harmonic(150, 3.0, period_jitter=j))["Jitter (Local)"] for j in levels]
    assert all(a < b for a, b in zip(jitters, jitters[1:])), jitters
    assert jitters[-1] > 1.0  # 4 % cycle jitter is clearly above the 1 % "normal" limit


def test_shimmer_increases_monotonically_with_amplitude_perturbation(sig):
    levels = [0.0, 0.02, 0.05, 0.10]
    shimmers = [
        _prosody(sig.perturbed_harmonic(150, 3.0, amplitude_shimmer=s))["Shimmer (Local)"] for s in levels
    ]
    assert all(a < b for a, b in zip(shimmers, shimmers[1:])), shimmers


def test_ddp_is_three_times_rap_and_dda_three_times_apq3(sig):
    # DDP = mean|dT_i+1 - dT_i| and RAP = mean|T_i - (T_i-1 + T_i + T_i+1)/3|,
    # so DDP = 3 RAP identically (likewise DDA = 3 APQ3).
    x = sig.perturbed_harmonic(150, 3.0, period_jitter=0.02, amplitude_shimmer=0.05)
    f0, _ = P.compute_f0_track(x, SR)
    assert P._jitter_ddp(f0) == pytest.approx(3 * P._jitter_rap(f0), rel=1e-9)
    assert P._shimmer_dda(x, SR, f0) == pytest.approx(3 * P._shimmer_apq3(x, SR, f0), rel=1e-9)


def test_nhr_rises_with_added_noise(sig):
    x = sig.harmonic(150, 2.0)
    clean = _prosody(x)["NHR (Noise-to-Harmonics Ratio)"]
    noisy = _prosody(x + sig.white_noise(2.0, std=0.03))["NHR (Noise-to-Harmonics Ratio)"]
    assert clean < 0.01
    assert noisy > clean


def test_insufficient_voicing_returns_only_summary_metrics(sig):
    v = _prosody(sig.white_noise(1.0, std=0.1))
    assert "F0 Mean" not in v
    assert v["F0 Estimator Backend"] == "autocorrelation"


# ─────────────────────────────────────────────────────────────────────────────
# psychoacoustic
# ─────────────────────────────────────────────────────────────────────────────

def _psy(fn, x, sr=SR):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {m.name: m.value for m in fn(x, sr)}


def test_sharpness_of_1khz_tone_is_about_one_acum(sig):
    # 1 acum is defined by narrow-band sound at 1 kHz (z ~ 8.5 Bark):
    # S = 0.11 * z with g(z) = 1 below 16 Bark.
    s = _psy(PA.compute_sharpness, sig.sine(1000, 2.0, amplitude=0.3))["Sharpness"]
    assert s == pytest.approx(0.11 * PA._hz_to_bark(np.array([1000.0]))[0], abs=0.05)


def test_sharpness_never_decreases_with_tone_frequency(sig):
    # g(z) must be non-decreasing; with its knee at 15 Bark it dropped to 0.86
    # and a 2.8 kHz tone read less sharp than a 2.6 kHz tone.
    freqs = [500, 1000, 2000, 2600, 2800, 3000, 3200, 3500, 5000, 7000]
    s = [_psy(PA.compute_sharpness, sig.sine(f, 1.0, amplitude=0.3))["Sharpness"] for f in freqs]
    assert all(a <= b for a, b in zip(s, s[1:])), dict(zip(freqs, s))


def test_tonality_of_tone_and_white_noise(sig):
    tone = _psy(PA.compute_tonality, sig.sine(1003, 2.0, amplitude=0.3))
    noise = _psy(PA.compute_tonality, sig.white_noise(2.0, std=0.1))
    assert tone["Tonality"] > 0.99
    # Exponentially distributed periodogram bins: E[SFM] = exp(-Euler gamma) = 0.5615.
    assert noise["Spectral Flatness (SFM)"] == pytest.approx(math.exp(-0.5772156649), abs=0.03)


def test_harmonicity_and_cepstral_f0(sig):
    harm = _psy(PA.compute_harmonic_strength, sig.harmonic(220, 2.0))
    assert harm["Harmonicity"] > 0.95
    assert harm["F0 (cepstral estimate)"] == pytest.approx(220, abs=3)  # integer quefrency


def test_harmonicity_of_white_noise_equals_tolerance_window_coverage(sig):
    # For a flat spectrum the "harmonic" energy share is simply the fraction of
    # 0..Nyquist covered by the +/-max(3 bins, 3 %) windows around k * F0
    # (k < 20), so noise does not read 0 - it reads that coverage fraction.
    noise = _psy(PA.compute_harmonic_strength, sig.white_noise(2.0, std=0.1, seed=4))
    f0 = noise["F0 (cepstral estimate)"]
    nyquist, bin_hz = SR / 2, SR / 4096
    covered = np.zeros(2049, dtype=bool)
    freqs = np.fft.rfftfreq(4096, 1 / SR)
    for k in range(1, 20):
        if k * f0 > nyquist:
            break
        tol = max(3 * bin_hz, 0.03 * k * f0)
        covered |= (freqs >= k * f0 - tol) & (freqs <= k * f0 + tol)
    assert noise["Harmonicity"] == pytest.approx(covered[1:].mean(), abs=0.05)


def test_roughness_and_dissonance_known_orderings(sig):
    def pair(f1, f2):
        x = sig.sine(f1, 2.0) + sig.sine(f2, 2.0)
        return 0.3 * x / np.max(np.abs(x))

    pure = 0.3 * sig.sine(1003, 2.0)
    beating = pair(2000, 2055)   # ~Sethares' maximum-dissonance spacing near 2 kHz
    octave = pair(2000, 4000)

    for fn, name in ((PA.compute_roughness, "Roughness"), (PA.compute_dissonance, "Sensory Dissonance")):
        r_pure = _psy(fn, pure)[name]
        r_beat = _psy(fn, beating)[name]
        r_oct = _psy(fn, octave)[name]
        assert r_pure == pytest.approx(0.0, abs=1e-3), name
        assert r_oct == pytest.approx(0.0, abs=1e-3), name
        assert r_beat > 10 * max(r_pure, r_oct, 1e-4), name


# ─────────────────────────────────────────────────────────────────────────────
# speaker
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("exponent, expected_db_per_octave", [(0, 0.0), (1, -3.01), (2, -6.02)])
def test_spectral_tilt_of_power_law_noise(sig, exponent, expected_db_per_octave):
    x = sig.power_law_noise(exponent, 8.0)
    _, tilt = S.compute_spectral_tilt(x, SR)
    assert tilt == pytest.approx(expected_db_per_octave, abs=0.3)


def _vowel(formants, bandwidths, f0=120.0, duration_s=1.0, sr=SR):
    """Impulse train at f0 through cascaded two-pole formant resonators."""
    n = int(duration_s * sr)
    src = np.zeros(n)
    src[:: int(round(sr / f0))] = 1.0
    y = src
    for freq, bw in zip(formants, bandwidths):
        r = math.exp(-math.pi * bw / sr)
        theta = 2 * math.pi * freq / sr
        y = lfilter([1 - r], [1, -2 * r * math.cos(theta), r * r], y)
    return 0.3 * y / np.max(np.abs(y))


def test_formants_of_synthetic_vowel():
    # /a/-like vowel: F1 700, F2 1220, F3 2600, F4 3500 Hz. LPC (order 12)
    # biases a low F1 towards nearby harmonics, so F1 gets a wider tolerance.
    x = _vowel([700, 1220, 2600, 3500], [80, 90, 120, 150])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = {m.name: m.value for m in S.compute_formants(x, SR)}

    f1, f2, f3, f4 = (v[f"F{k} (Formant Frequency)"] for k in range(1, 5))
    assert f1 < f2 < f3 < f4
    assert f1 == pytest.approx(700, rel=0.15)
    assert f2 == pytest.approx(1220, rel=0.05)
    assert f3 == pytest.approx(2600, rel=0.05)
    assert f4 == pytest.approx(3500, rel=0.05)


@pytest.mark.parametrize("f0, creaky", [(50.0, True), (65.0, True), (120.0, False), (220.0, False), (300.0, False)])
def test_creakiness_flags_only_sub_80hz_voices(sig, f0, creaky):
    # Creak = active frames with F0 < 80 Hz. Period multiples of a 220 Hz voice
    # (3 x 4.5 ms) used to land in the 20-80 Hz lag window and read 100 % creaky.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = {m.name: m.value for m in S.compute_voice_quality(sig.harmonic(f0, 2.0), SR)}
    ratio = v["Creakiness (Vocal Fry) Ratio"]
    if creaky:
        assert ratio > 90.0
    else:
        assert ratio < 5.0


def test_white_noise_is_not_creaky(sig):
    v = {m.name: m.value for m in S.compute_voice_quality(sig.white_noise(2.0, std=0.1), SR)}
    assert v["Creakiness (Vocal Fry) Ratio"] < 5.0


def test_cpp_of_periodic_voice_lands_in_documented_range(sig):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = {m.name: m for m in S.compute_voice_quality(sig.harmonic(150, 2.0), SR)}
    cpp = v["Cepstral Peak Prominence (CPP)"]
    lo, hi = cpp.reference_range
    assert lo <= cpp.value <= hi
    assert v["Breathiness Index"].value < 0.5


def _aspirated(sig, hnr_db, sr=SR, seed=3):
    voice = sig.harmonic(150, 2.0, sr=sr)
    noise = np.random.default_rng(seed).normal(size=voice.size)
    noise *= np.sqrt(np.mean(voice ** 2) / np.mean(noise ** 2)) * 10 ** (-hnr_db / 20)
    return voice + noise


def test_cpp_falls_monotonically_as_aspiration_noise_rises(sig):
    cpp = [S.compute_cpp(_aspirated(sig, hnr), SR)[1] for hnr in (30, 20, 10, 5, 0)]
    assert all(a > b for a, b in zip(cpp, cpp[1:])), cpp
    assert cpp[0] - cpp[-1] > 5.0


def test_cpp_is_sample_rate_invariant(sig):
    at_16k = S.compute_cpp(sig.harmonic(150, 2.0, sr=16_000), 16_000)[1]
    at_48k = S.compute_cpp(sig.harmonic(150, 2.0, sr=48_000), 48_000)[1]
    assert abs(at_16k - at_48k) < 1.0


def test_breathiness_index_is_linear_in_cpp_between_anchors(sig):
    def breathiness(audio):
        return {m.name: m.value for m in S.compute_voice_quality(audio, SR)}["Breathiness Index"]

    clear, breathy = breathiness(sig.harmonic(150, 2.0)), breathiness(_aspirated(sig, -5))
    assert clear == 0.0
    assert breathy > 0.6
    cpp = S.compute_cpp(_aspirated(sig, 5), SR)[1]
    expected = (S._BREATHINESS_CLEAR_DB - cpp) / (S._BREATHINESS_CLEAR_DB - S._BREATHINESS_BREATHY_DB)
    assert breathiness(_aspirated(sig, 5)) == pytest.approx(min(1.0, max(0.0, expected)), abs=1e-3)


def test_demographic_heuristics_follow_documented_bands():
    labels = {f0: S._estimate_gender(f0, 50.0)[0].value for f0 in (110.0, 160.0, 210.0, 280.0)}
    assert labels[110.0] == "male"
    assert labels[160.0].startswith("ambiguous")
    assert labels[210.0] == "female"
    assert labels[280.0].startswith("child")
    assert S._estimate_gender(200.0, 2.0)[0].value.startswith("indeterminate")
    assert S._estimate_age(300.0, 20.0, None, None, None, None)[0].value.startswith("child")
    young = S._estimate_age(150.0, 20.0, 0.2, 1.0, 25.0, -3.0)[0].value
    old = S._estimate_age(150.0, 5.0, 3.0, 8.0, 5.0, -16.0)[0].value
    assert young == "estimated 15–30 years"
    assert old == "estimated 60+ years"


# ─────────────────────────────────────────────────────────────────────────────
# speech-presence detector
# ─────────────────────────────────────────────────────────────────────────────

NON_SPEECH = {
    "sine 120 Hz": lambda s: s.sine(120, 3.0, amplitude=0.5),
    "sine 220 Hz": lambda s: s.sine(220, 3.0, amplitude=0.5),
    "sine 1 kHz": lambda s: s.sine(1000, 3.0, amplitude=0.5),
    "white noise": lambda s: s.white_noise(3.0, std=0.1),
    "pink noise": lambda s: s.power_law_noise(1, 3.0),
    "brown noise": lambda s: s.power_law_noise(2, 3.0),
    "silence": lambda s: np.zeros(3 * SR),
    "hard-clipped sine": lambda s: np.clip(1.4 * s.sine(440, 3.0), -1.0, 1.0),
}


@pytest.mark.parametrize("name", sorted(NON_SPEECH))
def test_non_speech_signals_are_not_classified_as_speech(sig, name):
    audio = NON_SPEECH[name](sig).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = detect_speech(audio, SR)
    assert not result.is_speech, (name, result.confidence, result.cues)
    assert result.content_type != "speech"
    assert 0.0 <= result.confidence <= 1.0


def test_silence_is_labelled_silence():
    result = detect_speech(np.zeros(SR, dtype=np.float32), SR)
    assert result.content_type == "silence"
    assert result.confidence == 0.0


def test_real_speech_is_classified_as_speech(sample_speech):
    audio, sr = sample_speech
    result = detect_speech(audio, sr)
    assert result.is_speech
    assert result.content_type == "speech"
    assert result.f0_hz is not None and 80 <= result.f0_hz <= 320
    assert "SPEECH" in result.summary()


def test_harmonic_vowel_is_still_speech(sig):
    # The single-tone guard must not reject a harmonic-rich voiced sound.
    rng = np.random.default_rng(2)
    n = 3 * SR
    src = np.zeros(n)
    pos = 0.0
    while pos < n:
        src[int(pos)] = 1.0 + 0.05 * rng.standard_normal()
        pos += SR / 120.0 * (1.0 + 0.01 * rng.standard_normal())
    y = src
    for freq, bw in zip((700, 1220, 2600), (80, 90, 120)):
        r = math.exp(-math.pi * bw / SR)
        y = lfilter([1 - r], [1, -2 * r * math.cos(2 * math.pi * freq / SR), r * r], y)
    y = (0.3 * y / np.max(np.abs(y))).astype(np.float32)
    assert detect_speech(y, SR).is_speech

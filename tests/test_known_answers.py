"""Known-answer tests for the signal-level metric groups.

Every expected value below is derived from first principles or from the
relevant standard (ITU-R BS.1770-4, EBU Tech 3341/3342), never from what the
code happened to output. Tolerances are stated next to each assertion.
"""
from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

from qualiax import gpu
from qualiax import metrics as M
from qualiax.models import FileResult

FS_SINE_RMS_DB = 20 * math.log10(1 / math.sqrt(2))  # -3.0103 dBFS: RMS of a full-scale sine


def _db(x: float) -> float:
    return 20 * math.log10(x)


# ─────────────────────────────────────────────────────────────────────────────
# basic
# ─────────────────────────────────────────────────────────────────────────────

def test_full_scale_sine_levels(sig, values):
    # 997 Hz is not a sub-multiple of 16 kHz, so samples do not sit on zero crossings.
    x = sig.sine(997, 2.0, amplitude=1.0)
    v = values(M.compute_basic(x, 16_000))

    assert v["Peak Level"] == pytest.approx(0.0, abs=0.01)
    assert v["RMS Level"] == pytest.approx(FS_SINE_RMS_DB, abs=0.01)
    assert v["Crest Factor"] == pytest.approx(-FS_SINE_RMS_DB, abs=0.01)  # 3.01 dB
    assert v["Dynamic Range (simple)"] == pytest.approx(3.0103, abs=0.01)
    assert v["DC Offset"] == pytest.approx(0.0, abs=1e-3)
    assert v["Silence Ratio"] < 1.0  # only samples within 0.001 of a zero crossing
    assert v["Duration"] == pytest.approx(2.0)
    assert v["Channels"] == 1


def test_scaled_sine_levels_follow_20log10(sig, values):
    x = sig.sine(997, 1.0, amplitude=0.5)
    v = values(M.compute_basic(x, 16_000))

    assert v["Peak Amplitude"] == pytest.approx(0.5, abs=1e-3)
    assert v["Peak Level"] == pytest.approx(_db(0.5), abs=0.01)             # -6.02 dBFS
    assert v["RMS Level"] == pytest.approx(_db(0.5) + FS_SINE_RMS_DB, abs=0.01)  # -9.03 dBFS
    assert v["Clipping Detected"] == 0


def test_zero_crossing_rate_of_sine_is_twice_frequency_over_sr(sig, values):
    x = sig.sine(997, 2.0, amplitude=0.5)
    v = values(M.compute_basic(x, 16_000))
    # Two sign changes per period.
    assert v["Zero Crossing Rate"] == pytest.approx(2 * 997 / 16_000, rel=0.01)


def test_hard_clipped_sine_sets_clipping_flag(sig, metrics_by_name):
    x = np.clip(1.5 * sig.sine(440, 1.0), -1.0, 1.0)
    m = metrics_by_name(M.compute_basic(x, 16_000))

    assert m["Clipping Detected"].value == 1
    assert m["Clipping Detected"].warning
    assert m["Peak Amplitude"].value == pytest.approx(1.0)


def test_half_silent_signal_has_fifty_percent_silence(sig, values):
    x = sig.sine(997, 2.0, amplitude=0.5)
    x[len(x) // 2:] = 0.0
    v = values(M.compute_basic(x, 16_000))

    assert v["Silence Ratio"] == pytest.approx(50.0, abs=1.0)


def test_dc_offset_is_measured(sig, metrics_by_name):
    x = sig.sine(997, 1.0, amplitude=0.3) + 0.05
    m = metrics_by_name(M.compute_basic(x, 16_000))

    assert m["DC Offset"].value == pytest.approx(0.05, abs=1e-3)
    assert m["DC Offset"].warning == "Significant DC offset detected"


def test_clipping_in_one_stereo_channel_is_detected(sig, values):
    left = np.clip(1.5 * sig.sine(440, 1.0), -1.0, 1.0)
    right = 0.05 * sig.sine(440, 1.0)
    v = values(M.compute_basic(np.vstack([left, right]), 16_000))

    # The (L+R)/2 downmix peaks at 0.525 and would hide the clipped left channel.
    assert v["Clipping Detected"] == 1
    assert v["Peak Amplitude"] == pytest.approx(1.0)
    assert v["Channel 1 Peak Level"] == pytest.approx(0.0, abs=1e-6)


def test_out_of_phase_stereo_peak_is_the_sample_peak(sig, values):
    x = 0.5 * sig.sine(440, 1.0)
    v = values(M.compute_basic(np.vstack([x, -x]), 16_000))

    assert v["Peak Level"] == pytest.approx(_db(0.5), abs=0.01)
    assert v["Stereo Phase Correlation"] == pytest.approx(-1.0, abs=1e-6)
    assert v["Interaural Level Difference (ILD)"] == pytest.approx(0.0, abs=1e-6)


def test_stereo_ild_and_itd_known_answers(sig, values):
    sr = 16_000
    left = 0.4 * sig.sine(300, 1.0, sr)
    delay = 8  # samples = 0.5 ms
    right = 0.2 * np.roll(left / 0.4, delay)
    v = values(M.compute_basic(np.vstack([left, right]), sr))

    assert v["Interaural Level Difference (ILD)"] == pytest.approx(_db(2.0), abs=0.05)  # 6.02 dB
    # Right lags left by 0.5 ms: the correlation peak sits at -delay.
    assert abs(v["Interaural Time Difference (ITD)"]) == pytest.approx(delay / sr * 1000, abs=1e-6)


def test_silence_has_no_nan_basic_values(values):
    v = values(M.compute_basic(np.zeros(16_000), 16_000))

    assert v["Crest Factor"] is None
    assert v["Dynamic Range (simple)"] is None
    for name, value in v.items():
        assert not (isinstance(value, float) and math.isnan(value)), name


# ─────────────────────────────────────────────────────────────────────────────
# loudness (ITU-R BS.1770-4 / EBU R128)
# ─────────────────────────────────────────────────────────────────────────────

def _lufs(x, sr):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {m.name: m for m in M.compute_loudness(x, sr)}


@pytest.mark.parametrize("sr, tol", [(48_000, 0.02), (44_100, 0.02), (16_000, 0.1)])
def test_full_scale_997hz_sine_reads_minus_3_01_lkfs(sig, sr, tol):
    # BS.1770-4: a 0 dBFS 997 Hz sine in one channel reads -3.01 LKFS. The filter
    # is specified at 48 kHz; at 16 kHz the bilinear redesign costs ~0.05 dB,
    # inside the EBU Tech 3341 +/-0.1 LU tolerance.
    x = sig.sine(997, 10.0, sr)
    m = _lufs(x, sr)

    assert m["Integrated Loudness (LUFS)"].value == pytest.approx(-3.01, abs=tol)


def test_loudness_scales_with_level(sig):
    sr = 48_000
    x = sig.sine(997, 10.0, sr, amplitude=10 ** (-20 / 20))
    m = _lufs(x, sr)
    assert m["Integrated Loudness (LUFS)"].value == pytest.approx(-23.01, abs=0.02)


@pytest.mark.parametrize("level_dbfs", [-23.0, -33.0])
def test_ebu_tech3341_stereo_sine_cases(sig, level_dbfs):
    # EBU Tech 3341 cases 1 and 2: stereo 1 kHz sine at X dBFS in both channels
    # reads X LUFS (+/-0.1). Channel energies are summed, not averaged.
    sr = 48_000
    x = sig.sine(1000, 10.0, sr, amplitude=10 ** (level_dbfs / 20))
    m = _lufs(np.vstack([x, x]), sr)

    assert m["Integrated Loudness (LUFS)"].value == pytest.approx(level_dbfs, abs=0.1)
    assert m["Max Short-Term Loudness"].value == pytest.approx(level_dbfs, abs=0.1)


def test_identical_stereo_reads_3db_louder_than_mono(sig):
    sr = 16_000
    x = sig.sine(997, 5.0, sr, amplitude=0.1)
    mono = _lufs(x, sr)["Integrated Loudness (LUFS)"].value
    stereo = _lufs(np.vstack([x, x]), sr)["Integrated Loudness (LUFS)"].value

    assert stereo - mono == pytest.approx(10 * math.log10(2), abs=0.01)


def test_stationary_signal_short_term_equals_integrated_and_lra_is_zero(sig):
    sr = 16_000
    m = _lufs(sig.sine(997, 8.0, sr, amplitude=0.25), sr)

    integrated = m["Integrated Loudness (LUFS)"].value
    assert m["Max Short-Term Loudness"].value == pytest.approx(integrated, abs=0.01)
    assert m["Loudness Range (LRA)"].value == pytest.approx(0.0, abs=0.05)


def test_relative_gate_ignores_quiet_passage(sig):
    # 5 s at -20 dBFS then 5 s at -60 dBFS, 400 ms blocks on a 100 ms hop:
    #   47 loud blocks (-23.01 LUFS), 3 boundary blocks holding 75/50/25 % of the
    #   loud energy, 47 quiet blocks (-63 LUFS).
    # Relative gate = energy mean of all blocks (-26.1 LUFS) - 10 LU = -36.1 LUFS:
    # quiet blocks are dropped, boundary blocks kept, so
    #   L = -23.01 + 10 log10((47 + 1.5) / 50) = -23.14 LUFS.
    # Without the relative gate the result would be about -26.1 LUFS.
    sr = 48_000
    loud = sig.sine(997, 5.0, sr, amplitude=10 ** (-20 / 20))
    quiet = sig.sine(997, 5.0, sr, amplitude=10 ** (-60 / 20))
    m = _lufs(np.concatenate([loud, quiet]), sr)

    expected = -23.0103 + 10 * math.log10((47 + 0.75 + 0.5 + 0.25) / 50)
    assert m["Integrated Loudness (LUFS)"].value == pytest.approx(expected, abs=0.02)


def test_ebu_tech3342_loudness_range_case(sig):
    # EBU Tech 3342 case 1: 20 s at -20 dBFS followed by 20 s at -30 dBFS -> LRA 10 +/- 1 LU.
    sr = 16_000
    a = sig.sine(1000, 20.0, sr, amplitude=10 ** (-20 / 20))
    b = sig.sine(1000, 20.0, sr, amplitude=10 ** (-30 / 20))
    m = _lufs(np.concatenate([a, b]), sr)

    assert m["Loudness Range (LRA)"].value == pytest.approx(10.0, abs=1.0)


def test_signal_below_absolute_gate_has_no_integrated_loudness(sig):
    sr = 16_000
    m = _lufs(sig.sine(997, 2.0, sr, amplitude=10 ** (-80 / 20)), sr)

    lufs = m["Integrated Loudness (LUFS)"]
    assert lufs.value is None
    assert "-70 LUFS" in lufs.warning
    assert m["Loudness Delta vs Spotify"].value is None


def test_short_file_reports_lra_and_short_term_consistently(sig):
    """2.3 s is too short for 3 s windows: both values N/A with the same explanation."""
    sr = 16_000
    m = _lufs(sig.am_tone(300, 3.0, 2.3, sr), sr)

    lra = m["Loudness Range (LRA)"]
    short_term = m["Max Short-Term Loudness"]
    assert lra.value is None
    assert short_term.value is None
    assert lra.warning == short_term.warning
    assert "3 s" in lra.warning and "2.30 s" in lra.warning
    assert m["Integrated Loudness (LUFS)"].value is not None
    assert lra.formatted_value() == "N/A"


def test_silence_loudness_has_no_nan(sig):
    m = _lufs(np.zeros(16_000 * 4), 16_000)

    for metric in m.values():
        assert not (isinstance(metric.value, float) and math.isnan(metric.value)), metric.name
    assert m["Integrated Loudness (LUFS)"].value is None
    assert m["Integrated Loudness (LUFS)"].warning
    assert m["Loudness Range (LRA)"].value is None
    assert m["Max Short-Term Loudness"].value is None


def test_short_file_loudness_serializes_without_nan(sig):
    sr = 16_000
    result = FileResult(path="short.wav", metrics=list(_lufs(sig.sine(997, 2.3, sr, amplitude=0.1), sr).values()))
    from qualiax.reporter import JsonReporter

    rendered = JsonReporter().render([result])
    assert "NaN" not in rendered and "Infinity" not in rendered
    json.loads(rendered)


def test_true_peak_detects_inter_sample_peak(sig):
    # fs/4 sine at 45 degrees: samples land at +/-0.7071 (-3.01 dBFS) while the
    # waveform peaks at 1.0 (0 dBTP). EBU Tech 3341 true-peak tolerance: +0.2/-0.4 dB.
    sr = 48_000
    x = sig.sine(sr / 4, 1.0, sr, phase=np.pi / 4)
    m = _lufs(x, sr)

    assert _db(np.max(np.abs(x))) == pytest.approx(-3.01, abs=0.01)
    assert -0.4 <= m["True Peak"].value <= 0.2
    assert m["True Peak"].warning == "Exceeds -1 dBTP streaming limit"


def test_true_peak_is_never_below_sample_peak(sig):
    x = sig.white_noise(1.0, std=0.2)
    tp = _lufs(x, 16_000)["True Peak"].value
    assert tp >= _db(np.max(np.abs(x))) - 1e-9


def test_true_peak_takes_the_loudest_channel(sig):
    sr = 16_000
    quiet = 0.1 * sig.sine(997, 1.0, sr)
    loud = 0.8 * sig.sine(997, 1.0, sr)
    m = _lufs(np.vstack([quiet, loud]), sr)
    assert m["True Peak"].value == pytest.approx(_db(0.8), abs=0.1)


def test_true_peak_warning_fires_at_exactly_zero_dbtp(sig, monkeypatch):
    # 0.0 dBTP is falsy; the old `if true_peak and ...` check skipped the warning.
    monkeypatch.setattr(gpu, "true_peak", lambda mono, sr, oversample=4: 0.0)
    m = _lufs(0.5 * sig.sine(997, 1.0), 16_000)
    assert m["True Peak"].warning == "Exceeds -1 dBTP streaming limit"


def test_platform_deltas_are_lufs_minus_target(sig):
    sr = 48_000
    m = _lufs(sig.sine(997, 6.0, sr, amplitude=10 ** (-20 / 20)), sr)
    lufs = m["Integrated Loudness (LUFS)"].value

    assert m["Loudness Delta vs Spotify"].value == pytest.approx(round(lufs + 14.0, 2))
    assert m["Loudness Delta vs EBU R128"].value == pytest.approx(round(lufs + 23.0, 2))
    assert m["Loudness Delta vs EBU R128"].warning is None  # -23.01 is within +/-1 LU


# ─────────────────────────────────────────────────────────────────────────────
# spectral
# ─────────────────────────────────────────────────────────────────────────────

def _spectral(x, sr=16_000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {m.name: m.value for m in M.compute_spectral(x, sr)}


def test_pure_tone_spectral_known_answers(sig):
    v = _spectral(sig.sine(1000, 2.0, amplitude=0.5))

    assert v["Spectral Centroid"] == pytest.approx(1000, rel=0.02)
    assert v["Spectral Rolloff (95%)"] == pytest.approx(1000, abs=50)
    assert v["Spectral Bandwidth"] < 50
    assert v["Spectral Flatness"] < -30.0  # linear flatness < 0.001
    assert v["Band Energy: Low-Mid (300–2000 Hz)"] > 99.0


def test_spectral_shape_is_scale_invariant(sig):
    # Centroid, rolloff and flatness describe spectral *shape*; a tone 40 dB
    # quieter must be exactly as tonal (flatness used an absolute floor before).
    loud = _spectral(sig.sine(1000, 2.0, amplitude=0.5))
    quiet = _spectral(sig.sine(1000, 2.0, amplitude=0.005))

    assert quiet["Spectral Flatness"] == pytest.approx(loud["Spectral Flatness"], abs=0.5)
    assert quiet["Spectral Centroid"] == pytest.approx(loud["Spectral Centroid"], rel=1e-3)
    assert quiet["Spectral Rolloff (95%)"] == pytest.approx(loud["Spectral Rolloff (95%)"], rel=1e-3)


def test_white_noise_spectral_known_answers(sig):
    sr = 16_000
    nyquist = sr / 2
    v = _spectral(sig.white_noise(4.0, sr), sr)

    # Flat spectrum on [0, Nyquist]: mean = Nyquist/2, std = Nyquist/sqrt(12).
    assert v["Spectral Centroid"] == pytest.approx(nyquist / 2, rel=0.03)
    assert v["Spectral Bandwidth"] == pytest.approx(nyquist / math.sqrt(12), rel=0.03)
    assert v["Spectral Rolloff (95%)"] == pytest.approx(0.95 * nyquist, rel=0.02)
    # Periodogram bins are exponential: expected flatness is exp(-gamma) = -2.51 dB.
    assert v["Spectral Flatness"] == pytest.approx(10 * math.log10(math.exp(-0.5772156649)), abs=0.3)
    # Band energy shares are proportional to bandwidth.
    assert v["Band Energy: Presence (2–6 kHz)"] == pytest.approx(100 * 4000 / nyquist, abs=2.0)
    assert v["Band Energy: Air (6–20 kHz)"] == pytest.approx(100 * 2000 / nyquist, abs=2.0)
    assert v["Band Energy: Low-Mid (300–2000 Hz)"] == pytest.approx(100 * 1700 / nyquist, abs=2.0)
    assert v["Band Energy: Bass (80–300 Hz)"] == pytest.approx(100 * 220 / nyquist, abs=1.0)
    assert v["Estimated Fundamental (F0)"] is None


def test_spectral_bandwidth_is_per_frame_spread(sig):
    # First half 500 Hz, second half 3 kHz. Each frame is a pure tone, so the
    # per-frame spread (librosa's definition) is small; measuring spread around
    # the file-wide mean centroid would report ~1250 Hz instead.
    sr = 16_000
    x = np.concatenate([sig.sine(500, 1.0, sr, 0.5), sig.sine(3000, 1.0, sr, 0.5)])
    v = _spectral(x, sr)
    assert v["Spectral Bandwidth"] < 200


def test_spectral_f0_of_harmonic_tone(sig):
    v = _spectral(sig.harmonic(220, 2.0))
    assert v["Estimated Fundamental (F0)"] == pytest.approx(220, abs=2)


# ─────────────────────────────────────────────────────────────────────────────
# temporal
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_pauses_and_centroid(sig, values):
    sr = 16_000
    tone = sig.sine(440, 3.0, sr, amplitude=0.3)
    # Three 300 ms silent pauses.
    for start in (0.6, 1.4, 2.2):
        s = int(start * sr)
        tone[s:s + int(0.3 * sr)] = 0.0
    v = values(M.compute_temporal(tone, sr))

    assert v["Num Pauses (>100ms)"] == 3
    # A 300 ms gap of zeros contains (300 - 25) / 10 + 1 fully-silent 25 ms frames.
    assert v["Max Pause Duration"] == pytest.approx(0.28, abs=0.03)
    assert v["Attack Time"] < 20.0


def test_temporal_centroid_of_first_half_tone(sig, values):
    sr = 16_000
    x = np.concatenate([sig.sine(440, 1.0, sr, 0.3), np.zeros(sr)])
    v = values(M.compute_temporal(x, sr))
    # Energy is uniform over the first second: centre of mass at ~0.5 s.
    assert v["Temporal Centroid"] == pytest.approx(0.5, abs=0.03)


# ─────────────────────────────────────────────────────────────────────────────
# noise
# ─────────────────────────────────────────────────────────────────────────────

def _noise(x, sr=16_000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {m.name: m for m in M.compute_noise(x, sr)}


def _gated_tone_plus_noise(sig, noise_std, sr=16_000, seed=3):
    # 0.5-amplitude tone on for half of every second (power S = 0.125), plus white noise.
    t = np.arange(4 * sr)
    tone = 0.5 * np.sin(2 * np.pi * 220 * t / sr) * ((t % sr) < sr // 2)
    return tone + sig.white_noise(4.0, sr, std=noise_std, seed=seed)


def test_estimated_snr_decreases_monotonically_with_noise(sig):
    snrs = [
        _noise(_gated_tone_plus_noise(sig, std))["Estimated SNR"].value
        for std in (0.001, 0.005, 0.02, 0.05, 0.2)
    ]
    assert all(a > b for a, b in zip(snrs, snrs[1:])), snrs


@pytest.mark.parametrize("noise_std", [0.002, 0.01, 0.05])
def test_estimated_snr_and_noise_floor_known_answers(sig, noise_std):
    # The estimator compares loud frames (tone + noise) with quiet frames
    # (noise only), i.e. it measures (S + N) / N with S = 0.125, N = std**2.
    m = _noise(_gated_tone_plus_noise(sig, noise_std))
    signal_power, noise_power = 0.125, noise_std ** 2

    expected_snr = 10 * math.log10((signal_power + noise_power) / noise_power)
    assert m["Estimated SNR"].value == pytest.approx(expected_snr, abs=1.5)
    assert m["Estimated Noise Floor"].value == pytest.approx(10 * math.log10(noise_power), abs=1.0)


def test_digital_silence_floor_reports_floor_limited_snr(sig):
    t = np.arange(4 * 16_000)
    x = 0.5 * np.sin(2 * np.pi * 220 * t / 16_000) * ((t % 16_000) < 8_000)
    m = _noise(x)

    assert m["Estimated SNR"].value == pytest.approx(M._SNR_CEILING_DB)
    assert "Floor-limited" in m["Estimated SNR"].calibration_note
    assert m["Estimated SNR"].warning is None
    assert m["Spectral SNR"].value <= M._SNR_CEILING_DB
    assert m["Estimated Noise Floor"].value is None
    assert m["Estimated Noise Floor"].calibration_note


def test_all_silent_noise_group_has_no_nan():
    m = _noise(np.zeros(16_000))
    for metric in m.values():
        assert not (isinstance(metric.value, float) and math.isnan(metric.value)), metric.name
    assert m["Estimated SNR"].value is None


def test_hnr_of_periodic_signal_is_high(sig):
    for f0 in (100, 150, 220, 300):
        hnr = M._compute_hnr(sig.harmonic(f0, 1.5), 16_000)
        # A perfectly periodic signal has no noise; the old biased estimate
        # was capped at 5-10 dB by the autocorrelation lag taper.
        assert hnr > 30.0, (f0, hnr)


@pytest.mark.parametrize("target_hnr_db", [0.0, 10.0, 20.0])
def test_hnr_matches_harmonic_to_noise_power_ratio(sig, target_hnr_db):
    # Boersma (1993): r(T0) = H / (H + N), so HNR = 10 log10(r / (1 - r)) = 10 log10(H / N).
    x = sig.harmonic(150, 2.0)
    noise_std = math.sqrt(np.mean(x ** 2) / 10 ** (target_hnr_db / 10))
    y = x + sig.white_noise(2.0, std=noise_std, seed=5)
    assert M._compute_hnr(y, 16_000) == pytest.approx(target_hnr_db, abs=2.0)


@pytest.mark.parametrize("n_gaps", [0, 1, 3, 5])
def test_dropouts_count_injected_zero_gaps(sig, n_gaps):
    sr = 16_000
    x = sig.sine(440, 4.0, sr, amplitude=0.5)
    for k in range(n_gaps):
        start = int((0.5 + 0.6 * k) * sr)
        x[start:start + int(0.2 * sr)] = 0.0
    m = _noise(x, sr)
    assert m["Detected Dropouts"].value == n_gaps
    assert bool(m["Detected Dropouts"].warning) == (n_gaps > 0)


def test_dropouts_ignore_gentle_dips_and_natural_fades(sig):
    sr = 16_000
    t = sig.time(4.0, sr)
    tone = sig.sine(440, 4.0, sr, amplitude=0.5)
    dips = tone * (1 - 0.9 * sum(np.exp(-((t - c) ** 2) / (2 * 0.05 ** 2)) for c in (0.8, 1.6, 2.4, 3.2)))
    assert M._detect_dropouts(dips, sr) == 0

    # Syllables that fade smoothly to exact digital zero between them are pauses, not dropouts.
    syllables = np.clip(np.sin(2 * np.pi * 3 * t), 0, None) ** 2 * sig.harmonic(150, 4.0, sr)
    assert M._detect_dropouts(syllables, sr) == 0

    # Leading / trailing silence is not a mid-signal dropout.
    padded = np.concatenate([np.zeros(sr), tone, np.zeros(sr)])
    assert M._detect_dropouts(padded, sr) == 0


def test_dropouts_on_clean_real_speech(sample_speech):
    audio, sr = sample_speech
    assert M._detect_dropouts(audio, sr) == 0


# ─────────────────────────────────────────────────────────────────────────────
# speech group
# ─────────────────────────────────────────────────────────────────────────────

def _speech(x, sr=16_000):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {m.name: m.value for m in M.compute_speech(x, sr)}


@pytest.mark.parametrize("rate", [3.0, 4.0, 5.0])
def test_speech_group_speaking_rate_counts_syllables(sig, rate):
    # One raised-cosine "syllable" per modulation cycle. The median-filtered
    # envelope has flat tops, which the old strict-peak test never detected.
    v = _speech(sig.am_tone(200, rate, 3.0))
    assert v["Estimated Speaking Rate"] == pytest.approx(rate, abs=0.35)


def test_active_speech_level_of_gated_tone(sig):
    # Active level = RMS over the active portion: 0.2 / sqrt(2) -> -16.99 dBFS.
    x = np.concatenate([np.zeros(16_000), sig.sine(180, 1.0, amplitude=0.2)])
    v = _speech(x)
    assert v["Active Speech Level (ASL)"] == pytest.approx(_db(0.2) + FS_SINE_RMS_DB, abs=1.0)


def test_voiced_unvoiced_ratio_separates_tone_from_noise(sig):
    assert _speech(sig.harmonic(150, 1.0))["Voiced/Unvoiced Ratio"] > 0.95
    assert _speech(sig.white_noise(1.0, std=0.2))["Voiced/Unvoiced Ratio"] < 0.05


def test_speech_band_snr_known_answer(sig):
    # White noise: energy share in 300-3400 Hz vs outside is 3100 / 4900 of 8 kHz.
    v = _speech(sig.white_noise(2.0, std=0.1))
    assert v["Speech-Band SNR (300 Hz – 3.4 kHz)"] == pytest.approx(0.0, abs=0.5)


# ─────────────────────────────────────────────────────────────────────────────
# perceptual (intrusive metrics against a reference)
# ─────────────────────────────────────────────────────────────────────────────

def _perceptual(y, ref, sr=16_000, ref_sr=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {m.name: m.value for m in M.compute_perceptual(y, sr, ref_audio=ref, ref_sr=ref_sr or sr)}


@pytest.mark.parametrize("snr_db", [0.0, 10.0, 20.0])
def test_si_sdr_and_sdr_equal_the_injected_snr(sig, snr_db):
    ref = sig.harmonic(150, 2.0)
    noise = sig.white_noise(2.0, std=1.0, seed=11)
    noise *= math.sqrt(np.mean(ref ** 2) / np.mean(noise ** 2) / 10 ** (snr_db / 10))
    v = _perceptual(ref + noise, ref)

    # Noise is (nearly) orthogonal to the reference, so both equal the SNR.
    assert v["SI-SDR"] == pytest.approx(snr_db, abs=0.3)
    assert v["SDR (Signal-to-Distortion Ratio)"] == pytest.approx(snr_db, abs=0.3)


def test_si_sdr_is_scale_invariant_but_sdr_is_not(sig):
    ref = sig.harmonic(150, 2.0)
    v = _perceptual(0.5 * ref, ref)
    assert v["SI-SDR"] > 100.0                     # a pure gain is not distortion
    # SDR = 10 log10(|r|^2 / |0.5 r - r|^2) = 10 log10(4) = 6.02 dB.
    assert v["SDR (Signal-to-Distortion Ratio)"] == pytest.approx(10 * math.log10(4), abs=0.01)


def test_identical_signal_has_zero_spectral_distance(sig):
    ref = sig.harmonic(150, 2.0)
    v = _perceptual(ref.copy(), ref)
    assert v["Log-Spectral Distance"] == pytest.approx(0.0, abs=1e-6)
    assert v["Spectral Correlation"] == pytest.approx(1.0, abs=1e-6)
    assert v["Cepstral Distance"] == pytest.approx(0.0, abs=1e-6)


def test_reference_at_other_sample_rate_is_resampled(sig):
    ref_8k = sig.harmonic(150, 2.0, sr=8_000, n_harmonics=8)
    y_16k = sig.harmonic(150, 2.0, sr=16_000, n_harmonics=8)
    v = _perceptual(y_16k, ref_8k, sr=16_000, ref_sr=8_000)
    assert v["Spectral Correlation"] > 0.95


def test_mos_proxies_never_improve_when_noise_is_added(sig):
    # Syllables separated by exact digital silence: the cleanest possible
    # floor. It used to be scored as 0 dB SNR, so the clean file got MOS 1.0
    # while the same file with a little added noise got 4.5.
    sr = 16_000
    t = sig.time(3.0, sr)
    clean = np.clip(np.sin(2 * np.pi * 2 * t), 0, None) * sig.harmonic(150, 3.0, sr)
    names = ["DNSMOS P.835 SIG (proxy)", "DNSMOS P.835 BAK (proxy)",
             "DNSMOS P.835 OVRL (proxy)", "Estimated MOS (non-intrusive proxy)"]
    scores = []
    for std in (0.0, 1e-3, 1e-2, 3e-2):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            v = {m.name: m.value for m in M.compute_perceptual(clean + sig.white_noise(3.0, sr, std=std), sr)}
        scores.append([v[name] for name in names])
    for name, series in zip(names, zip(*scores)):
        assert all(a >= b for a, b in zip(series, series[1:])), (name, series)
    assert scores[0][3] > scores[-1][3]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "P.563 proxy pins to its 1.0 floor on clean wideband speech: energy below "
        "200 Hz (a normal male F0 and its first harmonic) is scored as 'rumble' and "
        "only 300-3400 Hz counts as speech, as if the input were already "
        "telephone-band filtered. Retuning the cue bands is an owner decision."
    ),
)
def test_p563_proxy_scores_clean_speech_above_the_floor(sample_speech):
    audio, sr = sample_speech
    assert M._compute_p563_proxy(audio, sr).value >= 3.0


# ─────────────────────────────────────────────────────────────────────────────
# robustness: no group may emit NaN for silence or very short input
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("group", sorted(M._BUILTIN_METRIC_GROUPS))
@pytest.mark.parametrize("kind", ["silence", "short", "tone"])
def test_no_group_emits_nan(sig, group, kind):
    sr = 16_000
    audio = {
        "silence": np.zeros(sr),
        "short": sig.sine(440, 0.05, sr, amplitude=0.3),
        "tone": sig.sine(440, 1.0, sr, amplitude=0.3),
    }[kind]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = M.METRIC_GROUPS[group](audio, sr)
    for metric in results:
        value = metric.value
        assert not (isinstance(value, (float, np.floating)) and np.isnan(value)), (group, kind, metric.name)

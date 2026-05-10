import numpy as np
import pytest

from qualiax import metrics as metrics_module
from qualiax import metrics_speaker as speaker_module
from qualiax.metrics import compute_basic, compute_loudness, compute_perceptual, compute_speech


def test_speech_band_energies_are_non_negative():
    sr = 16_000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    results = compute_speech(audio, sr)
    values = {metric.name: metric.value for metric in results}

    assert values["F1-Region Energy (300–1000 Hz)"] >= 0.0
    assert values["F2-Region Energy (1–2.5 kHz)"] >= 0.0
    assert values["F3-Region Energy (2.5–3.5 kHz)"] >= 0.0


def test_speech_reports_active_speech_level():
    sr = 16_000
    t = np.linspace(0, 1, sr, endpoint=False)
    active = (0.2 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    audio = np.concatenate([np.zeros(sr, dtype=np.float32), active])

    results = compute_speech(audio, sr)
    values = {metric.name: metric.value for metric in results}

    assert values["Active Speech Level (ASL)"] is not None
    assert values["Active Speech Level (ASL)"] > -20.0


def test_spectral_correlation_warns_on_failure(monkeypatch):
    def fake_stft(*args, **kwargs):
        raise RuntimeError("stft broke")

    monkeypatch.setattr("scipy.signal.stft", fake_stft)

    with pytest.warns(UserWarning, match="Spectral Correlation"):
        value = metrics_module._spectral_correlation(
            np.zeros(512, dtype=np.float32),
            np.zeros(512, dtype=np.float32),
            16_000,
        )

    assert value is None


def test_speaker_hnr_fallback_warns_on_failure(monkeypatch):
    monkeypatch.setattr(speaker_module, "compute_formants", lambda mono, sr: [])
    monkeypatch.setattr(speaker_module, "compute_spectral_tilt", lambda mono, sr: ([], None))
    monkeypatch.setattr(speaker_module, "compute_voice_quality", lambda mono, sr: [])
    monkeypatch.setattr(speaker_module, "_to_mono", lambda audio: audio)
    monkeypatch.setattr(speaker_module, "_estimate_gender", lambda f0_mean, voiced_pct: [])
    monkeypatch.setattr(speaker_module, "_estimate_age", lambda *args, **kwargs: [])
    monkeypatch.setattr("qualiax.metrics_prosody.compute_f0_track", lambda mono, sr: (np.array([]), np.array([])))
    def broken_hnr(mono, sr):
        raise RuntimeError("hnr broke")

    monkeypatch.setattr(metrics_module, "_compute_hnr", broken_hnr)

    with pytest.warns(UserWarning, match="Speaker HNR fallback failed"):
        results = speaker_module.compute_speaker(np.zeros(1024, dtype=np.float32), 16_000)

    assert results == []


def test_loudness_includes_platform_target_advisor_metrics():
    sr = 16_000
    t = np.linspace(0, 4, sr * 4, endpoint=False)
    audio = (0.08 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    results = compute_loudness(audio, sr)
    names = {metric.name for metric in results}

    assert "Loudness Delta vs Spotify" in names
    assert "Loudness Delta vs Apple Music" in names
    assert "Loudness Delta vs EBU R128" in names


def test_perceptual_reports_v04_non_intrusive_metrics_without_reference():
    sr = 16_000
    t = np.linspace(0, 1, sr, endpoint=False)
    audio = (0.15 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)

    results = compute_perceptual(audio, sr)
    names = {metric.name for metric in results}

    assert "DNSMOS P.835 OVRL (proxy)" in names
    assert "AECMOS (proxy)" in names
    assert "P.563 Proxy (NB Quality Estimate)" in names
    assert "UTMOS (proxy)" in names
    assert "SHEET MOS (proxy)" in names
    assert "Codec Artifact Risk" in names


def test_improved_p563_proxy_penalizes_artifacts():
    sr = 8_000
    t = np.linspace(0, 1, sr, endpoint=False)
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 4 * t))
    clean = (
        envelope
        * (
            0.10 * np.sin(2 * np.pi * 180 * t)
            + 0.05 * np.sin(2 * np.pi * 360 * t)
            + 0.03 * np.sin(2 * np.pi * 540 * t)
        )
    ).astype(np.float32)
    degraded = clean.copy()
    degraded[::32] = 0.99
    degraded[1::32] = -0.99
    degraded += 0.12 * np.random.default_rng(0).standard_normal(len(degraded)).astype(np.float32)
    degraded = np.clip(degraded, -1.0, 1.0)

    clean_score = metrics_module._compute_p563_proxy(clean, sr).value
    degraded_score = metrics_module._compute_p563_proxy(degraded, sr).value

    assert clean_score is not None
    assert degraded_score is not None
    assert clean_score > degraded_score


def test_stereo_audio_reports_interchannel_metrics():
    sr = 16_000
    t = np.linspace(0, 1, sr, endpoint=False)
    left = 0.2 * np.sin(2 * np.pi * 220 * t)
    right = 0.1 * np.sin(2 * np.pi * 220 * t + 0.1)
    audio = np.vstack([left, right]).astype(np.float32)

    results = compute_basic(audio, sr)
    values = {metric.name: metric.value for metric in results}

    assert "Stereo Phase Correlation" in values
    assert "Stereo Width" in values
    assert "Interaural Level Difference (ILD)" in values
    assert "Interaural Time Difference (ITD)" in values


def test_codec_artifact_risk_penalizes_bandlimited_audio():
    sr = 16_000
    t = np.linspace(0, 1, sr, endpoint=False)
    fullband = (0.15 * np.sin(2 * np.pi * 220 * t) + 0.08 * np.sin(2 * np.pi * 6000 * t)).astype(np.float32)
    bandlimited = (0.15 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    full_results = {metric.name: metric.value for metric in compute_perceptual(fullband, sr)}
    limited_results = {metric.name: metric.value for metric in compute_perceptual(bandlimited, sr)}

    assert full_results["Estimated Codec Bandwidth Cutoff"] > limited_results["Estimated Codec Bandwidth Cutoff"]
    assert limited_results["Codec Artifact Risk"] >= full_results["Codec Artifact Risk"]

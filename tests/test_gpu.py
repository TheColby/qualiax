"""Tests for qualiax.gpu.

The numpy/scipy path is checked against straightforward reference
implementations written here. The torch path (CPU device) must match the numpy
path; those tests are skipped when torch is not installed.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
import pytest
from scipy.signal import freqz, lfilter
from scipy.signal import stft as scipy_stft

from qualiax import gpu

SR = 16_000


@pytest.fixture
def signal():
    rng = np.random.default_rng(7)
    t = np.arange(int(1.3 * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * rng.standard_normal(len(t))).astype(np.float32)


@pytest.fixture
def numpy_only(monkeypatch):
    """Force the numpy path even when torch is installed."""
    monkeypatch.setattr(gpu._ctx, "available", lambda: False)


def _frames(x, frame_len, hop):
    return [x[i:i + frame_len] for i in range(0, len(x) - frame_len, hop)]


# ─────────────────────────────────────────────────────────────────────────────
# numpy path vs reference implementations
# ─────────────────────────────────────────────────────────────────────────────

def test_n_frames_matches_python_range():
    assert gpu._n_frames(1000, 400, 160) == len(range(0, 600, 160))
    assert gpu._n_frames(400, 400, 160) == 0
    assert gpu._n_frames(100, 400, 160) == 0


def test_batch_energy_is_frame_mean_square(signal, numpy_only):
    expected = [np.mean(f.astype(np.float64) ** 2) for f in _frames(signal, 400, 160)]
    np.testing.assert_allclose(gpu.batch_energy(signal, SR, 25.0, 10.0), expected, rtol=1e-5)


def test_batch_zcr_is_sign_change_rate(signal, numpy_only):
    expected = [np.mean(np.abs(np.diff(np.sign(f))) > 0) for f in _frames(signal, 400, 160)]
    np.testing.assert_allclose(gpu.batch_zcr(signal, SR, 25.0, 10.0), expected, rtol=1e-6)


def test_batch_autocorr_matches_numpy_correlate(signal, numpy_only):
    acf = gpu.batch_autocorr(signal, SR, 40.0, 10.0, normalize=True)
    frames = _frames(signal.astype(np.float64), 640, 160)
    assert acf.shape == (len(frames), 640)
    for got, frame in zip(acf[::17], frames[::17]):
        centred = frame - frame.mean()
        ref = np.correlate(centred, centred, mode="full")[len(centred) - 1:]
        np.testing.assert_allclose(got, ref / ref[0], atol=1e-5)


def test_short_input_yields_empty_frames(numpy_only):
    tiny = np.zeros(10, dtype=np.float32)
    assert gpu.batch_energy(tiny, SR).shape == (0,)
    assert gpu.batch_zcr(tiny, SR).shape == (0,)
    assert gpu.batch_autocorr(tiny, SR).shape == (0, 640)


def test_stft_numpy_path_is_scipy_stft(signal, numpy_only):
    f, mag = gpu.stft(signal, SR, n_fft=512, hop_length=128)
    f_ref, _, z = scipy_stft(signal, fs=SR, nperseg=512, noverlap=384)
    np.testing.assert_allclose(f, f_ref)
    np.testing.assert_allclose(mag, np.abs(z), rtol=1e-6)


def _reference_spectral_features(x, sr, n_fft):
    """Per-frame textbook definitions, averaged over frames."""
    f, _, z = scipy_stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
    mag = np.abs(z).astype(np.float64)
    power = mag ** 2
    floor = power.max() * 1e-10
    cent, bw, roll, flat, ent, hfc = [], [], [], [], [], []
    for p in power.T:
        total = p.sum() + 1e-10
        w = p / total
        c = float(np.sum(f * w))
        cent.append(c)
        bw.append(math.sqrt(np.sum((f - c) ** 2 * w)))
        roll.append(f[int(np.argmax(np.cumsum(p) >= 0.95 * total))])
        flat.append(math.exp(np.mean(np.log(p + floor))) / (np.mean(p) + floor))
        ent.append(float(-np.sum(w * np.log2(w + 1e-10))))
        hfc.append(float(np.sum(f * p)))
    flux = np.mean([np.linalg.norm(mag[:, i + 1] - mag[:, i]) for i in range(mag.shape[1] - 1)])
    return {
        "centroid": np.mean(cent),
        "bandwidth": np.mean(bw),
        "rolloff": np.mean(roll),
        "flatness_db": 10 * math.log10(np.mean(flat)),
        "flux": flux,
        "entropy": np.mean(ent),
        "hfc": np.mean(hfc),
    }


def test_spectral_features_numpy_match_reference(signal, numpy_only):
    got = gpu.spectral_features(signal, SR, n_fft=1024)
    ref = _reference_spectral_features(signal, SR, 1024)
    for key, expected in ref.items():
        assert got[key] == pytest.approx(expected, rel=1e-6, abs=1e-9), key


def test_mfcc_c0_absorbs_gain_and_other_coefficients_do_not_change(numpy_only):
    # log(a^2 P) = log P + 2 ln a; the orthonormal DCT-II maps a constant c over
    # n_mels bands to c * sqrt(n_mels) in coefficient 0 and 0 elsewhere.
    # (Holds while every mel energy is far above the 1e-8 log floor, so use
    # broadband noise and skip the zero-padded edge frames.)
    rng = np.random.default_rng(3)
    x = 0.3 * rng.standard_normal(SR)
    n_mels = 40
    gain = 4.0
    base = gpu.mfcc(x, SR, n_mfcc=13, n_mels=n_mels)[:, 1:-1]
    louder = gpu.mfcc(x * gain, SR, n_mfcc=13, n_mels=n_mels)[:, 1:-1]
    assert base.shape == louder.shape and base.shape[0] == 13
    np.testing.assert_allclose(louder[0] - base[0], 2 * math.log(gain) * math.sqrt(n_mels), atol=1e-3)
    np.testing.assert_allclose(louder[1:], base[1:], atol=1e-3)


def test_mel_filterbank_is_triangular_and_normalised_to_peak_one():
    fb = gpu._mel_filterbank(SR, 40, 512, 80.0, 8000.0)
    assert fb.shape == (40, 257)
    assert np.all(fb >= 0.0)
    assert np.allclose(fb.max(axis=1), 1.0)
    centres = fb.argmax(axis=1)
    assert np.all(np.diff(centres) >= 0)  # centres ascend with band index


# ─────────────────────────────────────────────────────────────────────────────
# K-weighting, true peak, resampling
# ─────────────────────────────────────────────────────────────────────────────

BS1770_48K = {
    "b1": [1.53512485958697, -2.69169618940638, 1.19839281085285],
    "a1": [1.0, -1.69065929318241, 0.73248077421585],
    "b2": [1.0, -2.0, 1.0],
    "a2": [1.0, -1.99004745483398, 0.99007225036621],
}


def test_k_weighting_coefficients_match_bs1770_table_at_48k():
    b1, a1, b2, a2 = gpu.k_weighting_coefficients(48_000)
    np.testing.assert_allclose(b1, BS1770_48K["b1"], atol=1e-12)
    np.testing.assert_allclose(a1, BS1770_48K["a1"], atol=1e-12)
    np.testing.assert_allclose(b2, BS1770_48K["b2"], atol=1e-12)
    np.testing.assert_allclose(a2, BS1770_48K["a2"], atol=1e-12)


@pytest.mark.parametrize("sr, tol_db", [(48_000, 0.001), (44_100, 0.005), (16_000, 0.06)])
def test_k_weighting_gain_at_997hz_is_0_691_db(sr, tol_db):
    # The -0.691 dB constant in BS.1770 cancels the K-weighting gain at 997 Hz.
    b1, a1, b2, a2 = gpu.k_weighting_coefficients(sr)
    _, h1 = freqz(b1, a1, worN=[997.0], fs=sr)
    _, h2 = freqz(b2, a2, worN=[997.0], fs=sr)
    assert 20 * math.log10(abs(h1[0] * h2[0])) == pytest.approx(0.691, abs=tol_db)


def test_k_weighting_filter_applies_both_stages(signal):
    b1, a1, b2, a2 = gpu.k_weighting_coefficients(SR)
    expected = lfilter(b2, a2, lfilter(b1, a1, signal.astype(np.float64)))
    np.testing.assert_allclose(gpu.k_weighting_filter(signal, SR), expected, rtol=1e-12, atol=1e-15)


def test_true_peak_finds_inter_sample_peak():
    sr = 48_000
    x = np.sin(2 * np.pi * (sr / 4) * np.arange(sr) / sr + np.pi / 4)
    assert 20 * math.log10(np.max(np.abs(x))) == pytest.approx(-3.01, abs=0.01)
    assert -0.4 <= gpu.true_peak(x, sr) <= 0.2  # EBU Tech 3341 tolerance around 0 dBTP


def test_true_peak_edge_cases():
    assert gpu.true_peak(np.zeros(100), SR) == -math.inf
    assert gpu.true_peak(np.zeros(0), SR) == -math.inf
    impulse = np.zeros(64)
    impulse[10] = 0.5
    assert gpu.true_peak(impulse, SR) >= 20 * math.log10(0.5)


def test_resample_is_anti_aliased_and_preserves_in_band_tones():
    t = np.arange(SR) / SR
    in_band = gpu.resample(np.sin(2 * np.pi * 1000 * t), SR, 8_000)
    alias = gpu.resample(np.sin(2 * np.pi * 6000 * t), SR, 8_000)  # above the new Nyquist

    assert len(in_band) == 8_000
    core = slice(100, -100)
    assert np.sqrt(np.mean(in_band[core] ** 2)) == pytest.approx(1 / math.sqrt(2), rel=0.01)
    assert 20 * math.log10(np.sqrt(np.mean(alias[core] ** 2)) * math.sqrt(2)) < -40.0


def test_resample_same_rate_is_identity(signal):
    assert gpu.resample(signal, SR, SR) is signal


# ─────────────────────────────────────────────────────────────────────────────
# device management and fallback behaviour
# ─────────────────────────────────────────────────────────────────────────────

def test_device_reports_numpy_when_torch_is_missing(monkeypatch):
    ctx = gpu._DeviceContext()
    monkeypatch.setattr(ctx, "_tried", True)
    monkeypatch.setattr(ctx, "_torch", None)
    assert not ctx.available()
    assert ctx.device is None
    assert ctx.device_str == "cpu (numpy)"
    ctx.set("cuda")
    assert ctx._device is None


class _FakeDevice:
    def __init__(self, kind, index=None):
        self.type = kind
        self.index = index


def _fake_torch(cuda=False, mps=False):
    from types import SimpleNamespace

    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, get_device_name=lambda idx: "Fake GPU"),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        device=_FakeDevice,
    )


@pytest.mark.parametrize(
    "cuda, mps, expected_type, expected_str",
    [(True, True, "cuda", "cuda:0 (Fake GPU)"), (False, True, "mps", "mps (Apple Metal)"), (False, False, "cpu", "cpu")],
)
def test_device_auto_detection_priority(cuda, mps, expected_type, expected_str):
    ctx = gpu._DeviceContext()
    ctx._tried, ctx._torch = True, _fake_torch(cuda=cuda, mps=mps)
    assert ctx.device.type == expected_type
    assert ctx.device_str == expected_str


def test_requesting_unavailable_accelerator_warns_and_uses_cpu():
    ctx = gpu._DeviceContext()
    ctx._tried, ctx._torch = True, _fake_torch()
    with pytest.warns(UserWarning, match="CUDA not available"):
        ctx.set("cuda")
    assert ctx.device.type == "cpu"
    with pytest.warns(UserWarning, match="MPS"):
        ctx.set("mps")
    assert ctx.device.type == "cpu"
    ctx.set("cpu")
    assert ctx.device.type == "cpu"
    ctx.set("auto")
    assert ctx.device.type == "cpu"


def test_torch_kernel_failure_falls_back_to_numpy(monkeypatch, signal):
    monkeypatch.setattr(gpu._ctx, "available", lambda: True)

    def boom(*args, **kwargs):
        raise RuntimeError("device lost")

    for name in ("_batch_energy_torch", "_batch_zcr_torch", "_batch_autocorr_torch",
                 "_stft_torch", "_spectral_features_torch", "_mfcc_torch"):
        monkeypatch.setattr(gpu, name, boom)

    with pytest.warns(UserWarning, match="device lost"):
        energy = gpu.batch_energy(signal, SR)
    np.testing.assert_allclose(energy, gpu._batch_energy_numpy(signal, 400, 160))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert gpu.batch_zcr(signal, SR).shape == energy.shape
        assert gpu.batch_autocorr(signal, SR).shape[1] == 640
        assert gpu.stft(signal, SR, n_fft=512)[1].shape[0] == 257
        assert set(gpu.spectral_features(signal, SR)) >= {"centroid", "flatness_db", "hfc"}
        assert gpu.mfcc(signal, SR).shape[0] == 13


# ─────────────────────────────────────────────────────────────────────────────
# torch path parity (skipped without torch)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def torch_cpu():
    pytest.importorskip("torch")
    gpu.set_device("cpu")
    yield
    gpu.set_device("auto")


def test_torch_framed_kernels_match_numpy(torch_cpu, signal):
    np.testing.assert_allclose(gpu._batch_energy_torch(signal, 400, 160),
                               gpu._batch_energy_numpy(signal, 400, 160), rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(gpu._batch_zcr_torch(signal, 400, 160),
                               gpu._batch_zcr_numpy(signal, 400, 160), atol=1e-6)
    np.testing.assert_allclose(gpu._batch_autocorr_torch(signal, 640, 160, True),
                               gpu._batch_autocorr_numpy(signal, 640, 160, True), atol=1e-4)


def test_torch_stft_matches_scipy_framing_and_scaling(torch_cpu, signal):
    for n_fft, hop in ((2048, 1024), (512, 128), (512, 256)):
        f_t, m_t = gpu._stft_torch(signal, SR, n_fft, hop, "hann")
        f_n, m_n = gpu._stft_scipy(signal, SR, n_fft, hop)
        np.testing.assert_allclose(f_t, f_n)
        assert m_t.shape == m_n.shape
        np.testing.assert_allclose(m_t, m_n, rtol=1e-3, atol=1e-6)


def test_torch_spectral_features_and_mfcc_match_numpy(torch_cpu, signal):
    got = gpu._spectral_features_torch(signal, SR, 2048)
    ref = gpu._spectral_features_numpy(signal, SR, 2048)
    for key in ref:
        assert got[key] == pytest.approx(ref[key], rel=1e-3, abs=1e-6), key
    np.testing.assert_allclose(gpu._mfcc_torch(signal, SR, 13, 40, 512, 80.0, 8000.0),
                               gpu._mfcc_numpy(signal, SR, 13, 40, 512, 80.0, 8000.0), atol=1e-2)

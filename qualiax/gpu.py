"""
qualiax.gpu — Hardware acceleration layer.

Auto-selects the fastest available backend in priority order:
    CUDA  (NVIDIA GPU)  → torch.device("cuda")
    MPS   (Apple Metal) → torch.device("mps")
    CPU                 → numpy / scipy fallback

All accelerated kernels gracefully fall back to CPU numpy/scipy
if PyTorch is not installed or the requested device is unavailable.
The torch and numpy paths use identical framing and scaling, so metric values
do not depend on which backend ran. K-weighting, true-peak oversampling and
resampling always run on the CPU with scipy (see the notes on each function).

Usage
-----
    from qualiax.gpu import device_ctx, stft, batch_energy, mfcc, autocorr

    # The module is self-initializing; just import and call.
    # To override the device:
    from qualiax.gpu import set_device
    set_device("cpu")          # force CPU
    set_device("cuda")         # force CUDA
    set_device("mps")          # force Apple Metal
"""
from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np
from numpy.typing import NDArray

# ─────────────────────────────────────────────────────────────────────────────
# Device management
# ─────────────────────────────────────────────────────────────────────────────

class _DeviceContext:
    """Singleton that holds the active compute device."""

    def __init__(self):
        self._device = None   # lazy init
        self._torch = None
        self._tried = False

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def torch(self):
        """Return the torch module, or None if not installed."""
        if not self._tried:
            self._tried = True
            try:
                import torch as _torch
                self._torch = _torch
            except ImportError:
                self._torch = None
        return self._torch

    def available(self) -> bool:
        return self.torch is not None

    @property
    def device(self):
        if self._device is None:
            self._device = self._auto_detect()
        return self._device

    @property
    def device_str(self) -> str:
        if not self.available():
            return "cpu (numpy)"
        d = self.device
        if d.type == "cuda":
            try:
                idx = d.index or 0
                name = self.torch.cuda.get_device_name(idx)
                return f"cuda:{idx} ({name})"
            except Exception:
                return "cuda"
        if d.type == "mps":
            return "mps (Apple Metal)"
        return "cpu"

    def set(self, device: str) -> None:
        """Manually set device: 'auto', 'cpu', 'cuda', 'mps'."""
        if device == "auto":
            self._device = self._auto_detect()
            return
        if not self.available():
            self._device = None
            return
        torch = self.torch
        if device == "cuda":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            else:
                warnings.warn("CUDA not available — falling back to CPU")
                self._device = torch.device("cpu")
        elif device == "mps":
            if torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                warnings.warn("MPS (Metal) not available — falling back to CPU")
                self._device = torch.device("cpu")
        else:
            self._device = torch.device("cpu")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _auto_detect(self):
        torch = self.torch
        if torch is None:
            return None
        # Priority: CUDA > MPS > CPU
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def tensor(self, arr: NDArray, dtype=None) -> "torch.Tensor":
        """Move a numpy array to the active device."""
        torch = self.torch
        if dtype is None:
            dtype = torch.float32
        return torch.as_tensor(arr, dtype=dtype, device=self.device)

    def numpy(self, t: "torch.Tensor") -> NDArray:
        """Move a tensor back to CPU numpy."""
        return t.detach().cpu().numpy()


# Global singleton
_ctx = _DeviceContext()


def set_device(device: str) -> None:
    """Set the global compute device: 'auto' | 'cpu' | 'cuda' | 'mps'."""
    _ctx.set(device)


def get_device_str() -> str:
    """Human-readable name of the active device."""
    return _ctx.device_str


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared by all kernels
# ─────────────────────────────────────────────────────────────────────────────

def _eps32() -> float:
    return float(np.finfo(np.float32).eps)


def _n_frames(n_samples: int, frame_len: int, hop: int) -> int:
    """Frame count used by every framed kernel: ``len(range(0, n - frame_len, hop))``.

    The torch and numpy paths must agree on this so that results do not depend on
    whether torch is installed.
    """
    if hop <= 0:
        return 0
    return len(range(0, n_samples - frame_len, hop))


# ─────────────────────────────────────────────────────────────────────────────
# Accelerated STFT
#
# Returns: (freqs: NDArray[F], magnitude: NDArray[F, T])
#
# Both paths reproduce ``scipy.signal.stft`` defaults exactly (periodic Hann
# window, ``boundary='zeros'``, ``padded=True`` and magnitude scaled by
# ``1 / sum(window)``) so every downstream feature is backend-independent.
# ─────────────────────────────────────────────────────────────────────────────

def stft(
    mono: NDArray,
    sr: int,
    n_fft: int = 2048,
    hop_length: Optional[int] = None,
    window: str = "hann",
) -> tuple[NDArray, NDArray]:
    """
    Compute magnitude STFT.
    Uses torch.stft on CUDA/MPS when available, falls back to scipy.
    Returns (freqs, magnitude) arrays on CPU.
    """
    if hop_length is None:
        hop_length = n_fft // 2

    if _ctx.available():
        try:
            return _stft_torch(mono, sr, n_fft, hop_length, window)
        except Exception as e:
            warnings.warn(f"[gpu] torch STFT failed ({e}), falling back to scipy")

    return _stft_scipy(mono, sr, n_fft, hop_length)


def _stft_mag_torch(mono, n_fft, hop_length):
    """Magnitude STFT on the active torch device, framed and scaled like scipy.

    Returns a device tensor of shape (n_fft // 2 + 1, T).
    """
    torch = _ctx.torch
    x = _ctx.tensor(np.asarray(mono, dtype=np.float32))
    half = n_fft // 2
    # scipy.signal.stft: zero-extend n_fft//2 on both sides (boundary='zeros'),
    # then zero-pad the tail so the last frame is complete (padded=True).
    padded_len = x.shape[0] + 2 * half
    tail = (-(padded_len - n_fft) % hop_length) % n_fft
    x = torch.nn.functional.pad(x, (half, half + tail))
    win = torch.hann_window(n_fft, periodic=True, device=_ctx.device)
    stft_out = torch.stft(
        x,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=win,
        return_complex=True,
        center=False,
    )
    return stft_out.abs() / win.sum()


def _stft_torch(mono, sr, n_fft, hop_length, window_name):
    mag = _stft_mag_torch(mono, n_fft, hop_length)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return freqs, _ctx.numpy(mag).astype(np.float64)


def _stft_scipy(mono, sr, n_fft, hop_length):
    from scipy.signal import stft as scipy_stft
    f, _t, Zxx = scipy_stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length)
    return f, np.abs(Zxx)


# ─────────────────────────────────────────────────────────────────────────────
# Accelerated batch frame energy
#
# Splits signal into overlapping frames and computes per-frame energy (RMS²).
# Returns NDArray of shape (n_frames,).
# ─────────────────────────────────────────────────────────────────────────────

def batch_energy(
    mono: NDArray,
    sr: int,
    frame_len_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> NDArray:
    """
    Compute per-frame energy (mean squared amplitude) efficiently.
    GPU-accelerated via torch.Tensor.unfold when available.
    """
    frame_len = int(frame_len_ms * sr / 1000)
    hop = int(hop_ms * sr / 1000)

    if _ctx.available():
        try:
            return _batch_energy_torch(mono, frame_len, hop)
        except Exception as e:
            warnings.warn(f"[gpu] batch_energy torch failed ({e}), falling back")

    return _batch_energy_numpy(mono, frame_len, hop)


def _unfold_frames_torch(x, n_samples, frame_len, hop):
    """Frame a device tensor exactly like the numpy kernels (no zero padding).

    Zero-padding the tail would create artificial near-silent frames that bias
    noise-floor and SNR estimates.
    """
    n = _n_frames(n_samples, frame_len, hop)
    if n <= 0:
        return None
    return x.unfold(0, frame_len, hop)[:n]


def _batch_energy_torch(mono, frame_len, hop):
    x = _ctx.tensor(mono)
    frames = _unfold_frames_torch(x, len(mono), frame_len, hop)
    if frames is None:
        return np.array([], dtype=np.float32)
    energy = (frames ** 2).mean(dim=1)
    return _ctx.numpy(energy)


def _batch_energy_numpy(mono, frame_len, hop):
    energies = []
    for i in range(0, len(mono) - frame_len, hop):
        energies.append(float(np.mean(mono[i:i + frame_len] ** 2)))
    return np.array(energies, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Accelerated batch zero-crossing rate
# ─────────────────────────────────────────────────────────────────────────────

def batch_zcr(
    mono: NDArray,
    sr: int,
    frame_len_ms: float = 25.0,
    hop_ms: float = 10.0,
) -> NDArray:
    """Compute per-frame zero-crossing rate. GPU-accelerated when available."""
    frame_len = int(frame_len_ms * sr / 1000)
    hop = int(hop_ms * sr / 1000)

    if _ctx.available():
        try:
            return _batch_zcr_torch(mono, frame_len, hop)
        except Exception as e:
            warnings.warn(f"[gpu] batch_zcr torch failed ({e}), falling back")

    return _batch_zcr_numpy(mono, frame_len, hop)


def _batch_zcr_torch(mono, frame_len, hop):
    torch = _ctx.torch
    x = _ctx.tensor(np.sign(mono))
    frames = _unfold_frames_torch(x, len(mono), frame_len, hop)
    if frames is None:
        return np.array([], dtype=np.float32)
    # ZCR = mean of (|diff| > 0) → (|sign_diff| > 0)
    sign_diff = torch.abs(frames[:, 1:] - frames[:, :-1])
    zcr = (sign_diff > 0).float().mean(dim=1)
    return _ctx.numpy(zcr)


def _batch_zcr_numpy(mono, frame_len, hop):
    zcrs = []
    for i in range(0, len(mono) - frame_len, hop):
        f = np.sign(mono[i:i + frame_len])
        zcrs.append(float(np.mean(np.abs(np.diff(f)) > 0)))
    return np.array(zcrs, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Accelerated batch autocorrelation (for F0 / HNR / jitter estimation)
#
# Returns: NDArray of shape (n_frames, frame_len) — unnormalized autocorrelations
# ─────────────────────────────────────────────────────────────────────────────

def batch_autocorr(
    mono: NDArray,
    sr: int,
    frame_len_ms: float = 40.0,
    hop_ms: float = 10.0,
    normalize: bool = True,
) -> NDArray:
    """
    Compute autocorrelation for every frame simultaneously.
    Returns shape (n_frames, frame_len).
    GPU-accelerated via batched FFT convolution.
    """
    frame_len = int(frame_len_ms * sr / 1000)
    hop = int(hop_ms * sr / 1000)

    if _ctx.available():
        try:
            return _batch_autocorr_torch(mono, frame_len, hop, normalize)
        except Exception as e:
            warnings.warn(f"[gpu] batch_autocorr torch failed ({e}), falling back")

    return _batch_autocorr_numpy(mono, frame_len, hop, normalize)


def _batch_autocorr_torch(mono, frame_len, hop, normalize):
    torch = _ctx.torch
    x = _ctx.tensor(mono)

    frames = _unfold_frames_torch(x, len(mono), frame_len, hop)  # (N, frame_len)
    if frames is None:
        return np.zeros((0, frame_len), dtype=np.float32)
    # Remove mean per frame
    frames = frames - frames.mean(dim=1, keepdim=True)

    # Batched FFT autocorrelation: rfft → |·|² → irfft
    n_fft = 2 * frame_len
    F = torch.fft.rfft(frames, n=n_fft, dim=1)  # (N, n_fft//2+1)
    power = F * F.conj()                          # |F|²
    acf_full = torch.fft.irfft(power, n=n_fft, dim=1)  # (N, n_fft)
    acf = acf_full[:, :frame_len]                 # (N, frame_len) — keep lags

    if normalize:
        # Match the numpy path: only normalise frames with non-negligible energy.
        lag0 = acf[:, 0:1]
        norm = torch.where(lag0 > 1e-8, lag0, torch.ones_like(lag0))
        acf = acf / norm

    return _ctx.numpy(acf)


def _batch_autocorr_numpy(mono, frame_len, hop, normalize):
    results = []
    for i in range(0, len(mono) - frame_len, hop):
        frame = mono[i:i + frame_len].astype(np.float64)
        frame -= frame.mean()
        n = 2 * frame_len
        F = np.fft.rfft(frame, n=n)
        acf = np.fft.irfft(F * np.conj(F))[:frame_len]
        if normalize and acf[0] > 1e-8:
            acf /= acf[0]
        results.append(acf.astype(np.float32))
    return np.stack(results) if results else np.zeros((0, frame_len), dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Accelerated MFCC
# ─────────────────────────────────────────────────────────────────────────────

def mfcc(
    mono: NDArray,
    sr: int,
    n_mfcc: int = 13,
    n_mels: int = 40,
    n_fft: int = 512,
    fmin: float = 80.0,
    fmax: Optional[float] = None,
) -> Optional[NDArray]:
    """
    Compute MFCCs (n_mfcc, T).
    GPU-accelerated mel filterbank + DCT via torch.
    Falls back to scipy-based implementation.
    """
    if fmax is None:
        fmax = min(sr / 2.0, 8000.0)

    if _ctx.available():
        try:
            return _mfcc_torch(mono, sr, n_mfcc, n_mels, n_fft, fmin, fmax)
        except Exception as e:
            warnings.warn(f"[gpu] mfcc torch failed ({e}), falling back")

    return _mfcc_numpy(mono, sr, n_mfcc, n_mels, n_fft, fmin, fmax)


def _mel_filterbank(sr, n_mels, n_fft, fmin, fmax) -> NDArray:
    """Triangular mel filterbank shared by the torch and numpy MFCC paths."""
    n_freqs = n_fft // 2 + 1
    mel_min = 2595 * np.log10(1 + fmin / 700)
    mel_max = 2595 * np.log10(1 + fmax / 700)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_pts = np.floor(hz_pts * (n_fft + 1) / sr).astype(int)
    bin_pts = np.clip(bin_pts, 0, n_freqs - 1)

    fb = np.zeros((n_mels, n_freqs))
    for m in range(1, n_mels + 1):
        f_lo, f_c, f_hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(f_lo, f_c + 1):
            if f_c > f_lo and 0 <= k < n_freqs:
                fb[m - 1, k] = (k - f_lo) / (f_c - f_lo)
        for k in range(f_c, f_hi + 1):
            if f_hi > f_c and 0 <= k < n_freqs:
                fb[m - 1, k] = (f_hi - k) / (f_hi - f_c)
    return fb


def _mfcc_torch(mono, sr, n_mfcc, n_mels, n_fft, fmin, fmax):
    torch = _ctx.torch

    # --- STFT (scipy-compatible framing and scaling) ---
    power = _stft_mag_torch(mono, n_fft, n_fft // 2) ** 2  # (n_fft//2+1, T)

    # --- Mel filterbank ---
    fb = _ctx.tensor(_mel_filterbank(sr, n_mels, n_fft, fmin, fmax))
    mel_power = torch.mm(fb, power)  # (n_mels, T)
    log_mel = torch.log(mel_power + 1e-8)  # (n_mels, T)

    # --- DCT-II (orthonormal) ---
    # torch doesn't have dct, so we compute it as: dct[k] = sum_n log_mel[n] * cos(pi/N*(n+0.5)*k)
    N = n_mels
    n_idx = torch.arange(N, device=_ctx.device, dtype=torch.float32)
    k_idx = torch.arange(n_mfcc, device=_ctx.device, dtype=torch.float32)
    # DCT-II matrix: (n_mfcc, N)
    dct_mat = torch.cos(
        math.pi / N * (n_idx[None, :] + 0.5) * k_idx[:, None]
    )
    # Orthonormalization factors
    dct_mat[0] *= math.sqrt(1.0 / N)
    dct_mat[1:] *= math.sqrt(2.0 / N)

    mfccs = torch.mm(dct_mat, log_mel)  # (n_mfcc, T)
    return _ctx.numpy(mfccs)


def _mfcc_numpy(mono, sr, n_mfcc, n_mels, n_fft, fmin, fmax):
    """Pure numpy/scipy MFCC fallback."""
    try:
        from scipy.fft import dct as scipy_dct
        from scipy.signal import stft as scipy_stft

        _, _, Zxx = scipy_stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
        power = np.abs(Zxx) ** 2

        fb = _mel_filterbank(sr, n_mels, n_fft, fmin, fmax)
        mel_power = fb @ power
        log_mel = np.log(mel_power + 1e-8)
        mfccs_out = scipy_dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]
        return mfccs_out
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Accelerated K-weighting filter (BS.1770)
# ─────────────────────────────────────────────────────────────────────────────

# Analog prototype parameters that reproduce the ITU-R BS.1770-4 48 kHz
# coefficient tables exactly (same derivation as libebur128).
_KW_SHELF_F0 = 1681.974450955533
_KW_SHELF_GAIN_DB = 3.999843853973347
_KW_SHELF_Q = 0.7071752369554196
_KW_SHELF_VB_EXP = 0.4996667741545416
_KW_HP_F0 = 38.13547087602444
_KW_HP_Q = 0.5003270373238773


def k_weighting_coefficients(sr: int) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return ``(b1, a1, b2, a2)`` for the two K-weighting biquads at ``sr``.

    At 48 kHz these equal the BS.1770-4 tables. The RLB high-pass numerator is
    ``[1, -2, 1]`` at 48 kHz (a passband gain slightly above unity that the
    -0.691 dB constant is calibrated against); that reference gain is kept at
    every other sample rate.
    """
    K = math.tan(math.pi * _KW_SHELF_F0 / sr)
    Vh = 10 ** (_KW_SHELF_GAIN_DB / 20.0)
    Vb = Vh ** _KW_SHELF_VB_EXP
    a0 = 1 + K / _KW_SHELF_Q + K * K
    b1 = [
        (Vh + Vb * K / _KW_SHELF_Q + K * K) / a0,
        2 * (K * K - Vh) / a0,
        (Vh - Vb * K / _KW_SHELF_Q + K * K) / a0,
    ]
    a1 = [1.0, 2 * (K * K - 1) / a0, (1 - K / _KW_SHELF_Q + K * K) / a0]

    K2 = math.tan(math.pi * _KW_HP_F0 / sr)
    a0_2 = 1 + K2 / _KW_HP_Q + K2 * K2
    K48 = math.tan(math.pi * _KW_HP_F0 / 48000.0)
    gain_48k = 1 + K48 / _KW_HP_Q + K48 * K48
    b2 = [gain_48k / a0_2, -2.0 * gain_48k / a0_2, gain_48k / a0_2]
    a2 = [1.0, 2 * (K2 * K2 - 1) / a0_2, (1 - K2 / _KW_HP_Q + K2 * K2) / a0_2]
    return b1, a1, b2, a2


def k_weighting_filter(mono: NDArray, sr: int) -> NDArray:
    """
    Apply BS.1770 K-weighting (two cascaded IIR stages) in float64 on the CPU.

    A torch path is deliberately not used: IIR filtering is sequential, float32
    biquads are numerically poor for a 38 Hz high-pass, and ``torchaudio``'s
    ``biquad`` clamps its output to [-1, 1], which under-reads loud material.
    """
    return _k_weight_scipy(mono, sr)


def _k_weight_scipy(mono, sr):
    from scipy.signal import lfilter

    b1, a1, b2, a2 = k_weighting_coefficients(sr)
    s1 = lfilter(b1, a1, np.asarray(mono, dtype=np.float64))
    return lfilter(b2, a2, s1)


# ─────────────────────────────────────────────────────────────────────────────
# Accelerated spectral features (single-pass over STFT)
# Returns a dict of feature arrays computed together on GPU.
# ─────────────────────────────────────────────────────────────────────────────

def spectral_features(
    mono: NDArray,
    sr: int,
    n_fft: int = 2048,
) -> dict[str, float]:
    """
    Compute all spectral features in one STFT pass on GPU.
    Returns centroid, bandwidth, rolloff, flatness, flux, entropy (means).
    """
    if _ctx.available():
        try:
            return _spectral_features_torch(mono, sr, n_fft)
        except Exception as e:
            warnings.warn(f"[gpu] spectral_features torch failed ({e}), falling back")

    return _spectral_features_numpy(mono, sr, n_fft)


def _spectral_features_torch(mono, sr, n_fft):
    # The FFT work runs on the device; the per-frame reductions are shared with
    # the numpy path so both backends report identical features.
    mag = _ctx.numpy(_stft_mag_torch(mono, n_fft, n_fft // 2)).astype(np.float64)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return _spectral_features_from_mag(freqs, mag)


def _spectral_features_numpy(mono, sr, n_fft):
    from scipy.signal import stft as scipy_stft
    f, _t, Zxx = scipy_stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
    return _spectral_features_from_mag(f, np.abs(Zxx))


def _spectral_features_from_mag(freqs: NDArray, mag: NDArray) -> dict[str, float]:
    """Frame-wise spectral descriptors averaged over time.

    Centroid, bandwidth, rolloff, flatness and entropy are scale-invariant: the
    flatness floor is relative to the loudest bin (-100 dB), so a quiet tone is
    exactly as "tonal" as a loud one.
    """
    power = mag ** 2
    total = power.sum(axis=0, keepdims=True) + 1e-10
    norm = power / total

    centroid_frames = (freqs[:, None] * norm).sum(axis=0)
    centroid = float(np.mean(centroid_frames))
    # Bandwidth is the spread around each frame's own centroid.
    diff_sq = ((freqs[:, None] - centroid_frames[None, :]) ** 2) * norm
    bandwidth = float(np.mean(np.sqrt(diff_sq.sum(axis=0))))
    cumsum = np.cumsum(power, axis=0)
    rolloff_idx = np.argmax(cumsum >= 0.95 * total, axis=0).clip(0, len(freqs) - 1)
    rolloff = float(np.mean(freqs[rolloff_idx]))
    floor = max(float(np.max(power)) * 1e-10, np.finfo(np.float64).tiny)
    geo = np.exp(np.mean(np.log(power + floor), axis=0))
    arith = np.mean(power, axis=0) + floor
    flatness = float(np.mean(geo / arith))
    flatness_db = 10 * math.log10(flatness) if flatness > 0 else -100.0
    diff_mag = np.diff(mag, axis=1)
    flux = float(np.mean(np.sqrt(np.sum(diff_mag ** 2, axis=0)))) if diff_mag.shape[1] else 0.0
    entropy = float(np.mean(-(norm * np.log2(norm + 1e-10)).sum(axis=0)))
    hfc = float(np.mean((freqs[:, None] * power).sum(axis=0)))

    return {
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "flatness_db": flatness_db,
        "flux": flux,
        "entropy": entropy,
        "hfc": hfc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Accelerated true peak measurement (4× oversampling)
# ─────────────────────────────────────────────────────────────────────────────

def true_peak(mono: NDArray, sr: int, oversample: int = 4) -> float:
    """
    Compute true peak (dBTP) via band-limited (polyphase FIR) oversampling.

    Always runs on the CPU: linear interpolation (the previous torch path) can
    never exceed the sample peak, so it cannot detect inter-sample peaks at all.
    """
    return _true_peak_scipy(mono, sr, oversample)


def _true_peak_scipy(mono, sr, oversample):
    from scipy.signal import resample_poly
    mono = np.asarray(mono, dtype=np.float64)
    if mono.size == 0:
        return -math.inf
    up = resample_poly(mono, oversample, 1)
    # True peak can never be below the sample peak.
    peak = max(float(np.max(np.abs(up))), float(np.max(np.abs(mono))))
    return 20 * math.log10(peak) if peak > 0 else -math.inf


# ─────────────────────────────────────────────────────────────────────────────
# Resampling
# ─────────────────────────────────────────────────────────────────────────────

def resample(mono: NDArray, orig_sr: int, target_sr: int) -> NDArray:
    """Resample audio with an anti-aliased polyphase filter (CPU).

    Linear interpolation (the previous torch path) has no anti-aliasing filter,
    so downsampling folded out-of-band energy into the passband.
    """
    if orig_sr == target_sr:
        return mono
    return _resample_scipy(mono, orig_sr, target_sr)


def _resample_scipy(mono, orig_sr, target_sr):
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(orig_sr, target_sr)
    return resample_poly(mono, target_sr // g, orig_sr // g).astype(np.float32)

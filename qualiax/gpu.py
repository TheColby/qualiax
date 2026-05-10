"""
qualiax.gpu — Hardware acceleration layer.

Auto-selects the fastest available backend in priority order:
    CUDA  (NVIDIA GPU)  → torch.device("cuda")
    MPS   (Apple Metal) → torch.device("mps")
    CPU                 → numpy / scipy fallback

All accelerated kernels gracefully fall back to CPU numpy/scipy
if PyTorch is not installed or the requested device is unavailable.

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


# ─────────────────────────────────────────────────────────────────────────────
# Accelerated STFT
#
# Returns: (freqs: NDArray[F], magnitude: NDArray[F, T])
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


def _stft_torch(mono, sr, n_fft, hop_length, window_name):
    torch = _ctx.torch
    # Build window on device
    win_fn = {"hann": torch.hann_window, "hamming": torch.hamming_window}.get(
        window_name, torch.hann_window
    )
    win = win_fn(n_fft, device=_ctx.device)
    x = _ctx.tensor(mono.astype(np.float32))

    # torch.stft → (freq, time, 2) complex output
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*resized.*", category=UserWarning)
        stft_out = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=win,
            return_complex=True,
            pad_mode="reflect",
            center=True,
        )  # shape: (n_fft//2+1, T)

    mag = stft_out.abs()
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr).to(_ctx.device)
    return _ctx.numpy(freqs), _ctx.numpy(mag)


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


def _batch_energy_torch(mono, frame_len, hop):
    torch = _ctx.torch
    x = _ctx.tensor(mono)
    # Pad to make sure we get all frames
    pad = frame_len - (len(mono) - frame_len) % hop if len(mono) > frame_len else 0
    if pad > 0:
        x = torch.nn.functional.pad(x, (0, pad))
    # unfold: (n_frames, frame_len)
    frames = x.unfold(0, frame_len, hop)
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
    pad = frame_len - (len(mono) - frame_len) % hop if len(mono) > frame_len else 0
    if pad > 0:
        x = torch.nn.functional.pad(x, (0, pad))
    frames = x.unfold(0, frame_len, hop)
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

    pad = frame_len - (len(mono) - frame_len) % hop if len(mono) > frame_len else 0
    if pad > 0:
        x = torch.nn.functional.pad(x, (0, pad))

    frames = x.unfold(0, frame_len, hop)  # (N, frame_len)
    # Remove mean per frame
    frames = frames - frames.mean(dim=1, keepdim=True)

    # Batched FFT autocorrelation: rfft → |·|² → irfft
    n_fft = 2 * frame_len
    F = torch.fft.rfft(frames, n=n_fft, dim=1)  # (N, n_fft//2+1)
    power = F * F.conj()                          # |F|²
    acf_full = torch.fft.irfft(power, n=n_fft, dim=1)  # (N, n_fft)
    acf = acf_full[:, :frame_len]                 # (N, frame_len) — keep lags

    if normalize:
        norm = acf[:, 0:1].clamp(min=1e-8)
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


def _mfcc_torch(mono, sr, n_mfcc, n_mels, n_fft, fmin, fmax):
    torch = _ctx.torch

    # --- STFT ---
    hop_length = n_fft // 2
    win = torch.hann_window(n_fft, device=_ctx.device)
    x = _ctx.tensor(mono.astype(np.float32))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*resized.*", category=UserWarning)
        stft_out = torch.stft(
            x, n_fft=n_fft, hop_length=hop_length,
            win_length=n_fft, window=win,
            return_complex=True, pad_mode="reflect", center=False,
        )
    power = stft_out.abs() ** 2  # (n_fft//2+1, T)

    # --- Mel filterbank (computed on device) ---
    n_freqs = n_fft // 2 + 1
    mel_min = 2595.0 * math.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * math.log10(1.0 + fmax / 700.0)
    mel_pts = torch.linspace(mel_min, mel_max, n_mels + 2, device=_ctx.device)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)
    bin_pts = (hz_pts * (n_fft + 1) / sr).long().clamp(0, n_freqs - 1)

    fb = torch.zeros(n_mels, n_freqs, device=_ctx.device)
    for m in range(1, n_mels + 1):
        f_lo, f_c, f_hi = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        # Rising slope
        if f_c > f_lo:
            for k in range(int(f_lo), int(f_c) + 1):
                if 0 <= k < n_freqs:
                    fb[m - 1, k] = float(k - f_lo) / float(f_c - f_lo)
        # Falling slope
        if f_hi > f_c:
            for k in range(int(f_c), int(f_hi) + 1):
                if 0 <= k < n_freqs:
                    fb[m - 1, k] = float(f_hi - k) / float(f_hi - f_c)

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

        mel_power = fb @ power
        log_mel = np.log(mel_power + 1e-8)
        mfccs_out = scipy_dct(log_mel, type=2, axis=0, norm="ortho")[:n_mfcc]
        return mfccs_out
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Accelerated K-weighting filter (BS.1770)
# ─────────────────────────────────────────────────────────────────────────────

_torchaudio_available: Optional[bool] = None


def _check_torchaudio() -> bool:
    global _torchaudio_available
    if _torchaudio_available is None:
        try:
            import torchaudio.functional  # noqa: F401
            _torchaudio_available = True
        except ImportError:
            _torchaudio_available = False
    return _torchaudio_available


def k_weighting_filter(mono: NDArray, sr: int) -> NDArray:
    """
    Apply BS.1770 K-weighting (two cascaded IIR stages).
    Uses torchaudio functional biquad on GPU when available.
    Falls back to scipy.signal.lfilter.
    """
    if _ctx.available() and _check_torchaudio():
        try:
            return _k_weight_torch(mono, sr)
        except Exception as e:
            warnings.warn(f"[gpu] k-weight torch failed ({e}), falling back")

    return _k_weight_scipy(mono, sr)


def _k_weight_torch(mono, sr):
    import torchaudio.functional as AF
    torch = _ctx.torch

    x = _ctx.tensor(mono.astype(np.float64)).float()

    # Stage 1 coefficients (high-shelf pre-filter) — same as scipy version
    db  = 3.999843853973347
    f0  = 1681.9744509555319
    Q   = 0.7071752369554193
    K   = math.tan(math.pi * f0 / sr)
    Vh  = 10 ** (db / 20.0)
    Vb  = Vh ** 0.4845
    a0  = 1 + K / Q + K * K
    b0_1 = (Vh + Vb * K / Q + K * K) / a0
    b1_1 = 2 * (K * K - Vh) / a0
    b2_1 = (Vh - Vb * K / Q + K * K) / a0
    a1_1 = 2 * (K * K - 1) / a0
    a2_1 = (1 - K / Q + K * K) / a0

    # Stage 2 coefficients (high-pass RLB)
    f0_2 = 38.13547087602444
    Q_2  = 0.5003270373238773
    K2   = math.tan(math.pi * f0_2 / sr)
    a0_2 = 1 + K2 / Q_2 + K2 * K2
    b0_2 = 1.0 / a0_2
    b1_2 = -2.0 / a0_2
    b2_2 = 1.0 / a0_2
    a1_2 = 2 * (K2 * K2 - 1) / a0_2
    a2_2 = (1 - K2 / Q_2 + K2 * K2) / a0_2

    s1 = AF.biquad(x, b0_1, b1_1, b2_1, 1.0, a1_1, a2_1)
    s2 = AF.biquad(s1, b0_2, b1_2, b2_2, 1.0, a1_2, a2_2)
    return _ctx.numpy(s2).astype(np.float64)


def _k_weight_scipy(mono, sr):
    from scipy.signal import lfilter

    db  = 3.999843853973347
    f0  = 1681.9744509555319
    Q   = 0.7071752369554193
    K   = math.tan(math.pi * f0 / sr)
    Vh  = 10 ** (db / 20.0)
    Vb  = Vh ** 0.4845
    a0  = 1 + K / Q + K * K
    b0  = (Vh + Vb * K / Q + K * K) / a0
    b1  = 2 * (K * K - Vh) / a0
    b2  = (Vh - Vb * K / Q + K * K) / a0
    a1  = 2 * (K * K - 1) / a0
    a2  = (1 - K / Q + K * K) / a0
    s1  = lfilter([b0, b1, b2], [1, a1, a2], mono)

    f0_2 = 38.13547087602444
    Q_2  = 0.5003270373238773
    K2   = math.tan(math.pi * f0_2 / sr)
    a0_2 = 1 + K2 / Q_2 + K2 * K2
    b0_2 = 1.0 / a0_2
    b1_2 = -2.0 / a0_2
    b2_2 = 1.0 / a0_2
    a1_2 = 2 * (K2 * K2 - 1) / a0_2
    a2_2 = (1 - K2 / Q_2 + K2 * K2) / a0_2
    return lfilter([b0_2, b1_2, b2_2], [1, a1_2, a2_2], s1)


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
    torch = _ctx.torch
    hop = n_fft // 2
    win = torch.hann_window(n_fft, device=_ctx.device)
    x = _ctx.tensor(mono.astype(np.float32))

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*resized.*", category=UserWarning)
        stft_out = torch.stft(
            x, n_fft=n_fft, hop_length=hop,
            win_length=n_fft, window=win,
            return_complex=True, center=True, pad_mode="reflect",
        )
    mag = stft_out.abs()       # (F, T)
    power = mag ** 2            # (F, T)
    n_freqs = n_fft // 2 + 1
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sr).to(_ctx.device)  # (F,)

    total_power = power.sum(dim=0, keepdim=True).clamp(min=1e-10)  # (1, T)
    norm_power  = power / total_power                               # (F, T)

    # Centroid
    centroid_frames = (freqs[:, None] * norm_power).sum(dim=0)  # (T,)
    centroid = float(centroid_frames.mean())

    # Bandwidth
    diff_sq = ((freqs[:, None] - centroid_frames[None, :]) ** 2) * norm_power
    bw_frames = diff_sq.sum(dim=0).clamp(min=0).sqrt()
    bandwidth = float(bw_frames.mean())

    # Rolloff (95% energy)
    cumsum = power.cumsum(dim=0)
    total = power.sum(dim=0, keepdim=True).clamp(min=1e-10)
    rolloff_mask = (cumsum >= 0.95 * total).float()
    rolloff_idx = (rolloff_mask.cumsum(dim=0) <= 1).long().sum(dim=0).clamp(0, n_freqs - 1)
    rolloff = float(freqs[rolloff_idx].float().mean())

    # Spectral flatness
    log_power = torch.log(power + 1e-10)
    geo_mean = log_power.mean(dim=0).exp()           # (T,)
    arith_mean = power.mean(dim=0).clamp(min=1e-10)  # (T,)
    flatness = float((geo_mean / arith_mean).mean())
    flatness_db = 10 * math.log10(flatness) if flatness > 0 else -100.0

    # Spectral flux
    diff_mag = (mag[:, 1:] - mag[:, :-1])
    flux = float((diff_mag ** 2).sum(dim=0).sqrt().mean())

    # Spectral entropy
    entropy = float(-(norm_power * torch.log2(norm_power + 1e-10)).sum(dim=0).mean())

    # High-frequency content
    hfc = float((freqs[:, None] * power).sum(dim=0).mean())

    return {
        "centroid": centroid,
        "bandwidth": bandwidth,
        "rolloff": rolloff,
        "flatness_db": flatness_db,
        "flux": flux,
        "entropy": entropy,
        "hfc": hfc,
    }


def _spectral_features_numpy(mono, sr, n_fft):
    from scipy.signal import stft as scipy_stft
    f, _t, Zxx = scipy_stft(mono, fs=sr, nperseg=n_fft, noverlap=n_fft // 2)
    mag = np.abs(Zxx)
    power = mag ** 2
    total = power.sum(axis=0, keepdims=True) + 1e-10
    norm = power / total
    freqs = f

    centroid = float(np.mean((freqs[:, None] * norm).sum(axis=0)))
    diff_sq = ((freqs[:, None] - centroid) ** 2) * norm
    bandwidth = float(np.mean(np.sqrt(diff_sq.sum(axis=0))))
    cumsum = np.cumsum(power, axis=0)
    rolloff_idx = np.argmax(cumsum >= 0.95 * total, axis=0).clip(0, len(freqs) - 1)
    rolloff = float(np.mean(freqs[rolloff_idx]))
    geo = np.exp(np.mean(np.log(power + 1e-10), axis=0))
    arith = np.mean(power, axis=0) + 1e-10
    flatness = float(np.mean(geo / arith))
    flatness_db = 10 * math.log10(flatness) if flatness > 0 else -100.0
    diff_mag = np.diff(mag, axis=1)
    flux = float(np.mean(np.sqrt(np.sum(diff_mag ** 2, axis=0))))
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
    Compute true peak (dBTP) via oversampling.
    GPU-accelerated via torch interpolation.
    """
    if _ctx.available():
        try:
            return _true_peak_torch(mono, oversample)
        except Exception as e:
            warnings.warn(f"[gpu] true_peak torch failed ({e}), falling back")

    return _true_peak_scipy(mono, sr, oversample)


def _true_peak_torch(mono, oversample):
    torch = _ctx.torch
    x = _ctx.tensor(mono.astype(np.float32)).view(1, 1, -1)
    # Upsample via linear interpolation (faster than resample_poly for this purpose)
    up = torch.nn.functional.interpolate(
        x, scale_factor=float(oversample), mode="linear", align_corners=False
    )
    peak = float(up.abs().max())
    return 20 * math.log10(peak) if peak > 0 else -math.inf


def _true_peak_scipy(mono, sr, oversample):
    from scipy.signal import resample_poly
    up = resample_poly(mono, oversample, 1)
    peak = float(np.max(np.abs(up)))
    return 20 * math.log10(peak) if peak > 0 else -math.inf


# ─────────────────────────────────────────────────────────────────────────────
# Accelerated resampling
# ─────────────────────────────────────────────────────────────────────────────

def resample(mono: NDArray, orig_sr: int, target_sr: int) -> NDArray:
    """Resample audio. GPU-accelerated via torch interpolation when available."""
    if orig_sr == target_sr:
        return mono
    if _ctx.available():
        try:
            return _resample_torch(mono, orig_sr, target_sr)
        except Exception as e:
            warnings.warn(f"[gpu] resample torch failed ({e}), falling back")
    return _resample_scipy(mono, orig_sr, target_sr)


def _resample_torch(mono, orig_sr, target_sr):
    torch = _ctx.torch
    x = _ctx.tensor(mono.astype(np.float32)).view(1, 1, -1)
    n_out = int(len(mono) * target_sr / orig_sr)
    up = torch.nn.functional.interpolate(
        x, size=n_out, mode="linear", align_corners=False
    )
    return _ctx.numpy(up.squeeze())


def _resample_scipy(mono, orig_sr, target_sr):
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(orig_sr, target_sr)
    return resample_poly(mono, target_sr // g, orig_sr // g).astype(np.float32)

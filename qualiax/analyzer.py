"""
Audio loading and analysis orchestration.
"""
from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np

from .models import FileResult, MetricResult
from .metrics import METRIC_GROUPS


class AudioLoader:
    """Load audio files to numpy arrays using available backends."""

    @staticmethod
    def load(path: Path) -> tuple[np.ndarray, int]:
        """Returns (audio_ndarray, sample_rate). Audio shape: (samples,) or (channels, samples)."""
        suffix = path.suffix.lower()

        # Try soundfile first (handles WAV, FLAC, OGG, AIFF, etc.)
        try:
            import soundfile as sf
            data, sr = sf.read(str(path), always_2d=False)
            if data.ndim == 2:
                data = data.T  # (channels, samples)
            return data.astype(np.float32), sr
        except ImportError:
            pass
        except Exception as e:
            warnings.warn(f"soundfile failed for {path.name}: {e}")

        # Try pydub (handles MP3, AAC, M4A via ffmpeg)
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(str(path))
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32)
            samples /= 2 ** (seg.sample_width * 8 - 1)
            if seg.channels > 1:
                samples = samples.reshape((-1, seg.channels)).T
            return samples, seg.frame_rate
        except ImportError:
            pass
        except Exception as e:
            warnings.warn(f"pydub failed for {path.name}: {e}")

        # Try librosa (most permissive, handles almost everything)
        try:
            import librosa
            data, sr = librosa.load(str(path), sr=None, mono=False)
            return data.astype(np.float32), sr
        except ImportError:
            pass
        except Exception as e:
            warnings.warn(f"librosa failed for {path.name}: {e}")

        # Fallback: raw WAV via stdlib
        if suffix == ".wav":
            try:
                import wave
                import struct
                with wave.open(str(path)) as wf:
                    sr = wf.getframerate()
                    n_channels = wf.getnchannels()
                    sample_width = wf.getsampwidth()
                    n_frames = wf.getnframes()
                    raw = wf.readframes(n_frames)
                fmt = {1: "b", 2: "h", 4: "i"}.get(sample_width, "h")
                samples = np.array(struct.unpack(f"<{n_frames * n_channels}{fmt}", raw), dtype=np.float32)
                max_val = 2 ** (sample_width * 8 - 1)
                samples /= max_val
                if n_channels > 1:
                    samples = samples.reshape((-1, n_channels)).T
                return samples, sr
            except Exception as e:
                raise RuntimeError(f"All loaders failed. Last error: {e}")

        raise RuntimeError(
            f"Cannot load '{path.suffix}' file. Install `soundfile`, `pydub`, or `librosa`:\n"
            "  pip install soundfile pydub librosa"
        )


class AudioAnalyzer:
    def __init__(
        self,
        metric_groups: set[str],
        reference: Optional[Path] = None,
        verbose: bool = False,
        workers: int = 1,
    ):
        self.metric_groups = metric_groups
        self.reference = reference
        self.verbose = verbose
        self.workers = max(1, workers)
        self._ref_audio: Optional[np.ndarray] = None
        self._ref_sr: Optional[int] = None

    def _load_reference(self):
        if self.reference and self._ref_audio is None:
            try:
                self._ref_audio, self._ref_sr = AudioLoader.load(self.reference)
                if self.verbose:
                    print(f"[info] Loaded reference: {self.reference.name} "
                          f"({self._ref_sr} Hz, {self._ref_audio.shape})")
            except Exception as e:
                warnings.warn(f"Failed to load reference file: {e}")

    def _resolve_groups(self) -> list[str]:
        if "all" in self.metric_groups:
            return list(METRIC_GROUPS.keys())
        return [g for g in METRIC_GROUPS if g in self.metric_groups]

    def analyze_file(self, path: Path) -> FileResult:
        result = FileResult(path=str(path))

        # Load audio
        try:
            audio, sr = AudioLoader.load(path)
        except Exception as e:
            result.error = str(e)
            return result

        # Populate basic file info
        mono = audio if audio.ndim == 1 else audio.mean(axis=0)
        result.duration_s = len(mono) / sr
        result.sample_rate = sr
        result.channels = 1 if audio.ndim == 1 else audio.shape[0]

        if self.verbose:
            print(f"[info] Analyzing: {path.name} "
                  f"({sr} Hz, {result.channels}ch, {result.duration_s:.1f}s)")

        # Load reference once
        self._load_reference()

        # Run each metric group
        groups = self._resolve_groups()
        for group_name in groups:
            fn = METRIC_GROUPS[group_name]
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    metrics = fn(
                        audio, sr,
                        ref_audio=self._ref_audio,
                        ref_sr=self._ref_sr,
                    )
                    for w in caught:
                        if self.verbose:
                            print(f"  [warn] {w.message}")
                result.metrics.extend(metrics)
            except Exception as e:
                result.metrics.append(MetricResult(
                    name=f"{group_name} (group failed)",
                    value=None,
                    description=str(e),
                    group=group_name,
                ))

        return result

    def analyze_all(self, paths: list[Path]) -> list[FileResult]:
        if self.workers == 1 or len(paths) == 1:
            return [self.analyze_file(p) for p in paths]

        results = [None] * len(paths)
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = {ex.submit(self.analyze_file, p): i for i, p in enumerate(paths)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    results[idx] = FileResult(path=str(paths[idx]), error=str(e))
        return results

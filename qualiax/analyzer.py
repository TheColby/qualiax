"""
Audio loading and analysis orchestration.
"""
from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
from typing import Optional

import numpy as np

from .models import FileResult, MetricResult
from .metrics import METRIC_GROUPS
from .speech_detector import detect_speech

# Groups that require speech content to be meaningful
_SPEECH_ONLY_GROUPS = {"speech", "prosody", "speaker"}
_NON_SPEECH_PERCEPTUAL_METRICS = {
    "P.563 Proxy (NB Quality Estimate)",
    "PESQ (ITU-T P.862)",
    "PESQ proxy (install `pesq` for true score)",
    "STOI (Short-Time Objective Intelligibility)",
    "STOI proxy (install `pystoi` for true score)",
}
_MIXED_SPEECH_FRACTION_THRESHOLD = 0.20


class AudioLoader:
    """Load audio files to numpy arrays using available backends."""

    @staticmethod
    def _load_wav_stdlib(path: Path) -> tuple[np.ndarray, int]:
        """Load PCM WAV via the stdlib wave module."""
        import wave

        with wave.open(str(path), "rb") as wf:
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            n_frames = wf.getnframes()
            comp_type = wf.getcomptype()
            raw = wf.readframes(n_frames)

        if comp_type != "NONE":
            raise RuntimeError(f"Unsupported WAV compression type: {comp_type}")

        if sample_width == 1:
            samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            samples = (samples - 128.0) / 128.0
        elif sample_width == 2:
            samples = np.frombuffer(raw, dtype="<i2").astype(np.float32)
            samples /= float(1 << 15)
        elif sample_width == 3:
            packed = np.frombuffer(raw, dtype=np.uint8)
            if len(packed) % 3 != 0:
                raise RuntimeError("Malformed 24-bit WAV payload.")
            triplets = packed.reshape(-1, 3).astype(np.int32)
            samples = (
                triplets[:, 0]
                | (triplets[:, 1] << 8)
                | (triplets[:, 2] << 16)
            )
            sign_bit = 1 << 23
            samples = ((samples ^ sign_bit) - sign_bit).astype(np.float32)
            samples /= float(1 << 23)
        elif sample_width == 4:
            samples = np.frombuffer(raw, dtype="<i4").astype(np.float32)
            samples /= float(1 << 31)
        else:
            raise RuntimeError(f"Unsupported PCM sample width: {sample_width * 8} bits")

        if n_channels > 1:
            samples = samples.reshape((-1, n_channels)).T
        return samples, sr

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
                return AudioLoader._load_wav_stdlib(path)
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
        self._reference_error: Optional[str] = None
        self._reference_lock = threading.Lock()

    def _load_reference(self):
        if not self.reference:
            return
        if self._reference_error:
            raise RuntimeError(self._reference_error)
        if self._ref_audio is not None:
            return
        with self._reference_lock:
            if self._reference_error:
                raise RuntimeError(self._reference_error)
            if self._ref_audio is not None:
                return
            try:
                self._ref_audio, self._ref_sr = AudioLoader.load(self.reference)
                if self.verbose:
                    print(f"[info] Loaded reference: {self.reference.name} "
                          f"({self._ref_sr} Hz, {self._ref_audio.shape})")
            except Exception as e:
                self._reference_error = (
                    f"Failed to load reference file '{self.reference}': {e}"
                )
                raise RuntimeError(self._reference_error) from e

    @staticmethod
    def _should_run_speech_metrics(detection) -> bool:
        return (
            detection.content_type == "speech"
            or (
                detection.content_type == "mixed"
                and detection.speech_fraction >= _MIXED_SPEECH_FRACTION_THRESHOLD
            )
        )

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

        # Classify content type using the shared detector
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            detection = detect_speech(audio, sr)
            result.content_type = detection.content_type
            result.speech_confidence = detection.confidence
            for w in caught:
                result.notes.append(f"content_type: {w.message}")
        speech_metrics_allowed = self._should_run_speech_metrics(detection)
        if detection.content_type == "mixed" and speech_metrics_allowed:
            result.notes.append(
                "content_type: mixed audio with substantial speech "
                f"(speech_fraction={detection.speech_fraction:.2f}); "
                "retaining speech-oriented metrics"
            )
        if self.verbose:
            print(
                f"  [info] Content type: {result.content_type} "
                f"(speech_confidence={result.speech_confidence:.2f})"
            )

        # Load reference once
        try:
            self._load_reference()
        except Exception as e:
            result.error = str(e)
            return result

        # Run each metric group
        groups = self._resolve_groups()
        for group_name in groups:
            # Speech gate: skip voice-quality groups for non-speech content
            if group_name in _SPEECH_ONLY_GROUPS and not speech_metrics_allowed:
                result.notes.append(
                    f"{group_name}: skipped (content_type='{result.content_type}', "
                    "not speech)"
                )
                continue
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
                        result.notes.append(f"{group_name}: {w.message}")
                        if self.verbose:
                            print(f"  [warn] {w.message}")
                if group_name == "perceptual" and not speech_metrics_allowed:
                    filtered_metrics = [
                        m for m in metrics if m.name not in _NON_SPEECH_PERCEPTUAL_METRICS
                    ]
                    if len(filtered_metrics) != len(metrics):
                        result.notes.append(
                            "perceptual: omitted speech-specific metrics for "
                            f"content_type='{result.content_type}'"
                        )
                    metrics = filtered_metrics
                result.metrics.extend(metrics)
            except Exception as e:
                result.metrics.append(MetricResult(
                    name=f"{group_name} (group failed)",
                    value=None,
                    description=str(e),
                    group=group_name,
                ))
                result.notes.append(f"{group_name}: group failed ({e})")

        return result

    def analyze_all(self, paths: list[Path], on_progress=None) -> list[FileResult]:
        """Analyze all paths. on_progress(completed, total, path) called after each file."""
        total = len(paths)

        if self.reference:
            try:
                self._load_reference()
            except Exception:
                pass

        if self.workers == 1 or total == 1:
            results = []
            for i, p in enumerate(paths):
                r = self.analyze_file(p)
                results.append(r)
                if on_progress:
                    on_progress(i + 1, total, p)
            return results

        results = [None] * total
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = {ex.submit(self.analyze_file, p): i for i, p in enumerate(paths)}
            completed = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    results[idx] = FileResult(path=str(paths[idx]), error=str(e))
                completed += 1
                if on_progress:
                    on_progress(completed, total, paths[idx])
        return results

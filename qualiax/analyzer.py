"""
Audio loading and analysis orchestration.
"""
from __future__ import annotations

import asyncio
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Optional

import numpy as np

from .models import DiagnosticEntry, FileResult, GroupHealth, MetricResult
from .metrics import METRIC_GROUPS
from .provenance import build_provenance
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


@dataclass(frozen=True)
class _AnalysisTask:
    path: Path
    label: str
    audio: np.ndarray
    sr: int
    load_error: Optional[str] = None
    source_file: Optional[str] = None
    segment_index: Optional[int] = None
    total_segments: Optional[int] = None
    segment_start_s: Optional[float] = None
    segment_end_s: Optional[float] = None
    sample_start: Optional[int] = None
    sample_end: Optional[int] = None


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
        segment_seconds: Optional[float] = None,
        strict: bool = False,
        include_demographics: bool = False,
    ):
        self.metric_groups = metric_groups
        self.reference = reference
        self.verbose = verbose
        self.workers = max(1, workers)
        self.segment_seconds = segment_seconds if segment_seconds and segment_seconds > 0 else None
        self.strict = strict
        self.include_demographics = include_demographics
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

    @staticmethod
    def _apply_confidence_notes(result: FileResult) -> None:
        confidence_levels = {metric.confidence for metric in result.metrics if metric.confidence}
        calibration_notes: list[str] = []
        seen = set()
        for metric in result.metrics:
            if metric.calibration_note and metric.calibration_note not in seen:
                calibration_notes.append(metric.calibration_note)
                seen.add(metric.calibration_note)

        if "proxy" in confidence_levels:
            result.confidence_notes.append(
                "Includes proxy-derived metrics; use them for directional review rather than absolute scoring."
            )
        if "heuristic" in confidence_levels:
            result.confidence_notes.append(
                "Includes heuristic estimates; interpret demographic or inferred labels cautiously."
            )
        if "model" in confidence_levels:
            result.confidence_notes.append(
                "Includes model-backed metrics; results can shift with runtime, model file, or calibration changes."
            )
        if any(metric.value is None for metric in result.metrics) and not result.error:
            result.confidence_notes.append(
                "One or more metrics were unavailable; inspect notes and missing values before comparing runs."
            )
        for note in calibration_notes:
            result.confidence_notes.append(f"Calibration: {note}")

        deduped: list[str] = []
        seen_notes = set()
        for note in result.confidence_notes:
            if note in seen_notes:
                continue
            deduped.append(note)
            seen_notes.add(note)
        result.confidence_notes = deduped

    @staticmethod
    def _replace_nan_values(metrics: list[MetricResult]) -> None:
        """Report undefined (NaN) metric values as missing instead of a raw ``nan``.

        Built-in groups already return None for undefined values; this guards
        custom metric groups so every output format shows N/A / null.
        """
        for metric in metrics:
            value = metric.value
            if isinstance(value, (float, np.floating)) and np.isnan(value):
                metric.value = None
                if not metric.warning:
                    metric.warning = "Unavailable: value is undefined (NaN) for this input"

    @staticmethod
    def _append_note(result: FileResult, message: str) -> None:
        if message not in result.notes:
            result.notes.append(message)

    @staticmethod
    def _add_diagnostic(
        result: FileResult,
        *,
        code: str,
        severity: str,
        source: str,
        message: str,
        group: Optional[str] = None,
        metric: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> None:
        for diagnostic in result.diagnostics:
            if (
                diagnostic.code == code
                and diagnostic.source == source
                and diagnostic.message == message
                and diagnostic.group == group
                and diagnostic.metric == metric
            ):
                diagnostic.count += 1
                if context:
                    diagnostic.context.update(context)
                return
        result.diagnostics.append(
            DiagnosticEntry(
                code=code,
                severity=severity,
                source=source,
                message=message,
                group=group,
                metric=metric,
                context=context or {},
            )
        )

    @staticmethod
    def _summarize_group_health(result: FileResult) -> None:
        groups = result.metrics_by_group()
        health_by_group: dict[str, GroupHealth] = {}
        for group, metrics in groups.items():
            warning_count = sum(1 for metric in metrics if metric.warning)
            missing_count = sum(1 for metric in metrics if metric.value is None)
            status = "ok"
            if warning_count:
                status = "warn"
            elif missing_count == len(metrics):
                status = "missing"
            elif missing_count:
                status = "partial"
            health_by_group[group] = GroupHealth(
                group=group,
                status=status,
                metric_count=len(metrics),
                warning_count=warning_count,
                missing_count=missing_count,
            )

        for diagnostic in result.diagnostics:
            if diagnostic.group:
                health = health_by_group.get(diagnostic.group)
                if health is None:
                    health = GroupHealth(group=diagnostic.group, status="skipped")
                    health_by_group[diagnostic.group] = health
                health.diagnostic_count += diagnostic.count
                if diagnostic.severity == "error":
                    health.status = "error"
                elif diagnostic.severity == "warn" and health.status == "ok":
                    health.status = "warn"
                elif diagnostic.code.endswith("_skipped") and health.metric_count == 0:
                    health.status = "skipped"

        result.group_health = [health_by_group[key] for key in sorted(health_by_group)]

    @staticmethod
    def _finalize_result(result: FileResult) -> None:
        AudioAnalyzer._apply_confidence_notes(result)
        AudioAnalyzer._summarize_group_health(result)
        result.provenance = build_provenance(result.metrics)

    def _analyze_loaded_audio(
        self,
        audio: np.ndarray,
        sr: int,
        *,
        label: str,
        source_file: Optional[str] = None,
        segment_index: Optional[int] = None,
        total_segments: Optional[int] = None,
        segment_start_s: Optional[float] = None,
        segment_end_s: Optional[float] = None,
        sample_start: Optional[int] = None,
        sample_end: Optional[int] = None,
    ) -> FileResult:
        result = FileResult(
            path=label,
            source_file=source_file,
            segment_index=segment_index,
            total_segments=total_segments,
            segment_start_s=segment_start_s,
            segment_end_s=segment_end_s,
        )

        mono = audio if audio.ndim == 1 else audio.mean(axis=0)
        result.duration_s = len(mono) / sr
        result.sample_rate = sr
        result.channels = 1 if audio.ndim == 1 else audio.shape[0]

        if self.verbose:
            print(f"[info] Analyzing: {label} "
                  f"({sr} Hz, {result.channels}ch, {result.duration_s:.1f}s)")

        # Classify content type using the shared detector
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            detection = detect_speech(audio, sr)
            result.content_type = detection.content_type
            result.speech_confidence = detection.confidence
            for w in caught:
                message = f"content_type: {w.message}"
                self._append_note(result, message)
                self._add_diagnostic(
                    result,
                    code="content_type_warning",
                    severity="warn",
                    source="content_type",
                    message=str(w.message),
                    context={"content_type": detection.content_type},
                )
        speech_metrics_allowed = self._should_run_speech_metrics(detection)
        if detection.content_type == "mixed" and speech_metrics_allowed:
            message = (
                "content_type: mixed audio with substantial speech "
                f"(speech_fraction={detection.speech_fraction:.2f}); "
                "retaining speech-oriented metrics"
            )
            self._append_note(result, message)
            self._add_diagnostic(
                result,
                code="mixed_content_retained",
                severity="info",
                source="content_type",
                message=message,
                context={"speech_fraction": detection.speech_fraction},
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
            self._add_diagnostic(
                result,
                code="reference_load_failed",
                severity="error",
                source="loader",
                message=str(e),
            )
            result.provenance = build_provenance(result.metrics)
            return result
        ref_audio = self._ref_audio
        ref_sr = self._ref_sr
        if sample_start is not None and sample_end is not None:
            ref_audio, ref_sr = self._reference_slice(sample_start, sample_end, sr)

        # Run each metric group
        groups = self._resolve_groups()
        for group_name in groups:
            # Speech gate: skip voice-quality groups for non-speech content
            if group_name in _SPEECH_ONLY_GROUPS and not speech_metrics_allowed:
                message = (
                    f"{group_name}: skipped (content_type='{result.content_type}', "
                    "not speech)"
                )
                self._append_note(result, message)
                self._add_diagnostic(
                    result,
                    code="metric_group_skipped",
                    severity="info",
                    source="metric_group",
                    message=message,
                    group=group_name,
                )
                continue
            fn = METRIC_GROUPS[group_name]
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    if group_name == "speaker":
                        from .metrics_speaker import compute_speaker

                        metrics = compute_speaker(
                            audio,
                            sr,
                            ref_audio=ref_audio,
                            ref_sr=ref_sr,
                            include_demographics=self.include_demographics,
                        )
                    else:
                        metrics = fn(
                            audio, sr,
                            ref_audio=ref_audio,
                            ref_sr=ref_sr,
                        )
                    for w in caught:
                        message = f"{group_name}: {w.message}"
                        self._append_note(result, message)
                        self._add_diagnostic(
                            result,
                            code="metric_group_warning",
                            severity="warn",
                            source="metric_group",
                            message=str(w.message),
                            group=group_name,
                        )
                        if self.verbose:
                            print(f"  [warn] {w.message}")
                if group_name == "perceptual" and not speech_metrics_allowed:
                    filtered_metrics = [
                        m for m in metrics if m.name not in _NON_SPEECH_PERCEPTUAL_METRICS
                    ]
                    if len(filtered_metrics) != len(metrics):
                        message = (
                            "perceptual: omitted speech-specific metrics for "
                            f"content_type='{result.content_type}'"
                        )
                        self._append_note(result, message)
                        self._add_diagnostic(
                            result,
                            code="speech_specific_metrics_omitted",
                            severity="info",
                            source="metric_group",
                            message=message,
                            group="perceptual",
                        )
                    metrics = filtered_metrics
                self._replace_nan_values(metrics)
                result.metrics.extend(metrics)
            except Exception as e:
                if self.strict:
                    raise RuntimeError(f"{group_name} group failed for {label}: {e}") from e
                result.metrics.append(MetricResult(
                    name=f"{group_name} (group failed)",
                    value=None,
                    description=str(e),
                    group=group_name,
                    confidence="heuristic",
                    calibration_note="A whole metric group failed; compare this run cautiously.",
                ))
                message = f"{group_name}: group failed ({e})"
                self._append_note(result, message)
                self._add_diagnostic(
                    result,
                    code="metric_group_failed",
                    severity="error",
                    source="metric_group",
                    message=str(e),
                    group=group_name,
                )

        self._finalize_result(result)
        return result

    def _reference_slice(
        self,
        sample_start: int,
        sample_end: int,
        sr: int,
    ) -> tuple[Optional[np.ndarray], Optional[int]]:
        if self._ref_audio is None:
            return None, None
        if self._ref_sr != sr:
            # Map the segment's time span onto the reference's own sample grid;
            # returning the whole reference compared every segment with the
            # start of the reference.
            sample_start = int(round(sample_start * self._ref_sr / sr))
            sample_end = int(round(sample_end * self._ref_sr / sr))
        if self._ref_audio.ndim == 1:
            return self._ref_audio[sample_start:sample_end], self._ref_sr
        return self._ref_audio[:, sample_start:sample_end], self._ref_sr

    def analyze_file(self, path: Path) -> FileResult:
        try:
            audio, sr = AudioLoader.load(path)
        except Exception as e:
            result = FileResult(path=str(path), error=str(e))
            self._add_diagnostic(
                result,
                code="audio_load_failed",
                severity="error",
                source="loader",
                message=str(e),
            )
            result.provenance = build_provenance(result.metrics)
            return result
        return self._analyze_loaded_audio(audio, sr, label=str(path))

    def _build_tasks(self, path: Path) -> list[_AnalysisTask]:
        try:
            audio, sr = AudioLoader.load(path)
        except Exception as e:
            return [
                _AnalysisTask(
                    path=path,
                    label=str(path),
                    audio=np.array([], dtype=np.float32),
                    sr=0,
                    load_error=str(e),
                )
            ]

        if not self.segment_seconds:
            return [
                _AnalysisTask(
                    path=path,
                    label=str(path),
                    audio=audio,
                    sr=sr,
                )
            ]

        segment_samples = max(1, int(round(self.segment_seconds * sr)))
        total_samples = audio.shape[-1] if audio.ndim > 1 else len(audio)
        if total_samples <= segment_samples:
            return [
                _AnalysisTask(
                    path=path,
                    label=str(path),
                    audio=audio,
                    sr=sr,
                    source_file=str(path),
                    segment_index=1,
                    total_segments=1,
                    segment_start_s=0.0,
                    segment_end_s=total_samples / sr,
                    sample_start=0,
                    sample_end=total_samples,
                )
            ]

        tasks: list[_AnalysisTask] = []
        bounds = list(range(0, total_samples, segment_samples))
        total_segments = len(bounds)
        for idx, start in enumerate(bounds, start=1):
            end = min(start + segment_samples, total_samples)
            segment_audio = audio[start:end] if audio.ndim == 1 else audio[:, start:end]
            start_s = start / sr
            end_s = end / sr
            label = f"{path} [segment {idx}/{total_segments} {start_s:.2f}-{end_s:.2f}s]"
            tasks.append(
                _AnalysisTask(
                    path=path,
                    label=label,
                    audio=segment_audio,
                    sr=sr,
                    source_file=str(path),
                    segment_index=idx,
                    total_segments=total_segments,
                    segment_start_s=start_s,
                    segment_end_s=end_s,
                    sample_start=start,
                    sample_end=end,
                )
            )
        return tasks

    def _run_task(self, task: _AnalysisTask) -> FileResult:
        if task.sr == 0:
            message = task.load_error or "failed to load audio"
            result = FileResult(path=task.label, source_file=task.source_file, error=message)
            self._add_diagnostic(result, code="audio_load_failed", severity="error", source="loader", message=message)
            return result
        return self._analyze_loaded_audio(
            task.audio,
            task.sr,
            label=task.label,
            source_file=task.source_file,
            segment_index=task.segment_index,
            total_segments=task.total_segments,
            segment_start_s=task.segment_start_s,
            segment_end_s=task.segment_end_s,
            sample_start=task.sample_start,
            sample_end=task.sample_end,
        )

    def analyze_all(self, paths: list[Path], on_progress=None) -> list[FileResult]:
        """Analyze all paths. on_progress(completed, total, path) called after each file."""
        if self.reference:
            try:
                self._load_reference()
            except Exception:
                if self.strict:
                    raise

        if not self.segment_seconds:
            total = len(paths)
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
                        if self.strict:
                            raise
                        results[idx] = FileResult(path=str(paths[idx]), source_file=str(paths[idx]), error=str(e))
                    completed += 1
                    if on_progress:
                        on_progress(completed, total, paths[idx])
            return results

        tasks: list[_AnalysisTask] = []
        for path in paths:
            tasks.extend(self._build_tasks(path))
        total = len(tasks)

        if self.workers == 1 or total == 1:
            results = []
            for i, task in enumerate(tasks):
                r = self._run_task(task)
                results.append(r)
                if on_progress:
                    on_progress(i + 1, total, task.path)
            return results

        results = [None] * total
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futures = {ex.submit(self._run_task, task): i for i, task in enumerate(tasks)}
            completed = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    if self.strict:
                        raise
                    task = tasks[idx]
                    results[idx] = FileResult(path=task.label, source_file=task.source_file, error=str(e))
                completed += 1
                if on_progress:
                    on_progress(completed, total, tasks[idx].path)
        return results

    async def analyze_all_async(self, paths: list[Path]) -> list[FileResult]:
        """Analyze all paths asynchronously while preserving result ordering."""
        if self.reference:
            try:
                await asyncio.to_thread(self._load_reference)
            except Exception:
                if self.strict:
                    raise

        if not self.segment_seconds:
            return await self._run_async_paths(paths)

        built_tasks = await asyncio.gather(
            *(asyncio.to_thread(self._build_tasks, path) for path in paths)
        )
        tasks = [task for group in built_tasks for task in group]
        return await self._run_async_analysis_tasks(tasks)

    async def _run_async_paths(self, paths: list[Path]) -> list[FileResult]:
        semaphore = asyncio.Semaphore(max(1, self.workers))
        results: list[Optional[FileResult]] = [None] * len(paths)

        async def _run(index: int, path: Path) -> None:
            async with semaphore:
                try:
                    result = await asyncio.to_thread(self.analyze_file, path)
                except Exception as exc:
                    if self.strict:
                        raise
                    result = FileResult(path=str(path), source_file=str(path), error=str(exc))
                results[index] = result

        await asyncio.gather(*(_run(idx, path) for idx, path in enumerate(paths)))
        return [result for result in results if result is not None]

    async def _run_async_analysis_tasks(self, tasks: list[_AnalysisTask]) -> list[FileResult]:
        semaphore = asyncio.Semaphore(max(1, self.workers))
        results: list[Optional[FileResult]] = [None] * len(tasks)

        async def _run(index: int, task: _AnalysisTask) -> None:
            async with semaphore:
                try:
                    result = await asyncio.to_thread(self._run_task, task)
                except Exception as exc:
                    if self.strict:
                        raise
                    result = FileResult(path=task.label, source_file=task.source_file, error=str(exc))
                results[index] = result

        await asyncio.gather(*(_run(idx, task) for idx, task in enumerate(tasks)))
        return [result for result in results if result is not None]

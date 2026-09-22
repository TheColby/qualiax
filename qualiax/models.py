"""
Data models for qualiax results.
"""
from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import Any, Optional

from .version import OUTPUT_SCHEMA_VERSION, __version__


@dataclass
class DiagnosticEntry:
    """Structured diagnostic emitted during analysis or validation."""
    code: str
    severity: str                    # info | warn | error
    source: str                      # loader | content_type | metric_group | rules | watch | runtime
    message: str
    group: Optional[str] = None
    metric: Optional[str] = None
    count: int = 1
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "source": self.source,
            "message": self.message,
            "group": self.group,
            "metric": self.metric,
            "count": self.count,
            "context": self.context,
        }


@dataclass
class GroupHealth:
    """Per-group health summary for downstream automation."""
    group: str
    status: str                      # ok | warn | partial | missing | error | skipped
    metric_count: int = 0
    warning_count: int = 0
    missing_count: int = 0
    diagnostic_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "status": self.status,
            "metric_count": self.metric_count,
            "warning_count": self.warning_count,
            "missing_count": self.missing_count,
            "diagnostic_count": self.diagnostic_count,
        }


@dataclass
class ProvenanceInfo:
    """Runtime and model provenance for reproducibility."""
    compute_backend: Optional[str] = None
    python_version: Optional[str] = None
    platform: Optional[str] = None
    runtime_fingerprint: Optional[str] = None
    model_runtime: Optional[str] = None
    model_assets: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "compute_backend": self.compute_backend,
            "python_version": self.python_version,
            "platform": self.platform,
            "runtime_fingerprint": self.runtime_fingerprint,
            "model_runtime": self.model_runtime,
            "model_assets": self.model_assets,
        }


@dataclass
class MetricResult:
    """A single measured metric."""
    name: str
    value: Any                    # float, int, str, or None
    unit: str = ""               # e.g. "dB", "Hz", "%", "s"
    description: str = ""        # human-readable explanation
    group: str = ""              # metric group name
    higher_is_better: Optional[bool] = None  # None = N/A
    warning: Optional[str] = None            # e.g. "below recommended threshold"
    reference_range: Optional[tuple[float, float]] = None  # (min, max) good range
    confidence: Optional[str] = None         # measured | model | proxy | heuristic
    calibration_note: Optional[str] = None   # caveat about trust / applicability

    def formatted_value(self) -> str:
        value = self.value
        if value is None:
            return "N/A"
        if isinstance(value, numbers.Integral):  # int, bool, numpy integers
            return str(value)
        if isinstance(value, numbers.Real):  # float, numpy floating
            number = float(value)
            if math.isnan(number):
                # NaN means "undefined"; render it like any other missing value.
                return "N/A"
            return f"{number:.4f}"
        return str(value)

    def value_with_unit(self) -> str:
        fv = self.formatted_value()
        if self.unit:
            return f"{fv} {self.unit}"
        return fv


@dataclass
class FileResult:
    """All metrics for a single audio file."""
    path: str
    source_file: Optional[str] = None
    segment_index: Optional[int] = None
    total_segments: Optional[int] = None
    segment_start_s: Optional[float] = None
    segment_end_s: Optional[float] = None
    metrics: list[MetricResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    confidence_notes: list[str] = field(default_factory=list)
    diagnostics: list[DiagnosticEntry] = field(default_factory=list)
    group_health: list[GroupHealth] = field(default_factory=list)
    provenance: Optional[ProvenanceInfo] = None
    insights: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    duration_s: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    bit_depth: Optional[int] = None
    content_type: Optional[str] = None   # "speech" | "music" | "noise" | "silence" | "mixed" | "unknown"
    speech_confidence: Optional[float] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileResult":
        """Rebuild a result from ``to_dict()`` output (or a parsed JSON report item)."""
        provenance = data.get("provenance")
        return cls(
            path=str(data.get("file", "")),
            source_file=data.get("source_file"),
            segment_index=data.get("segment_index"),
            total_segments=data.get("total_segments"),
            segment_start_s=data.get("segment_start_s"),
            segment_end_s=data.get("segment_end_s"),
            metrics=[
                MetricResult(
                    name=item["name"],
                    value=item.get("value"),
                    unit=item.get("unit", ""),
                    description=item.get("description", ""),
                    group=item.get("group", ""),
                    higher_is_better=item.get("higher_is_better"),
                    warning=item.get("warning"),
                    reference_range=(
                        tuple(item["reference_range"]) if item.get("reference_range") is not None else None
                    ),
                    confidence=item.get("confidence"),
                    calibration_note=item.get("calibration_note"),
                )
                for item in data.get("metrics", [])
            ],
            notes=list(data.get("notes", [])),
            confidence_notes=list(data.get("confidence_notes", [])),
            diagnostics=[DiagnosticEntry(**item) for item in data.get("diagnostics", [])],
            group_health=[GroupHealth(**item) for item in data.get("group_health", [])],
            provenance=ProvenanceInfo(**provenance) if provenance else None,
            insights=dict(data.get("insights") or {}),
            error=data.get("error"),
            duration_s=float(data.get("duration_s") or 0.0),
            sample_rate=int(data.get("sample_rate") or 0),
            channels=int(data.get("channels") or 0),
            bit_depth=data.get("bit_depth"),
            content_type=data.get("content_type"),
            speech_confidence=data.get("speech_confidence"),
        )

    def metrics_by_group(self) -> dict[str, list[MetricResult]]:
        groups: dict[str, list[MetricResult]] = {}
        for m in self.metrics:
            groups.setdefault(m.group, []).append(m)
        return groups

    def to_dict(self) -> dict:
        return {
            "schema_version": OUTPUT_SCHEMA_VERSION,
            "tool_version": __version__,
            "file": self.path,
            "source_file": self.source_file,
            "segment_index": self.segment_index,
            "total_segments": self.total_segments,
            "segment_start_s": self.segment_start_s,
            "segment_end_s": self.segment_end_s,
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "content_type": self.content_type,
            "speech_confidence": self.speech_confidence,
            "notes": self.notes,
            "confidence_notes": self.confidence_notes,
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "group_health": [health.to_dict() for health in self.group_health],
            "provenance": self.provenance.to_dict() if self.provenance is not None else None,
            "insights": self.insights,
            "error": self.error,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "unit": m.unit,
                    "description": m.description,
                    "group": m.group,
                    "higher_is_better": m.higher_is_better,
                    "warning": m.warning,
                    "reference_range": m.reference_range,
                    "confidence": m.confidence,
                    "calibration_note": m.calibration_note,
                }
                for m in self.metrics
            ],
        }

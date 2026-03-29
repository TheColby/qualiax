"""
Data models for qualiax results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


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

    def formatted_value(self) -> str:
        if self.value is None:
            return "N/A"
        if isinstance(self.value, float):
            return f"{self.value:.4f}"
        return str(self.value)

    def value_with_unit(self) -> str:
        fv = self.formatted_value()
        if self.unit:
            return f"{fv} {self.unit}"
        return fv


@dataclass
class FileResult:
    """All metrics for a single audio file."""
    path: str
    metrics: list[MetricResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    error: Optional[str] = None
    duration_s: float = 0.0
    sample_rate: int = 0
    channels: int = 0
    bit_depth: Optional[int] = None

    def metrics_by_group(self) -> dict[str, list[MetricResult]]:
        groups: dict[str, list[MetricResult]] = {}
        for m in self.metrics:
            groups.setdefault(m.group, []).append(m)
        return groups

    def to_dict(self) -> dict:
        return {
            "file": self.path,
            "duration_s": self.duration_s,
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "notes": self.notes,
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
                }
                for m in self.metrics
            ],
        }

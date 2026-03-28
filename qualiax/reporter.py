"""
Output reporters for qualiax results.
"""
from __future__ import annotations

import json
import csv
import io
import math
from typing import Optional

from .models import FileResult, MetricResult

# ─────────────────────────────────────────────────────────────────────────────
# ANSI color helpers
# ─────────────────────────────────────────────────────────────────────────────

class Color:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    RED     = "\033[91m"
    GREEN   = "\033[92m"
    YELLOW  = "\033[93m"
    BLUE    = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN    = "\033[96m"
    WHITE   = "\033[97m"
    GRAY    = "\033[90m"

    @classmethod
    def apply(cls, text: str, *codes: str) -> str:
        return "".join(codes) + text + cls.RESET


def _fmt_value(m: MetricResult, color: bool) -> str:
    fv = m.value_with_unit()
    if not color or m.value is None:
        return fv
    if m.warning:
        return Color.apply(fv, Color.YELLOW)
    if m.higher_is_better is not None and m.reference_range and m.value is not None:
        lo, hi = m.reference_range
        try:
            v = float(m.value)
            if lo <= v <= hi:
                return Color.apply(fv, Color.GREEN)
            else:
                return Color.apply(fv, Color.YELLOW)
        except (ValueError, TypeError):
            pass
    return fv


GROUP_ORDER = ["basic", "loudness", "spectral", "temporal", "noise", "speech", "perceptual"]

GROUP_LABELS = {
    "basic":      "📊  Basic File Info",
    "loudness":   "🔊  Loudness (EBU R128 / BS.1770)",
    "spectral":   "🎵  Spectral Analysis",
    "temporal":   "⏱️  Temporal Analysis",
    "noise":      "🔇  Noise & Distortion",
    "speech":     "🗣️  Speech Features",
    "perceptual": "👁️  Perceptual Quality Metrics",
}


# ─────────────────────────────────────────────────────────────────────────────
# Console Reporter
# ─────────────────────────────────────────────────────────────────────────────

class ConsoleReporter:
    def __init__(self, color: bool = True):
        self.color = color

    def _c(self, text: str, *codes: str) -> str:
        if not self.color:
            return text
        return Color.apply(text, *codes)

    def render(self, results: list[FileResult]) -> str:
        parts = []
        for result in results:
            parts.append(self._render_file(result))
        return "\n".join(parts)

    def _render_file(self, result: FileResult) -> str:
        lines = []
        sep = "═" * 72

        # Header
        lines.append(self._c(sep, Color.CYAN, Color.BOLD))
        fname = result.path
        lines.append(self._c(f"  FILE: {fname}", Color.WHITE, Color.BOLD))
        if result.error:
            lines.append(self._c(f"  ERROR: {result.error}", Color.RED))
            lines.append(self._c(sep, Color.CYAN, Color.BOLD))
            return "\n".join(lines)

        lines.append(self._c(
            f"  {result.duration_s:.3f}s  |  {result.sample_rate} Hz  |  "
            f"{result.channels}ch", Color.GRAY
        ))
        lines.append(self._c(sep, Color.CYAN, Color.BOLD))

        # Group metrics
        groups = result.metrics_by_group()
        ordered = [g for g in GROUP_ORDER if g in groups]
        ordered += [g for g in groups if g not in ordered]

        for group in ordered:
            metrics = groups[group]
            label = GROUP_LABELS.get(group, group.upper())
            lines.append("")
            lines.append(self._c(f"  {label}", Color.BOLD, Color.BLUE))
            lines.append(self._c("  " + "─" * 68, Color.DIM))

            for m in metrics:
                # Name column (left-padded)
                name = f"    {m.name}"
                val_str = _fmt_value(m, self.color)

                # Right-align value
                pad = max(2, 64 - len(m.name))
                line = name + " " * pad + val_str

                lines.append(line)

                # Show description if present
                if m.description:
                    lines.append(self._c(f"      {m.description}", Color.GRAY))

                # Show warning
                if m.warning:
                    lines.append(self._c(f"      ⚠️  {m.warning}", Color.YELLOW))

        lines.append("")
        lines.append(self._c(sep, Color.CYAN))
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# JSON Reporter
# ─────────────────────────────────────────────────────────────────────────────

class JsonReporter:
    def render(self, results: list[FileResult]) -> str:
        data = [r.to_dict() for r in results]
        return json.dumps(data, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(obj):
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ─────────────────────────────────────────────────────────────────────────────
# CSV Reporter
# ─────────────────────────────────────────────────────────────────────────────

class CsvReporter:
    def render(self, results: list[FileResult]) -> str:
        buf = io.StringIO()

        # Collect all metric names across all files
        all_metric_names: list[str] = []
        seen = set()
        for r in results:
            for m in r.metrics:
                if m.name not in seen:
                    all_metric_names.append(m.name)
                    seen.add(m.name)

        fieldnames = ["file", "duration_s", "sample_rate", "channels", "error"] + all_metric_names
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row: dict = {
                "file": r.path,
                "duration_s": r.duration_s,
                "sample_rate": r.sample_rate,
                "channels": r.channels,
                "error": r.error or "",
            }
            metric_lookup = {m.name: m.value for m in r.metrics}
            for name in all_metric_names:
                v = metric_lookup.get(name)
                if v is None:
                    row[name] = ""
                elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                    row[name] = ""
                else:
                    row[name] = v
            writer.writerow(row)

        return buf.getvalue()

"""
Output reporters for qualiax results.
"""
from __future__ import annotations

import json
import csv
import io
import math
from html import escape
from typing import Any, Optional

from .models import FileResult, MetricResult
from .version import OUTPUT_SCHEMA_VERSION, __version__

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


GROUP_ORDER = [
    "basic",
    "loudness",
    "spectral",
    "temporal",
    "noise",
    "speech",
    "perceptual",
    "prosody",
    "psychoacoustic",
    "speaker",
]

GROUP_LABELS = {
    "basic":      "📊  Basic File Info",
    "loudness":   "🔊  Loudness (EBU R128 / BS.1770)",
    "spectral":   "🎵  Spectral Analysis",
    "temporal":   "⏱️  Temporal Analysis",
    "noise":      "🔇  Noise & Distortion",
    "speech":     "🗣️  Speech Features",
    "perceptual": "👁️  Perceptual Quality Metrics",
    "prosody":    "🎙️  Prosody & Voice Dynamics",
    "psychoacoustic": "🧠  Psychoacoustic Analysis",
    "speaker":    "🪪  Speaker Characteristics",
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

        if result.content_type:
            if result.speech_confidence is not None:
                ct_tag = (
                    f"  [{result.content_type}, "
                    f"speech={result.speech_confidence:.2f}]"
                )
            else:
                ct_tag = f"  [{result.content_type}]"
        else:
            ct_tag = ""
        lines.append(self._c(
            f"  {result.duration_s:.3f}s  |  {result.sample_rate} Hz  |  "
            f"{result.channels}ch{ct_tag}", Color.GRAY
        ))
        if result.segment_index is not None:
            seg_range = (
                f"{result.segment_start_s:.2f}-{result.segment_end_s:.2f}s"
                if result.segment_start_s is not None and result.segment_end_s is not None
                else "range unavailable"
            )
            lines.append(self._c(
                f"  segment {result.segment_index}/{result.total_segments or '?'}  |  "
                f"source={result.source_file or result.path}  |  {seg_range}",
                Color.GRAY,
            ))
        lines.append(self._c(sep, Color.CYAN, Color.BOLD))

        if result.notes:
            lines.append("")
            lines.append(self._c("  Notes", Color.BOLD, Color.YELLOW))
            lines.append(self._c("  " + "─" * 68, Color.DIM))
            for note in result.notes:
                lines.append(self._c(f"    - {note}", Color.YELLOW))
        if result.confidence_notes:
            lines.append("")
            lines.append(self._c("  Confidence", Color.BOLD, Color.MAGENTA))
            lines.append(self._c("  " + "─" * 68, Color.DIM))
            for note in result.confidence_notes:
                lines.append(self._c(f"    - {note}", Color.MAGENTA))
        if result.diagnostics:
            lines.append("")
            lines.append(self._c("  Diagnostics", Color.BOLD, Color.YELLOW))
            lines.append(self._c("  " + "─" * 68, Color.DIM))
            for diagnostic in result.diagnostics:
                lines.append(
                    self._c(
                        f"    - [{diagnostic.severity}] {diagnostic.source}: {diagnostic.message}",
                        Color.YELLOW if diagnostic.severity != "error" else Color.RED,
                    )
                )
        if result.provenance:
            lines.append("")
            lines.append(self._c("  Provenance", Color.BOLD, Color.CYAN))
            lines.append(self._c("  " + "─" * 68, Color.DIM))
            lines.append(self._c(f"    - backend: {result.provenance.compute_backend or 'unknown'}", Color.CYAN))
            lines.append(self._c(f"    - runtime: {result.provenance.model_runtime or 'none'}", Color.CYAN))
            lines.append(
                self._c(
                    f"    - fingerprint: {result.provenance.runtime_fingerprint or 'unavailable'}",
                    Color.CYAN,
                )
            )

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
                if m.confidence:
                    lines.append(self._c(f"      trust: {m.confidence}", Color.MAGENTA))
                if m.calibration_note:
                    lines.append(self._c(f"      calibration: {m.calibration_note}", Color.GRAY))

        lines.append("")
        lines.append(self._c(sep, Color.CYAN))
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# JSON Reporter
# ─────────────────────────────────────────────────────────────────────────────

class JsonReporter:
    def render(self, results: list[FileResult]) -> str:
        data = _sanitize_for_json([r.to_dict() for r in results])
        return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)


class JsonlReporter:
    def render(self, results: list[FileResult]) -> str:
        return "\n".join(
            json.dumps(_sanitize_for_json(result.to_dict()), ensure_ascii=False, allow_nan=False)
            for result in results
        ) + ("\n" if results else "")


def _sanitize_for_json(obj):
    if isinstance(obj, dict):
        return {key: _sanitize_for_json(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(value) for value in obj]
    if isinstance(obj, tuple):
        return [_sanitize_for_json(value) for value in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _metric_status(metric: MetricResult) -> str:
    if metric.value is None:
        return "missing"
    if metric.warning:
        return "warn"
    return "ok"


def _group_status(metrics: list[MetricResult]) -> str:
    statuses = {_metric_status(metric) for metric in metrics}
    if "warn" in statuses:
        return "warn"
    if statuses == {"missing"}:
        return "missing"
    if "missing" in statuses:
        return "partial"
    return "ok"


def _result_status(result: FileResult) -> str:
    if result.error:
        return "error"
    if any(metric.warning for metric in result.metrics):
        return "warn"
    if any(metric.value is None for metric in result.metrics):
        return "partial"
    return "ok"


def _status_label(status: str) -> str:
    return {
        "ok": "OK",
        "warn": "Needs Review",
        "partial": "Partial",
        "missing": "Missing",
        "error": "Error",
    }.get(status, status.title())


def _metric_trust_text(metric: MetricResult) -> str:
    parts = []
    if metric.confidence:
        parts.append(metric.confidence)
    if metric.calibration_note:
        parts.append(metric.calibration_note)
    return " - ".join(parts)


class HtmlReporter:
    def render(self, results: list[FileResult]) -> str:
        cards = "\n".join(self._render_file(result) for result in results)
        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>qualiax Report</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1f1b17;
      --muted: #6b6258;
      --line: #d9cdbd;
      --accent: #b0572b;
      --accent-soft: #f5dfcf;
      --ok: #1b7f4d;
      --ok-soft: #d8f2e5;
      --warn: #a25a10;
      --warn-soft: #fde6c7;
      --error: #a12727;
      --error-soft: #f9d6d6;
      --partial: #5f55a2;
      --partial-soft: #e4def9;
      --missing: #6a7280;
      --missing-soft: #e8ebef;
      --shadow: 0 18px 40px rgba(49, 33, 16, 0.10);
      --radius: 18px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", Georgia, serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, #fff6e8 0, transparent 34%),
        radial-gradient(circle at top right, #f6e0cc 0, transparent 28%),
        linear-gradient(180deg, #f8f4ed 0%, var(--bg) 100%);
      line-height: 1.45;
    }}
    .page {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 40px 20px 64px;
    }}
    .hero {{
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 28px;
    }}
    .hero h1 {{
      margin: 0;
      font-size: clamp(2rem, 4vw, 3.4rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      margin: 8px 0 0;
      color: var(--muted);
      max-width: 760px;
    }}
    .meta {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .chip {{
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.75);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 0.92rem;
      color: var(--muted);
    }}
    .result-card {{
      background: var(--panel);
      border: 1px solid rgba(151, 123, 96, 0.22);
      border-radius: calc(var(--radius) + 4px);
      box-shadow: var(--shadow);
      padding: 24px;
      margin-bottom: 22px;
    }}
    .result-top {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      justify-content: space-between;
      align-items: flex-start;
      margin-bottom: 18px;
    }}
    .file-title h2 {{
      margin: 0 0 6px;
      font-size: 1.55rem;
      line-height: 1.05;
      word-break: break-word;
    }}
    .submeta {{
      color: var(--muted);
      font-size: 0.98rem;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 0.9rem;
      font-weight: 600;
      border: 1px solid transparent;
      white-space: nowrap;
    }}
    .badge.ok {{ color: var(--ok); background: var(--ok-soft); border-color: rgba(27,127,77,0.18); }}
    .badge.warn {{ color: var(--warn); background: var(--warn-soft); border-color: rgba(162,90,16,0.18); }}
    .badge.error {{ color: var(--error); background: var(--error-soft); border-color: rgba(161,39,39,0.18); }}
    .badge.partial, .badge.missing {{ color: var(--partial); background: var(--partial-soft); border-color: rgba(95,85,162,0.18); }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 10px;
      margin: 0 0 18px;
    }}
    .summary-item {{
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: linear-gradient(180deg, rgba(255,255,255,0.86), rgba(250,246,239,0.96));
    }}
    .summary-item .label {{
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .summary-item .value {{
      margin-top: 5px;
      font-size: 1.08rem;
      font-weight: 700;
    }}
    .notes {{
      margin: 0 0 18px;
      padding: 14px 16px;
      border-radius: 16px;
      background: #fff6ea;
      border: 1px solid #f2d1aa;
    }}
    .notes h3, .groups h3 {{
      margin: 0 0 10px;
      font-size: 1rem;
    }}
    .notes ul {{
      margin: 0;
      padding-left: 18px;
    }}
    .groups {{
      display: grid;
      gap: 16px;
    }}
    .group-card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      overflow: hidden;
      background: #fffdfa;
    }}
    .group-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      padding: 14px 16px;
      background: linear-gradient(90deg, rgba(176,87,43,0.12), rgba(176,87,43,0.03));
      border-bottom: 1px solid var(--line);
    }}
    .group-head h4 {{
      margin: 0;
      font-size: 1rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 12px 16px;
      vertical-align: top;
      border-bottom: 1px solid #efe4d8;
      text-align: left;
    }}
    th {{
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      background: rgba(244, 237, 227, 0.58);
    }}
    td.metric-name {{
      width: 29%;
      font-weight: 700;
    }}
    td.metric-value {{
      width: 16%;
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
      font-weight: 700;
    }}
    td.metric-desc {{
      color: var(--muted);
    }}
    td.metric-warning {{
      width: 22%;
      color: var(--warn);
    }}
    .error-box {{
      padding: 14px 16px;
      border-radius: 16px;
      background: var(--error-soft);
      color: var(--error);
      border: 1px solid rgba(161,39,39,0.18);
    }}
    @media (max-width: 760px) {{
      .page {{ padding: 28px 14px 40px; }}
      .result-card {{ padding: 18px; }}
      .group-head {{ align-items: flex-start; flex-direction: column; }}
      table, thead, tbody, tr, th, td {{ display: block; }}
      thead {{ display: none; }}
      tr {{ border-bottom: 1px solid #efe4d8; }}
      td {{
        border-bottom: none;
        padding-top: 6px;
        padding-bottom: 6px;
      }}
      td.metric-name {{ padding-top: 14px; }}
      td.metric-name::before,
      td.metric-value::before,
      td.metric-desc::before,
      td.metric-warning::before {{
        display: block;
        font-size: 0.73rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
        margin-bottom: 3px;
      }}
      td.metric-name::before {{ content: "Metric"; }}
      td.metric-value::before {{ content: "Value"; }}
      td.metric-desc::before {{ content: "Description"; }}
      td.metric-warning::before {{ content: "Warning"; }}
    }}
  </style>
</head>
<body>
  <main class="page">
    <section class="hero">
      <div>
        <h1>qualiax HTML Report</h1>
        <p>Self-contained analysis output with grouped metric tables, status badges, notes, and file-level summary metadata.</p>
      </div>
      <div class="meta">
        <span class="chip">tool {escape(__version__)}</span>
        <span class="chip">schema {escape(OUTPUT_SCHEMA_VERSION)}</span>
        <span class="chip">{len(results)} file{'s' if len(results) != 1 else ''}</span>
      </div>
    </section>
    {cards}
  </main>
</body>
</html>"""

    def _render_file(self, result: FileResult) -> str:
        status = _result_status(result)
        groups = result.metrics_by_group()
        ordered = [group for group in GROUP_ORDER if group in groups]
        ordered += [group for group in groups if group not in ordered]

        summary_items = [
            ("Duration", f"{result.duration_s:.3f}s" if result.duration_s else "N/A"),
            ("Sample Rate", f"{result.sample_rate} Hz" if result.sample_rate else "N/A"),
            ("Channels", str(result.channels) if result.channels else "N/A"),
            ("Content Type", result.content_type or "unknown"),
            (
                "Speech Confidence",
                f"{result.speech_confidence:.2f}" if result.speech_confidence is not None else "N/A",
            ),
        ]
        if result.segment_index is not None:
            summary_items.extend(
                [
                    ("Segment", f"{result.segment_index}/{result.total_segments or '?'}"),
                    (
                        "Segment Range",
                        (
                            f"{result.segment_start_s:.2f}-{result.segment_end_s:.2f}s"
                            if result.segment_start_s is not None and result.segment_end_s is not None
                            else "N/A"
                        ),
                    ),
                    ("Source File", result.source_file or result.path),
                ]
            )
        summary_html = "\n".join(
            f'<div class="summary-item"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div></div>'
            for label, value in summary_items
        )

        notes_html = ""
        if result.notes:
            notes_html = (
                '<section class="notes"><h3>Notes</h3><ul>'
                + "".join(f"<li>{escape(note)}</li>" for note in result.notes)
                + "</ul></section>"
            )
        confidence_html = ""
        if result.confidence_notes:
            confidence_html = (
                '<section class="notes"><h3>Confidence &amp; Calibration</h3><ul>'
                + "".join(f"<li>{escape(note)}</li>" for note in result.confidence_notes)
                + "</ul></section>"
            )
        diagnostics_html = ""
        if result.diagnostics:
            diagnostics_html = (
                '<section class="notes"><h3>Diagnostics</h3><ul>'
                + "".join(
                    f"<li>[{escape(diagnostic.severity)}] {escape(diagnostic.source)}: {escape(diagnostic.message)}</li>"
                    for diagnostic in result.diagnostics
                )
                + "</ul></section>"
            )
        provenance_html = ""
        if result.provenance:
            provenance_html = (
                '<section class="notes"><h3>Provenance</h3><ul>'
                + f"<li>Backend: {escape(result.provenance.compute_backend or 'unknown')}</li>"
                + f"<li>Model runtime: {escape(result.provenance.model_runtime or 'none')}</li>"
                + f"<li>Fingerprint: {escape(result.provenance.runtime_fingerprint or 'unavailable')}</li>"
                + "</ul></section>"
            )

        error_html = ""
        if result.error:
            error_html = f'<div class="error-box">{escape(result.error)}</div>'

        groups_html = "\n".join(self._render_group(group, groups[group]) for group in ordered)
        return f"""
<section class="result-card">
  <div class="result-top">
    <div class="file-title">
      <h2>{escape(result.path)}</h2>
      <div class="submeta">Grouped metrics and review-oriented status summary.</div>
    </div>
    <div class="badge {status}">{escape(_status_label(status))}</div>
  </div>
  <div class="summary-grid">
    {summary_html}
  </div>
  {notes_html}
  {confidence_html}
  {diagnostics_html}
  {provenance_html}
  {error_html}
  <section class="groups">
    {groups_html}
  </section>
</section>"""

    def _render_group(self, group: str, metrics: list[MetricResult]) -> str:
        status = _group_status(metrics)
        rows = "".join(
            f"""
      <tr>
        <td class="metric-name">{escape(metric.name)}</td>
        <td class="metric-value">{escape(metric.value_with_unit())}</td>
        <td class="metric-desc">{escape(metric.description or '')}</td>
        <td class="metric-warning">{escape(metric.warning or '')}</td>
        <td class="metric-desc">{escape(_metric_trust_text(metric))}</td>
      </tr>"""
            for metric in metrics
        )
        return f"""
<article class="group-card">
  <div class="group-head">
    <h4>{escape(GROUP_LABELS.get(group, group.upper()))}</h4>
    <div class="badge {status}">{escape(_status_label(status))}</div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Metric</th>
        <th>Value</th>
        <th>Description</th>
        <th>Warning</th>
        <th>Trust</th>
      </tr>
    </thead>
    <tbody>
      {rows}
    </tbody>
  </table>
</article>"""


class MarkdownReporter:
    def render(self, results: list[FileResult]) -> str:
        lines = [
            "# qualiax Report",
            "",
            f"- Tool version: `{__version__}`",
            f"- Schema version: `{OUTPUT_SCHEMA_VERSION}`",
            f"- Files: `{len(results)}`",
        ]
        for result in results:
            lines.extend(self._render_file(result))
        return "\n".join(lines) + "\n"

    def _render_file(self, result: FileResult) -> list[str]:
        lines = [
            "",
            f"## `{result.path}`",
            "",
            f"- Status: **{_status_label(_result_status(result))}**",
            f"- Duration: `{result.duration_s:.3f}s`" if result.duration_s else "- Duration: `N/A`",
            f"- Sample rate: `{result.sample_rate} Hz`" if result.sample_rate else "- Sample rate: `N/A`",
            f"- Channels: `{result.channels}`" if result.channels else "- Channels: `N/A`",
            f"- Content type: `{result.content_type or 'unknown'}`",
            (
                f"- Speech confidence: `{result.speech_confidence:.2f}`"
                if result.speech_confidence is not None
                else "- Speech confidence: `N/A`"
            ),
        ]
        if result.segment_index is not None:
            lines.extend(
                [
                    f"- Segment: `{result.segment_index}/{result.total_segments or '?'}`",
                    (
                        f"- Segment range: `{result.segment_start_s:.2f}-{result.segment_end_s:.2f}s`"
                        if result.segment_start_s is not None and result.segment_end_s is not None
                        else "- Segment range: `N/A`"
                    ),
                    f"- Source file: `{_md_escape(result.source_file or result.path)}`",
                ]
            )
        if result.error:
            lines.extend(["", f"> Error: {result.error}"])
            return lines
        if result.notes:
            lines.extend(["", "### Notes", ""])
            lines.extend(f"- {_md_escape(note)}" for note in result.notes)
        if result.confidence_notes:
            lines.extend(["", "### Confidence & Calibration", ""])
            lines.extend(f"- {_md_escape(note)}" for note in result.confidence_notes)
        if result.diagnostics:
            lines.extend(["", "### Diagnostics", ""])
            lines.extend(
                f"- [{_md_escape(diagnostic.severity)}] {_md_escape(diagnostic.source)}: {_md_escape(diagnostic.message)}"
                for diagnostic in result.diagnostics
            )
        if result.provenance:
            lines.extend(
                [
                    "",
                    "### Provenance",
                    "",
                    f"- Backend: `{_md_escape(result.provenance.compute_backend or 'unknown')}`",
                    f"- Model runtime: `{_md_escape(result.provenance.model_runtime or 'none')}`",
                    f"- Fingerprint: `{_md_escape(result.provenance.runtime_fingerprint or 'unavailable')}`",
                ]
            )
        groups = result.metrics_by_group()
        ordered = [group for group in GROUP_ORDER if group in groups]
        ordered += [group for group in groups if group not in ordered]
        for group in ordered:
            lines.extend(["", f"### {_md_escape(GROUP_LABELS.get(group, group.title()))}", ""])
            lines.append("| Metric | Value | Description | Warning | Trust |")
            lines.append("| --- | --- | --- | --- | --- |")
            for metric in groups[group]:
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            _md_escape(metric.name),
                            _md_escape(metric.value_with_unit()),
                            _md_escape(metric.description or ""),
                            _md_escape(metric.warning or ""),
                            _md_escape(_metric_trust_text(metric)),
                        ]
                    )
                    + " |"
                )
        return lines


def _md_escape(value: Any) -> str:
    text = str(value)
    return text.replace("\\", "\\\\").replace("|", "\\|").replace("\n", " ")


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

        fieldnames = [
            "schema_version",
            "tool_version",
            "file",
            "source_file",
            "segment_index",
            "total_segments",
            "segment_start_s",
            "segment_end_s",
            "duration_s",
            "sample_rate",
            "channels",
            "content_type",
            "speech_confidence",
            "notes",
            "confidence_notes",
            "diagnostics",
            "group_health",
            "provenance_backend",
            "provenance_runtime",
            "runtime_fingerprint",
            "error",
        ] + all_metric_names
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row: dict = {
                "schema_version": OUTPUT_SCHEMA_VERSION,
                "tool_version": __version__,
                "file": r.path,
                "source_file": r.source_file or "",
                "segment_index": r.segment_index if r.segment_index is not None else "",
                "total_segments": r.total_segments if r.total_segments is not None else "",
                "segment_start_s": r.segment_start_s if r.segment_start_s is not None else "",
                "segment_end_s": r.segment_end_s if r.segment_end_s is not None else "",
                "duration_s": r.duration_s,
                "sample_rate": r.sample_rate,
                "channels": r.channels,
                "content_type": r.content_type or "",
                "speech_confidence": r.speech_confidence if r.speech_confidence is not None else "",
                "notes": " | ".join(r.notes),
                "confidence_notes": " | ".join(r.confidence_notes),
                "diagnostics": " | ".join(
                    f"[{diagnostic.severity}] {diagnostic.source}: {diagnostic.message}"
                    for diagnostic in r.diagnostics
                ),
                "group_health": " | ".join(
                    f"{health.group}={health.status}"
                    for health in r.group_health
                ),
                "provenance_backend": r.provenance.compute_backend if r.provenance else "",
                "provenance_runtime": r.provenance.model_runtime if r.provenance else "",
                "runtime_fingerprint": r.provenance.runtime_fingerprint if r.provenance else "",
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

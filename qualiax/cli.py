"""
qualiax - Perceptual speech & audio quality analyzer
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional

import click

from .analyzer import AudioAnalyzer
from .reporter import ConsoleReporter, JsonReporter, CsvReporter

# ANSI color codes
_GREEN  = "\033[32m"
_CYAN   = "\033[36m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"
_UP     = "\033[1A"
_ERASE  = "\033[2K"


def _progress_bar(completed: int, total: int, path: Path, color: bool) -> str:
    width = 28
    filled = int(width * completed / total) if total else width
    if color:
        bar = _GREEN + "█" * filled + _DIM + "░" * (width - filled) + _RESET
        counter = f"{_CYAN}{completed}/{total}{_RESET}"
        name = _DIM + path.name[:40] + _RESET
    else:
        bar = "█" * filled + "░" * (width - filled)
        counter = f"{completed}/{total}"
        name = path.name[:40]
    return f"\r  [{bar}] {counter}  {name}"

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aac", ".m4a", ".opus", ".aiff", ".aif"}


def collect_files(path: Path) -> list[Path]:
    """Collect all supported audio files from a path (file or directory)."""
    if path.is_file():
        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [path]
        else:
            click.echo(f"[warning] Unsupported file type: {path.suffix}", err=True)
            return []
    elif path.is_dir():
        files = []
        for ext in SUPPORTED_EXTENSIONS:
            files.extend(path.rglob(f"*{ext}"))
            files.extend(path.rglob(f"*{ext.upper()}"))
        return sorted(set(files))
    else:
        click.echo(f"[error] Path not found: {path}", err=True)
        return []


@click.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--silent", is_flag=True, help="Suppress console output (useful with --output).")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Write results to a file (format determined by extension: .json, .csv).")
@click.option("--format", "-f", "fmt", type=click.Choice(["pretty", "json", "csv"]),
              default="pretty", help="Output format (default: pretty).")
@click.option("--reference", "-r", type=click.Path(exists=True), default=None,
              help="Reference audio file for intrusive metrics (PESQ, STOI, SI-SDR, etc.).")
@click.option("--metrics", "-m", default=None,
              help="Comma-separated list of metric groups to include. "
                   "Groups: basic, loudness, spectral, temporal, noise, speech, perceptual, all. "
                   "Default: all.")
@click.option("--no-color", is_flag=True, help="Disable colored output.")
@click.option("--verbose", "-v", is_flag=True, help="Show analysis progress and warnings.")
@click.option("--workers", "-w", default=1, type=int,
              help="Number of parallel worker threads (default: 1).")
def main(
    paths: tuple[str, ...],
    silent: bool,
    output: Optional[str],
    fmt: str,
    reference: Optional[str],
    metrics: Optional[str],
    no_color: bool,
    verbose: bool,
    workers: int,
) -> None:
    """
    \b
    qualiax — Perceptual Speech & Audio Quality Analyzer
    ═══════════════════════════════════════════════════════
    Analyze one or more audio files for a comprehensive set of
    perceptual speech and audio quality metrics.

    \b
    PATHS can be individual audio files or directories (searched recursively).

    \b
    Examples:
      qualiax recording.wav
      qualiax ./recordings/ --format json --output results.json
      qualiax call.wav --reference clean_ref.wav
      qualiax *.wav --metrics basic,loudness,spectral --no-color
    """
    # --- Collect files ---
    all_files: list[Path] = []
    for p in paths:
        all_files.extend(collect_files(Path(p)))

    if not all_files:
        click.echo("[error] No supported audio files found.", err=True)
        sys.exit(1)

    # --- Parse metric groups ---
    requested_groups = _parse_metric_groups(metrics)

    # --- Parse reference ---
    ref_path = Path(reference) if reference else None

    # --- Run analysis ---
    analyzer = AudioAnalyzer(
        metric_groups=requested_groups,
        reference=ref_path,
        verbose=verbose,
        workers=workers,
    )

    use_color = not no_color and sys.stderr.isatty()

    def _on_progress(completed, total, path):
        if not silent:
            line = _progress_bar(completed, total, path, color=use_color)
            click.echo(line, nl=(completed == total), err=True)

    results = analyzer.analyze_all(all_files, on_progress=_on_progress)

    # --- Report ---
    if output:
        out_path = Path(output)
        out_fmt = _detect_format(out_path, fmt)
        reporter = _make_reporter(out_fmt, color=False)
        text = reporter.render(results)
        out_path.write_text(text, encoding="utf-8")
        if not silent:
            click.echo(f"Results written to {out_path}", err=True)

    if not silent:
        console_fmt = fmt if fmt != "pretty" else "pretty"
        reporter = _make_reporter(console_fmt, color=not no_color)
        click.echo(reporter.render(results))

    # Exit code: 0 if all succeeded, 1 if any file failed
    if any(r.error for r in results):
        sys.exit(1)


def _parse_metric_groups(metrics: Optional[str]) -> set[str]:
    valid = {"basic", "loudness", "spectral", "temporal", "noise", "speech", "perceptual", "all"}
    if not metrics:
        return {"all"}
    groups = {g.strip().lower() for g in metrics.split(",")}
    unknown = groups - valid
    if unknown:
        click.echo(f"[warning] Unknown metric groups: {', '.join(unknown)}. "
                   f"Valid groups: {', '.join(sorted(valid))}", err=True)
    return groups or {"all"}


def _detect_format(path: Path, fallback: str) -> str:
    ext = path.suffix.lower()
    if ext == ".json":
        return "json"
    if ext == ".csv":
        return "csv"
    return fallback


def _make_reporter(fmt: str, color: bool):
    if fmt == "json":
        return JsonReporter()
    if fmt == "csv":
        return CsvReporter()
    return ConsoleReporter(color=color)

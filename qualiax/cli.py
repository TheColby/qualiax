"""
qualiax - Perceptual speech & audio quality analyzer
"""
from __future__ import annotations

import sys
import tempfile
import time
import warnings
import wave
import json
from pathlib import Path
from queue import Empty, Queue
from typing import Optional

import click
import numpy as np

from .analyzer import AudioAnalyzer
from .discovery import SUPPORTED_EXTENSIONS, collect_files as _collect_files
from .diffing import diff_reports, render_diff
from .insights import (
    build_insight_summary_payload,
    enrich_existing_report,
    enrich_results,
    export_flagged_segment_snippets,
    validate_insights_report,
)
from .metrics import available_metric_groups
from .presets import available_presets, get_preset, lint_preset
from .reporter import ConsoleReporter, JsonReporter, JsonlReporter, CsvReporter, HtmlReporter, MarkdownReporter
from .rules import apply_threshold_rules, lint_threshold_rules, load_threshold_rules
from .scorecards import build_scorecard, render_scorecard
from .validation import validate_report_file

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

_MAX_PENDING_WATCH_ROUNDS = 120
_DEFAULT_WATCH_DEBOUNCE = 0.0
_DEFAULT_WATCH_RETRIES = 2
_DEFAULT_WATCH_BACKOFF = 2.0
_DEFAULT_WATCH_MAX_PENDING = 512

def collect_files(path: Path) -> list[Path]:
    """Collect all supported audio files from a path (file or directory)."""
    try:
        return _collect_files(path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc


@click.command()
@click.argument("paths", nargs=-1, required=False, type=click.Path())
@click.option("--silent", is_flag=True, help="Suppress console output (useful with --output).")
@click.option("--save-sidecar", is_flag=True,
              help="Write per-file JSON sidecars next to each analyzed source.")
@click.option("--force", is_flag=True,
              help="Allow overwriting existing output or sidecar files.")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Write results to a file (format determined by extension: .json, .jsonl, .csv, .html, .md).")
@click.option("--format", "-f", "fmt", type=click.Choice(["pretty", "json", "jsonl", "csv", "html", "markdown"]),
              default="pretty", help="Output format (default: pretty).")
@click.option("--reference", "-r", type=click.Path(exists=True), default=None,
              help="Reference audio file for intrusive metrics (PESQ, STOI, SI-SDR, etc.).")
@click.option("--metrics", "-m", default=None,
              help="Comma-separated list of metric groups to include. "
                   "Groups: basic, loudness, spectral, temporal, noise, speech, perceptual, "
                   "prosody, psychoacoustic, speaker, all. "
                   "Default: all.")
@click.option("--no-color", is_flag=True, help="Disable colored output.")
@click.option("--verbose", "-v", is_flag=True, help="Show analysis progress and warnings.")
@click.option("--workers", "-w", default=1, type=int,
              help="Number of parallel worker threads (default: 1).")
@click.option("--preset", type=click.Choice(available_presets(), case_sensitive=False), default=None,
              help="Built-in analysis preset: podcast, call-center-qa, speech-enhancement, music-mastering.")
@click.option("--scorecard", type=click.Path(), default=None,
              help="Write an aggregate scorecard (.json, .html, .md) for the analyzed batch.")
@click.option("--rules", type=click.Path(exists=True), default=None,
              help="Threshold rules config (.json or .toml). Metric violations return exit code 2.")
@click.option("--lint-rules", is_flag=True,
              help="Validate preset/rule configuration and exit without analyzing audio.")
@click.option("--segment-seconds", type=float, default=None,
              help="Split each input into fixed-length segments before analysis.")
@click.option("--mic-seconds", type=float, default=None,
              help="Capture microphone input for N seconds and analyze it.")
@click.option("--mic-sample-rate", type=int, default=16_000,
              help="Microphone capture sample rate for --mic-seconds (default: 16000).")
@click.option("--watch", is_flag=True,
              help="Watch input directories for new audio files and analyze them on arrival.")
@click.option("--watch-interval", type=float, default=2.0,
              help="Polling interval in seconds for --watch (default: 2.0).")
@click.option("--watch-limit", type=int, default=None,
              help="Stop --watch after this many new files have been analyzed.")
@click.option("--watch-debounce", type=float, default=_DEFAULT_WATCH_DEBOUNCE,
              help="Seconds a new file must remain stable before watch mode analyzes it.")
@click.option("--watch-retries", type=int, default=_DEFAULT_WATCH_RETRIES,
              help="Retry count for watch-mode analysis failures before dropping a file.")
@click.option("--watch-backoff", type=float, default=_DEFAULT_WATCH_BACKOFF,
              help="Base backoff in seconds between watch-mode retries.")
@click.option("--watch-max-pending", type=int, default=_DEFAULT_WATCH_MAX_PENDING,
              help="Maximum number of pending watch candidates kept in memory.")
@click.option("--validate-output", is_flag=True,
              help="Validate JSON/JSONL report or scorecard outputs against the built-in schema.")
@click.option("--strict", is_flag=True,
              help="Fail the run on unexpected metric-group failures instead of degrading gracefully.")
@click.option("--include-demographics", is_flag=True,
              help="Include heuristic speaker age/gender estimates in the speaker group.")
@click.option("--insights", is_flag=True,
              help="Add composite fingerprints, defect labels, suggestions, drift, audit, CI, and heatmap payloads.")
@click.option("--baseline", type=click.Path(exists=True), default=None,
              help="Compare --insights output against a prior JSON report baseline.")
@click.option("--ci", is_flag=True,
              help="Add CI-oriented quality gate checks to --insights output.")
@click.option("--drift", is_flag=True,
              help="Add sequential drift-monitor comparisons to --insights output.")
@click.option("--drift-state", type=click.Path(), default=None,
              help="Persist --drift comparison state across runs or watch sessions.")
@click.option("--fingerprint-sensitivity",
              type=click.Choice(["coarse", "balanced", "strict"], case_sensitive=False),
              default="balanced",
              help="Quality fingerprint bucket sensitivity for --insights.")
@click.option("--insight-rules", type=click.Path(exists=True), default=None,
              help="Custom JSON/TOML rules for --insights labels, suggestions, and CI gates.")
@click.option("--insight-snippets", type=click.Path(), default=None,
              help="Export WAV snippets for flagged insight segments into this directory.")
@click.option("--insights-summary", type=click.Path(), default=None,
              help="Write compact pipeline-oriented insights JSON summary.")
def main(
    paths: tuple[str, ...],
    silent: bool,
    save_sidecar: bool,
    force: bool,
    output: Optional[str],
    fmt: str,
    reference: Optional[str],
    metrics: Optional[str],
    no_color: bool,
    verbose: bool,
    workers: int,
    preset: Optional[str],
    scorecard: Optional[str],
    rules: Optional[str],
    lint_rules: bool,
    segment_seconds: Optional[float],
    mic_seconds: Optional[float],
    mic_sample_rate: int,
    watch: bool,
    watch_interval: float,
    watch_limit: Optional[int],
    watch_debounce: float,
    watch_retries: int,
    watch_backoff: float,
    watch_max_pending: int,
    validate_output: bool,
    strict: bool,
    include_demographics: bool,
    insights: bool,
    baseline: Optional[str],
    ci: bool,
    drift: bool,
    drift_state: Optional[str],
    fingerprint_sensitivity: str,
    insight_rules: Optional[str],
    insight_snippets: Optional[str],
    insights_summary: Optional[str],
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
      qualiax - --output report.md --format markdown
      qualiax ./calls --output results.json --scorecard scorecard.md --silent
      qualiax long_call.wav --segment-seconds 30 --output segments.json
      qualiax --mic-seconds 5 --output mic.json --silent
      qualiax ./incoming --watch --watch-limit 3 --output arrivals.json --silent
      qualiax call.wav --reference clean_ref.wav
      qualiax episode.wav --preset podcast --output report.json --silent
      qualiax diff before.json after.json --format markdown
      qualiax *.wav --metrics basic,loudness,spectral --no-color
    """
    if paths and paths[0] == "insights":
        _run_insights_subcommand(
            args=paths[1:],
            output=output,
            silent=silent,
            force=force,
            baseline=baseline,
            ci=ci,
            drift=drift,
            drift_state=drift_state,
            fingerprint_sensitivity=fingerprint_sensitivity,
            preset=preset,
            insight_rules=insight_rules,
        )
        return

    if paths and paths[0] == "diff":
        if (
            segment_seconds is not None
            or mic_seconds is not None
            or watch
            or watch_limit is not None
            or watch_interval != 2.0
            or watch_debounce != _DEFAULT_WATCH_DEBOUNCE
            or watch_retries != _DEFAULT_WATCH_RETRIES
            or watch_backoff != _DEFAULT_WATCH_BACKOFF
            or watch_max_pending != _DEFAULT_WATCH_MAX_PENDING
            or mic_sample_rate != 16_000
            or lint_rules
            or validate_output
            or insights
            or baseline
            or ci
            or drift
            or drift_state
            or fingerprint_sensitivity != "balanced"
            or insight_rules
            or insight_snippets
            or insights_summary
        ):
            raise click.UsageError(
                "diff mode does not support segment, microphone, watch, lint, validation, or insights options."
            )
        _run_diff(
            diff_args=paths[1:],
            silent=silent,
            force=force,
            output=output,
            fmt=fmt,
            no_color=no_color,
            save_sidecar=save_sidecar,
            reference=reference,
            metrics=metrics,
            workers=workers,
            preset=preset,
            scorecard=scorecard,
            rules=rules,
            lint_rules=lint_rules,
            validate_output=validate_output,
            strict=strict,
            include_demographics=include_demographics,
            insights=insights,
            baseline=baseline,
            ci=ci,
            drift=drift,
            drift_state=drift_state,
            fingerprint_sensitivity=fingerprint_sensitivity,
            insight_rules=insight_rules,
            insight_snippets=insight_snippets,
            insights_summary=insights_summary,
        )
        return

    if mic_seconds is not None and paths:
        raise click.UsageError("--mic-seconds cannot be combined with input paths.")
    if watch and mic_seconds is not None:
        raise click.UsageError("--watch cannot be combined with --mic-seconds.")
    if not paths and mic_seconds is None and not lint_rules:
        raise click.UsageError("Provide at least one input path, use '-', or set --mic-seconds.")

    # --- Collect files ---
    all_files: list[Path] = []
    stdin_paths: list[Path] = []
    display_labels: dict[str, str] = {}
    watch_dirs: list[Path] = []

    if mic_seconds is not None:
        if save_sidecar:
            raise click.UsageError("--save-sidecar is not supported for microphone input.")
        temp_path = _capture_microphone_wav(mic_seconds, mic_sample_rate, silent=silent)
        stdin_paths.append(temp_path)
        all_files.append(temp_path)
        display_labels[str(temp_path)] = "microphone"
    elif watch:
        for raw_path in paths:
            path = Path(raw_path)
            if raw_path == "-":
                raise click.UsageError("--watch does not support stdin input.")
            if not path.exists():
                raise click.ClickException(f"Path not found: {path}")
            if not path.is_dir():
                raise click.UsageError("--watch only supports directory inputs.")
            watch_dirs.append(path)
    elif not lint_rules:
        for p in paths:
            if p == "-":
                if save_sidecar:
                    raise click.UsageError("--save-sidecar is not supported for stdin input.")
                temp_path = _materialize_stdin_wav()
                stdin_paths.append(temp_path)
                all_files.append(temp_path)
                display_labels[str(temp_path)] = "stdin"
                continue
            all_files.extend(collect_files(Path(p)))

        if not all_files:
            click.echo("[error] No supported audio files found.", err=True)
            sys.exit(1)

    try:
        if silent and not output and not save_sidecar and not scorecard and not insights_summary:
            raise click.UsageError("--silent requires --output, --scorecard, or --save-sidecar (or --insights-summary).")
        if _paths_conflict(output, scorecard):
            raise click.UsageError("--output and --scorecard must point to different paths.")
        if _paths_conflict(output, insights_summary) or _paths_conflict(scorecard, insights_summary):
            raise click.UsageError("--insights-summary must point to a different path than --output or --scorecard.")

        if save_sidecar and not force:
            existing = [str(p.with_suffix(".json")) for p in all_files if p.with_suffix(".json").exists()]
            if existing:
                raise click.ClickException(
                    "Refusing to overwrite existing sidecar file(s) without --force: "
                    + ", ".join(existing[:3])
                    + (" ..." if len(existing) > 3 else "")
                )

        # --- Parse metric groups ---
        selected_preset = get_preset(preset) if preset else None
        requested_groups = _parse_metric_groups(metrics, selected_preset.metric_groups if selected_preset else None)

        # --- Parse reference ---
        ref_path = Path(reference) if reference else None
        preset_rule_defs = list(selected_preset.rules) if selected_preset else []
        user_rule_defs = []
        if rules:
            user_rule_defs = load_threshold_rules(Path(rules))
        if segment_seconds is not None and segment_seconds <= 0:
            raise click.BadParameter("--segment-seconds must be greater than 0.", param_hint="--segment-seconds")
        if mic_seconds is not None and mic_seconds <= 0:
            raise click.BadParameter("--mic-seconds must be greater than 0.", param_hint="--mic-seconds")
        if mic_sample_rate <= 0:
            raise click.BadParameter("--mic-sample-rate must be greater than 0.", param_hint="--mic-sample-rate")
        if watch_interval <= 0:
            raise click.BadParameter("--watch-interval must be greater than 0.", param_hint="--watch-interval")
        if watch_limit is not None and watch_limit <= 0:
            raise click.BadParameter("--watch-limit must be greater than 0.", param_hint="--watch-limit")
        if watch_debounce < 0:
            raise click.BadParameter("--watch-debounce must be >= 0.", param_hint="--watch-debounce")
        if watch_retries < 0:
            raise click.BadParameter("--watch-retries must be >= 0.", param_hint="--watch-retries")
        if watch_backoff < 0:
            raise click.BadParameter("--watch-backoff must be >= 0.", param_hint="--watch-backoff")
        if watch_max_pending <= 0:
            raise click.BadParameter("--watch-max-pending must be greater than 0.", param_hint="--watch-max-pending")

        preset_lints = lint_preset(selected_preset) if selected_preset else []
        user_rule_lints = lint_threshold_rules(user_rule_defs, valid_groups=set(available_metric_groups()))
        if lint_rules:
            issues = preset_lints + user_rule_lints
            if not issues and not (selected_preset or user_rule_defs):
                raise click.UsageError("--lint-rules requires --rules or --preset.")
            for issue in issues:
                label = "error" if issue.level == "error" else "warn"
                click.echo(f"[{label}] {issue.message}", err=True)
            sys.exit(2 if any(issue.level == "error" for issue in issues) else 0)
        error_lints = [issue for issue in preset_lints + user_rule_lints if issue.level == "error"]
        if error_lints:
            raise click.ClickException(error_lints[0].message)

        # --- Run analysis ---
        analyzer = AudioAnalyzer(
            metric_groups=requested_groups,
            reference=ref_path,
            verbose=verbose,
            workers=workers,
            segment_seconds=segment_seconds,
            strict=strict,
            include_demographics=include_demographics,
        )

        use_color = not no_color and sys.stderr.isatty()

        def _on_progress(completed, total, path):
            if not silent:
                line = _progress_bar(completed, total, Path(display_labels.get(str(path), str(path))), color=use_color)
                click.echo(line, nl=(completed == total), err=True)

        def _normalize_results(results):
            for result in results:
                result.path = display_labels.get(result.path, result.path)
                if result.source_file:
                    result.source_file = display_labels.get(result.source_file, result.source_file)
            return results

        def _write_sidecars(batch_results, *, allow_overwrite: bool) -> None:
            if not save_sidecar:
                return
            results_by_source: dict[str, list] = {}
            for result in batch_results:
                key = result.source_file or result.path
                results_by_source.setdefault(key, []).append(result)
            if not allow_overwrite:
                existing = [
                    str(Path(source_file).with_suffix(".json"))
                    for source_file in results_by_source
                    if Path(source_file).with_suffix(".json").exists()
                ]
                if existing:
                    raise click.ClickException(
                        "Refusing to overwrite existing sidecar file(s) without --force: "
                        + ", ".join(existing[:3])
                        + (" ..." if len(existing) > 3 else "")
                    )
            for source_file, grouped_results in results_by_source.items():
                auto_path = Path(source_file).with_suffix(".json")
                auto_path.write_text(JsonReporter().render(grouped_results), encoding="utf-8")
            if not silent:
                click.echo("JSON sidecars saved alongside source file(s)", err=True)

        def _validate_written_path(path: Path) -> None:
            issues = validate_report_file(path)
            if issues:
                raise click.ClickException(
                    "Output validation failed: "
                    + "; ".join(f"{issue.path}: {issue.message}" for issue in issues[:6])
                )

        def _write_output(render_results, *, allow_overwrite: bool, append: bool = False) -> None:
            if not output:
                return
            out_path = Path(output)
            if out_path.exists() and not allow_overwrite:
                raise click.ClickException(
                    f"Refusing to overwrite existing output file without --force: {out_path}"
                )
            out_fmt = _detect_format(out_path, fmt)
            reporter = _make_reporter(out_fmt, color=False)
            rendered = reporter.render(render_results)
            if append and out_fmt == "jsonl":
                with out_path.open("a", encoding="utf-8") as handle:
                    handle.write(rendered)
            else:
                out_path.write_text(rendered, encoding="utf-8")
            if validate_output and out_fmt in {"json", "jsonl"}:
                _validate_written_path(out_path)
            if not silent:
                click.echo(f"Results written to {out_path}", err=True)

        def _write_scorecard(render_results, *, allow_overwrite: bool) -> None:
            if not scorecard:
                return
            scorecard_path = Path(scorecard)
            if scorecard_path.exists() and not allow_overwrite:
                raise click.ClickException(
                    f"Refusing to overwrite existing scorecard without --force: {scorecard_path}"
                )
            scorecard_fmt = _detect_format(scorecard_path, "markdown")
            if scorecard_fmt not in {"json", "html", "markdown"}:
                raise click.ClickException("Scorecard output must be .json, .html, or .md")
            payload = build_scorecard(render_results)
            scorecard_path.write_text(render_scorecard(payload, scorecard_fmt), encoding="utf-8")
            if validate_output and scorecard_fmt == "json":
                _validate_written_path(scorecard_path)
            if not silent:
                click.echo(f"Scorecard written to {scorecard_path}", err=True)

        def _write_insights_summary(render_results, *, allow_overwrite: bool) -> None:
            if not insights_summary:
                return
            summary_path = Path(insights_summary)
            if summary_path.exists() and not allow_overwrite:
                raise click.ClickException(
                    f"Refusing to overwrite existing insights summary without --force: {summary_path}"
                )
            summary_path.write_text(
                json.dumps(
                    build_insight_summary_payload(render_results),
                    indent=2,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                encoding="utf-8",
            )
            if not silent:
                click.echo(f"Insights summary written to {summary_path}", err=True)

        def _render_console(batch_results) -> None:
            if silent:
                return
            console_fmt = fmt if fmt != "pretty" else "pretty"
            reporter = _make_reporter(console_fmt, color=not no_color)
            click.echo(reporter.render(batch_results))

        def _batch_exit_code(batch_results, violations) -> int:
            if any(result.error for result in batch_results):
                return 1
            if violations:
                return 2
            if any(
                check.get("status") == "fail"
                for result in batch_results
                for check in result.insights.get("ci_checks", [])
            ):
                return 2
            return 0

        if watch:
            if output and Path(output).exists() and not force:
                raise click.ClickException(
                    f"Refusing to overwrite existing output file without --force: {output}"
                )
            if scorecard and Path(scorecard).exists() and not force:
                raise click.ClickException(
                    f"Refusing to overwrite existing scorecard without --force: {scorecard}"
                )
            processed_signatures = {
                str(candidate.resolve()): _file_signature(candidate.resolve())
                for watch_dir in watch_dirs
                for candidate in collect_files(watch_dir)
                if _file_signature(candidate.resolve()) is not None
            }
            retry_state: dict[str, dict[str, object]] = {}
            aggregate_results = []
            processed = 0
            exit_code = 0
            out_fmt = _detect_format(Path(output), fmt) if output else fmt
            if not silent:
                click.echo(
                    f"Watching {len(watch_dirs)} director{'y' if len(watch_dirs) == 1 else 'ies'} for new audio files...",
                    err=True,
                )
            try:
                for new_files in _iter_watch_batches(
                    watch_dirs,
                    processed_signatures,
                    watch_interval,
                    retry_state=retry_state,
                    debounce_seconds=watch_debounce,
                    max_pending=watch_max_pending,
                ):
                    if new_files:
                        try:
                            batch_results = _normalize_results(analyzer.analyze_all(new_files, on_progress=_on_progress))
                        except Exception as exc:
                            _record_watch_batch_failure(
                                new_files,
                                retry_state,
                                watch_retries=watch_retries,
                                watch_backoff=watch_backoff,
                            )
                            if not silent:
                                click.echo(f"[warn] Watch batch failed: {exc}", err=True)
                            continue
                        preset_violations = apply_threshold_rules(batch_results, preset_rule_defs)
                        user_violations = apply_threshold_rules(
                            batch_results,
                            user_rule_defs,
                            unmatched_behavior="violation" if user_rule_defs else "ignore",
                        )
                        violations = preset_violations + user_violations
                        _finalize_watch_batch(
                            batch_results,
                            processed_signatures,
                            retry_state,
                            watch_retries=watch_retries,
                            watch_backoff=watch_backoff,
                        )
                        aggregate_results.extend(batch_results)
                        if insights:
                            enrich_results(
                                aggregate_results,
                                baseline_path=baseline,
                                ci=ci,
                                drift=drift,
                                drift_state_path=drift_state,
                                fingerprint_sensitivity=fingerprint_sensitivity,
                                preset=preset,
                                insight_rules_path=insight_rules,
                            )
                            if insight_snippets:
                                export_flagged_segment_snippets(aggregate_results, insight_snippets)
                        _write_sidecars(batch_results, allow_overwrite=force)
                        if out_fmt == "jsonl":
                            _write_output(batch_results, allow_overwrite=True, append=True)
                        else:
                            _write_output(aggregate_results, allow_overwrite=True)
                        _write_scorecard(aggregate_results, allow_overwrite=True)
                        _write_insights_summary(aggregate_results, allow_overwrite=True)
                        _render_console(batch_results)
                        batch_exit = _batch_exit_code(batch_results, violations)
                        if batch_exit == 1:
                            exit_code = 1
                        elif batch_exit == 2 and exit_code == 0:
                            exit_code = 2
                        processed += len(new_files)
                        if watch_limit is not None and processed >= watch_limit:
                            break
            except KeyboardInterrupt:
                if not silent:
                    click.echo("Watch mode stopped.", err=True)
            sys.exit(exit_code)

        results = _normalize_results(analyzer.analyze_all(all_files, on_progress=_on_progress))
        preset_violations = apply_threshold_rules(results, preset_rule_defs)
        user_violations = apply_threshold_rules(
            results,
            user_rule_defs,
            unmatched_behavior="violation" if user_rule_defs else "ignore",
        )
        violations = preset_violations + user_violations
        if insights:
            enrich_results(
                results,
                baseline_path=baseline,
                ci=ci,
                drift=drift,
                drift_state_path=drift_state,
                fingerprint_sensitivity=fingerprint_sensitivity,
                preset=preset,
                insight_rules_path=insight_rules,
            )
            if insight_snippets:
                export_flagged_segment_snippets(results, insight_snippets)
        _write_sidecars(results, allow_overwrite=force)
        _write_output(results, allow_overwrite=force)
        _write_scorecard(results, allow_overwrite=force)
        _write_insights_summary(results, allow_overwrite=force)
        _render_console(results)
        sys.exit(_batch_exit_code(results, violations))
    finally:
        for temp_path in stdin_paths:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass


def _parse_metric_groups(metrics: Optional[str], preset_groups: tuple[str, ...] | None = None) -> set[str]:
    valid = {
        "basic", "loudness", "spectral", "temporal", "noise", "speech",
        "perceptual", "prosody", "psychoacoustic", "speaker", "all",
    }
    if not metrics:
        if preset_groups:
            return set(preset_groups)
        return {"all"}
    groups = {g.strip().lower() for g in metrics.split(",") if g.strip()}
    if not groups:
        raise click.BadParameter(
            "Provide at least one metric group.",
            param_hint="--metrics",
        )
    unknown = groups - valid
    if unknown:
        raise click.BadParameter(
            f"Unknown metric groups: {', '.join(sorted(unknown))}. "
            f"Valid groups: {', '.join(sorted(valid))}",
            param_hint="--metrics",
        )
    return groups


def _detect_format(path: Path, fallback: str) -> str:
    ext = path.suffix.lower()
    if ext == ".json":
        return "json"
    if ext in {".jsonl", ".ndjson"}:
        return "jsonl"
    if ext == ".csv":
        return "csv"
    if ext in {".html", ".htm"}:
        return "html"
    if ext in {".md", ".markdown"}:
        return "markdown"
    return fallback


def _paths_conflict(output: Optional[str], scorecard: Optional[str]) -> bool:
    if not output or not scorecard:
        return False
    return Path(output).expanduser().resolve(strict=False) == Path(scorecard).expanduser().resolve(strict=False)


def _make_reporter(fmt: str, color: bool):
    if fmt == "json":
        return JsonReporter()
    if fmt == "jsonl":
        return JsonlReporter()
    if fmt == "csv":
        return CsvReporter()
    if fmt == "html":
        return HtmlReporter()
    if fmt == "markdown":
        return MarkdownReporter()
    return ConsoleReporter(color=color)


def _run_insights_subcommand(
    args: tuple[str, ...],
    *,
    output: Optional[str],
    silent: bool,
    force: bool,
    baseline: Optional[str],
    ci: bool,
    drift: bool,
    drift_state: Optional[str],
    fingerprint_sensitivity: str,
    preset: Optional[str],
    insight_rules: Optional[str],
) -> None:
    if args and args[0] == "validate":
        if len(args) != 2:
            raise click.UsageError("Usage: qualiax insights validate REPORT.json")
        issues = validate_insights_report(Path(args[1]))
        for issue in issues:
            click.echo(f"[error] {issue.path}: {issue.message}", err=True)
        sys.exit(2 if issues else 0)
    if len(args) != 1:
        raise click.UsageError("Usage: qualiax insights REPORT.json --output ENRICHED.json")
    if not output:
        raise click.UsageError("qualiax insights requires --output.")
    input_path = Path(args[0])
    output_path = Path(output)
    if not input_path.exists():
        raise click.ClickException(f"Report not found: {input_path}")
    if output_path.exists() and not force:
        raise click.ClickException(
            f"Refusing to overwrite existing output file without --force: {output_path}"
        )
    enrich_existing_report(
        input_path,
        output_path,
        baseline_path=baseline,
        ci=ci,
        drift=drift,
        drift_state_path=drift_state,
        fingerprint_sensitivity=fingerprint_sensitivity,
        preset=preset,
        insight_rules_path=insight_rules,
    )
    if not silent:
        click.echo(f"Insights written to {output_path}", err=True)


def _run_diff(
    diff_args: tuple[str, ...],
    silent: bool,
    force: bool,
    output: Optional[str],
    fmt: str,
    no_color: bool,
    save_sidecar: bool,
    reference: Optional[str],
    metrics: Optional[str],
    workers: int,
    preset: Optional[str],
    scorecard: Optional[str],
    rules: Optional[str],
    lint_rules: bool,
    validate_output: bool,
    strict: bool,
    include_demographics: bool,
    insights: bool,
    baseline: Optional[str],
    ci: bool,
    drift: bool,
    drift_state: Optional[str],
    fingerprint_sensitivity: str,
    insight_rules: Optional[str],
    insight_snippets: Optional[str],
    insights_summary: Optional[str],
) -> None:
    if (
        save_sidecar
        or reference
        or metrics
        or workers != 1
        or preset
        or scorecard
        or rules
        or lint_rules
        or validate_output
        or strict
        or include_demographics
        or insights
        or baseline
        or ci
        or drift
        or drift_state
        or fingerprint_sensitivity != "balanced"
        or insight_rules
        or insight_snippets
        or insights_summary
    ):
        raise click.UsageError("diff mode only supports --output, --format, --silent, --force, and --no-color.")
    if len(diff_args) != 2:
        raise click.UsageError("Usage: qualiax diff BEFORE.json AFTER.json")
    before_path = Path(diff_args[0])
    after_path = Path(diff_args[1])
    if not before_path.exists() or not after_path.exists():
        raise click.ClickException("Both diff inputs must exist.")
    diff_fmt = _detect_format(Path(output), fmt) if output else fmt
    if diff_fmt == "pretty":
        diff_fmt = "pretty"
    elif diff_fmt not in {"json", "markdown", "pretty"}:
        raise click.UsageError("diff mode supports pretty, json, or markdown output.")
    summary = diff_reports(before_path, after_path)
    rendered = render_diff(summary, diff_fmt)
    if output:
        out_path = Path(output)
        if out_path.exists() and not force:
            raise click.ClickException(
                f"Refusing to overwrite existing output file without --force: {out_path}"
            )
        out_path.write_text(rendered, encoding="utf-8")
        if not silent:
            click.echo(f"Results written to {out_path}", err=True)
    elif silent:
        raise click.UsageError("--silent requires --output or --save-sidecar.")
    else:
        click.echo(rendered)


def _materialize_stdin_wav() -> Path:
    payload = sys.stdin.buffer.read()
    if not payload:
        raise click.ClickException("No stdin audio received.")
    with tempfile.NamedTemporaryFile(prefix="qualiax-stdin-", suffix=".wav", delete=False) as tmp:
        tmp.write(payload)
        return Path(tmp.name)


def _capture_microphone_wav(duration_s: float, sample_rate: int, *, silent: bool) -> Path:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise click.ClickException(
            "Microphone mode requires the optional `sounddevice` package. "
            "Install it with `pip install sounddevice` or include the live extras."
        ) from exc

    if not silent:
        click.echo(
            f"Recording microphone for {duration_s:.1f}s at {sample_rate} Hz...",
            err=True,
        )
    frames = max(1, int(round(duration_s * sample_rate)))
    try:
        recording = sd.rec(frames, samplerate=sample_rate, channels=1, dtype="float32")
        sd.wait()
    except Exception as exc:
        raise click.ClickException(f"Microphone capture failed: {exc}") from exc

    mono = np.asarray(recording, dtype=np.float32).reshape(-1)
    pcm = np.clip(mono * 32767.0, -32768, 32767).astype("<i2")
    with tempfile.NamedTemporaryFile(prefix="qualiax-mic-", suffix=".wav", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    with wave.open(str(tmp_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return tmp_path


def _iter_watch_batches(
    watch_dirs: list[Path],
    processed_signatures: dict[str, tuple[int, int]] | set[str],
    watch_interval: float,
    *,
    retry_state: Optional[dict[str, dict[str, object]]] = None,
    debounce_seconds: float = _DEFAULT_WATCH_DEBOUNCE,
    max_pending: int = _DEFAULT_WATCH_MAX_PENDING,
):
    pending: dict[str, dict[str, object]] = {}
    retry_state = retry_state or {}
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        yield from _iter_watch_batches_polling(
            watch_dirs,
            processed_signatures,
            watch_interval,
            retry_state=retry_state,
            debounce_seconds=debounce_seconds,
            max_pending=max_pending,
        )
        return

    queue: Queue[Path] = Queue()

    class _ArrivalHandler(FileSystemEventHandler):
        def on_created(self, event):
            self._enqueue(getattr(event, "src_path", None))

        def on_moved(self, event):
            self._enqueue(getattr(event, "dest_path", None))

        def _enqueue(self, raw_path):
            if not raw_path:
                return
            candidate = Path(raw_path)
            if candidate.suffix.lower() not in SUPPORTED_EXTENSIONS:
                return
            queue.put(candidate)

    observer = Observer()
    handler = _ArrivalHandler()
    for watch_dir in watch_dirs:
        observer.schedule(handler, str(watch_dir), recursive=True)
    observer.start()
    try:
        while True:
            _rescan_watch_dirs(
                watch_dirs,
                pending,
                processed_signatures,
                retry_state=retry_state,
                max_pending=max_pending,
            )
            ready = _drain_pending_watch_files(
                pending,
                debounce_seconds=debounce_seconds,
            )
            yield ready
            try:
                first = queue.get(timeout=watch_interval)
            except Empty:
                continue
            batch = [first]
            while True:
                try:
                    batch.append(queue.get_nowait())
                except Empty:
                    break
            deduped: list[Path] = []
            batch_seen = set()
            for candidate in batch:
                try:
                    resolved = candidate.resolve()
                except FileNotFoundError:
                    continue
                resolved_key = str(resolved)
                if resolved_key in batch_seen:
                    continue
                batch_seen.add(resolved_key)
                _register_watch_candidate(
                    resolved,
                    pending,
                    processed_signatures,
                    retry_state=retry_state,
                    max_pending=max_pending,
                )
            deduped = _drain_pending_watch_files(
                pending,
                debounce_seconds=debounce_seconds,
            )
            yield deduped
    finally:
        observer.stop()
        observer.join(timeout=2.0)


def _iter_watch_batches_polling(
    watch_dirs: list[Path],
    processed_signatures: dict[str, tuple[int, int]] | set[str],
    watch_interval: float,
    *,
    retry_state: Optional[dict[str, dict[str, object]]] = None,
    debounce_seconds: float = _DEFAULT_WATCH_DEBOUNCE,
    max_pending: int = _DEFAULT_WATCH_MAX_PENDING,
):
    pending: dict[str, dict[str, object]] = {}
    retry_state = retry_state or {}
    while True:
        yield _drain_pending_watch_files(
            pending,
            debounce_seconds=debounce_seconds,
        )
        _rescan_watch_dirs(
            watch_dirs,
            pending,
            processed_signatures,
            retry_state=retry_state,
            max_pending=max_pending,
        )
        time.sleep(watch_interval)


def _register_watch_candidate(
    candidate: Path,
    pending: dict[str, dict[str, object]],
    processed_signatures: dict[str, tuple[int, int]] | set[str],
    *,
    retry_state: Optional[dict[str, dict[str, object]]] = None,
    max_pending: int = _DEFAULT_WATCH_MAX_PENDING,
) -> None:
    retry_state = retry_state or {}
    try:
        resolved_path = candidate.resolve()
    except FileNotFoundError:
        return
    if not resolved_path.is_file():
        return
    resolved = str(resolved_path)
    signature = _file_signature(resolved_path)
    if signature is None:
        return
    if _watch_signature_already_processed(processed_signatures, resolved, signature):
        return
    retry_info = retry_state.get(resolved)
    now = time.monotonic()
    if retry_info and retry_info.get("dropped"):
        return
    if retry_info and float(retry_info.get("next_retry_at", 0.0)) > now:
        return
    state = pending.get(resolved)
    if state is None:
        if len(pending) >= max_pending:
            warnings.warn(f"watch: dropping candidate because pending queue is full: {resolved_path}")
            retry_state[resolved] = {"dropped": True}
            return
        pending[resolved] = {
            "path": resolved_path,
            "signature": signature,
            "stable_rounds": 0,
            "age_rounds": 0,
            "first_seen_at": now,
        }
        return
    state["path"] = resolved_path
    if signature == state["signature"]:
        state["stable_rounds"] = int(state["stable_rounds"]) + 1
    else:
        state["signature"] = signature
        state["stable_rounds"] = 0
        state["first_seen_at"] = now
    state["age_rounds"] = int(state.get("age_rounds", 0))


def _drain_pending_watch_files(
    pending: dict[str, dict[str, object]],
    *,
    debounce_seconds: float = _DEFAULT_WATCH_DEBOUNCE,
) -> list[Path]:
    ready: list[Path] = []
    now = time.monotonic()
    for resolved, state in list(pending.items()):
        path = state["path"]
        state["age_rounds"] = int(state.get("age_rounds", 0)) + 1
        signature = _file_signature(path)
        if signature is None:
            pending.pop(resolved, None)
            continue
        if signature == state["signature"]:
            state["stable_rounds"] = int(state["stable_rounds"]) + 1
        else:
            state["signature"] = signature
            state["stable_rounds"] = 0
        if int(state["age_rounds"]) > _MAX_PENDING_WATCH_ROUNDS:
            warnings.warn(
                f"watch: dropping unstable file after {_MAX_PENDING_WATCH_ROUNDS} rounds: {path}"
            )
            pending.pop(resolved, None)
            continue
        if (
            int(state["stable_rounds"]) >= 1
            and now - float(state.get("first_seen_at", now)) >= debounce_seconds
        ):
            ready.append(path)
            pending.pop(resolved, None)
    return ready


def _rescan_watch_dirs(
    watch_dirs: list[Path],
    pending: dict[str, dict[str, object]],
    processed_signatures: dict[str, tuple[int, int]] | set[str],
    *,
    retry_state: Optional[dict[str, dict[str, object]]] = None,
    max_pending: int = _DEFAULT_WATCH_MAX_PENDING,
) -> None:
    for watch_dir in watch_dirs:
        for candidate in collect_files(watch_dir):
            _register_watch_candidate(
                candidate,
                pending,
                processed_signatures,
                retry_state=retry_state,
                max_pending=max_pending,
            )


def _record_watch_batch_failure(
    paths: list[Path],
    retry_state: dict[str, dict[str, object]],
    *,
    watch_retries: int,
    watch_backoff: float,
) -> None:
    now = time.monotonic()
    for path in paths:
        resolved = str(path.resolve())
        state = retry_state.setdefault(resolved, {"attempts": 0, "next_retry_at": 0.0, "dropped": False})
        attempts = int(state.get("attempts", 0)) + 1
        state["attempts"] = attempts
        if attempts > watch_retries:
            state["dropped"] = True
            warnings.warn(f"watch: dropping file after {watch_retries} retry attempt(s): {path}")
            continue
        state["next_retry_at"] = now + (watch_backoff * (2 ** (attempts - 1)))


def _finalize_watch_batch(
    batch_results: list,
    processed_signatures: dict[str, tuple[int, int]] | set[str],
    retry_state: dict[str, dict[str, object]],
    *,
    watch_retries: int,
    watch_backoff: float,
) -> None:
    failed_paths: list[Path] = []
    for result in batch_results:
        try:
            result_path = Path(result.source_file or result.path)
        except TypeError:
            continue
        if result.error:
            failed_paths.append(result_path)
            continue
        try:
            resolved = result_path.resolve()
        except FileNotFoundError:
            continue
        signature = _file_signature(resolved)
        if signature is None:
            continue
        _mark_watch_signature_processed(processed_signatures, str(resolved), signature)
        retry_state.pop(str(resolved), None)
    if failed_paths:
        _record_watch_batch_failure(
            failed_paths,
            retry_state,
            watch_retries=watch_retries,
            watch_backoff=watch_backoff,
        )


def _watch_signature_already_processed(
    processed_signatures: dict[str, tuple[int, int]] | set[str],
    resolved: str,
    signature: tuple[int, int],
) -> bool:
    if isinstance(processed_signatures, dict):
        return processed_signatures.get(resolved) == signature
    return resolved in processed_signatures


def _mark_watch_signature_processed(
    processed_signatures: dict[str, tuple[int, int]] | set[str],
    resolved: str,
    signature: tuple[int, int],
) -> None:
    if isinstance(processed_signatures, dict):
        processed_signatures[resolved] = signature
        return
    processed_signatures.add(resolved)


def _file_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return None
    if not path.is_file():
        return None
    return stat.st_size, stat.st_mtime_ns


if __name__ == "__main__":
    main()

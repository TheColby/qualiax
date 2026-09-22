import json
from itertools import cycle
from pathlib import Path

from click.testing import CliRunner

from qualiax.cli import _finalize_watch_batch, _iter_watch_batches_polling, collect_files, main
from qualiax.migrations import ExitCode
from qualiax.models import FileResult, MetricResult
from qualiax.version import OUTPUT_SCHEMA_VERSION


class DummyAnalyzer:
    def __init__(self, *args, **kwargs):
        pass

    def analyze_all(self, paths, on_progress=None):
        results = [FileResult(path=str(path), duration_s=1.0, sample_rate=16_000, channels=1) for path in paths]
        for index, path in enumerate(paths, start=1):
            if on_progress:
                on_progress(index, len(paths), path)
        return results


def test_help_lists_all_public_metric_groups():
    runner = CliRunner()

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "prosody" in result.output
    assert "psychoacoustic" in result.output
    assert "speaker" in result.output
    assert "--mic-seconds" in result.output
    assert "--watch" in result.output
    assert "--lint-rules" in result.output
    assert "--validate-output" in result.output
    assert "--watch-debounce" in result.output
    assert "--strict" in result.output
    assert "--include-demographics" in result.output
    assert "--insights" in result.output
    assert "--baseline" in result.output
    assert "--ci" in result.output
    assert "--drift" in result.output
    assert "--drift-state" in result.output
    assert "--fingerprint-sensitivity" in result.output
    assert "--insight-rules" in result.output
    assert "--insight-snippets" in result.output
    assert "--insights-summary" in result.output
    assert "speech_quality" not in result.output


def test_module_has_main_entrypoint():
    import qualiax.cli as cli_module

    assert hasattr(cli_module, "__name__")


def test_collect_files_accepts_mixed_case_extensions(tmp_path):
    audio = tmp_path / "Voice.Wav"
    audio.write_bytes(b"audio")

    files = collect_files(tmp_path)

    assert files == [audio]


def test_collect_files_accepts_video_container_extensions(tmp_path):
    video = tmp_path / "clip.MP4"
    video.write_bytes(b"video")

    files = collect_files(tmp_path)

    assert files == [video]


def test_cli_rejects_missing_input_path():
    runner = CliRunner()

    result = runner.invoke(main, ["missing.wav"])

    assert result.exit_code != 0
    assert "Path not found: missing.wav" in result.output


def test_cli_rejects_unsupported_input_file(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", DummyAnalyzer)

    monkeypatch.chdir(tmp_path)
    Path("notes.txt").write_text("not audio")
    result = runner.invoke(main, ["notes.txt"])

    assert result.exit_code != 0
    assert "Unsupported file type: .txt" in result.output


def test_cli_does_not_write_sidecar_by_default(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", DummyAnalyzer)

    monkeypatch.chdir(tmp_path)
    Path("input.wav").write_bytes(b"audio")
    result = runner.invoke(main, ["input.wav"])

    assert result.exit_code == 0
    assert not Path("input.json").exists()


def test_cli_can_write_sidecar_when_requested(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", DummyAnalyzer)

    monkeypatch.chdir(tmp_path)
    Path("input.wav").write_bytes(b"audio")
    result = runner.invoke(main, ["input.wav", "--save-sidecar", "--silent"])

    assert result.exit_code == 0
    payload = json.loads(Path("input.json").read_text())
    assert payload[0]["schema_version"] == OUTPUT_SCHEMA_VERSION


def test_silent_requires_output_destination(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", DummyAnalyzer)

    monkeypatch.chdir(tmp_path)
    Path("input.wav").write_bytes(b"audio")
    result = runner.invoke(main, ["input.wav", "--silent"])

    assert result.exit_code != 0
    assert "--silent requires --output, --scorecard, or --save-sidecar" in result.output


def test_output_refuses_to_overwrite_without_force(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", DummyAnalyzer)

    monkeypatch.chdir(tmp_path)
    Path("input.wav").write_bytes(b"audio")
    Path("report.json").write_text("{}")
    result = runner.invoke(main, ["input.wav", "--output", "report.json", "--silent"])

    assert result.exit_code != 0
    assert "Refusing to overwrite existing output file without --force" in result.output


def test_sidecar_refuses_to_overwrite_without_force(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", DummyAnalyzer)

    monkeypatch.chdir(tmp_path)
    Path("input.wav").write_bytes(b"audio")
    Path("input.json").write_text("{}")
    result = runner.invoke(main, ["input.wav", "--save-sidecar", "--silent"])

    assert result.exit_code != 0
    assert "Refusing to overwrite existing sidecar file(s) without --force" in result.output


def test_unknown_metric_groups_are_rejected(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", DummyAnalyzer)

    monkeypatch.chdir(tmp_path)
    Path("input.wav").write_bytes(b"audio")
    result = runner.invoke(main, ["input.wav", "--metrics", "basic,speach"])

    assert result.exit_code != 0
    assert "Unknown metric groups: speach" in result.output


def test_cli_preset_applies_default_groups_and_rules(monkeypatch, tmp_path):
    captured = {}

    class PresetAnalyzer:
        def __init__(self, *args, **kwargs):
            captured["metric_groups"] = kwargs["metric_groups"]

        def analyze_all(self, paths, on_progress=None):
            results = [
                FileResult(
                    path=str(path),
                    duration_s=1.0,
                    sample_rate=16_000,
                    channels=1,
                    metrics=[
                        MetricResult(
                            name="Integrated Loudness (LUFS)",
                            value=-13.0,
                            unit="LUFS",
                            group="loudness",
                        ),
                        MetricResult(
                            name="True Peak",
                            value=-0.5,
                            unit="dBTP",
                            group="loudness",
                        ),
                    ],
                )
                for path in paths
            ]
            for index, path in enumerate(paths, start=1):
                if on_progress:
                    on_progress(index, len(paths), path)
            return results

    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", PresetAnalyzer)

    monkeypatch.chdir(tmp_path)
    Path("input.wav").write_bytes(b"audio")
    result = runner.invoke(
        main,
        ["input.wav", "--preset", "podcast", "--output", "report.json", "--silent"],
    )

    assert result.exit_code == 2
    assert captured["metric_groups"] == {"basic", "loudness", "speech", "perceptual"}
    payload = json.loads(Path("report.json").read_text())
    assert payload[0]["notes"] == ["threshold rules: 2 violation(s)"]
    assert payload[0]["metrics"][0]["warning"] is not None
    assert payload[0]["metrics"][1]["warning"] is not None


def test_cli_can_write_scorecard(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", DummyAnalyzer)

    monkeypatch.chdir(tmp_path)
    Path("input.wav").write_bytes(b"audio")
    result = runner.invoke(
        main,
        ["input.wav", "--output", "report.json", "--scorecard", "scorecard.md", "--silent"],
    )

    assert result.exit_code == 0
    scorecard = Path("scorecard.md").read_text()
    assert "# qualiax Scorecard" in scorecard
    assert "| Group | Metric | Count |" in scorecard


def test_cli_rejects_shared_output_and_scorecard_paths(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", DummyAnalyzer)

    monkeypatch.chdir(tmp_path)
    Path("input.wav").write_bytes(b"audio")
    result = runner.invoke(
        main,
        ["input.wav", "--output", "report.json", "--scorecard", "report.json", "--silent"],
    )

    assert result.exit_code != 0
    assert "--output and --scorecard must point to different paths" in result.output


def test_cli_can_write_html_report(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", DummyAnalyzer)

    monkeypatch.chdir(tmp_path)
    Path("input.wav").write_bytes(b"audio")
    result = runner.invoke(main, ["input.wav", "--output", "report.html", "--silent"])

    assert result.exit_code == 0
    html = Path("report.html").read_text()
    assert "<!doctype html>" in html.lower()
    assert "qualiax HTML Report" in html


def test_cli_can_write_markdown_report(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", DummyAnalyzer)

    monkeypatch.chdir(tmp_path)
    Path("input.wav").write_bytes(b"audio")
    result = runner.invoke(main, ["input.wav", "--output", "report.md", "--silent"])

    assert result.exit_code == 0
    report = Path("report.md").read_text()
    assert "# qualiax Report" in report
    assert "## `input.wav`" in report


def test_cli_diff_mode_can_write_markdown_report(tmp_path, monkeypatch):
    runner = CliRunner()

    monkeypatch.chdir(tmp_path)
    Path("before.json").write_text(
        json.dumps(
            [
                {
                    "file": "input.wav",
                    "error": None,
                    "metrics": [
                        {
                            "name": "True Peak",
                            "group": "loudness",
                            "value": -2.0,
                            "unit": "dBTP",
                            "higher_is_better": False,
                            "warning": None,
                        }
                    ],
                }
            ]
        )
    )
    Path("after.json").write_text(
        json.dumps(
            [
                {
                    "file": "input.wav",
                    "error": None,
                    "metrics": [
                        {
                            "name": "True Peak",
                            "group": "loudness",
                            "value": -0.1,
                            "unit": "dBTP",
                            "higher_is_better": False,
                            "warning": "Exceeds ceiling",
                        }
                    ],
                }
            ]
        )
    )
    result = runner.invoke(
        main,
        ["diff", "before.json", "after.json", "--output", "delta.md", "--silent"],
    )

    assert result.exit_code == 0
    report = Path("delta.md").read_text()
    assert "# qualiax Diff Report" in report
    assert "Regression detected" in report


def test_cli_rules_can_fail_compliance(monkeypatch, tmp_path):
    class RulesAnalyzer(DummyAnalyzer):
        def analyze_all(self, paths, on_progress=None):
            results = super().analyze_all(paths, on_progress=on_progress)
            results[0].metrics = [
                __import__("qualiax.models", fromlist=["MetricResult"]).MetricResult(
                    name="True Peak",
                    value=-0.2,
                    unit="dBTP",
                    group="loudness",
                )
            ]
            return results

    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", RulesAnalyzer)

    monkeypatch.chdir(tmp_path)
    Path("input.wav").write_bytes(b"audio")
    Path("rules.json").write_text(
        json.dumps(
            {
                "rules": [
                    {
                        "metric": "True Peak",
                        "group": "loudness",
                        "max": -1.0,
                        "message": "Streaming ceiling exceeded",
                    }
                ]
            }
        )
    )
    result = runner.invoke(
        main,
        ["input.wav", "--rules", "rules.json", "--output", "report.json", "--silent"],
    )

    assert result.exit_code == 2
    payload = json.loads(Path("report.json").read_text())
    assert payload[0]["notes"] == ["threshold rules: 1 violation(s)"]
    assert payload[0]["metrics"][0]["warning"] == "Streaming ceiling exceeded"


def test_cli_lint_rules_can_fail_without_inputs(tmp_path, monkeypatch):
    runner = CliRunner()

    monkeypatch.chdir(tmp_path)
    Path("rules.json").write_text(
        json.dumps({"rules": [{"metric": "True Peak", "group": "missing-group", "max": -1.0}]})
    )
    result = runner.invoke(main, ["--rules", "rules.json", "--lint-rules"])

    assert result.exit_code == ExitCode.INVALID_INPUT
    assert "unknown group" in result.output


def test_cli_validate_output_checks_json_contract(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", DummyAnalyzer)

    monkeypatch.chdir(tmp_path)
    Path("input.wav").write_bytes(b"audio")
    result = runner.invoke(main, ["input.wav", "--output", "report.json", "--validate-output", "--silent"])

    assert result.exit_code == 0


def test_segmented_sidecar_groups_results_by_source(monkeypatch, tmp_path):
    runner = CliRunner()

    class SegmentAnalyzer:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_all(self, paths, on_progress=None):
            source = str(paths[0])
            first = FileResult(
                path=f"{source} [segment 1/2 0.00-1.00s]",
                source_file=source,
                segment_index=1,
                total_segments=2,
                segment_start_s=0.0,
                segment_end_s=1.0,
                duration_s=1.0,
                sample_rate=16_000,
                channels=1,
            )
            second = FileResult(
                path=f"{source} [segment 2/2 1.00-2.00s]",
                source_file=source,
                segment_index=2,
                total_segments=2,
                segment_start_s=1.0,
                segment_end_s=2.0,
                duration_s=1.0,
                sample_rate=16_000,
                channels=1,
            )
            if on_progress:
                on_progress(1, 2, paths[0])
                on_progress(2, 2, paths[0])
            return [first, second]

    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", SegmentAnalyzer)

    monkeypatch.chdir(tmp_path)
    Path("input.wav").write_bytes(b"audio")
    result = runner.invoke(
        main,
        ["input.wav", "--segment-seconds", "1", "--save-sidecar", "--silent"],
    )

    assert result.exit_code == 0
    payload = json.loads(Path("input.json").read_text())
    assert len(payload) == 2
    assert payload[0]["source_file"].endswith("input.wav")
    assert payload[1]["segment_index"] == 2


def test_cli_can_capture_microphone_input(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", DummyAnalyzer)

    monkeypatch.chdir(tmp_path)
    def fake_capture(duration_s, sample_rate, *, silent):
        path = Path("mic.wav")
        path.write_bytes(b"audio")
        return path

    monkeypatch.setattr("qualiax.cli._capture_microphone_wav", fake_capture)
    result = runner.invoke(
        main,
        ["--mic-seconds", "1.5", "--output", "mic.json", "--silent"],
    )

    assert result.exit_code == 0
    payload = json.loads(Path("mic.json").read_text())
    assert payload[0]["file"] == "microphone"


def test_cli_watch_mode_processes_new_files(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", DummyAnalyzer)

    monkeypatch.chdir(tmp_path)
    incoming = Path("incoming")
    incoming.mkdir()
    new_file = incoming / "new.wav"
    new_file.write_bytes(b"audio")
    calls = {"count": 0}

    def fake_collect(path):
        calls["count"] += 1
        if calls["count"] == 1:
            return []
        return [new_file]

    monkeypatch.setattr("qualiax.cli.collect_files", fake_collect)
    monkeypatch.setattr("qualiax.cli.time.sleep", lambda _: None)

    result = runner.invoke(
        main,
        ["incoming", "--watch", "--watch-limit", "1", "--output", "watch.json", "--silent"],
    )

    assert result.exit_code == 0
    payload = json.loads(Path("watch.json").read_text())
    assert payload[0]["file"].endswith("new.wav")


def test_watch_batches_wait_for_stable_file(monkeypatch, tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    new_file = incoming / "new.wav"
    new_file.write_bytes(b"audio")

    signature_cycle = cycle([(1, 1), (2, 2), (2, 2), (2, 2)])

    def fake_collect(path):
        return [new_file]

    def fake_signature(path):
        return next(signature_cycle)

    monkeypatch.setattr("qualiax.cli.collect_files", fake_collect)
    monkeypatch.setattr("qualiax.cli._file_signature", fake_signature)
    monkeypatch.setattr("qualiax.cli.time.sleep", lambda _: None)

    batches = _iter_watch_batches_polling([incoming], set(), 0.1)

    assert next(batches) == []
    assert next(batches) == []
    assert next(batches) == [new_file]


def test_watch_batches_reprocess_file_when_signature_changes(monkeypatch, tmp_path):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    target = incoming / "new.wav"
    target.write_bytes(b"audio")

    processed = {str(target.resolve()): (1, 1)}
    signature_cycle = cycle([(1, 1), (2, 2), (2, 2), (2, 2)])

    def fake_collect(path):
        return [target]

    def fake_signature(path):
        return next(signature_cycle)

    monkeypatch.setattr("qualiax.cli.collect_files", fake_collect)
    monkeypatch.setattr("qualiax.cli._file_signature", fake_signature)
    monkeypatch.setattr("qualiax.cli.time.sleep", lambda _: None)

    batches = _iter_watch_batches_polling([incoming], processed, 0.1)

    assert next(batches) == []
    assert next(batches) == []
    assert next(batches) == [target]


def test_cli_watch_mode_can_append_jsonl(monkeypatch, tmp_path):
    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", DummyAnalyzer)

    monkeypatch.chdir(tmp_path)
    incoming = Path("incoming")
    incoming.mkdir()
    first = incoming / "first.wav"
    second = incoming / "second.wav"
    calls = {"count": 0}

    def fake_watch_batches(watch_dirs, seen, watch_interval, **kwargs):
        calls["count"] += 1
        yield [first]
        yield [second]

    first.write_bytes(b"audio")
    second.write_bytes(b"audio")
    monkeypatch.setattr("qualiax.cli._iter_watch_batches", fake_watch_batches)

    result = runner.invoke(
        main,
        ["incoming", "--watch", "--watch-limit", "2", "--output", "watch.jsonl", "--silent"],
    )

    assert result.exit_code == 0
    lines = Path("watch.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["file"].endswith("first.wav")
    assert json.loads(lines[1])["file"].endswith("second.wav")


def test_cli_forwards_strict_and_include_demographics(monkeypatch, tmp_path):
    captured = {}

    class CapturingAnalyzer(DummyAnalyzer):
        def __init__(self, *args, **kwargs):
            captured["strict"] = kwargs["strict"]
            captured["include_demographics"] = kwargs["include_demographics"]

    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", CapturingAnalyzer)

    monkeypatch.chdir(tmp_path)
    Path("input.wav").write_bytes(b"audio")
    result = runner.invoke(
        main,
        ["input.wav", "--strict", "--include-demographics", "--output", "report.json", "--silent"],
    )

    assert result.exit_code == 0
    assert captured["strict"] is True
    assert captured["include_demographics"] is True


def test_finalize_watch_batch_retries_failures_and_marks_successes(tmp_path):
    good = tmp_path / "good.wav"
    bad = tmp_path / "bad.wav"
    good.write_bytes(b"ok")
    bad.write_bytes(b"bad")
    processed = {}
    retries = {}

    _finalize_watch_batch(
        [
            FileResult(path=str(good)),
            FileResult(path=str(bad), error="decode failed"),
        ],
        processed,
        retries,
        watch_retries=2,
        watch_backoff=0.1,
    )

    assert str(good.resolve()) in processed
    assert retries[str(bad.resolve())]["attempts"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# Option validation and real end-to-end output
# ─────────────────────────────────────────────────────────────────────────────

import csv
import io
import re
import wave

import numpy as np
import pytest


def _write_tone_wav(path: Path, seconds: float = 2.3, sr: int = 16_000) -> None:
    t = np.arange(int(seconds * sr)) / sr
    audio = 0.3 * np.sin(2 * np.pi * 220 * t) * (0.5 * (1 - np.cos(2 * np.pi * 3 * t)))
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())


@pytest.mark.parametrize(
    "extra_args, message",
    [
        (["--mic-seconds", "1"], "--mic-seconds cannot be combined with input paths"),
        (["--silent"], "--silent requires --output"),
        (["--output", "same.json", "--scorecard", "same.json"], "must point to different paths"),
        (["--segment-seconds", "0", "--output", "o.json"], "--segment-seconds must be greater than 0"),
        (["--watch-interval", "0", "--output", "o.json"], "--watch-interval must be greater than 0"),
        (["--watch-debounce", "-1", "--output", "o.json"], "--watch-debounce must be >= 0"),
        (["--watch-max-pending", "0", "--output", "o.json"], "--watch-max-pending must be greater than 0"),
        (["--watch"], "--watch only supports directory inputs"),
    ],
)
def test_cli_rejects_invalid_option_combinations(monkeypatch, tmp_path, extra_args, message):
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", DummyAnalyzer)
    monkeypatch.chdir(tmp_path)
    Path("input.wav").write_bytes(b"audio")
    result = CliRunner().invoke(main, ["input.wav", *extra_args])
    assert result.exit_code != 0
    assert message in result.output


def test_cli_requires_an_input():
    result = CliRunner().invoke(main, [])
    assert result.exit_code != 0
    assert "Provide at least one input path" in result.output


def test_cli_reports_directory_without_audio(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    result = CliRunner().invoke(main, [str(tmp_path)])
    assert result.exit_code == 1
    assert "No supported audio files found" in result.output


@pytest.mark.parametrize("ext, fmt", [(".json", "json"), (".jsonl", "jsonl"), (".csv", "csv"),
                                      (".html", "html"), (".md", "markdown")])
def test_cli_real_short_file_writes_every_format_without_nan(tmp_path, ext, fmt):
    wav = tmp_path / "short.wav"
    _write_tone_wav(wav)
    out = tmp_path / f"report{ext}"

    result = CliRunner().invoke(
        main, [str(wav), "--metrics", "basic,loudness,noise", "--output", str(out), "--silent"]
    )

    assert result.exit_code == 0, result.output
    text = out.read_text()
    assert not re.search(r"\bnan\b", text, re.IGNORECASE), fmt
    if fmt == "json":
        payload = json.loads(text, parse_constant=lambda token: pytest.fail(f"literal {token}"))
        metrics = {m["name"]: m for m in payload[0]["metrics"]}
        assert metrics["Loudness Range (LRA)"]["value"] is None
        assert metrics["Loudness Range (LRA)"]["warning"] == metrics["Max Short-Term Loudness"]["warning"]
    elif fmt == "csv":
        row = next(csv.DictReader(io.StringIO(text)))
        assert row["Loudness Range (LRA)"] == ""
        assert row["Max Short-Term Loudness"] == ""
        assert float(row["Integrated Loudness (LUFS)"]) < 0
    elif fmt in {"html", "markdown"}:
        assert "N/A LU" in text


def test_cli_pretty_console_shows_na_not_nan_for_short_file(tmp_path):
    wav = tmp_path / "short.wav"
    _write_tone_wav(wav)
    result = CliRunner().invoke(main, [str(wav), "--metrics", "loudness", "--no-color"])

    assert result.exit_code == 0, result.output
    assert not re.search(r"\bnan\b", result.output, re.IGNORECASE)
    lra_line = next(line for line in result.output.splitlines() if "Loudness Range (LRA)" in line)
    st_line = next(line for line in result.output.splitlines() if "Max Short-Term Loudness" in line)
    assert lra_line.rstrip().endswith("N/A LU")
    assert st_line.rstrip().endswith("N/A LUFS")


def test_cli_json_to_stdout_validates_against_contract(tmp_path):
    from qualiax.validation import validate_report_payload

    wav = tmp_path / "tone.wav"
    _write_tone_wav(wav, seconds=1.0)
    result = CliRunner().invoke(main, [str(wav), "--metrics", "basic,spectral", "--format", "json"])

    assert result.exit_code == 0, result.output
    # Click < 8.2 (the only option on Python 3.9) mixes the stderr progress bar into result.stdout.
    payload = json.loads(result.stdout[result.stdout.index("[\n"):])
    assert validate_report_payload(payload) == []


def test_python_dash_m_runs_the_cli():
    import subprocess
    import sys

    result = subprocess.run([sys.executable, "-m", "qualiax", "--help"], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.startswith("Usage: qualiax ")


def test_readme_lists_every_cli_option():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    section = readme[readme.index("### All Options"):readme.index("## Python API")]
    missing = [opt for param in main.params for opt in param.opts if opt.startswith("--") and opt not in section]
    assert not missing, f"README 'All Options' is missing {missing}; paste the output of `qualiax --help`"

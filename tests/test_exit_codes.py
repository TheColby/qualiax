"""Every CLI outcome maps to the documented ExitCode value."""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

from qualiax.cli import _most_severe, main
from qualiax.metrics import METRIC_GROUPS
from qualiax.migrations import ExitCode
from qualiax.models import DiagnosticEntry, FileResult, MetricResult


def _tone(path: Path, seconds: float = 1.0, sr: int = 16_000) -> Path:
    t = np.arange(int(seconds * sr)) / sr
    audio = 0.3 * np.sin(2 * np.pi * 220 * t)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())
    return path


def _run(*args):
    return CliRunner().invoke(main, [str(arg) for arg in args])


@pytest.mark.parametrize(
    "args",
    [
        ["--no-such-option"],
        ["{wav}", "--silent"],
        ["{wav}", "--ci"],
        ["{wav}", "--metrics", "not-a-group"],
    ],
)
def test_usage_errors_are_invalid_input_not_quality_gate(tmp_path, args):
    wav = _tone(tmp_path / "a.wav")
    result = _run(*[arg.format(wav=wav) for arg in args])
    assert result.exit_code == ExitCode.INVALID_INPUT, result.output


def test_undecodable_audio_is_invalid_input(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"not a wav file at all")
    result = _run(bad, "--output", tmp_path / "r.json", "--silent")
    assert result.exit_code == ExitCode.INVALID_INPUT
    report = json.loads((tmp_path / "r.json").read_text())
    assert report[0]["error"]
    assert report[0]["diagnostics"][0]["code"] == "audio_load_failed"


def test_strict_metric_group_failure_is_analysis_failed_without_traceback(tmp_path, monkeypatch):
    def broken(audio, sr, ref_audio=None, ref_sr=None):
        raise ValueError("kaboom")

    monkeypatch.setitem(METRIC_GROUPS, "temporal", broken)
    wav = _tone(tmp_path / "a.wav")

    lenient = _run(wav, "--metrics", "basic,temporal", "--output", tmp_path / "a.json", "--silent")
    strict = _run(wav, "--metrics", "basic,temporal", "--output", tmp_path / "b.json", "--silent", "--strict")

    assert lenient.exit_code == ExitCode.OK
    assert strict.exit_code == ExitCode.ANALYSIS_FAILED
    assert isinstance(strict.exception, SystemExit)
    assert "kaboom" in strict.output


def test_output_contract_violation_exit_code(tmp_path, monkeypatch):
    from qualiax import cli
    from qualiax.validation import ValidationIssue

    monkeypatch.setattr(cli, "validate_report_file", lambda path: [ValidationIssue("$[0].file", "broken")])
    result = _run(_tone(tmp_path / "a.wav"), "--metrics", "basic", "--output", tmp_path / "r.json", "--silent", "--validate-output")
    assert result.exit_code == ExitCode.CONTRACT_VIOLATION
    assert "Output validation failed" in result.output


def _fake_analyzer(results):
    class FakeAnalyzer:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_all(self, paths, on_progress=None):
            return results

    return FakeAnalyzer


def _load_failure(path):
    return FileResult(
        path=path,
        error="cannot decode",
        diagnostics=[DiagnosticEntry(code="audio_load_failed", severity="error", source="loader", message="cannot decode")],
    )


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ([FileResult(path="a.wav", metrics=[MetricResult("RMS Level", -20.0, group="basic")])], ExitCode.OK),
        ([_load_failure("a.wav")], ExitCode.INVALID_INPUT),
        ([FileResult(path="a.wav", error="worker crashed")], ExitCode.ANALYSIS_FAILED),
        ([_load_failure("a.wav"), FileResult(path="b.wav", error="worker crashed")], ExitCode.ANALYSIS_FAILED),
    ],
)
def test_batch_errors_distinguish_input_from_analysis_failures(tmp_path, monkeypatch, results, expected):
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", _fake_analyzer(results))
    wav = _tone(tmp_path / "a.wav")
    result = _run(wav, "--output", tmp_path / "r.json", "--silent")
    assert result.exit_code == expected


def test_lint_rules_errors_are_invalid_input(tmp_path):
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"rules": [{"metric": "", "max": "loud"}]}))
    result = _run("--rules", rules, "--lint-rules")
    assert result.exit_code == ExitCode.INVALID_INPUT


def test_watch_mode_reports_the_most_severe_batch_outcome():
    assert _most_severe(ExitCode.OK, ExitCode.QUALITY_GATE_FAILED) == ExitCode.QUALITY_GATE_FAILED
    assert _most_severe(ExitCode.QUALITY_GATE_FAILED, ExitCode.INVALID_INPUT) == ExitCode.INVALID_INPUT
    assert _most_severe(ExitCode.INVALID_INPUT, ExitCode.ANALYSIS_FAILED) == ExitCode.ANALYSIS_FAILED
    assert _most_severe(ExitCode.ANALYSIS_FAILED, ExitCode.OK) == ExitCode.ANALYSIS_FAILED

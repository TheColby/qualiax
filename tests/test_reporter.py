import csv
import io
import json
import re
import warnings
import wave

import numpy as np
import pytest

from qualiax.cli import _make_reporter
from qualiax.models import DiagnosticEntry, FileResult, GroupHealth, MetricResult, ProvenanceInfo
from qualiax.reporter import (
    ConsoleReporter,
    CsvReporter,
    HtmlReporter,
    JsonlReporter,
    JsonReporter,
    MarkdownReporter,
    _sanitize_for_json,
)
from qualiax.validation import validate_report_payload
from qualiax.version import OUTPUT_SCHEMA_VERSION, __version__


def test_json_reporter_serializes_nan_as_null():
    result = FileResult(
        path="example.wav",
        metrics=[MetricResult(name="LRA", value=float("nan"), unit="LU", group="loudness")],
    )

    rendered = JsonReporter().render([result])

    assert "NaN" not in rendered
    payload = json.loads(rendered)
    assert payload[0]["schema_version"] == OUTPUT_SCHEMA_VERSION
    assert payload[0]["tool_version"] == __version__
    assert payload[0]["file"] == "example.wav"
    assert payload[0]["notes"] == []
    assert payload[0]["metrics"][0]["value"] is None


def test_json_reporter_includes_notes_field():
    result = FileResult(
        path="example.wav",
        source_file="example.wav",
        segment_index=1,
        total_segments=3,
        segment_start_s=0.0,
        segment_end_s=1.0,
        notes=["perceptual: proxy fallback used"],
        confidence_notes=["Includes proxy-derived metrics; use directionally."],
        diagnostics=[
            DiagnosticEntry(
                code="proxy_fallback",
                severity="warn",
                source="metric_group",
                message="proxy fallback used",
                group="perceptual",
            )
        ],
        provenance=ProvenanceInfo(
            compute_backend="cpu",
            model_runtime="none",
            runtime_fingerprint="abc123",
        ),
        metrics=[MetricResult(name="MOS", value=3.2, group="perceptual")],
    )

    payload = json.loads(JsonReporter().render([result]))

    assert payload[0]["notes"] == ["perceptual: proxy fallback used"]
    assert payload[0]["confidence_notes"] == ["Includes proxy-derived metrics; use directionally."]
    assert payload[0]["diagnostics"][0]["code"] == "proxy_fallback"
    assert payload[0]["provenance"]["runtime_fingerprint"] == "abc123"
    assert "speech_confidence" in payload[0]
    assert payload[0]["segment_index"] == 1
    assert validate_report_payload(payload) == []


def test_html_reporter_renders_status_and_metric_content():
    result = FileResult(
        path="example.wav",
        duration_s=2.5,
        sample_rate=16_000,
        channels=1,
        content_type="speech",
        speech_confidence=0.93,
        notes=["perceptual: proxy fallback used"],
        confidence_notes=["Includes proxy-derived metrics; use directionally."],
        diagnostics=[
            DiagnosticEntry(
                code="proxy_fallback",
                severity="warn",
                source="metric_group",
                message="proxy fallback used",
            )
        ],
        provenance=ProvenanceInfo(
            compute_backend="cpu",
            model_runtime="none",
            runtime_fingerprint="abc123",
        ),
        metrics=[
            MetricResult(
                name="Integrated Loudness (LUFS)",
                value=-16.1,
                unit="LUFS",
                description="Program loudness",
                group="loudness",
            ),
            MetricResult(
                name="True Peak",
                value=-0.4,
                unit="dBTP",
                description="Inter-sample peak level",
                group="loudness",
                warning="Exceeds -1 dBTP streaming limit",
                confidence="proxy",
                calibration_note="Directional only.",
            ),
        ],
    )

    rendered = HtmlReporter().render([result])

    assert "<!doctype html>" in rendered.lower()
    assert "qualiax HTML Report" in rendered
    assert "example.wav" in rendered
    assert "Needs Review" in rendered
    assert "Integrated Loudness (LUFS)" in rendered
    assert "Exceeds -1 dBTP streaming limit" in rendered
    assert "Confidence &amp; Calibration" in rendered
    assert "Diagnostics" in rendered
    assert "Provenance" in rendered
    assert "proxy - Directional only." in rendered


def test_markdown_reporter_renders_tables_and_notes():
    result = FileResult(
        path="example.wav",
        source_file="session.wav",
        segment_index=2,
        total_segments=4,
        segment_start_s=30.0,
        segment_end_s=45.0,
        duration_s=2.5,
        sample_rate=16_000,
        channels=1,
        notes=["threshold rules: 1 violation(s)"],
        confidence_notes=["Includes proxy-derived metrics; use directionally."],
        diagnostics=[
            DiagnosticEntry(
                code="threshold_rules_violations",
                severity="warn",
                source="rules",
                message="1 threshold rule violation(s)",
            )
        ],
        provenance=ProvenanceInfo(
            compute_backend="cpu",
            model_runtime="none",
            runtime_fingerprint="abc123",
        ),
        metrics=[
            MetricResult(
                name="Integrated Loudness (LUFS)",
                value=-16.1,
                unit="LUFS",
                description="Program loudness",
                group="loudness",
                confidence="measured",
            ),
        ],
    )

    rendered = MarkdownReporter().render([result])

    assert "# qualiax Report" in rendered
    assert "## `example.wav`" in rendered
    assert "- Segment: `2/4`" in rendered
    assert "### Notes" in rendered
    assert "### Confidence & Calibration" in rendered
    assert "### Diagnostics" in rendered
    assert "### Provenance" in rendered
    assert "| Metric | Value | Description | Warning | Trust |" in rendered


# ─────────────────────────────────────────────────────────────────────────────
# Every output format, including awkward values
# ─────────────────────────────────────────────────────────────────────────────

ALL_FORMATS = ["pretty", "json", "jsonl", "csv", "html", "markdown"]
FIXED_CSV_COLUMNS = [
    "schema_version", "tool_version", "file", "source_file", "segment_index", "total_segments",
    "segment_start_s", "segment_end_s", "duration_s", "sample_rate", "channels", "content_type",
    "speech_confidence", "notes", "confidence_notes", "diagnostics", "group_health",
    "provenance_backend", "provenance_runtime", "runtime_fingerprint", "insights", "error",
]
NAN_TOKEN = re.compile(r"\bnan\b", re.IGNORECASE)


def _strict_json_loads(text):
    """json.loads that rejects the non-standard NaN / Infinity / -Infinity literals."""
    def reject(token):
        raise ValueError(f"non-standard JSON literal: {token}")
    return json.loads(text, parse_constant=reject)


def _awkward_result(path="awkward.wav"):
    return FileResult(
        path=path,
        duration_s=2.3,
        sample_rate=16_000,
        channels=1,
        content_type="speech",
        speech_confidence=np.float32(0.8),
        notes=["loudness: short file"],
        group_health=[GroupHealth(group="loudness", status="partial", metric_count=4, missing_count=2)],
        provenance=ProvenanceInfo(compute_backend="cpu (numpy)", runtime_fingerprint="f00"),
        insights={
            "quality_fingerprint": {"signature": "abc", "score": float("nan")},
            "defect_labels": [{"id": "noisy_floor", "evidence": "snr"}],
        },
        metrics=[
            MetricResult("Loudness Range (LRA)", float("nan"), "LU", "range", "loudness"),
            MetricResult("Max Short-Term Loudness", None, "LUFS", "st", "loudness",
                         warning="Unavailable: requires at least 3 s of audio (got 2.30 s)"),
            MetricResult("Peak Level", float("-inf"), "dBFS", "peak", "basic"),
            MetricResult("Weird Ratio", float("inf"), "", "", "basic"),
            MetricResult("Numpy Undefined", np.float32("nan"), "dB", "", "noise"),
            MetricResult("Numpy Float", np.float32(1.5), "dB", "", "noise", reference_range=(0.0, 2.0)),
            MetricResult("Numpy Int", np.int64(3), "", "", "noise"),
            MetricResult("Label", "male", "", "", "speaker"),
            MetricResult("Integrated Loudness (LUFS)", -16.2, "LUFS", "loud", "loudness",
                         reference_range=(-16.0, -14.0), higher_is_better=True),
        ],
    )


@pytest.mark.parametrize("fmt", ALL_FORMATS)
def test_every_format_renders_without_nan(fmt):
    rendered = _make_reporter(fmt, color=False).render([_awkward_result(), _awkward_result("second.wav")])

    assert rendered.strip()
    assert "awkward.wav" in rendered
    if fmt in {"json", "jsonl"}:
        assert "NaN" not in rendered and "Infinity" not in rendered
        payload = (
            _strict_json_loads(rendered)
            if fmt == "json"
            else [_strict_json_loads(line) for line in rendered.splitlines() if line.strip()]
        )
        assert len(payload) == 2
        values = {m["name"]: m["value"] for m in payload[0]["metrics"]}
        assert values["Loudness Range (LRA)"] is None
        assert values["Numpy Undefined"] is None
        assert values["Peak Level"] is None       # -inf has no JSON literal
        assert values["Numpy Float"] == 1.5
        assert values["Numpy Int"] == 3
        assert payload[0]["insights"]["quality_fingerprint"]["score"] is None
        assert validate_report_payload(payload) == []
    else:
        assert not NAN_TOKEN.search(rendered), fmt


@pytest.mark.parametrize("fmt", ["pretty", "html", "markdown"])
def test_human_formats_render_missing_values_as_na(fmt):
    rendered = _make_reporter(fmt, color=False).render([_awkward_result()])
    assert "N/A LU" in rendered        # NaN LRA
    assert "N/A LUFS" in rendered      # None short-term
    assert "Unavailable: requires at least 3 s of audio" in rendered
    assert "1.5000 dB" in rendered     # numpy float formatted like a float


def test_formatted_value_handles_numbers_consistently():
    assert MetricResult("x", float("nan")).formatted_value() == "N/A"
    assert MetricResult("x", np.float64("nan")).formatted_value() == "N/A"
    assert MetricResult("x", None).formatted_value() == "N/A"
    assert MetricResult("x", float("-inf"), "dBFS").value_with_unit() == "-inf dBFS"
    assert MetricResult("x", np.float32(0.25)).formatted_value() == "0.2500"
    assert MetricResult("x", 16_000, "Hz").value_with_unit() == "16000 Hz"
    assert MetricResult("x", True).formatted_value() == "True"
    assert MetricResult("x", "female").formatted_value() == "female"


def test_sanitize_for_json_converts_numpy_containers():
    data = {"a": np.array([1.0, np.nan]), "b": (np.int32(2), np.bool_(True)), "c": [float("inf")]}
    assert _sanitize_for_json(data) == {"a": [1.0, None], "b": [2, True], "c": [None]}


def test_csv_header_is_stable_and_blank_for_missing_values():
    results = [_awkward_result("a.wav"), FileResult(path="b.wav", metrics=[MetricResult("Extra", 1.0)])]
    first = CsvReporter().render(results)
    second = CsvReporter().render(results)
    assert first == second

    rows = list(csv.reader(io.StringIO(first)))
    header = rows[0]
    assert header[: len(FIXED_CSV_COLUMNS)] == FIXED_CSV_COLUMNS
    metric_columns = header[len(FIXED_CSV_COLUMNS):]
    assert metric_columns == [m.name for m in results[0].metrics] + ["Extra"]  # first-seen order

    record = dict(zip(header, rows[1]))
    assert record["Loudness Range (LRA)"] == ""
    assert record["Numpy Undefined"] == ""
    assert record["Peak Level"] == ""
    assert record["Numpy Float"] == "1.5"
    assert record["Numpy Int"] == "3"
    assert record["Extra"] == ""
    assert dict(zip(header, rows[2]))["Extra"] == "1.0"
    assert record["group_health"] == "loudness=partial"


def test_html_report_is_self_contained():
    html = HtmlReporter().render([_awkward_result()])
    lowered = html.lower()
    assert "<link" not in lowered
    assert not re.search(r"<script[^>]+src=", lowered)
    assert not re.search(r"(src|href)\s*=\s*[\"']?(https?:)?//", lowered)
    assert "@import" not in lowered
    assert not re.search(r"url\(\s*[\"']?(https?:)?//", lowered)
    assert lowered.count("<script>") == 1  # the inline filter script only


def test_html_escapes_untrusted_text():
    result = FileResult(path="<script>alert(1)</script>.wav",
                        metrics=[MetricResult("A & B", "<b>", warning="x < y")])
    html = HtmlReporter().render([result])
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;.wav" in html
    assert "A &amp; B" in html and "x &lt; y" in html


def test_markdown_escapes_pipes_in_cells():
    result = FileResult(path="x.wav", metrics=[MetricResult("a|b", "c|d", description="line1\nline2")])
    md = MarkdownReporter().render([result])
    assert "| a\\|b | c\\|d | line1 line2 |" in md


def test_error_results_render_in_every_format():
    failed = FileResult(path="broken.wav", error="All loaders failed")
    for fmt in ALL_FORMATS:
        rendered = _make_reporter(fmt, color=False).render([failed])
        assert "All loaders failed" in rendered, fmt


def test_jsonl_of_no_results_is_empty():
    assert JsonlReporter().render([]) == ""


def test_console_colours_follow_reference_range_and_warnings():
    in_range = MetricResult("SNR", 25.0, "dB", group="noise", higher_is_better=True, reference_range=(20.0, 40.0))
    out_of_range = MetricResult("SNR2", 10.0, "dB", group="noise", higher_is_better=True,
                                reference_range=(20.0, 40.0))
    warned = MetricResult("SNR3", 25.0, "dB", group="noise", warning="check", reference_range=(20.0, 40.0))
    missing = MetricResult("SNR4", None, "dB", group="noise", warning="n/a")
    rendered = ConsoleReporter(color=True).render(
        [FileResult(path="c.wav", metrics=[in_range, out_of_range, warned, missing])]
    )

    assert "\033[92m25.0000 dB" in rendered   # green: inside range
    assert "\033[93m10.0000 dB" in rendered   # yellow: outside range
    assert "\033[93m25.0000 dB" in rendered   # yellow: has a warning
    assert "N/A dB" in rendered and "\033[93mN/A" not in rendered


def test_console_renders_segments_insights_and_provenance():
    result = _awkward_result()
    result.segment_index, result.total_segments = 2, 3
    result.segment_start_s, result.segment_end_s = 1.0, 2.0
    result.source_file = "long.wav"
    rendered = ConsoleReporter(color=False).render([result])
    assert "segment 2/3" in rendered and "source=long.wav" in rendered and "1.00-2.00s" in rendered
    assert "fingerprint: abc" in rendered
    assert "label: noisy_floor" in rendered
    assert "backend: cpu (numpy)" in rendered


def test_real_short_file_renders_na_in_every_format(tmp_path):
    """End to end: a 2.3 s file through the analyzer shows N/A, never nan, for LRA and short-term."""
    from qualiax.analyzer import AudioAnalyzer

    sr = 16_000
    t = np.arange(int(2.3 * sr)) / sr
    audio = 0.3 * np.sin(2 * np.pi * 220 * t) * (0.5 * (1 - np.cos(2 * np.pi * 3 * t)))
    path = tmp_path / "short.wav"
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes((audio * 32767).astype("<i2").tobytes())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = AudioAnalyzer(metric_groups={"basic", "loudness", "noise"}).analyze_file(path)
    metrics = {m.name: m for m in result.metrics}
    assert metrics["Loudness Range (LRA)"].value is None
    assert metrics["Max Short-Term Loudness"].value is None
    assert metrics["Loudness Range (LRA)"].warning == metrics["Max Short-Term Loudness"].warning

    for fmt in ALL_FORMATS:
        rendered = _make_reporter(fmt, color=False).render([result])
        if fmt in {"json", "jsonl"}:
            _strict_json_loads(rendered.splitlines()[0] if fmt == "jsonl" else rendered)
        assert not NAN_TOKEN.search(rendered), fmt

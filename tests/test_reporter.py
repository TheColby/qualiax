import json

from qualiax.models import DiagnosticEntry, FileResult, MetricResult, ProvenanceInfo
from qualiax.reporter import HtmlReporter, JsonReporter, MarkdownReporter
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

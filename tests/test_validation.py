import json

from qualiax import build_scorecard, get_json_schema
from qualiax.models import FileResult, MetricResult
from qualiax.reporter import JsonReporter
from qualiax.validation import validate_report_payload, validate_scorecard_payload


def test_report_payload_validates_against_builtin_contract():
    payload = json.loads(
        JsonReporter().render(
            [
                FileResult(
                    path="example.wav",
                    metrics=[MetricResult(name="RMS Level", value=-18.0, unit="dBFS", group="basic")],
                )
            ]
        )
    )

    assert validate_report_payload(payload) == []


def test_scorecard_payload_validates_against_builtin_contract():
    scorecard = build_scorecard(
        [
            FileResult(
                path="example.wav",
                metrics=[MetricResult(name="RMS Level", value=-18.0, unit="dBFS", group="basic")],
            )
        ]
    )

    assert validate_scorecard_payload(scorecard.to_dict()) == []


def test_get_json_schema_exposes_known_contracts():
    report_schema = get_json_schema("report")
    scorecard_schema = get_json_schema("scorecard")

    assert report_schema["title"] == "qualiax report"
    assert scorecard_schema["title"] == "qualiax scorecard"


def _item_with_insights(insights):
    return [{**json.loads(JsonReporter().render([FileResult(path="x.wav")]))[0], "insights": insights}]


def test_partial_insights_without_version_only_type_checks_present_fields():
    label = {"id": "hum", "severity": "warn", "confidence": 0.8, "evidence": "50 Hz peak", "evidence_metrics": []}
    assert validate_report_payload(_item_with_insights({"defect_labels": [label]})) == []

    bad = validate_report_payload(_item_with_insights({"defect_labels": [{"id": "hum", "severity": "loud"}]}))
    assert {issue.path for issue in bad} >= {
        "$[0].insights.defect_labels[0].severity",
        "$[0].insights.defect_labels[0].confidence",
    }
    unexpected = validate_report_payload(_item_with_insights({"custom": True}))
    assert [issue.message for issue in unexpected] == ["Unexpected field."]


def test_versioned_insights_must_be_a_complete_payload():
    issues = validate_report_payload(_item_with_insights({"version": "1.0", "defect_labels": []}))
    missing = {issue.path for issue in issues if issue.message == "Missing required field."}
    assert "$[0].insights.quality_fingerprint" in missing
    assert "$[0].insights.triage" in missing

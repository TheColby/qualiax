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

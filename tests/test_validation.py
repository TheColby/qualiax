import json

import pytest

from qualiax import build_scorecard, get_json_schema
from qualiax.models import FileResult, MetricResult
from qualiax.reporter import JsonReporter
from qualiax.validation import (
    assert_valid_report_payload,
    assert_valid_scorecard_payload,
    validate_report_file,
    validate_report_payload,
    validate_scorecard_payload,
)


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


# ─────────────────────────────────────────────────────────────────────────────
# Rejection paths: every constraint type must actually produce an issue
# ─────────────────────────────────────────────────────────────────────────────

def _valid_report():
    return json.loads(
        JsonReporter().render(
            [
                FileResult(
                    path="example.wav",
                    duration_s=1.0,
                    sample_rate=16_000,
                    channels=1,
                    metrics=[
                        MetricResult(name="RMS Level", value=-18.0, unit="dBFS", group="basic",
                                     reference_range=(-20.0, -9.0)),
                    ],
                )
            ]
        )
    )


def _issue_paths(issues):
    return {issue.path for issue in issues}


def test_report_payload_must_be_a_list():
    issues = validate_report_payload({"file": "x.wav"})
    assert [(i.path, i.message) for i in issues] == [("$", "Report payload must be a list.")]


def test_missing_required_field_is_reported():
    payload = _valid_report()
    del payload[0]["metrics"]
    assert "$[0].metrics" in _issue_paths(validate_report_payload(payload))


@pytest.mark.parametrize(
    "mutate, path",
    [
        (lambda p: p[0].__setitem__("sample_rate", "16000"), "$[0].sample_rate"),
        (lambda p: p[0].__setitem__("sample_rate", True), "$[0].sample_rate"),       # bool is not an integer
        (lambda p: p[0].__setitem__("duration_s", None), "$[0].duration_s"),
        (lambda p: p[0].__setitem__("schema_version", "0.0"), "$[0].schema_version"),  # const
        (lambda p: p[0].__setitem__("unexpected", 1), "$[0].unexpected"),               # additionalProperties
        (lambda p: p[0]["metrics"][0].__setitem__("unit", 3), "$[0].metrics[0].unit"),
        (lambda p: p[0]["metrics"][0].__setitem__("higher_is_better", "yes"),
         "$[0].metrics[0].higher_is_better"),
        (lambda p: p[0]["metrics"][0].__setitem__("reference_range", ["a", 1]),
         "$[0].metrics[0].reference_range[0]"),
        (lambda p: p[0].__setitem__("notes", "not a list"), "$[0].notes"),
        (lambda p: p[0].__setitem__("notes", [1]), "$[0].notes[0]"),
    ],
)
def test_type_const_and_shape_violations_are_reported(mutate, path):
    payload = _valid_report()
    mutate(payload)
    assert path in _issue_paths(validate_report_payload(payload))


def test_diagnostic_enum_and_minimum_are_enforced():
    payload = _valid_report()
    payload[0]["diagnostics"] = [
        {"code": "x", "severity": "fatal", "source": "s", "message": "m", "group": None,
         "metric": None, "count": 0, "context": {}}
    ]
    paths = _issue_paths(validate_report_payload(payload))
    assert "$[0].diagnostics[0].severity" in paths
    assert "$[0].diagnostics[0].count" in paths


def test_provenance_null_is_allowed_but_wrong_type_is_not():
    payload = _valid_report()
    payload[0]["provenance"] = None
    assert validate_report_payload(payload) == []
    payload[0]["provenance"] = "cpu"
    assert "$[0].provenance" in _issue_paths(validate_report_payload(payload))


def test_reference_range_must_have_exactly_two_items():
    payload = _valid_report()
    payload[0]["metrics"][0]["reference_range"] = [0.0, 1.0, 2.0]
    assert "$[0].metrics[0].reference_range" in _issue_paths(validate_report_payload(payload))


def test_scorecard_validation_rejects_bad_counts():
    payload = build_scorecard([FileResult(path="a.wav")]).to_dict()
    payload["file_count"] = -1
    assert "$.file_count" in _issue_paths(validate_scorecard_payload(payload))


def test_validate_report_file_handles_json_jsonl_and_scorecards(tmp_path):
    report = _valid_report()
    json_path = tmp_path / "report.json"
    json_path.write_text(json.dumps(report))
    assert validate_report_file(json_path) == []

    jsonl_path = tmp_path / "report.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(item) for item in report + report) + "\n\n")
    assert validate_report_file(jsonl_path) == []

    broken = tmp_path / "broken.jsonl"
    item = dict(report[0])
    del item["file"]
    broken.write_text(json.dumps(item) + "\n")
    assert "$[0].file" in _issue_paths(validate_report_file(broken))

    scorecard_path = tmp_path / "scorecard.json"
    scorecard_path.write_text(json.dumps(build_scorecard([FileResult(path="a.wav")]).to_dict()))
    assert validate_report_file(scorecard_path) == []


def test_assert_helpers_raise_with_readable_messages():
    assert_valid_report_payload(_valid_report())
    with pytest.raises(ValueError, match=r"\$\[0\]\.file: Missing required field"):
        payload = _valid_report()
        del payload[0]["file"]
        assert_valid_report_payload(payload)
    with pytest.raises(ValueError, match="file_count"):
        assert_valid_scorecard_payload({"schema_version": "x"})


def test_unknown_schema_name_is_rejected():
    with pytest.raises(ValueError, match="Unknown schema"):
        get_json_schema("nope")

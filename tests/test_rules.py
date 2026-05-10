from qualiax.models import FileResult, MetricResult
from qualiax.rules import ThresholdRule, apply_threshold_rules, lint_threshold_rules


def test_rules_match_case_insensitively():
    result = FileResult(
        path="clip.wav",
        metrics=[MetricResult(name="True Peak", value=-0.1, unit="dBTP", group="loudness")],
    )

    violations = apply_threshold_rules(
        [result],
        [ThresholdRule(metric=" true peak ", group="LOUDNESS", max=-1.0)],
    )

    assert len(violations) == 1
    assert "threshold rules: 1 violation(s)" in result.notes


def test_rules_note_unmatched_metric_names():
    result = FileResult(
        path="clip.wav",
        metrics=[MetricResult(name="RMS Level", value=-18.0, unit="dBFS", group="basic")],
    )

    apply_threshold_rules(
        [result],
        [ThresholdRule(metric="Missing Metric", max=0.0)],
        unmatched_behavior="note",
    )

    assert "threshold rules: unmatched metric names: Missing Metric" in result.notes


def test_rules_can_treat_unmatched_metric_names_as_violations():
    result = FileResult(
        path="clip.wav",
        metrics=[MetricResult(name="RMS Level", value=-18.0, unit="dBFS", group="basic")],
    )

    violations = apply_threshold_rules(
        [result],
        [ThresholdRule(metric="Missing Metric", max=0.0)],
        unmatched_behavior="violation",
    )

    assert len(violations) == 1
    assert violations[0].file == "*"
    assert result.diagnostics[0].code == "threshold_rules_unmatched"


def test_rule_lint_flags_invalid_ranges_and_duplicates():
    issues = lint_threshold_rules(
        [
            ThresholdRule(metric="True Peak", max=-1.0, group="loudness"),
            ThresholdRule(metric="True Peak", max=-1.0, group="loudness"),
            ThresholdRule(metric="LUFS", min=-10.0, max=-20.0, group="loudness"),
        ],
        valid_groups={"loudness", "basic"},
    )

    assert any(issue.level == "warn" and "duplicates" in issue.message for issue in issues)
    assert any(issue.level == "error" and "min > max" in issue.message for issue in issues)

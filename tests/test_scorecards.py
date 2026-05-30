import json

from qualiax import FileResult, MetricResult, build_scorecard, render_scorecard


def test_build_scorecard_rolls_up_metrics_and_status():
    results = [
        FileResult(
            path="a.wav",
            confidence_notes=["Includes proxy-derived metrics; use directionally."],
            metrics=[
                MetricResult(
                    name="Integrated Loudness (LUFS)",
                    value=-16.0,
                    unit="LUFS",
                    group="loudness",
                ),
                MetricResult(
                    name="True Peak",
                    value=-0.8,
                    unit="dBTP",
                    group="loudness",
                    warning="Ceiling exceeded",
                ),
            ],
        ),
        FileResult(
            path="b.wav",
            metrics=[
                MetricResult(
                    name="Integrated Loudness (LUFS)",
                    value=-14.0,
                    unit="LUFS",
                    group="loudness",
                ),
                MetricResult(
                    name="True Peak",
                    value=-1.2,
                    unit="dBTP",
                    group="loudness",
                ),
            ],
        ),
    ]

    scorecard = build_scorecard(results)

    assert scorecard.file_count == 2
    assert scorecard.status_counts["warn"] == 1
    assert scorecard.status_counts["ok"] == 1
    assert scorecard.confidence_notes == ("Includes proxy-derived metrics; use directionally.",)
    loudness = {
        (rollup.group, rollup.name): rollup
        for rollup in scorecard.metric_rollups
    }[("loudness", "Integrated Loudness (LUFS)")]
    assert loudness.mean == -15.0
    assert loudness.min_file == "a.wav"
    assert loudness.max_file == "b.wav"


def test_render_scorecard_json_and_markdown():
    scorecard = build_scorecard(
        [
            FileResult(
                path="a.wav",
                metrics=[MetricResult(name="RMS Level", value=-18.0, unit="dBFS", group="basic")],
            )
        ]
    )

    payload = json.loads(render_scorecard(scorecard, "json"))
    markdown = render_scorecard(scorecard, "markdown")

    assert payload["file_count"] == 1
    assert payload["metric_rollups"][0]["name"] == "RMS Level"
    assert "# qualiax Scorecard" in markdown
    assert "| Group | Metric | Count |" in markdown


def test_scorecard_json_treats_nan_values_as_missing():
    scorecard = build_scorecard(
        [
            FileResult(
                path="a.wav",
                metrics=[MetricResult(name="LRA", value=float("nan"), unit="LU", group="loudness")],
            )
        ]
    )

    payload = json.loads(render_scorecard(scorecard, "json"))
    rollup = payload["metric_rollups"][0]

    assert rollup["name"] == "LRA"
    assert rollup["measured_count"] == 0
    assert rollup["missing_count"] == 1
    assert rollup["mean"] is None


def test_scorecard_aggregates_segments_by_source_file():
    scorecard = build_scorecard(
        [
            FileResult(
                path="call.wav [segment 1/2]",
                source_file="call.wav",
                metrics=[MetricResult(name="RMS Level", value=-20.0, unit="dBFS", group="basic", warning="low")],
            ),
            FileResult(
                path="call.wav [segment 2/2]",
                source_file="call.wav",
                metrics=[MetricResult(name="RMS Level", value=-10.0, unit="dBFS", group="basic")],
            ),
            FileResult(
                path="other.wav",
                metrics=[MetricResult(name="RMS Level", value=-6.0, unit="dBFS", group="basic")],
            ),
        ]
    )

    rollup = scorecard.metric_rollups[0]

    assert scorecard.file_count == 2
    assert rollup.source_count == 2
    assert rollup.measured_count == 2
    assert rollup.warning_count == 1
    assert rollup.mean == -10.5


def test_scorecard_captures_categorical_rollups_and_confidence_counts():
    scorecard = build_scorecard(
        [
            FileResult(
                path="speech.wav",
                metrics=[
                    MetricResult(
                        name="Content Type",
                        value="speech",
                        group="speech",
                        confidence="model",
                    )
                ],
            ),
            FileResult(
                path="music.wav",
                metrics=[
                    MetricResult(
                        name="Content Type",
                        value="music",
                        group="speech",
                        confidence="model",
                    )
                ],
            ),
            FileResult(
                path="speech-2.wav",
                metrics=[
                    MetricResult(
                        name="Content Type",
                        value="speech",
                        group="speech",
                        confidence="model",
                    )
                ],
            ),
        ]
    )

    rollup = scorecard.metric_rollups[0]
    payload = json.loads(render_scorecard(scorecard, "json"))

    assert rollup.value_kind == "categorical"
    assert rollup.measured_count == 3
    assert rollup.category_counts == {"music": 1, "speech": 2}
    assert rollup.dominant_category == "speech"
    assert rollup.confidence_counts == {"model": 3}
    assert payload["metric_rollups"][0]["category_counts"]["speech"] == 2


def test_scorecard_rolls_up_insight_labels_ci_and_drift():
    scorecard = build_scorecard(
        [
            FileResult(
                path="a.wav",
                insights={
                    "defect_labels": [
                        {"id": "noisy_floor", "severity": "fail"},
                        {"id": "clipping_risk", "severity": "warn"},
                    ],
                    "ci_checks": [{"id": "audio_quality_gate", "status": "fail"}],
                    "drift_monitor": {"status": "drift"},
                    "dataset_audit": [{"id": "duplicate_quality_fingerprint", "severity": "warn"}],
                },
            ),
            FileResult(
                path="b.wav",
                insights={
                    "defect_labels": [{"id": "noisy_floor", "severity": "fail"}],
                    "ci_checks": [{"id": "audio_quality_gate", "status": "pass"}],
                    "drift_monitor": {"status": "stable"},
                    "dataset_audit": [],
                },
            ),
        ]
    )

    payload = json.loads(render_scorecard(scorecard, "json"))
    markdown = render_scorecard(scorecard, "markdown")

    assert payload["insight_summary"]["defect_label_counts"]["noisy_floor"] == 2
    assert payload["insight_summary"]["label_severity_counts"]["fail"] == 2
    assert payload["insight_summary"]["ci_status_counts"]["fail"] == 1
    assert payload["insight_summary"]["drift_status_counts"]["drift"] == 1
    assert "Insight Rollups" in markdown
    assert "noisy_floor" in markdown

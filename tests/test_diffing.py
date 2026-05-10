import json

from qualiax.models import FileResult, MetricResult
from qualiax.diffing import diff_reports, render_diff


def test_diff_reports_surfaces_regressions_and_improvements(tmp_path):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(
        json.dumps(
            [
                {
                    "file": "clip.wav",
                    "error": None,
                    "metrics": [
                        {
                            "name": "True Peak",
                            "group": "loudness",
                            "value": -2.0,
                            "unit": "dBTP",
                            "higher_is_better": False,
                            "warning": None,
                        },
                        {
                            "name": "STOI",
                            "group": "perceptual",
                            "value": 0.82,
                            "unit": "",
                            "higher_is_better": True,
                            "warning": None,
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    after.write_text(
        json.dumps(
            [
                {
                    "file": "clip.wav",
                    "error": None,
                    "metrics": [
                        {
                            "name": "True Peak",
                            "group": "loudness",
                            "value": -0.3,
                            "unit": "dBTP",
                            "higher_is_better": False,
                            "warning": "Exceeds ceiling",
                        },
                        {
                            "name": "STOI",
                            "group": "perceptual",
                            "value": 0.90,
                            "unit": "",
                            "higher_is_better": True,
                            "warning": None,
                        },
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )

    summary = diff_reports(before, after)

    assert summary.regressions == 1
    assert summary.improvements == 1
    rendered = render_diff(summary, "markdown")
    assert "qualiax Diff Report" in rendered
    assert "Regression detected" in rendered


def test_diff_uses_reference_ranges_from_real_serialized_results(tmp_path):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(
        json.dumps(
            [
                FileResult(
                    path="clip.wav",
                    metrics=[
                        MetricResult(
                            name="Integrated Loudness (LUFS)",
                            group="loudness",
                            value=-15.0,
                            unit="LUFS",
                            reference_range=(-16.0, -14.0),
                        )
                    ],
                ).to_dict()
            ]
        ),
        encoding="utf-8",
    )
    after.write_text(
        json.dumps(
            [
                FileResult(
                    path="clip.wav",
                    metrics=[
                        MetricResult(
                            name="Integrated Loudness (LUFS)",
                            group="loudness",
                            value=-10.0,
                            unit="LUFS",
                            reference_range=(-16.0, -14.0),
                        )
                    ],
                ).to_dict()
            ]
        ),
        encoding="utf-8",
    )

    summary = diff_reports(before, after)

    assert len(summary.entries) == 1
    assert summary.entries[0].classification == "regression"
    assert summary.entries[0].note == "Regression detected"


def test_diff_accepts_jsonl_reports(tmp_path):
    before = tmp_path / "before.jsonl"
    after = tmp_path / "after.jsonl"
    before.write_text(
        json.dumps(
            {
                "file": "clip.wav",
                "error": None,
                "metrics": [{"name": "True Peak", "group": "loudness", "value": -2.0, "unit": "dBTP", "higher_is_better": False, "warning": None}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    after.write_text(
        json.dumps(
            {
                "file": "clip.wav",
                "error": None,
                "metrics": [{"name": "True Peak", "group": "loudness", "value": -0.2, "unit": "dBTP", "higher_is_better": False, "warning": "Exceeds ceiling"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = diff_reports(before, after)

    assert summary.regressions == 1

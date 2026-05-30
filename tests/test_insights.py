import json
from pathlib import Path

from click.testing import CliRunner

from qualiax.insights import (
    INSIGHT_SCHEMA_VERSION,
    InsightPayload,
    build_insight_summary_payload,
    enrich_existing_report,
    enrich_results,
    export_flagged_segment_snippets,
    validate_insights_report,
)
from qualiax.models import FileResult, MetricResult
from qualiax.cli import main
from qualiax.reporter import HtmlReporter


def _result(path: str, metrics: list[MetricResult], *, segment_index=None, source_file=None) -> FileResult:
    return FileResult(
        path=path,
        source_file=source_file,
        segment_index=segment_index,
        total_segments=2 if segment_index is not None else None,
        segment_start_s=float(segment_index - 1) if segment_index else None,
        segment_end_s=float(segment_index) if segment_index else None,
        duration_s=1.0,
        sample_rate=16_000,
        channels=1,
        metrics=metrics,
    )


def _write_wav(path: Path, *, sr: int = 16_000, seconds: float = 1.0) -> None:
    import wave

    import numpy as np

    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    pcm = np.clip(audio * 32767, -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def test_enrich_results_adds_quality_fingerprints_labels_and_suggestions():
    results = [
        _result(
            "voice.wav",
            [
                MetricResult(name="SNR", value=8.0, unit="dB", group="noise"),
                MetricResult(name="True Peak", value=0.1, unit="dBTP", group="loudness"),
                MetricResult(name="DNSMOS OVRL", value=2.1, group="perceptual"),
            ],
        )
    ]

    enrich_results(results, ci=True)

    insights = results[0].insights
    assert insights["version"] == INSIGHT_SCHEMA_VERSION
    assert insights["quality_fingerprint"]["signature"]
    assert "noise.snr" in insights["quality_fingerprint"]["features"]
    labels = {label["id"]: label for label in insights["defect_labels"]}
    assert set(labels) >= {"noisy_floor", "clipping_risk"}
    assert labels["noisy_floor"]["severity"] == "fail"
    assert labels["noisy_floor"]["evidence_metrics"][0]["metric"] == "SNR"
    assert any("denoise" in suggestion.lower() for suggestion in insights["repair_suggestions"])
    assert insights["mos_explanation"][0]["metric"] == "DNSMOS OVRL"
    assert insights["ci_checks"][0]["status"] == "fail"
    payload = InsightPayload.from_dict(insights)
    assert payload.version == INSIGHT_SCHEMA_VERSION
    assert payload.to_dict()["version"] == INSIGHT_SCHEMA_VERSION


def test_fingerprint_sensitivity_and_weights_are_configurable():
    loose = [_result("a.wav", [MetricResult(name="SNR", value=10.9, unit="dB", group="noise")])]
    strict = [_result("a.wav", [MetricResult(name="SNR", value=10.9, unit="dB", group="noise")])]

    enrich_results(loose, fingerprint_sensitivity="coarse", fingerprint_weights={"noise.snr": 2.0})
    enrich_results(strict, fingerprint_sensitivity="strict", fingerprint_weights={"noise.snr": 2.0})

    loose_fp = loose[0].insights["quality_fingerprint"]
    strict_fp = strict[0].insights["quality_fingerprint"]
    assert loose_fp["sensitivity"] == "coarse"
    assert loose_fp["weights"] == {"noise.snr": 2.0}
    assert loose_fp["buckets"] != strict_fp["buckets"]
    assert loose_fp["signature"] != strict_fp["signature"]


def test_enrich_results_compares_against_baseline_and_builds_dataset_audit():
    baseline = [
        _result(
            "baseline.wav",
            [
                MetricResult(name="SNR", value=28.0, unit="dB", group="noise"),
                MetricResult(name="Integrated Loudness (LUFS)", value=-16.0, unit="LUFS", group="loudness"),
            ],
        )
    ]
    results = [
        _result(
            "current-a.wav",
            [
                MetricResult(name="SNR", value=12.0, unit="dB", group="noise"),
                MetricResult(name="Integrated Loudness (LUFS)", value=-24.0, unit="LUFS", group="loudness"),
            ],
        ),
        _result(
            "current-b.wav",
            [
                MetricResult(name="SNR", value=12.0, unit="dB", group="noise"),
                MetricResult(name="Integrated Loudness (LUFS)", value=-24.0, unit="LUFS", group="loudness"),
            ],
        ),
    ]

    enrich_results(results, baseline_results=baseline, dataset_audit=True)

    assert results[0].insights["baseline_comparison"]["status"] == "regressed"
    assert results[0].insights["baseline_comparison"]["baseline"]["profile_size"] == 2
    assert results[0].insights["baseline_comparison"]["baseline"]["fingerprint"]
    assert results[0].insights["baseline_comparison"]["comparisons"][0]["baseline_p05"] is not None
    assert results[0].insights["baseline_comparison"]["comparisons"][0]["percentile_status"] == "below_band"
    assert results[0].insights["baseline_comparison"]["largest_regressions"]
    assert any(issue["id"] == "duplicate_quality_fingerprint" for issue in results[0].insights["dataset_audit"])
    assert any(issue["id"] == "duplicate_quality_fingerprint" for issue in results[1].insights["dataset_audit"])


def test_preset_specific_rules_adjust_defect_labels():
    results = [
        _result(
            "podcast.wav",
            [MetricResult(name="Integrated Loudness (LUFS)", value=-20.0, unit="LUFS", group="loudness")],
        )
    ]

    enrich_results(results, preset="podcast")

    labels = {label["id"]: label for label in results[0].insights["defect_labels"]}
    assert labels["under_loud"]["severity"] == "warn"
    assert labels["under_loud"]["evidence_metrics"][0]["metric"] == "Integrated Loudness (LUFS)"


def test_custom_insight_rules_add_labels_suggestions_and_ci_failures(tmp_path):
    rules_path = tmp_path / "insight-rules.json"
    rules_path.write_text(
        json.dumps(
            {
                "labels": [
                    {
                        "id": "house_snr_floor",
                        "feature": "noise.snr",
                        "min": 20.0,
                        "severity": "fail",
                        "message": "House SNR floor missed",
                        "suggestion": "Reject or denoise before publishing.",
                        "ci_fail": True,
                    }
                ]
            }
        )
    )
    results = [_result("custom.wav", [MetricResult(name="SNR", value=18.0, unit="dB", group="noise")])]

    enrich_results(results, insight_rules_path=rules_path, ci=True)

    labels = {label["id"]: label for label in results[0].insights["defect_labels"]}
    assert labels["house_snr_floor"]["severity"] == "fail"
    assert any("Reject or denoise" in suggestion for suggestion in results[0].insights["repair_suggestions"])
    assert any(check["id"] == "custom_insight_rules_gate" and check["status"] == "fail" for check in results[0].insights["ci_checks"])


def test_enrich_results_adds_segment_heatmap_and_drift_monitor():
    results = [
        _result(
            "call.wav [segment 1/2 0.00-1.00s]",
            [MetricResult(name="SNR", value=26.0, unit="dB", group="noise")],
            segment_index=1,
            source_file="call.wav",
        ),
        _result(
            "call.wav [segment 2/2 1.00-2.00s]",
            [MetricResult(name="SNR", value=9.0, unit="dB", group="noise")],
            segment_index=2,
            source_file="call.wav",
        ),
    ]

    enrich_results(results, drift=True)

    assert results[0].insights["segment_heatmap"]["source_file"] == "call.wav"
    assert results[1].insights["segment_heatmap"]["cells"][1]["status"] == "warn"
    assert results[1].insights["drift_monitor"]["status"] == "drift"


def test_drift_state_persists_across_runs(tmp_path):
    state_path = tmp_path / "drift-state.json"
    first = [_result("a.wav", [MetricResult(name="SNR", value=30.0, unit="dB", group="noise")])]
    second = [_result("b.wav", [MetricResult(name="SNR", value=10.0, unit="dB", group="noise")])]

    enrich_results(first, drift=True, drift_state_path=state_path)
    enrich_results(second, drift=True, drift_state_path=state_path)

    assert second[0].insights["drift_monitor"]["status"] == "drift"
    assert json.loads(state_path.read_text())["last_features"]["noise.snr"] == 10.0


def test_triage_rank_orders_highest_risk_first():
    results = [
        _result("clean.wav", [MetricResult(name="SNR", value=28.0, unit="dB", group="noise")]),
        _result(
            "bad.wav",
            [
                MetricResult(name="SNR", value=7.0, unit="dB", group="noise"),
                MetricResult(name="True Peak", value=0.2, unit="dBTP", group="loudness"),
            ],
        ),
    ]

    enrich_results(results, ci=True)

    by_path = {result.path: result.insights["triage"] for result in results}
    assert by_path["bad.wav"]["rank"] == 1
    assert by_path["bad.wav"]["score"] > by_path["clean.wav"]["score"]


def test_enrich_existing_report_adds_insights_without_audio(tmp_path):
    report_path = tmp_path / "report.json"
    out_path = tmp_path / "enriched.json"
    report_path.write_text(
        json.dumps(
            [
                FileResult(
                    path="stored.wav",
                    metrics=[MetricResult(name="SNR", value=8.0, unit="dB", group="noise")],
                ).to_dict()
            ]
        )
    )

    enrich_existing_report(report_path, out_path, ci=True)

    payload = json.loads(out_path.read_text())
    assert payload[0]["file"] == "stored.wav"
    assert payload[0]["insights"]["ci_checks"][0]["status"] == "fail"


def test_validate_insights_report_rejects_missing_version(tmp_path):
    report_path = tmp_path / "report.json"
    result = FileResult(
        path="stored.wav",
        metrics=[MetricResult(name="SNR", value=8.0, unit="dB", group="noise")],
        insights={"defect_labels": []},
    )
    report_path.write_text(json.dumps([result.to_dict()]))

    issues = validate_insights_report(report_path)

    assert issues
    assert issues[0].path.endswith(".version")


def test_html_report_exposes_interactive_label_filter_and_heatmap_details():
    results = [
        _result(
            "call.wav [segment 1/1]",
            [MetricResult(name="SNR", value=7.0, unit="dB", group="noise")],
            segment_index=1,
            source_file="call.wav",
        )
    ]
    enrich_results(results)

    html = HtmlReporter().render(results)

    assert 'id="labelFilter"' in html
    assert "data-labels=" in html
    assert "data-segment" in html
    assert "location.hash" in html


def test_cli_insights_writes_enriched_json(monkeypatch):
    class InsightAnalyzer:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_all(self, paths, on_progress=None):
            return [
                FileResult(
                    path=str(paths[0]),
                    duration_s=1.0,
                    sample_rate=16_000,
                    channels=1,
                    metrics=[
                        MetricResult(name="SNR", value=7.0, unit="dB", group="noise"),
                    ],
                )
            ]

    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", InsightAnalyzer)

    with runner.isolated_filesystem():
        Path("input.wav").write_bytes(b"audio")
        result = runner.invoke(main, ["input.wav", "--insights", "--output", "report.json", "--silent"])

        assert result.exit_code == 0
        payload = json.loads(Path("report.json").read_text())
        assert payload[0]["insights"]["quality_fingerprint"]["signature"]
        assert payload[0]["insights"]["defect_labels"][0]["id"] == "noisy_floor"


def test_export_flagged_segment_snippets_writes_wav_and_links_heatmap(tmp_path):
    source = tmp_path / "call.wav"
    _write_wav(source, seconds=2.0)
    results = [
        _result(
            str(source) + " [segment 1/2]",
            [MetricResult(name="SNR", value=7.0, unit="dB", group="noise")],
            segment_index=1,
            source_file=str(source),
        )
    ]
    enrich_results(results)

    snippets = export_flagged_segment_snippets(results, tmp_path / "snippets")

    assert len(snippets) == 1
    assert Path(snippets[0]["path"]).exists()
    assert results[0].insights["segment_heatmap"]["cells"][0]["snippet"].endswith(".wav")


def test_compact_summary_payload_contains_only_pipeline_fields():
    results = [_result("bad.wav", [MetricResult(name="SNR", value=7.0, unit="dB", group="noise")])]
    enrich_results(results, ci=True)

    summary = build_insight_summary_payload(results)

    assert summary[0]["file"] == "bad.wav"
    assert summary[0]["fingerprint"]
    assert summary[0]["labels"] == ["noisy_floor"]
    assert "metrics" not in summary[0]


def test_cli_insights_subcommand_enriches_existing_report():
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("report.json").write_text(
            json.dumps(
                [
                    FileResult(
                        path="stored.wav",
                        metrics=[MetricResult(name="SNR", value=8.0, unit="dB", group="noise")],
                    ).to_dict()
                ]
            )
        )
        result = runner.invoke(main, ["insights", "report.json", "--output", "enriched.json", "--silent", "--ci"])

        assert result.exit_code == 0
        payload = json.loads(Path("enriched.json").read_text())
        assert payload[0]["insights"]["quality_fingerprint"]["signature"]


def test_cli_insights_validate_mode_checks_existing_report():
    runner = CliRunner()

    with runner.isolated_filesystem():
        Path("report.json").write_text(json.dumps([FileResult(path="stored.wav", insights={"defect_labels": []}).to_dict()]))
        result = runner.invoke(main, ["insights", "validate", "report.json"])

        assert result.exit_code != 0
        assert "version" in result.output


def test_cli_can_write_compact_insights_summary(monkeypatch):
    class InsightAnalyzer:
        def __init__(self, *args, **kwargs):
            pass

        def analyze_all(self, paths, on_progress=None):
            return [
                FileResult(
                    path=str(paths[0]),
                    duration_s=1.0,
                    sample_rate=16_000,
                    channels=1,
                    metrics=[MetricResult(name="SNR", value=7.0, unit="dB", group="noise")],
                )
            ]

    runner = CliRunner()
    monkeypatch.setattr("qualiax.cli.AudioAnalyzer", InsightAnalyzer)

    with runner.isolated_filesystem():
        Path("input.wav").write_bytes(b"audio")
        result = runner.invoke(
            main,
            [
                "input.wav",
                "--insights",
                "--insights-summary",
                "summary.json",
                "--output",
                "report.json",
                "--silent",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(Path("summary.json").read_text())
        assert payload[0]["labels"] == ["noisy_floor"]


def test_golden_noisy_clipped_fixture_keeps_labels_stable(tmp_path):
    source = Path(__file__).parent / "fixtures" / "insights" / "noisy_clipped_report.json"
    out_path = tmp_path / "fixture-enriched.json"

    enrich_existing_report(source, out_path, ci=True)

    payload = json.loads(out_path.read_text())
    labels = {label["id"] for label in payload[0]["insights"]["defect_labels"]}
    assert labels == {"clipping_risk", "noisy_floor"}
    assert payload[0]["insights"]["ci_checks"][0]["status"] == "fail"


def test_calibration_fixture_set_covers_core_cases():
    fixture_dir = Path(__file__).parent / "fixtures" / "insights"
    fixture_names = {path.name for path in fixture_dir.glob("*.json")}

    assert {
        "clean_report.json",
        "noisy_clipped_report.json",
        "quiet_low_mos_report.json",
        "duplicate_a_report.json",
        "duplicate_b_report.json",
        "drift_baseline_report.json",
    } <= fixture_names

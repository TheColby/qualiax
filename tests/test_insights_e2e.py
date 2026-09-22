"""End-to-end checks of ``--insights`` on a small generated defect corpus.

Every fixture is synthesized into a temp dir at test time from ``sample.wav``
(real 16 kHz speech) or a deterministic speech-like signal, then run through
the real CLI and metric stack via CliRunner. Only numpy/scipy/click are needed.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, lfilter

from qualiax.cli import main
from qualiax.migrations import ExitCode
from qualiax.validation import validate_report_file

SR = 16_000
SAMPLE_WAV = Path(__file__).resolve().parents[1] / "sample.wav"
SEGMENT_S = 3.0

pytestmark = pytest.mark.skipif(not SAMPLE_WAV.exists(), reason="sample.wav fixture not available")


# ── corpus ───────────────────────────────────────────────────────────────────

def _write(path: Path, audio: np.ndarray) -> Path:
    pcm = np.clip(np.round(audio * 32768.0), -32768, 32767).astype(np.int16)
    wavfile.write(path, SR, pcm)
    return path


def _speechlike(seconds: float, seed: int = 7) -> np.ndarray:
    """Harmonic source with formant-ish bands, a ~4 Hz syllable envelope and pauses."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    f0 = 140 + 30 * np.sin(2 * np.pi * 0.7 * t) + 15 * np.sin(2 * np.pi * 2.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    voiced = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 25))
    shaped = np.zeros(n)
    for lo, hi, gain in ((300, 900, 1.0), (1000, 2200, 0.6), (2400, 3400, 0.3)):
        b, a = butter(2, [lo / (SR / 2), hi / (SR / 2)], btype="band")
        shaped += gain * lfilter(b, a, voiced)
    envelope = 0.5 * (1 + np.sin(2 * np.pi * 4.0 * t - np.pi / 2)) ** 1.5
    pauses = uniform_filter1d((np.sin(2 * np.pi * t / 1.5) > -0.8).astype(float), int(0.03 * SR))
    x = shaped * envelope * pauses
    x = x / np.max(np.abs(x)) * 0.5
    return x + rng.normal(0, 10 ** (-60 / 20), n)  # -60 dBFS room floor


@pytest.fixture(scope="module")
def corpus(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("insight-corpus")
    sr, raw = wavfile.read(SAMPLE_WAV)
    assert sr == SR
    clean = raw.astype(np.float64) / 32768.0
    rng = np.random.default_rng(1234)
    files = {"clean": _write(root / "clean.wav", clean)}
    files["clipped"] = _write(root / "clipped.wav", np.clip(clean * 6.0, -1.0, 1.0))
    noisy = clean + rng.normal(0, np.sqrt(np.mean(clean**2)), clean.shape)  # ~0 dB SNR
    files["noisy"] = _write(root / "noisy.wav", noisy / np.max(np.abs(noisy)) * 0.6)
    files["quiet"] = _write(root / "quiet.wav", clean * 10 ** (-23 / 20))  # ~-40 LUFS
    dropouts = clean.copy()
    for start_s in (0.5, 1.1, 1.7):
        start = int(start_s * SR)
        dropouts[start:start + int(0.2 * SR)] = 0.0
    files["dropouts"] = _write(root / "dropouts.wav", dropouts)
    files["silent"] = _write(root / "silent.wav", np.zeros(2 * SR))

    long_clean = _speechlike(12.0)
    files["long_clean"] = _write(root / "long_clean.wav", long_clean)
    local = long_clean.copy()
    seg3 = slice(int(6.0 * SR), int(9.0 * SR))  # segment 3 of 4 at 3 s segments
    local[seg3] = np.clip(local[seg3] * 8.0, -1.0, 1.0)
    files["local_clip"] = _write(root / "local_clip.wav", local)
    tail = long_clean.copy()
    tail[int(9.0 * SR):] = np.clip(tail[int(9.0 * SR):] * 8.0, -1.0, 1.0)  # last segment only
    files["tail_clip"] = _write(root / "tail_clip.wav", tail)
    return files


def _invoke(*args) -> "object":
    return CliRunner().invoke(main, [str(arg) for arg in args])


def _report(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _labels(item: dict) -> dict[str, dict]:
    return {label["id"]: label for label in item["insights"]["defect_labels"]}


@pytest.fixture(scope="module")
def reports(corpus, tmp_path_factory) -> dict[str, list[dict]]:
    """One ``--insights --ci`` run per short variant; exit codes kept alongside."""
    out_dir = tmp_path_factory.mktemp("insight-reports")
    runs = {}
    for name in ("clean", "clipped", "noisy", "quiet", "dropouts", "silent"):
        out = out_dir / f"{name}.json"
        result = _invoke(corpus[name], "--insights", "--ci", "--output", out, "--silent")
        assert result.exception is None or isinstance(result.exception, SystemExit), result.output
        runs[name] = {"exit": result.exit_code, "report": _report(out), "path": out}
    return runs


# ── labels, suggestions, CI gate ─────────────────────────────────────────────

SUGGESTION_KEYWORDS = {
    "noisy_floor": "denoise",
    "clipping_risk": "true-peak",
    "hard_clipping": "declipping",
    "under_loud": "normalize",
    "over_loud": "reduce program loudness",
    "low_perceptual_quality": "codec",
    "silence_heavy": "silence",
    "dropouts": "conceal the gaps",
}


@pytest.mark.parametrize(
    ("name", "required", "forbidden"),
    [
        ("clean", set(), set(SUGGESTION_KEYWORDS)),
        ("clipped", {"clipping_risk", "hard_clipping"}, {"noisy_floor", "under_loud", "silence_heavy"}),
        ("noisy", {"noisy_floor"}, {"hard_clipping", "under_loud", "over_loud", "silence_heavy"}),
        ("quiet", {"under_loud"}, {"noisy_floor", "clipping_risk", "hard_clipping", "low_perceptual_quality"}),
        ("silent", {"silence_heavy"}, {"low_perceptual_quality", "noisy_floor", "hard_clipping"}),
        ("dropouts", {"dropouts"}, {"hard_clipping", "clipping_risk", "noisy_floor", "over_loud"}),
    ],
)
def test_defect_labels_match_injected_defect(reports, name, required, forbidden):
    item = reports[name]["report"][0]
    labels = _labels(item)
    assert required <= set(labels), labels
    assert not (forbidden & set(labels)), labels
    for label in labels.values():
        assert label["severity"] in {"info", "warn", "fail"}
        assert 0.0 < label["confidence"] <= 1.0
        assert label["evidence"]


@pytest.mark.parametrize("name", ["clean", "clipped", "noisy", "quiet", "silent", "dropouts"])
def test_repair_suggestions_follow_labels(reports, name):
    insights = reports[name]["report"][0]["insights"]
    suggestions = [s.lower() for s in insights["repair_suggestions"]]
    label_ids = [label["id"] for label in insights["defect_labels"]]
    # One suggestion per distinct built-in label, each on topic, and nothing extra.
    assert len(suggestions) == len(set(label_ids))
    for label_id in label_ids:
        assert any(SUGGESTION_KEYWORDS[label_id] in s for s in suggestions), (label_id, suggestions)


def test_clipped_file_labels_carry_metric_evidence(reports):
    labels = _labels(reports["clipped"]["report"][0])
    hard = labels["hard_clipping"]
    assert hard["severity"] == "fail"
    assert hard["evidence_metrics"][0]["metric"] == "Near-Clipped Samples"
    assert hard["evidence_metrics"][0]["value"] > 1000
    assert labels["clipping_risk"]["evidence_metrics"][0]["metric"] == "True Peak"


def test_mos_label_uses_dnsmos_rather_than_p563_floor(reports):
    fingerprint = reports["clean"]["report"][0]["insights"]["quality_fingerprint"]
    assert fingerprint["version"] == 2
    assert {"perceptual.dnsmos_ovrl", "speech.hnr", "noise.clipping_ratio"} <= set(fingerprint["features"])
    noisy = _labels(reports["noisy"]["report"][0])["low_perceptual_quality"]
    assert noisy["evidence_metrics"][0]["metric"].startswith("DNSMOS P.835 OVRL")


def test_zeroed_gaps_get_a_dropout_label(reports):
    labels = _labels(reports["dropouts"]["report"][0])
    assert labels["dropouts"]["severity"] == "warn"
    assert labels["dropouts"]["evidence_metrics"][0]["metric"] == "Detected Dropouts"
    assert "dropouts" not in _labels(reports["clean"]["report"][0])


@pytest.mark.parametrize(
    ("name", "expected_exit", "gate"),
    [
        ("clean", ExitCode.OK, "pass"),
        ("quiet", ExitCode.OK, "pass"),  # under_loud is a warning, not a blocking label
        ("silent", ExitCode.OK, "pass"),
        ("clipped", ExitCode.QUALITY_GATE_FAILED, "fail"),
        ("noisy", ExitCode.QUALITY_GATE_FAILED, "fail"),
    ],
)
def test_ci_gate_exit_codes(reports, name, expected_exit, gate):
    run = reports[name]
    assert run["exit"] == expected_exit
    checks = {check["id"]: check for check in run["report"][0]["insights"]["ci_checks"]}
    assert checks["audio_quality_gate"]["status"] == gate


def test_without_ci_flag_degraded_audio_still_exits_zero(corpus, tmp_path):
    result = _invoke(corpus["clipped"], "--insights", "--output", tmp_path / "r.json", "--silent")
    assert result.exit_code == ExitCode.OK
    assert _report(tmp_path / "r.json")[0]["insights"]["ci_checks"] == []


def test_ci_gate_and_threshold_rules_share_the_quality_gate_exit_code(corpus, tmp_path):
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"rules": [{"metric": "Integrated Loudness (LUFS)", "min": -30}]}))
    rule_run = _invoke(corpus["quiet"], "--rules", rules, "--output", tmp_path / "a.json", "--silent")
    gate_run = _invoke(corpus["clipped"], "--insights", "--ci", "--output", tmp_path / "b.json", "--silent")
    # README: rule violations and failing --ci gates both return exit code 2.
    assert rule_run.exit_code == gate_run.exit_code == ExitCode.QUALITY_GATE_FAILED


@pytest.mark.parametrize(
    "flags",
    [
        ["--ci"],
        ["--baseline", "{clean_report}"],
        ["--drift"],
        ["--fingerprint-sensitivity", "strict"],
        ["--insights-summary", "{tmp}/s.json"],
        ["--insight-snippets", "{tmp}/snips"],
    ],
)
def test_insight_flags_without_insights_are_rejected(corpus, reports, tmp_path, flags):
    flags = [f.format(clean_report=reports["clean"]["path"], tmp=tmp_path) for f in flags]
    result = _invoke(corpus["clipped"], *flags, "--output", tmp_path / "r.json", "--silent")
    assert result.exit_code != ExitCode.OK
    assert "requires --insights" in result.output
    assert not (tmp_path / "r.json").exists()


def test_drift_state_without_drift_is_rejected(corpus, tmp_path):
    result = _invoke(corpus["clean"], "--insights", "--drift-state", tmp_path / "d.json", "--output", tmp_path / "r.json")
    assert result.exit_code != ExitCode.OK
    assert "--drift-state requires --drift" in result.output


def test_insights_subcommand_honors_ci_exit_code(corpus, tmp_path):
    for name, expected in (("clipped", ExitCode.QUALITY_GATE_FAILED), ("clean", ExitCode.OK)):
        plain = tmp_path / f"{name}-plain.json"
        assert _invoke(corpus[name], "--output", plain, "--silent").exit_code == ExitCode.OK
        enriched = tmp_path / f"{name}-enriched.json"
        result = _invoke("insights", plain, "--ci", "--output", enriched, "--silent")
        assert result.exit_code == expected, result.output
        assert _report(enriched)[0]["insights"]["ci_checks"][0]["id"] == "audio_quality_gate"


# ── baseline and drift ───────────────────────────────────────────────────────

def test_baseline_from_clean_report_flags_degraded_runs(corpus, reports, tmp_path):
    baseline = reports["clean"]["path"]
    expectations = {
        "clean": (ExitCode.OK, "ok", None),
        "clipped": (ExitCode.QUALITY_GATE_FAILED, "regressed", "loudness.true_peak"),
        "noisy": (ExitCode.QUALITY_GATE_FAILED, "regressed", "noise.snr"),
        "quiet": (ExitCode.QUALITY_GATE_FAILED, "regressed", "loudness.integrated_lufs"),
    }
    for name, (exit_code, status, feature) in expectations.items():
        out = tmp_path / f"{name}.json"
        result = _invoke(corpus[name], "--insights", "--ci", "--baseline", baseline, "--output", out, "--silent")
        assert result.exit_code == exit_code, (name, result.output)
        comparison = _report(out)[0]["insights"]["baseline_comparison"]
        assert comparison["status"] == status
        assert comparison["baseline"]["name"] == baseline.name
        gate = {c["id"]: c["status"] for c in _report(out)[0]["insights"]["ci_checks"]}
        assert gate["baseline_regression_gate"] == ("pass" if status == "ok" else "fail")
        if feature:
            assert feature in {item["feature"] for item in comparison["largest_regressions"]}


def test_baseline_that_is_not_a_report_fails_before_analysis(corpus, tmp_path):
    scorecard = tmp_path / "scorecard.json"
    assert _invoke(corpus["clean"], "--scorecard", scorecard, "--output", tmp_path / "x.json", "--silent").exit_code == 0
    broken = tmp_path / "broken.json"
    broken.write_text('[{"file": ')
    for baseline, message in ((scorecard, "not a JSON dict"), (broken, "not valid JSON")):
        out = tmp_path / "r.json"
        result = _invoke(corpus["clipped"], "--insights", "--ci", "--baseline", baseline, "--output", out, "--silent")
        assert result.exit_code == ExitCode.INVALID_INPUT
        assert isinstance(result.exception, SystemExit)  # clean ClickException, no traceback
        assert message in result.output
        assert not out.exists()


def test_drift_state_persists_across_invocations(corpus, tmp_path):
    state = tmp_path / "drift.json"
    steps = [("clean", "baseline"), ("quiet", "drift"), ("quiet", "stable")]
    for index, (name, status) in enumerate(steps):
        out = tmp_path / f"run{index}.json"
        result = _invoke(corpus[name], "--insights", "--drift", "--drift-state", state, "--output", out, "--silent")
        assert result.exit_code == 0, result.output
        monitor = _report(out)[0]["insights"]["drift_monitor"]
        assert monitor["status"] == status
        saved = json.loads(state.read_text())
        assert saved["last_features"] == _report(out)[0]["insights"]["quality_fingerprint"]["features"]
        if status == "drift":
            change = {c["feature"]: c for c in monitor["changes"]}["loudness.integrated_lufs"]
            assert change["delta"] == pytest.approx(-23.0, abs=0.5)


# ── segments and snippets ────────────────────────────────────────────────────

def _read_wav(path: Path) -> tuple[np.ndarray, int, int]:
    with wave.open(str(path), "rb") as wf:
        assert wf.getsampwidth() == 2
        frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2")
        return frames, wf.getframerate(), wf.getnchannels()


def test_localized_defect_is_attributed_to_its_segment_with_snippet(corpus, tmp_path):
    out = tmp_path / "segments.json"
    snippets = tmp_path / "snips"
    result = _invoke(
        corpus["local_clip"], "--insights", "--ci", "--segment-seconds", SEGMENT_S,
        "--insight-snippets", snippets, "--output", out, "--silent",
    )
    assert result.exit_code == ExitCode.QUALITY_GATE_FAILED
    report = _report(out)
    flagged = {item["segment_index"]: set(_labels(item)) for item in report if _labels(item)}
    assert list(flagged) == [3]
    assert {"hard_clipping", "clipping_risk"} <= flagged[3]
    # Segments of one recording are not "duplicates" of each other.
    assert all(item["insights"]["dataset_audit"] == [] for item in report)

    cells = report[0]["insights"]["segment_heatmap"]["cells"]
    assert [(c["segment_index"], c["status"]) for c in cells] == [(1, "ok"), (2, "ok"), (3, "warn"), (4, "ok")]
    [snippet] = report[2]["insights"]["snippets"]
    assert (snippet["start_s"], snippet["end_s"]) == (5.75, 9.25)  # 6-9 s plus 0.25 s padding
    assert cells[2]["snippet"] == snippet["path"]
    assert sorted(p.name for p in snippets.iterdir()) == [Path(snippet["path"]).name]

    audio, sr, channels = _read_wav(Path(snippet["path"]))
    assert (sr, channels, len(audio)) == (SR, 1, int(3.5 * SR))
    _, source = wavfile.read(corpus["local_clip"])
    assert np.array_equal(audio, source[int(5.75 * SR):int(9.25 * SR)])


def test_snippets_are_not_overwritten_without_force(corpus, tmp_path):
    snippets = tmp_path / "snips"
    base = [corpus["local_clip"], "--insights", "--segment-seconds", SEGMENT_S, "--insight-snippets", snippets, "--silent"]
    assert _invoke(*base, "--output", tmp_path / "a.json").exit_code == 0
    [snippet] = list(snippets.iterdir())
    snippet.write_bytes(b"sentinel")

    refused = _invoke(*base, "--output", tmp_path / "b.json")
    assert refused.exit_code == ExitCode.INVALID_INPUT
    assert "Refusing to overwrite existing snippet" in refused.output
    assert snippet.read_bytes() == b"sentinel"

    forced = _invoke(*base, "--output", tmp_path / "c.json", "--force")
    assert forced.exit_code == 0
    assert _read_wav(snippet)[1] == SR


def test_snippet_for_last_segment_is_clamped_to_file_end(corpus, tmp_path):
    out = tmp_path / "tail.json"
    result = _invoke(
        corpus["tail_clip"], "--insights", "--segment-seconds", SEGMENT_S,
        "--insight-snippets", tmp_path / "snips", "--output", out, "--silent",
    )
    assert result.exit_code == 0
    [snippet] = [s for item in _report(out) for s in item["insights"]["snippets"]]
    assert snippet["segment_index"] == 4
    assert (snippet["start_s"], snippet["end_s"]) == (8.75, 12.0)
    assert len(_read_wav(Path(snippet["path"]))[0]) == int(3.25 * SR)


def test_watch_mode_rewrites_its_own_snippets_between_batches(corpus, tmp_path, monkeypatch):
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    first = incoming / "first.wav"
    second = incoming / "second.wav"
    first.write_bytes(corpus["local_clip"].read_bytes())
    second.write_bytes(corpus["tail_clip"].read_bytes())

    def fake_watch_batches(watch_dirs, seen, watch_interval, **kwargs):
        yield [first]
        yield [second]

    monkeypatch.setattr("qualiax.cli._iter_watch_batches", fake_watch_batches)
    out = tmp_path / "watch.json"
    snippets = tmp_path / "snips"
    result = _invoke(
        incoming, "--watch", "--watch-limit", "2", "--insights", "--ci", "--segment-seconds", SEGMENT_S,
        "--insight-snippets", snippets, "--output", out, "--silent",
    )
    assert result.exit_code == ExitCode.QUALITY_GATE_FAILED, result.output
    assert sorted(p.name for p in snippets.iterdir()) == ["first-segment-3.wav", "second-segment-4.wav"]
    linked = [s["path"] for item in _report(out) for s in item["insights"]["snippets"]]
    assert sorted(Path(p).name for p in linked) == ["first-segment-3.wav", "second-segment-4.wav"]


# ── summary, contract, fingerprint sensitivity ───────────────────────────────

def test_insights_summary_is_compact_and_matches_full_report(corpus, tmp_path):
    out = tmp_path / "full.json"
    summary = tmp_path / "summary.json"
    result = _invoke(
        corpus["clipped"], corpus["clean"], "--insights", "--ci", "--insights-summary", summary,
        "--output", out, "--silent",
    )
    assert result.exit_code == ExitCode.QUALITY_GATE_FAILED
    rows = json.loads(summary.read_text())
    full = _report(out)
    assert [set(row) for row in rows] == [{"file", "source_file", "fingerprint", "labels", "ci_checks", "triage"}] * 2
    for row, item in zip(rows, full):
        assert row["labels"] == [label["id"] for label in item["insights"]["defect_labels"]]
        assert row["fingerprint"] == item["insights"]["quality_fingerprint"]["signature"]
    assert rows[0]["triage"]["rank"] == 1 and rows[1]["triage"]["rank"] == 2
    assert summary.stat().st_size * 10 < out.stat().st_size


def test_validate_output_enforces_insight_contract(corpus, tmp_path):
    out = tmp_path / "full.json"
    result = _invoke(
        corpus["local_clip"], corpus["clipped"], "--insights", "--ci", "--segment-seconds", SEGMENT_S,
        "--validate-output", "--output", out, "--silent",
    )
    assert result.exit_code == ExitCode.QUALITY_GATE_FAILED, result.output
    assert validate_report_file(out) == []
    assert _invoke("insights", "validate", out).exit_code == ExitCode.OK

    payload = _report(out)
    del payload[0]["insights"]["triage"]["rank"]
    payload[-1]["insights"]["defect_labels"][0]["severity"] = "catastrophic"
    tampered = tmp_path / "tampered.json"
    tampered.write_text(json.dumps(payload))
    issue_paths = {issue.path for issue in validate_report_file(tampered)}
    assert "$[0].insights.triage.rank" in issue_paths
    assert f"$[{len(payload) - 1}].insights.defect_labels[0].severity" in issue_paths
    validate = _invoke("insights", "validate", tampered)
    assert validate.exit_code == ExitCode.CONTRACT_VIOLATION
    assert "severity" in validate.output


def test_fingerprint_sensitivity_changes_bucketing(tmp_path):
    sweep = tmp_path / "sweep"
    sweep.mkdir()
    _, raw = wavfile.read(SAMPLE_WAV)
    for index, gain_db in enumerate((0.0, -0.4, -0.8, -1.2, -1.6, -2.0)):
        _write(sweep / f"g{index}.wav", raw / 32768.0 * 10 ** (gain_db / 20))
    distinct = {}
    duplicates = {}
    for level in ("coarse", "balanced", "strict"):
        out = tmp_path / f"{level}.json"
        assert _invoke(sweep, "--insights", "--fingerprint-sensitivity", level, "--output", out, "--silent").exit_code == 0
        report = _report(out)
        assert {item["insights"]["quality_fingerprint"]["sensitivity"] for item in report} == {level}
        distinct[level] = len({item["insights"]["quality_fingerprint"]["signature"] for item in report})
        duplicates[level] = sum(bool(item["insights"]["dataset_audit"]) for item in report)
    assert distinct["coarse"] <= distinct["balanced"] <= distinct["strict"]
    assert distinct["coarse"] < distinct["strict"]
    assert duplicates["coarse"] > duplicates["strict"]


# ── custom rules ─────────────────────────────────────────────────────────────

RULE = {
    "id": "team_loudness_floor",
    "feature": "loudness.integrated_lufs",
    "min": -30,
    "severity": "fail",
    "message": "below team loudness floor",
    "suggestion": "Apply the team gain preset.",
    "ci_fail": True,
}
RULE_TOML = """
[[labels]]
id = "team_loudness_floor"
feature = "loudness.integrated_lufs"
min = -30
severity = "fail"
message = "below team loudness floor"
suggestion = "Apply the team gain preset."
ci_fail = true
"""


@pytest.mark.parametrize("fmt", ["json", "toml"])
def test_custom_rules_are_honored(corpus, tmp_path, fmt):
    rules = tmp_path / f"rules.{fmt}"
    rules.write_text(json.dumps({"labels": [RULE]}) if fmt == "json" else RULE_TOML)
    quiet = _invoke(corpus["quiet"], "--insights", "--ci", "--insight-rules", rules, "--output", tmp_path / "q.json", "--silent")
    clean = _invoke(corpus["clean"], "--insights", "--ci", "--insight-rules", rules, "--output", tmp_path / "c.json", "--silent")

    assert quiet.exit_code == ExitCode.QUALITY_GATE_FAILED
    insights = _report(tmp_path / "q.json")[0]["insights"]
    label = _labels({"insights": insights})["team_loudness_floor"]
    assert (label["severity"], label["evidence"]) == ("fail", "below team loudness floor")
    assert label["evidence_metrics"][0]["metric"] == "Integrated Loudness (LUFS)"
    assert "Apply the team gain preset." in insights["repair_suggestions"]
    gates = {c["id"]: c["status"] for c in insights["ci_checks"]}
    assert gates["custom_insight_rules_gate"] == "fail"

    assert clean.exit_code == ExitCode.OK
    assert "team_loudness_floor" not in _labels(_report(tmp_path / "c.json")[0])


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("bad.json", '{"labels": [', "Invalid JSON in insight rules"),
        ("bad.toml", "[[labels]\nid = ", "Invalid TOML in insight rules"),
        ("list.json", "[1, 2]", "expected an object/table"),
        ("shape.json", '{"labels": {"id": "x"}}', "'labels' must be a list"),
        ("typo.json", '{"labels": [{"feature": "loudness.lufs", "min": -30}]}', "not a known insight feature"),
        ("nobound.json", '{"labels": [{"feature": "noise.snr"}]}', "needs a numeric 'min' and/or 'max'"),
        ("severity.json", '{"labels": [{"feature": "noise.snr", "min": 10, "severity": "high"}]}', "severity must be one of"),
        ("conf.json", '{"labels": [{"feature": "noise.snr", "min": 10, "confidence": "high"}]}', "confidence must be a number"),
        ("toplevel.json", '{"label": []}', "unknown top-level key(s) label"),
        ("rules.yaml", "labels: []", "must be a .json or .toml file"),
    ],
)
def test_malformed_rules_fail_cleanly_before_analysis(corpus, tmp_path, filename, content, message):
    rules = tmp_path / filename
    rules.write_text(content)
    out = tmp_path / "r.json"
    result = _invoke(corpus["clean"], "--insights", "--insight-rules", rules, "--output", out, "--silent")
    assert result.exit_code == ExitCode.INVALID_INPUT
    assert isinstance(result.exception, SystemExit), result.exception  # no traceback
    assert "Traceback" not in result.output
    assert message in result.output
    assert not out.exists()

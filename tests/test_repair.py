import hashlib
import os
import shutil
import subprocess
import wave

import numpy as np
import pytest

import qualiax
from qualiax.models import FileResult, MetricResult
from qualiax.repair import RepairPlan, apply_repair_plan, build_repair_plan, evaluate_repair

requires_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


def _labeled(path, labels, *, sample_rate=16000):
    return FileResult(
        path=str(path),
        sample_rate=sample_rate,
        insights={"defect_labels": [{"id": label} for label in labels]},
    )


def _no_subprocess(*args, **kwargs):
    raise AssertionError("ffmpeg must not run")


def test_plan_orders_filters_by_defect(tmp_path):
    result = _labeled(tmp_path / "in.wav", ["over_loud", "noisy_floor", "hard_clipping", "clipping_risk"], sample_rate=22050)

    plan = build_repair_plan(result, output_path=tmp_path / "out.wav", loudness_target_lufs=-14, true_peak_dbtp=-1.0)

    names = [item.split("=")[0] for item in plan.filters]
    assert names == ["adeclip", "afftdn", "alimiter", "loudnorm", "aresample"]
    assert plan.filters[2] == "alimiter=limit=0.891:level=disabled"
    assert plan.filters[3] == "loudnorm=I=-14:TP=-1:LRA=11"
    assert plan.filters[4] == "aresample=22050"
    assert len(plan.reasons) == len(plan.filters)


def test_plan_without_defects_is_a_review_copy(tmp_path):
    plan = build_repair_plan(_labeled(tmp_path / "in.wav", []), output_path=tmp_path / "out.wav")
    assert plan.filters == ("anull",)


def test_limiter_never_uses_auto_level(tmp_path):
    # alimiter's default level=enabled renormalizes peaks back to 0 dBFS.
    plan = build_repair_plan(_labeled(tmp_path / "in.wav", ["clipping_risk"]), output_path=tmp_path / "out.wav")
    assert plan.filters == ("alimiter=limit=0.841:level=disabled",)


def test_ffmpeg_args_refuse_to_overwrite_outputs_by_default(tmp_path):
    result = _labeled(tmp_path / "in.wav", ["noisy_floor"])
    safe = build_repair_plan(result, output_path=tmp_path / "out.wav")
    forced = build_repair_plan(result, output_path=tmp_path / "out.wav", overwrite_output=True)

    assert safe.ffmpeg_args[:4] == ["ffmpeg", "-hide_banner", "-nostdin", "-n"]
    assert forced.ffmpeg_args[3] == "-y"
    assert safe.ffmpeg_args[-1] == str(tmp_path / "out.wav")


def test_leading_dash_paths_are_not_parsed_as_ffmpeg_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    plan = build_repair_plan(_labeled("-in.wav", ["noisy_floor"]), output_path="-out.wav")
    assert plan.ffmpeg_args[5] == os.path.join(".", "-in.wav")
    assert plan.ffmpeg_args[-1] == os.path.join(".", "-out.wav")


def test_source_overwrite_is_blocked_everywhere(tmp_path):
    source = tmp_path / "in.wav"
    source.write_bytes(b"audio")
    with pytest.raises(ValueError, match="overwrite the source"):
        build_repair_plan(_labeled(source, ["noisy_floor"]), output_path=tmp_path / "sub" / ".." / "in.wav")
    # Constructing the dataclass directly used to bypass the check.
    with pytest.raises(ValueError, match="overwrite the source"):
        RepairPlan(source=source, output=source, filters=("anull",), reasons=("x",))
    hardlink = tmp_path / "alias.wav"
    os.link(source, hardlink)
    with pytest.raises(ValueError, match="overwrite the source"):
        build_repair_plan(_labeled(source, ["noisy_floor"]), output_path=hardlink)
    symlink = tmp_path / "link.wav"
    symlink.symlink_to(source)
    with pytest.raises(ValueError, match="overwrite the source"):
        build_repair_plan(_labeled(source, ["noisy_floor"]), output_path=symlink)


def test_segment_results_repair_the_source_file(tmp_path):
    segment = FileResult(
        path="long.wav [segment 1/2]",
        source_file=str(tmp_path / "long.wav"),
        insights={"defect_labels": [{"id": "noisy_floor"}]},
    )
    plan = build_repair_plan(segment, output_path=tmp_path / "fixed.wav")
    assert plan.source == tmp_path / "long.wav"


def test_apply_is_dry_run_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr("qualiax.repair.subprocess.run", _no_subprocess)
    plan = build_repair_plan(_labeled(tmp_path / "in.wav", ["noisy_floor"]), output_path=tmp_path / "out.wav")

    applied = apply_repair_plan(plan)

    assert applied == {"executed": False, "command": plan.ffmpeg_args, "output": str(tmp_path / "out.wav")}
    assert not (tmp_path / "out.wav").exists()


def test_apply_refuses_existing_output_missing_source_and_missing_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr("qualiax.repair.subprocess.run", _no_subprocess)
    source = tmp_path / "in.wav"
    output = tmp_path / "out.wav"
    plan = build_repair_plan(_labeled(source, ["noisy_floor"]), output_path=output)

    with pytest.raises(FileNotFoundError):
        apply_repair_plan(plan, dry_run=False)
    source.write_bytes(b"audio")
    output.write_bytes(b"keep me")
    with pytest.raises(FileExistsError):
        apply_repair_plan(plan, dry_run=False)
    assert output.read_bytes() == b"keep me"
    output.unlink()
    monkeypatch.setattr("qualiax.repair.shutil.which", lambda name: None)
    with pytest.raises(RuntimeError, match="ffmpeg is required"):
        apply_repair_plan(plan, dry_run=False)


def test_apply_surfaces_ffmpeg_failures(tmp_path, monkeypatch):
    source = tmp_path / "in.wav"
    source.write_bytes(b"audio")
    plan = build_repair_plan(_labeled(source, ["noisy_floor"]), output_path=tmp_path / "out.wav")
    monkeypatch.setattr("qualiax.repair.shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(
        "qualiax.repair.subprocess.run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1, stdout="", stderr="Invalid data found"),
    )
    with pytest.raises(RuntimeError, match="Invalid data found"):
        apply_repair_plan(plan, dry_run=False)


def _metrics(**values):
    directions = {"snr": True, "true_peak": False, "centroid": None, "lufs": None}
    names = {
        "snr": "Estimated SNR",
        "true_peak": "True Peak",
        "centroid": "Spectral Centroid",
        "lufs": "Integrated Loudness (LUFS)",
    }
    return FileResult(
        path="x.wav",
        metrics=[
            MetricResult(name=names[key], value=value, higher_is_better=directions[key])
            for key, value in values.items()
        ],
    )


def test_evaluate_repair_uses_declared_directions_and_ignores_neutral_metrics():
    before = _metrics(snr=12.0, true_peak=0.3, centroid=1800.0)
    after = _metrics(snr=20.0, true_peak=-1.2, centroid=1650.0)

    report = evaluate_repair(before, after)

    assert [item["metric"] for item in report["improvements"]] == ["Estimated SNR", "True Peak"]
    assert report["regressions"] == []
    assert [item["metric"] for item in report["neutral"]] == ["Spectral Centroid"]
    assert report["accepted"] is True


def test_evaluate_repair_flags_real_regressions_and_missing_metrics():
    before = _metrics(snr=20.0, true_peak=-3.0, lufs=-20.0)
    after = FileResult(
        path="x.wav",
        metrics=[
            MetricResult(name="Estimated SNR", value=14.0, higher_is_better=True),
            MetricResult(name="True Peak", value=-3.0, higher_is_better=False),
        ],
    )

    report = evaluate_repair(before, after)

    assert [item["metric"] for item in report["regressions"]] == ["Estimated SNR"]
    assert report["regressions"][0]["delta"] == -6.0
    assert report["missing_after"] == ["Integrated Loudness (LUFS)"]
    assert report["accepted"] is False


def test_evaluate_repair_judges_targeted_metrics_by_distance():
    over_loud_fixed = evaluate_repair(
        _metrics(lufs=-8.0), _metrics(lufs=-15.5), targets={"Integrated Loudness (LUFS)": -16.0}
    )
    overshoot = evaluate_repair(
        _metrics(lufs=-17.0), _metrics(lufs=-25.0), targets={"Integrated Loudness (LUFS)": -16.0}
    )

    assert over_loud_fixed["improvements"][0]["target"] == -16.0
    assert over_loud_fixed["accepted"] is True
    assert overshoot["regressions"][0]["metric"] == "Integrated Loudness (LUFS)"


def test_evaluate_repair_tolerance_and_name_fallback():
    before = FileResult(path="x", metrics=[MetricResult(name="Jitter (Local)", value=1.0), MetricResult(name="Estimated SNR", value=10.0)])
    after = FileResult(path="x", metrics=[MetricResult(name="Jitter (Local)", value=0.5), MetricResult(name="Estimated SNR", value=10.0000001)])

    report = evaluate_repair(before, after)

    assert [item["metric"] for item in report["improvements"]] == ["Jitter (Local)"]
    assert report["regressions"] == []
    assert report["neutral"] == []


def _write_hot_noisy_wav(path, sample_rate=16000):
    rng = np.random.default_rng(7)
    t = np.arange(int(sample_rate * 2.0)) / sample_rate
    signal = 1.3 * np.sin(2 * np.pi * 220 * t) + 0.08 * rng.standard_normal(t.size)
    pcm = (np.clip(signal, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@requires_ffmpeg
def test_end_to_end_repair_with_ffmpeg(tmp_path):
    source = tmp_path / "hot.wav"
    _write_hot_noisy_wav(source)
    source_digest = _sha256(source)
    before = qualiax.analyze(source, metrics="basic,loudness,noise", strict=False)[0]
    before.insights = {"defect_labels": [{"id": "hard_clipping"}, {"id": "clipping_risk"}, {"id": "over_loud"}]}
    output = tmp_path / "repaired" / "hot.wav"

    plan = build_repair_plan(before, output_path=output, loudness_target_lufs=-16.0, true_peak_dbtp=-1.5)
    dry = apply_repair_plan(plan)
    assert dry["executed"] is False and not output.exists()

    applied = apply_repair_plan(plan, dry_run=False, timeout=120)
    after = qualiax.analyze(output, metrics="basic,loudness,noise", strict=False)[0]
    report = evaluate_repair(before, after, targets={"Integrated Loudness (LUFS)": -16.0})

    assert applied["executed"] is True
    assert _sha256(source) == source_digest  # input untouched
    with wave.open(str(output)) as handle:
        assert handle.getframerate() == 16000  # loudnorm's 192 kHz output was resampled back
    true_peak = {item["metric"]: item for item in report["improvements"]}.get("True Peak")
    assert true_peak is not None and true_peak["after"] < true_peak["before"]
    assert "Integrated Loudness (LUFS)" in {item["metric"] for item in report["improvements"]}
    with pytest.raises(FileExistsError):
        apply_repair_plan(plan, dry_run=False)

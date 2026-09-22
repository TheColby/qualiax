"""Official DNSMOS path, model assets, and the `qualiax models` command."""
from __future__ import annotations

import hashlib
import threading
import warnings
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from click.testing import CliRunner

from qualiax import assets
from qualiax import metrics as M
from qualiax.cli import main
from qualiax.dnsmos import DnsmosScorer, _hz_to_mel, _mel_to_hz, log_mel_features, resample_to_16k
from qualiax.migrations import ExitCode

# Microsoft's dnsmos_local.py (librosa 0.8.1 behaviour) on tests' sample.wav, 16 kHz.
_REFERENCE_SAMPLE_SCORES = {"SIG": 3.450382, "BAK": 4.027663, "OVRL": 3.161694, "P808_MOS": 4.004066}


# ── official models (skipped unless downloaded) ──────────────────────────────

def test_scores_match_microsoft_reference_on_sample(official_dnsmos, sample_speech):
    audio, sr = sample_speech
    assert sr == 16_000
    scorer = DnsmosScorer(assets.asset_path("dnsmos-p835"), assets.asset_path("dnsmos-p808"))
    scores = scorer.score(audio)
    for name, expected in _REFERENCE_SAMPLE_SCORES.items():
        assert scores[name] == pytest.approx(expected, abs=1e-4), name


def test_perceptual_group_reports_model_metrics(official_dnsmos, sample_speech):
    audio, sr = sample_speech
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_name = {m.name: m for m in M.compute_perceptual(audio, sr)}
    for name in ("DNSMOS P.835 SIG", "DNSMOS P.835 BAK", "DNSMOS P.835 OVRL", "DNSMOS P.808 MOS"):
        assert by_name[name].confidence == "model"
        assert 1.0 <= by_name[name].value <= 5.0
    assert "DNSMOS P.835 OVRL (proxy)" not in by_name


def test_short_and_silent_input(official_dnsmos, sig):
    scorer = DnsmosScorer(assets.asset_path("dnsmos-p835"))
    assert scorer.score(np.zeros(16_000)) is None
    assert scorer.score(np.array([])) is None
    # Like the reference, a 0.5 s clip is doubled until it covers 9.01 s (16 s), giving 7 hops.
    short = scorer.score(sig.harmonic(150, 0.5))
    assert short["windows"] == 7 and "P808_MOS" not in short


def test_checksum_mismatch_falls_back_to_proxy(tmp_path, monkeypatch, sample_speech):
    pytest.importorskip("onnxruntime")
    (tmp_path / "dnsmos").mkdir()
    (tmp_path / "dnsmos" / "sig_bak_ovr.onnx").write_bytes(b"tampered")
    monkeypatch.setenv("QUALIAX_MODEL_DIR", str(tmp_path))
    audio, sr = sample_speech
    with pytest.warns(UserWarning, match="does not match its pinned SHA-256"):
        names = {m.name for m in M._compute_dnsmos_p835(audio, sr)}
    assert "DNSMOS P.835 OVRL (proxy)" in names


# ── feature extraction and resampling (no models needed) ────────────────────

def test_log_mel_features_match_librosa_0_8_1(monkeypatch):
    librosa = pytest.importorskip("librosa")
    import librosa.filters

    # librosa 0.8.1 (pinned by DNS-Challenge) spaced odd-n_fft bins evenly up to sr/2.
    monkeypatch.setattr(
        librosa.filters, "fft_frequencies",
        lambda *, sr=22050, n_fft=2048: np.linspace(0, float(sr) / 2, int(1 + n_fft // 2)),
    )
    audio = np.random.default_rng(0).normal(scale=0.1, size=144_000)
    mel = librosa.feature.melspectrogram(y=audio, sr=16_000, n_fft=321, hop_length=160, n_mels=120, pad_mode="reflect")
    expected = ((librosa.power_to_db(mel, ref=np.max) + 40) / 40).T
    assert log_mel_features(audio).shape == (900, 120)
    assert np.max(np.abs(log_mel_features(audio) - expected)) < 1e-5


def test_slaney_mel_scale_round_trips():
    hz = np.array([0.0, 200.0, 999.0, 1000.0, 4000.0, 8000.0])
    assert _hz_to_mel(1000.0) == pytest.approx(15.0)
    assert np.allclose(_mel_to_hz(_hz_to_mel(hz)), hz)


@pytest.mark.parametrize("sr", [8_000, 22_050, 44_100, 48_000])
def test_resampler_length_passband_and_alias_rejection(sr):
    t = np.arange(int(2.0 * sr)) / sr
    passband = resample_to_16k(np.sin(2 * np.pi * 1000 * t), sr)
    assert len(passband) == int(len(t) * 16_000 / sr)
    core = passband[2000:-2000]
    assert np.sqrt(2 * np.mean(core ** 2)) == pytest.approx(1.0, abs=0.01)
    if sr > 16_000:
        aliased = resample_to_16k(np.sin(2 * np.pi * 9_000 * t), sr)[2000:-2000]
        assert 20 * np.log10(np.sqrt(np.mean(aliased ** 2)) + 1e-12) < -60


# ── asset download / verify ─────────────────────────────────────────────────

@pytest.fixture
def served(tmp_path, monkeypatch):
    """Serve files from a local directory and point a one-asset registry at it."""
    root = tmp_path / "srv"
    root.mkdir()
    payload = b"fake-onnx-weights"
    (root / "model.onnx").write_bytes(payload)
    (root / "wrong.onnx").write_bytes(b"something else")
    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(SimpleHTTPRequestHandler, directory=str(root)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"

    def register(url_name):
        asset = assets.ModelAsset(
            name="fake", relative_path="fake/model.onnx", url=f"{base}/{url_name}",
            sha256=hashlib.sha256(payload).hexdigest(), size=len(payload),
            license="test", source="local", description="fake model",
        )
        monkeypatch.setattr(assets, "MODEL_ASSETS", {"fake": asset})

    yield register
    server.shutdown()


def test_download_verifies_and_skips_present_files(served, tmp_path):
    served("model.onnx")
    target = tmp_path / "models"
    assert [row["status"] for row in assets.download_models(directory=target)] == ["downloaded"]
    assert [row["status"] for row in assets.download_models(directory=target)] == ["present"]
    assert [row["status"] for row in assets.download_models(directory=target, force=True)] == ["downloaded"]
    assert assets.verified_asset_path("fake", target) == target / "fake" / "model.onnx"


def test_download_rejects_checksum_mismatch_and_leaves_nothing(served, tmp_path):
    served("wrong.onnx")
    with pytest.raises(assets.ModelAssetError, match="expected 17 bytes"):
        assets.download_models(directory=tmp_path / "models")
    assert not any((tmp_path / "models").rglob("*.onnx*"))


def test_download_rejects_unknown_names(served, tmp_path):
    served("model.onnx")
    with pytest.raises(assets.ModelAssetError, match="Unknown model asset"):
        assets.download_models(["nope"], directory=tmp_path)


def test_verify_reports_missing_mismatched_and_ok(served, tmp_path):
    served("model.onnx")
    target = tmp_path / "models"
    assert assets.verify_models(target)[0]["status"] == "missing"
    (target / "fake").mkdir(parents=True)
    (target / "fake" / "model.onnx").write_bytes(b"corrupted-weights")
    assert assets.verify_models(target)[0]["status"] == "checksum_mismatch"
    assets.download_models(directory=target, force=True)
    assert assets.verify_models(target)[0]["status"] == "ok"


def test_model_dir_honours_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("QUALIAX_MODEL_DIR", str(tmp_path / "explicit"))
    assert assets.model_dir() == tmp_path / "explicit"
    monkeypatch.delenv("QUALIAX_MODEL_DIR")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert assets.model_dir() == tmp_path / "xdg" / "qualiax" / "models"


def test_models_cli(served, tmp_path):
    served("model.onnx")
    runner = CliRunner()
    target = str(tmp_path / "models")
    missing = runner.invoke(main, ["models", "verify", target])
    downloaded = runner.invoke(main, ["models", "download", target])
    verified = runner.invoke(main, ["models", "verify", target])
    bad = runner.invoke(main, ["models", "frobnicate"])

    assert missing.exit_code == ExitCode.INVALID_INPUT and "missing" in missing.output
    assert downloaded.exit_code == 0 and "downloaded" in downloaded.output
    assert verified.exit_code == 0 and " ok " in verified.output
    assert bad.exit_code == ExitCode.INVALID_INPUT

"""Proxy trust labels come from the packaged benchmark evidence."""
from __future__ import annotations

import warnings

import pytest

from qualiax import metrics as M
from qualiax.models import MetricResult


def test_packaged_calibration_matches_current_proxy_code():
    calibration = M._proxy_calibration()
    assert calibration, "qualiax/data/proxy_calibration.json is missing"
    assert calibration["proxy_fingerprint"] == M.proxy_code_fingerprint(), (
        "A benchmarked proxy changed since the calibration was measured; "
        "re-run `python benchmarks/proxy_benchmark.py CORPUS_DIR`."
    )


def test_every_benchmarked_proxy_has_complete_evidence():
    evidence = M._proxy_calibration()["metrics"]
    assert set(evidence) >= {
        "DNSMOS P.835 SIG (proxy)",
        "DNSMOS P.835 BAK (proxy)",
        "DNSMOS P.835 OVRL (proxy)",
        "Estimated MOS (non-intrusive proxy)",
        "P.563 Proxy (NB Quality Estimate)",
        "UTMOS (proxy)",
        "SHEET MOS (proxy)",
    }
    for name, item in evidence.items():
        low, high = item["pearson_ci95"]
        assert -1.0 <= low <= item["pearson"] <= high <= 1.0, name
        assert item["mae"] >= 0 and item["reference"].startswith("DNSMOS"), name


@pytest.mark.parametrize(("ci_low", "expected"), [(0.75, "proxy"), (0.69, "heuristic")])
def test_confidence_follows_the_lower_correlation_bound(monkeypatch, ci_low, expected):
    fake = {
        "corpus": "Test corpus",
        "clips": 10,
        "metrics": {"Thing (proxy)": {"reference": "DNSMOS P.808 MOS", "pearson": 0.8,
                                       "pearson_ci95": [ci_low, 0.9], "mae": 0.3, "bias": -0.1}},
    }
    monkeypatch.setattr(M, "_proxy_calibration", lambda: fake)
    metric = M._with_trust(MetricResult("Thing (proxy)", 3.0), confidence="proxy", calibration_note="fallback")
    assert metric.confidence == expected
    assert "Pearson r 0.80" in metric.calibration_note and "Test corpus" in metric.calibration_note


def test_unbenchmarked_proxies_keep_their_own_note(monkeypatch):
    monkeypatch.setattr(M, "_proxy_calibration", lambda: {})
    metric = M._with_trust(MetricResult("Other (proxy)", 3.0), confidence="proxy", calibration_note="fallback")
    assert (metric.confidence, metric.calibration_note) == ("proxy", "fallback")


def test_perceptual_proxies_carry_measured_labels(sample_speech):
    audio, sr = sample_speech
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_name = {m.name: m for m in M.compute_perceptual(audio, sr)}
    evidence = M._proxy_calibration()["metrics"]
    for name, item in evidence.items():
        expected = "proxy" if item["pearson_ci95"][0] >= M.PROXY_MIN_CORRELATION else "heuristic"
        assert by_name[name].confidence == expected, name
        assert "docs/proxy-benchmark.md" in by_name[name].calibration_note
    assert by_name["AECMOS (proxy)"].confidence == "heuristic"

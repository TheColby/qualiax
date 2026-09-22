"""Measure qualiax's MOS proxies against Microsoft's official DNSMOS models.

Usage:
    python benchmarks/proxy_benchmark.py CORPUS_DIR [--workers N] [--limit N]

CORPUS_DIR is the unpacked VoiceBank-DEMAND test set (Valentini-Botinhao, 2017;
https://doi.org/10.7488/ds/2117, CC BY 4.0) with ``clean_testset_wav/`` and
``noisy_testset_wav/``. The official models must be downloaded first
(``qualiax models download``).

Writes benchmarks/results/voicebank-demand-test.json, the packaged summary in
qualiax/data/proxy_calibration.json, and docs/proxy-benchmark.md.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from qualiax import assets  # noqa: E402
from qualiax import metrics as M  # noqa: E402
from qualiax.analyzer import AudioLoader  # noqa: E402
from qualiax.dnsmos import DnsmosScorer, resample_to_16k  # noqa: E402
from qualiax.version import __version__  # noqa: E402

# proxy metric name -> official reference score
PAIRS = {
    "DNSMOS P.835 SIG (proxy)": "SIG",
    "DNSMOS P.835 BAK (proxy)": "BAK",
    "DNSMOS P.835 OVRL (proxy)": "OVRL",
    "Estimated MOS (non-intrusive proxy)": "P808_MOS",
    "P.563 Proxy (NB Quality Estimate)": "P808_MOS",
    "UTMOS (proxy)": "P808_MOS",
    "SHEET MOS (proxy)": "P808_MOS",
}
REFERENCE_NAMES = {
    "SIG": "DNSMOS P.835 SIG",
    "BAK": "DNSMOS P.835 BAK",
    "OVRL": "DNSMOS P.835 OVRL",
    "P808_MOS": "DNSMOS P.808 MOS",
}
UNPAIRED = ("AECMOS (proxy)",)
CORPUS = {
    "name": "VoiceBank-DEMAND test set",
    "citation": "Valentini-Botinhao, C. (2017). Noisy speech database for training speech enhancement "
    "algorithms and TTS models. University of Edinburgh. https://doi.org/10.7488/ds/2117",
    "license": "CC BY 4.0",
}
BOOTSTRAP_ROUNDS = 2000

_SCORER = None


def _score_clip(path: str) -> dict:
    global _SCORER
    warnings.simplefilter("ignore")
    if _SCORER is None:
        _SCORER = DnsmosScorer(assets.asset_path("dnsmos-p835"), assets.asset_path("dnsmos-p808"))
    audio, sr = AudioLoader.load(Path(path))
    mono = audio if audio.ndim == 1 else audio.mean(axis=0)
    official = _SCORER.score(resample_to_16k(mono, sr))
    proxies = {}
    for metric in (
        *M._compute_dnsmos_proxy(mono, sr),
        M._compute_aecmos(mono, sr),
        M._compute_pseudo_mos(mono, sr),
        M._compute_p563_proxy(mono, sr),
        *M._compute_learned_mos(mono, sr),
    ):
        if isinstance(metric.value, (int, float)):
            proxies[metric.name] = float(metric.value)
    return {
        "file": Path(path).name,
        "condition": Path(path).parent.name.split("_")[0],
        "utterance": Path(path).stem,
        "sample_rate": sr,
        "official": {key: official[key] for key in REFERENCE_NAMES} if official else None,
        "proxies": proxies,
    }


def _agreement(proxy: np.ndarray, reference: np.ndarray) -> dict:
    error = proxy - reference
    return {
        "pearson": float(pearsonr(proxy, reference)[0]),
        "spearman": float(spearmanr(proxy, reference)[0]),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "bias": float(np.mean(error)),
    }


def _with_intervals(proxy, reference, clusters, rng) -> dict:
    point = _agreement(proxy, reference)
    ids = np.unique(clusters)
    members = {cluster: np.flatnonzero(clusters == cluster) for cluster in ids}
    samples = {key: [] for key in point}
    for _ in range(BOOTSTRAP_ROUNDS):
        index = np.concatenate([members[c] for c in rng.choice(ids, size=len(ids), replace=True)])
        for key, value in _agreement(proxy[index], reference[index]).items():
            samples[key].append(value)
    return {
        key: {"value": round(point[key], 4), "ci95": [round(float(np.percentile(samples[key], q)), 4) for q in (2.5, 97.5)]}
        for key in point
    }


def summarize(rows: list[dict]) -> dict:
    rows = [row for row in rows if row["official"]]
    rng = np.random.default_rng(0)
    clusters = np.array([row["utterance"] for row in rows])
    conditions = np.array([row["condition"] for row in rows])
    pairs = {}
    for proxy_name, reference_key in PAIRS.items():
        keep = np.array([proxy_name in row["proxies"] for row in rows])
        proxy = np.array([row["proxies"].get(proxy_name, np.nan) for row in rows])[keep]
        reference = np.array([row["official"][reference_key] for row in rows])[keep]
        by_condition = {
            condition: {
                "proxy_mean": round(float(np.mean(proxy[conditions[keep] == condition])), 3),
                "reference_mean": round(float(np.mean(reference[conditions[keep] == condition])), 3),
            }
            for condition in ("clean", "noisy")
        }
        pairs[proxy_name] = {
            "reference": REFERENCE_NAMES[reference_key],
            "n": int(keep.sum()),
            **_with_intervals(proxy, reference, clusters[keep], rng),
            "by_condition": by_condition,
        }
    unpaired = {
        name: {
            condition: {
                "mean": round(float(np.mean(values)), 3),
                "p10": round(float(np.percentile(values, 10)), 3),
            }
            for condition in ("clean", "noisy")
            if (values := [row["proxies"][name] for row in rows if row["condition"] == condition and name in row["proxies"]])
        }
        for name in UNPAIRED
    }
    return {"pairs": pairs, "unpaired": unpaired, "clips": len(rows)}


def calibration_summary(summary: dict, meta: dict) -> dict:
    return {
        "corpus": meta["corpus"]["name"],
        "clips": summary["clips"],
        "qualiax_version": meta["qualiax_version"],
        "proxy_fingerprint": meta["proxy_fingerprint"],
        "metrics": {
            name: {
                "reference": pair["reference"],
                "pearson": pair["pearson"]["value"],
                "pearson_ci95": pair["pearson"]["ci95"],
                "mae": pair["mae"]["value"],
                "bias": pair["bias"]["value"],
            }
            for name, pair in summary["pairs"].items()
        },
    }


def markdown(summary: dict, meta: dict) -> str:
    lines = [
        "# Proxy benchmark",
        "",
        f"qualiax {meta['qualiax_version']} proxies measured against Microsoft's official DNSMOS models "
        f"on the {meta['corpus']['name']} ({summary['clips']} clips: every clean and noisy utterance). "
        "Generated by `benchmarks/proxy_benchmark.py`; raw numbers are in "
        "`benchmarks/results/voicebank-demand-test.json`.",
        "",
        "Intervals are 95% bootstrap intervals that resample utterances, keeping each clean/noisy pair together. "
        "The test set has only two speakers, so these numbers say little about speaker variety.",
        "",
        "| Proxy | Reference | Pearson r | Spearman ρ | MAE | Bias | Clean mean (ref) | Noisy mean (ref) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, pair in summary["pairs"].items():
        def fmt(key):
            value, (lo, hi) = pair[key]["value"], pair[key]["ci95"]
            return f"{value:.2f} [{lo:.2f}, {hi:.2f}]"
        clean, noisy = pair["by_condition"]["clean"], pair["by_condition"]["noisy"]
        lines.append(
            f"| {name} | {pair['reference']} | {fmt('pearson')} | {fmt('spearman')} | {fmt('mae')} | {fmt('bias')} "
            f"| {clean['proxy_mean']:.2f} ({clean['reference_mean']:.2f}) | {noisy['proxy_mean']:.2f} ({noisy['reference_mean']:.2f}) |"
        )
    lines += [
        "",
        "## Not benchmarked against a model",
        "",
        "Microsoft's AECMOS needs the far-end reference, microphone, and processed signals, so it can't score a "
        "single recording, and this corpus has no echo. On it, the echo proxy scores:",
        "",
    ]
    for name, by_condition in summary["unpaired"].items():
        for condition, stats in by_condition.items():
            lines.append(f"- {name}, {condition}: mean {stats['mean']:.2f}, 10th percentile {stats['p10']:.2f}")
    lines += [
        "",
        "## Models and data",
        "",
        *(f"- `{a['name']}`: {a['description']}, {a['source']}, SHA-256 `{a['sha256']}` ({a['license']})" for a in meta["models"]),
        f"- Corpus: {meta['corpus']['citation']} ({meta['corpus']['license']})",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--limit", type=int, default=None, help="Use only the first N utterances (for a quick run).")
    args = parser.parse_args()

    if any(assets.asset_status(name) != "ok" for name in ("dnsmos-p835", "dnsmos-p808")):
        sys.exit("Official DNSMOS models are missing or unverified; run `qualiax models download`.")
    utterances = sorted(path.name for path in (args.corpus / "clean_testset_wav").glob("*.wav"))[: args.limit]
    files = [str(args.corpus / f"{condition}_testset_wav" / name) for name in utterances for condition in ("clean", "noisy")]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(_score_clip, files, chunksize=8))

    meta = {
        "generated": dt.date.today().isoformat(),
        "qualiax_version": __version__,
        "proxy_fingerprint": M.proxy_code_fingerprint(),
        "corpus": CORPUS,
        "models": [row for row in assets.verify_models()],
        "bootstrap_rounds": BOOTSTRAP_ROUNDS,
    }
    summary = summarize(rows)
    results_dir = REPO / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "voicebank-demand-test.json").write_text(
        json.dumps({"meta": meta, "summary": summary, "clips": rows}, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    data_dir = REPO / "qualiax" / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "proxy_calibration.json").write_text(
        json.dumps(calibration_summary(summary, meta), indent=2) + "\n", encoding="utf-8"
    )
    (REPO / "docs" / "proxy-benchmark.md").write_text(markdown(summary, meta), encoding="utf-8")
    print(markdown(summary, meta))


if __name__ == "__main__":
    main()

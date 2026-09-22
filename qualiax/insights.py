"""
Composite quality insights derived from existing qualiax metrics.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from .models import FileResult, MetricResult

INSIGHT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class InsightValidationIssue:
    path: str
    message: str


@dataclass(frozen=True)
class InsightPayload:
    version: str = INSIGHT_SCHEMA_VERSION
    quality_fingerprint: dict[str, Any] = field(default_factory=dict)
    defect_labels: list[dict[str, Any]] = field(default_factory=list)
    repair_suggestions: list[str] = field(default_factory=list)
    mos_explanation: list[dict[str, Any]] = field(default_factory=list)
    baseline_comparison: dict[str, Any] = field(default_factory=dict)
    drift_monitor: dict[str, Any] = field(default_factory=dict)
    dataset_audit: list[dict[str, Any]] = field(default_factory=list)
    ci_checks: list[dict[str, Any]] = field(default_factory=list)
    segment_heatmap: dict[str, Any] = field(default_factory=dict)
    triage: dict[str, Any] = field(default_factory=dict)
    snippets: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "InsightPayload":
        return cls(
            version=str(payload.get("version", "")),
            quality_fingerprint=dict(payload.get("quality_fingerprint", {})),
            defect_labels=list(payload.get("defect_labels", [])),
            repair_suggestions=list(payload.get("repair_suggestions", [])),
            mos_explanation=list(payload.get("mos_explanation", [])),
            baseline_comparison=dict(payload.get("baseline_comparison", {})),
            drift_monitor=dict(payload.get("drift_monitor", {})),
            dataset_audit=list(payload.get("dataset_audit", [])),
            ci_checks=list(payload.get("ci_checks", [])),
            segment_heatmap=dict(payload.get("segment_heatmap", {})),
            triage=dict(payload.get("triage", {})),
            snippets=list(payload.get("snippets", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "quality_fingerprint": self.quality_fingerprint,
            "defect_labels": self.defect_labels,
            "repair_suggestions": self.repair_suggestions,
            "mos_explanation": self.mos_explanation,
            "baseline_comparison": self.baseline_comparison,
            "drift_monitor": self.drift_monitor,
            "dataset_audit": self.dataset_audit,
            "ci_checks": self.ci_checks,
            "segment_heatmap": self.segment_heatmap,
            "triage": self.triage,
            "snippets": self.snippets,
        }


# Metric name -> insight feature. Keys must match the names emitted by the
# metric modules exactly; the short aliases are kept for hand-built reports.
_FINGERPRINT_FEATURES = {
    "SNR": "noise.snr",
    "Estimated SNR": "noise.snr",
    "Integrated Loudness (LUFS)": "loudness.integrated_lufs",
    "True Peak": "loudness.true_peak",
    "DNSMOS OVRL": "perceptual.dnsmos_ovrl",
    "DNSMOS P.835 OVRL": "perceptual.dnsmos_ovrl",
    "DNSMOS P.835 OVRL (proxy)": "perceptual.dnsmos_ovrl",
    "P.563 Proxy (NB Quality Estimate)": "perceptual.p563_proxy",
    "Spectral Centroid": "spectral.centroid",
    "Zero Crossing Rate": "temporal.zcr",
    "Clipping Ratio": "noise.clipping_ratio",
    "Harmonic-to-Noise Ratio": "speech.hnr",
    "Harmonic-to-Noise Ratio (HNR)": "speech.hnr",
    "F0 Mean": "prosody.f0_mean",
}

# The metric modules report clipping as a sample count plus a 0/1 full-scale
# flag rather than a ratio, so the ratio feature is derived from these.
_NEAR_CLIPPED_METRIC = "Near-Clipped Samples"
_CLIPPING_FLAG_METRIC = "Clipping Detected"

KNOWN_INSIGHT_FEATURES = frozenset(_FINGERPRINT_FEATURES.values())
_VALID_LABEL_SEVERITIES = ("info", "warn", "fail")
FINGERPRINT_VERSION = 2


class InsightInputError(ValueError):
    """Raised when an insight input (rules, baseline) cannot be used."""


def enrich_results(
    results: list[FileResult],
    *,
    baseline_results: list[FileResult] | None = None,
    baseline_path: str | Path | None = None,
    dataset_audit: bool = True,
    drift: bool = False,
    drift_state_path: str | Path | None = None,
    ci: bool = False,
    fingerprint_sensitivity: str = "balanced",
    fingerprint_weights: dict[str, float] | None = None,
    preset: str | None = None,
    insight_rules_path: str | Path | None = None,
) -> list[FileResult]:
    """Attach composite insight payloads to file results in place."""
    baseline = baseline_results or _load_baseline_results(baseline_path)
    baseline_profile = _profile(baseline) if baseline else {}
    baseline_meta = _baseline_metadata(baseline_profile, baseline_path, baseline)
    previous_features = _load_drift_state(drift_state_path) if drift else None
    last_features: dict[str, float] | None = None
    custom_rules = _load_insight_rules(insight_rules_path)

    for result in results:
        features, metric_lookup = _result_features(result)
        fingerprint = _quality_fingerprint(
            features,
            sensitivity=fingerprint_sensitivity,
            weights=fingerprint_weights,
        )
        defect_labels = _defect_labels(
            result,
            features,
            metric_lookup,
            preset=preset,
        )
        custom_labels, custom_suggestions, custom_ci_fail = _apply_custom_rules(
            features,
            metric_lookup,
            custom_rules,
        )
        defect_labels.extend(custom_labels)
        baseline_comparison = _baseline_comparison(features, baseline_profile, baseline_meta)
        drift_monitor = _drift_monitor(features, previous_features) if drift else {}
        ci_checks = _ci_checks(defect_labels, baseline_comparison, custom_ci_fail) if ci else []
        previous_features = features
        last_features = features

        result.insights = InsightPayload(
            quality_fingerprint=fingerprint,
            defect_labels=defect_labels,
            repair_suggestions=_repair_suggestions(defect_labels) + custom_suggestions,
            mos_explanation=_mos_explanation(result.metrics),
            baseline_comparison=baseline_comparison,
            drift_monitor=drift_monitor,
            ci_checks=ci_checks,
        ).to_dict()

    if dataset_audit:
        _attach_dataset_audit(results)
    _attach_segment_heatmaps(results)
    _attach_triage(results)
    if drift and drift_state_path is not None and last_features is not None:
        _write_drift_state(drift_state_path, last_features)
    return results


def enrich_existing_report(
    input_path: str | Path,
    output_path: str | Path,
    *,
    baseline_path: str | Path | None = None,
    drift: bool = False,
    drift_state_path: str | Path | None = None,
    ci: bool = False,
    fingerprint_sensitivity: str = "balanced",
    fingerprint_weights: dict[str, float] | None = None,
    preset: str | None = None,
    insight_rules_path: str | Path | None = None,
) -> list[FileResult]:
    """Enrich an existing JSON/JSONL report without re-reading audio."""
    results = _load_report_results(input_path)
    enrich_results(
        results,
        baseline_path=baseline_path,
        drift=drift,
        drift_state_path=drift_state_path,
        ci=ci,
        fingerprint_sensitivity=fingerprint_sensitivity,
        fingerprint_weights=fingerprint_weights,
        preset=preset,
        insight_rules_path=insight_rules_path,
    )
    from .reporter import JsonReporter

    Path(output_path).write_text(JsonReporter().render(results), encoding="utf-8")
    return results


def validate_insight_inputs(
    *,
    baseline_path: str | Path | None = None,
    insight_rules_path: str | Path | None = None,
) -> None:
    """Fail fast on unusable --baseline / --insight-rules inputs.

    Raises InsightInputError with a user-facing message; intended to run
    before any audio is analyzed.
    """
    _load_baseline_results(baseline_path)
    _load_insight_rules(insight_rules_path)


def validate_insights_report(path: str | Path) -> list[InsightValidationIssue]:
    """Validate only the insight payload portions of an existing report."""
    issues: list[InsightValidationIssue] = []
    try:
        payload = _load_raw_report_payload(path)
    except ValueError as exc:
        return [InsightValidationIssue("$", f"Report is not valid JSON/JSONL: {exc}")]
    if not isinstance(payload, list):
        return [InsightValidationIssue("$", "Report payload must be a list.")]
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            issues.append(InsightValidationIssue(f"$[{index}]", "Report item must be an object."))
            continue
        insights = item.get("insights")
        if not isinstance(insights, dict):
            issues.append(InsightValidationIssue(f"$[{index}].insights", "Insights must be an object."))
            continue
        issues.extend(_validate_insight_payload(insights, f"$[{index}].insights"))
    return issues


def build_insight_summary_payload(results: list[FileResult]) -> list[dict[str, Any]]:
    """Return compact pipeline-friendly insight summaries."""
    summaries = []
    for result in results:
        insights = result.insights or {}
        summaries.append(
            {
                "file": result.path,
                "source_file": result.source_file,
                "fingerprint": insights.get("quality_fingerprint", {}).get("signature"),
                "labels": [label.get("id") for label in insights.get("defect_labels", [])],
                "ci_checks": insights.get("ci_checks", []),
                "triage": insights.get("triage", {}),
            }
        )
    return summaries


def export_flagged_segment_snippets(
    results: list[FileResult],
    output_dir: str | Path,
    *,
    padding_s: float = 0.25,
    overwrite: bool = True,
    session_paths: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Export WAV snippets for flagged segment results and link them into insights.

    With ``overwrite=False`` a FileExistsError is raised before anything is
    written if a snippet path already exists, unless that path is listed in
    ``session_paths`` (snippets this process wrote earlier, e.g. in watch
    mode). Paths written by this call are added to ``session_paths``.
    """
    from .analyzer import AudioLoader

    output_dir = Path(output_dir)
    planned: list[tuple[FileResult, str, Path]] = []
    name_owner: dict[str, str] = {}
    for result in results:
        if result.segment_index is None:
            continue
        if not (result.insights or {}).get("defect_labels"):
            continue
        source = result.source_file or result.path
        if not Path(source).exists():
            continue
        name = _snippet_name(result)
        if name_owner.setdefault(name, source) != source:
            # Same stem from a different directory; keep both snippets.
            name = _snippet_name(result, disambiguate=True)
        planned.append((result, source, output_dir / name))

    if not overwrite:
        allowed = session_paths or set()
        existing = [str(path) for _, _, path in planned if path.exists() and str(path) not in allowed]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing snippet file(s) without --force: "
                + ", ".join(existing[:3])
                + (" ..." if len(existing) > 3 else "")
            )

    if planned:
        output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[dict[str, Any]] = []
    audio_cache: dict[str, tuple[Any, int]] = {}

    for result, source, snippet_path in planned:
        labels = result.insights.get("defect_labels", [])
        if source not in audio_cache:
            audio_cache[source] = AudioLoader.load(Path(source))
        audio, sr = audio_cache[source]
        total = audio.shape[-1] if hasattr(audio, "shape") else len(audio)
        duration = total / sr if sr else 0.0
        end_s = min(duration, float(result.segment_end_s or result.segment_start_s or 0.0) + padding_s)
        start_s = min(end_s, max(0.0, float(result.segment_start_s or 0.0) - padding_s))
        start = int(round(start_s * sr))
        end = min(int(round(end_s * sr)), total)
        snippet_audio = audio[start:end] if getattr(audio, "ndim", 1) == 1 else audio[:, start:end]
        _write_wav(snippet_path, snippet_audio, sr)
        if session_paths is not None:
            session_paths.add(str(snippet_path))
        snippet = {
            "path": str(snippet_path),
            "source_file": source,
            "segment_index": result.segment_index,
            "start_s": start_s,
            "end_s": end_s,
            "labels": [label.get("id") for label in labels],
        }
        result.insights.setdefault("snippets", []).append(snippet)
        cells = result.insights.get("segment_heatmap", {}).get("cells", [])
        for cell in cells:
            if cell.get("segment_index") == result.segment_index:
                cell["snippet"] = str(snippet_path)
        exported.append(snippet)
    return exported


def _load_baseline_results(path: str | Path | None) -> list[FileResult]:
    if path is None:
        return []
    results = _load_report_results(path, role="baseline report")
    if not _profile(results):
        raise InsightInputError(
            f"Baseline report {path} has no metrics insights can compare against; "
            "pass a qualiax JSON/JSONL report produced with the default metric groups."
        )
    return results


def _load_report_results(path: str | Path | None, *, role: str = "report") -> list[FileResult]:
    if path is None:
        return []
    path = Path(path)
    try:
        payload = _load_raw_report_payload(path)
    except OSError as exc:
        raise InsightInputError(f"Cannot read {role} {path}: {exc}") from exc
    except ValueError as exc:
        raise InsightInputError(f"{role.capitalize()} {path} is not valid JSON/JSONL: {exc}") from exc
    if not isinstance(payload, list):
        raise InsightInputError(
            f"{role.capitalize()} {path} must be a qualiax JSON/JSONL report (a list of per-file results), "
            f"not a JSON {type(payload).__name__}."
        )

    loaded: list[FileResult] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        metrics = [
            MetricResult(
                name=str(metric.get("name", "")),
                value=metric.get("value"),
                unit=str(metric.get("unit", "")),
                description=str(metric.get("description", "")),
                group=str(metric.get("group", "")),
                higher_is_better=metric.get("higher_is_better"),
                warning=metric.get("warning"),
                reference_range=tuple(metric["reference_range"]) if metric.get("reference_range") else None,
                confidence=metric.get("confidence"),
                calibration_note=metric.get("calibration_note"),
            )
            for metric in item.get("metrics", [])
            if isinstance(metric, dict)
        ]
        loaded.append(
            FileResult(
                path=str(item.get("file", "")),
                source_file=item.get("source_file"),
                segment_index=item.get("segment_index"),
                total_segments=item.get("total_segments"),
                segment_start_s=item.get("segment_start_s"),
                segment_end_s=item.get("segment_end_s"),
                duration_s=float(item.get("duration_s") or 0.0),
                sample_rate=int(item.get("sample_rate") or 0),
                channels=int(item.get("channels") or 0),
                bit_depth=item.get("bit_depth"),
                content_type=item.get("content_type"),
                speech_confidence=item.get("speech_confidence"),
                notes=list(item.get("notes", [])),
                confidence_notes=list(item.get("confidence_notes", [])),
                error=item.get("error"),
                metrics=metrics,
            )
        )
    return loaded


def _load_raw_report_payload(path: str | Path) -> Any:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _validate_insight_payload(payload: dict[str, Any], path: str) -> list[InsightValidationIssue]:
    # Single source of truth: the published INSIGHTS_SCHEMA contract, which
    # --validate-output applies to full reports as well.
    from .validation import validate_insight_payload

    return [InsightValidationIssue(issue.path, issue.message) for issue in validate_insight_payload(payload, path=path)]


def _load_drift_state(path: str | Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    state_path = Path(path)
    if not state_path.exists():
        return None
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    features = payload.get("last_features") if isinstance(payload, dict) else None
    if not isinstance(features, dict):
        return None
    return {
        str(name): float(value)
        for name, value in features.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }


def _write_drift_state(path: str | Path, features: dict[str, float]) -> None:
    Path(path).write_text(
        json.dumps({"version": 1, "last_features": features}, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _result_features(result: FileResult) -> tuple[dict[str, float], dict[str, MetricResult]]:
    """Features and their evidence metrics for one result, including derived ones."""
    features = _extract_features(result.metrics)
    lookup = _feature_metric_lookup(result.metrics)
    if "noise.clipping_ratio" not in features:
        derived = _derived_clipping_ratio(result)
        if derived is not None:
            features["noise.clipping_ratio"], lookup["noise.clipping_ratio"] = derived
    return features, lookup


def _derived_clipping_ratio(result: FileResult) -> tuple[float, MetricResult] | None:
    """Fraction of samples pinned at full scale, from the noise/basic clipping metrics.

    "Near-Clipped Samples" counts samples within 0.1% of the file's own peak,
    which also catches the crests of an unclipped loud sine, so only count
    them when "Clipping Detected" confirms the peak actually reached full scale.
    """
    by_name = {metric.name: metric for metric in result.metrics}
    count_metric = by_name.get(_NEAR_CLIPPED_METRIC)
    flag_metric = by_name.get(_CLIPPING_FLAG_METRIC)
    if count_metric is None or flag_metric is None:
        return None
    count, flag = count_metric.value, flag_metric.value
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in (count, flag)):
        return None
    total_samples = float(result.duration_s or 0.0) * float(result.sample_rate or 0)
    if total_samples <= 0:
        return None
    ratio = min(1.0, float(count) / total_samples) if flag else 0.0
    return round(ratio, 4), count_metric


def _extract_features(metrics: Iterable[MetricResult]) -> dict[str, float]:
    features: dict[str, float] = {}
    for metric in metrics:
        feature_name = _FINGERPRINT_FEATURES.get(metric.name)
        if not feature_name:
            continue
        if not isinstance(metric.value, (int, float)) or isinstance(metric.value, bool):
            continue
        value = float(metric.value)
        if math.isfinite(value):
            features[feature_name] = round(value, 4)
    return features


def _feature_metric_lookup(metrics: Iterable[MetricResult]) -> dict[str, MetricResult]:
    lookup: dict[str, MetricResult] = {}
    for metric in metrics:
        feature = _FINGERPRINT_FEATURES.get(metric.name)
        if feature:
            lookup[feature] = metric
    return lookup


def load_insight_rules(path: str | Path) -> dict[str, Any]:
    """Load and validate a JSON/TOML insight rules file.

    Raises InsightInputError with a readable message for unreadable files,
    parse errors, and rules that could never fire (unknown feature, no bound).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in {".json", ".toml"}:
        raise InsightInputError(f"Insight rules must be a .json or .toml file: {path}")
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InsightInputError(f"Cannot read insight rules {path}: {exc}") from exc
    try:
        if suffix == ".json":
            payload = json.loads(text)
        else:
            try:
                import tomllib
            except ImportError:  # pragma: no cover - Python < 3.11
                import tomli as tomllib
            payload = tomllib.loads(text)
    except ValueError as exc:  # JSONDecodeError and TOMLDecodeError both subclass ValueError
        raise InsightInputError(f"Invalid {suffix[1:].upper()} in insight rules {path}: {exc}") from exc
    problems = _insight_rule_problems(payload)
    if problems:
        shown = "; ".join(problems[:5]) + (f"; ... ({len(problems) - 5} more)" if len(problems) > 5 else "")
        raise InsightInputError(f"Invalid insight rules {path}: {shown}")
    return payload


def _load_insight_rules(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return load_insight_rules(path)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _insight_rule_problems(payload: Any) -> list[str]:
    if not isinstance(payload, dict):
        return [f"expected an object/table with a 'labels' list, got {type(payload).__name__}"]
    problems = []
    unknown = sorted(set(payload) - {"labels", "version"})
    if unknown:
        problems.append(f"unknown top-level key(s) {', '.join(unknown)} (supported: labels)")
    labels = payload.get("labels", [])
    if not isinstance(labels, list):
        return problems + ["'labels' must be a list of rule objects"]
    for index, rule in enumerate(labels):
        where = f"labels[{index}]"
        if not isinstance(rule, dict):
            problems.append(f"{where} must be an object")
            continue
        feature = rule.get("feature")
        if feature not in KNOWN_INSIGHT_FEATURES:
            problems.append(
                f"{where}.feature {feature!r} is not a known insight feature "
                f"(valid: {', '.join(sorted(KNOWN_INSIGHT_FEATURES))})"
            )
        bounds = {key: rule[key] for key in ("min", "max") if key in rule}
        if not bounds:
            problems.append(f"{where} needs a numeric 'min' and/or 'max'")
        for key, value in bounds.items():
            if not _is_number(value):
                problems.append(f"{where}.{key} must be a number")
        if len(bounds) == 2 and all(_is_number(v) for v in bounds.values()) and bounds["min"] > bounds["max"]:
            problems.append(f"{where} has min > max, so it would fire on every value")
        if "severity" in rule and rule["severity"] not in _VALID_LABEL_SEVERITIES:
            problems.append(f"{where}.severity must be one of {', '.join(_VALID_LABEL_SEVERITIES)}")
        if "confidence" in rule and not (_is_number(rule["confidence"]) and 0.0 <= rule["confidence"] <= 1.0):
            problems.append(f"{where}.confidence must be a number between 0 and 1")
        for key in ("id", "message", "suggestion"):
            if key in rule and not isinstance(rule[key], str):
                problems.append(f"{where}.{key} must be a string")
        if "ci_fail" in rule and not isinstance(rule["ci_fail"], bool):
            problems.append(f"{where}.ci_fail must be true or false")
    return problems


def _apply_custom_rules(
    features: dict[str, float],
    metric_lookup: dict[str, MetricResult],
    rules: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[str], bool]:
    labels: list[dict[str, Any]] = []
    suggestions: list[str] = []
    ci_fail = False
    for rule in rules.get("labels", []):
        if not isinstance(rule, dict):
            continue
        feature = str(rule.get("feature", ""))
        if feature not in features:
            continue
        value = features[feature]
        min_value = rule.get("min")
        max_value = rule.get("max")
        triggered = False
        if isinstance(min_value, (int, float)) and value < float(min_value):
            triggered = True
        if isinstance(max_value, (int, float)) and value > float(max_value):
            triggered = True
        if not triggered:
            continue
        label = _label(
            str(rule.get("id", "custom_rule")),
            str(rule.get("severity", "warn")),
            float(rule.get("confidence", 0.9)),
            str(rule.get("message", f"{feature} violated custom rule")),
            metric_lookup.get(feature),
        )
        labels.append(label)
        suggestion = rule.get("suggestion")
        if isinstance(suggestion, str) and suggestion not in suggestions:
            suggestions.append(suggestion)
        ci_fail = ci_fail or bool(rule.get("ci_fail"))
    return labels, suggestions, ci_fail


def _quality_fingerprint(
    features: dict[str, float],
    *,
    sensitivity: str,
    weights: dict[str, float] | None,
) -> dict[str, Any]:
    sensitivity = sensitivity if sensitivity in {"coarse", "balanced", "strict"} else "balanced"
    weights = {
        name: float(value)
        for name, value in (weights or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    buckets = {
        name: _bucket(value * weights.get(name, 1.0), sensitivity)
        for name, value in sorted(features.items())
    }
    signature_source = json.dumps(
        {"sensitivity": sensitivity, "weights": weights, "buckets": buckets},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "version": FINGERPRINT_VERSION,
        "sensitivity": sensitivity,
        "weights": weights,
        "signature": hashlib.sha1(signature_source.encode("utf-8")).hexdigest()[:16],
        "features": features,
        "buckets": buckets,
    }


def _fingerprint_dict(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:16]


def _bucket(value: float, sensitivity: str) -> float:
    if abs(value) >= 100:
        step = 10.0
    elif abs(value) >= 10:
        step = 2.0
    elif abs(value) >= 1:
        step = 0.5
    else:
        step = 0.05
    if sensitivity == "coarse":
        step *= 2.0
    elif sensitivity == "strict":
        step *= 0.25
    return round(round(value / step) * step, 4)


def _defect_labels(
    result: FileResult,
    features: dict[str, float],
    metric_lookup: dict[str, MetricResult],
    *,
    preset: str | None,
) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    thresholds = _preset_thresholds(preset)
    snr = features.get("noise.snr")
    true_peak = features.get("loudness.true_peak")
    loudness = features.get("loudness.integrated_lufs")
    clipping = features.get("noise.clipping_ratio")
    mos_feature = "perceptual.dnsmos_ovrl" if "perceptual.dnsmos_ovrl" in features else "perceptual.p563_proxy"
    mos = features.get(mos_feature)
    is_silence = result.content_type == "silence" or result.duration_s == 0

    if snr is not None and snr < thresholds["snr_min"]:
        severity = "fail" if snr < thresholds["snr_fail"] else "warn"
        labels.append(_label("noisy_floor", severity, 0.95 if severity == "fail" else 0.72, f"SNR is {snr:.1f} dB", metric_lookup.get("noise.snr")))
    if true_peak is not None and true_peak > -1.0:
        labels.append(_label("clipping_risk", "fail", 0.86, f"true peak is {true_peak:.1f} dBTP", metric_lookup.get("loudness.true_peak")))
    if clipping is not None and clipping > 0.005:
        labels.append(_label("hard_clipping", "fail", 0.90, f"clipping ratio is {clipping:.3f}", metric_lookup.get("noise.clipping_ratio")))
    if loudness is not None and loudness < thresholds["lufs_min"]:
        labels.append(_label("under_loud", "warn", 0.78, f"integrated loudness is {loudness:.1f} LUFS", metric_lookup.get("loudness.integrated_lufs")))
    if loudness is not None and loudness > thresholds["lufs_max"]:
        labels.append(_label("over_loud", "warn", 0.78, f"integrated loudness is {loudness:.1f} LUFS", metric_lookup.get("loudness.integrated_lufs")))
    # MOS models/proxies score silence as "bad speech"; silence_heavy covers that case.
    if mos is not None and mos < 2.8 and not is_silence:
        labels.append(_label("low_perceptual_quality", "warn", 0.82, f"MOS proxy/model score is {mos:.2f}", metric_lookup.get(mos_feature)))
    if is_silence:
        labels.append(_label("silence_heavy", "warn", 0.88, "content detector or duration indicates little usable audio", None))
    return labels


def _preset_thresholds(preset: str | None) -> dict[str, float]:
    thresholds = {"snr_min": 15.0, "snr_fail": 10.0, "lufs_min": -23.0, "lufs_max": -10.0}
    if preset == "podcast":
        thresholds.update({"lufs_min": -18.0, "lufs_max": -14.0})
    elif preset == "call-center-qa":
        thresholds.update({"snr_min": 12.0, "snr_fail": 8.0})
    elif preset == "speech-enhancement":
        thresholds.update({"snr_min": 15.0, "snr_fail": 12.0})
    elif preset == "music-mastering":
        thresholds.update({"lufs_min": -16.0, "lufs_max": -8.0})
    return thresholds


def _label(
    label_id: str,
    severity: str,
    confidence: float,
    evidence: str,
    metric: MetricResult | None,
) -> dict[str, Any]:
    evidence_metrics = []
    if metric is not None:
        evidence_metrics.append(
            {
                "metric": metric.name,
                "group": metric.group,
                "value": metric.value,
                "unit": metric.unit,
            }
        )
    return {
        "id": label_id,
        "severity": severity,
        "confidence": round(confidence, 3),
        "evidence": evidence,
        "evidence_metrics": evidence_metrics,
    }


def _repair_suggestions(labels: list[dict[str, Any]]) -> list[str]:
    suggestions_by_label = {
        "noisy_floor": "Run denoise or improve gain staging before downstream speech analysis.",
        "clipping_risk": "Lower input gain or re-export with true-peak limiting below -1 dBTP.",
        "hard_clipping": "Re-record if possible; otherwise use declipping before normalization.",
        "under_loud": "Normalize toward the target preset before publishing or comparing.",
        "over_loud": "Reduce program loudness and check limiter settings.",
        "low_perceptual_quality": "Inspect codec, noise, and speech enhancement stages before accepting the file.",
        "silence_heavy": "Trim leading/trailing silence or verify the file is the intended recording.",
    }
    suggestions = []
    for label in labels:
        suggestion = suggestions_by_label.get(label["id"])
        if suggestion and suggestion not in suggestions:
            suggestions.append(suggestion)
    return suggestions


def _mos_explanation(metrics: Iterable[MetricResult]) -> list[dict[str, Any]]:
    explanations = []
    for metric in metrics:
        if "MOS" not in metric.name and "P.563" not in metric.name:
            continue
        if not isinstance(metric.value, (int, float)) or isinstance(metric.value, bool):
            continue
        value = float(metric.value)
        if value < 2.8:
            direction = "negative"
            reason = "low score suggests audible degradation from noise, distortion, codec loss, or enhancement artifacts"
        elif value >= 3.8:
            direction = "positive"
            reason = "score is in a range usually associated with cleaner speech quality"
        else:
            direction = "mixed"
            reason = "score is usable but leaves room for audible defects"
        explanations.append(
            {
                "metric": metric.name,
                "score": round(value, 4),
                "direction": direction,
                "reason": reason,
                "confidence": metric.confidence or "unknown",
            }
        )
    return explanations


def _profile(results: list[FileResult]) -> dict[str, dict[str, float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for result in results:
        for name, value in _result_features(result)[0].items():
            values[name].append(value)
    return {
        name: {
            "mean": round(mean(items), 4),
            "p05": round(_percentile(items, 5), 4),
            "p95": round(_percentile(items, 95), 4),
        }
        for name, items in values.items()
        if items
    }


def _baseline_metadata(
    baseline_profile: dict[str, dict[str, float]],
    baseline_path: str | Path | None,
    baseline_results: list[FileResult],
) -> dict[str, Any]:
    if not baseline_profile:
        return {}
    return {
        "source": str(baseline_path) if baseline_path is not None else "inline",
        "name": Path(baseline_path).name if baseline_path is not None else "inline",
        "profile_size": len(baseline_profile),
        "file_count": len({result.source_file or result.path for result in baseline_results}),
        "fingerprint": _fingerprint_dict(baseline_profile),
    }


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * (percentile / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return float(ordered[int(rank)])
    return float(ordered[low] + (ordered[high] - ordered[low]) * (rank - low))


def _baseline_comparison(
    features: dict[str, float],
    baseline: dict[str, dict[str, float]],
    baseline_meta: dict[str, Any],
) -> dict[str, Any]:
    if not baseline:
        return {}
    comparisons = []
    regression_score = 0.0
    for name, current in sorted(features.items()):
        if name not in baseline:
            continue
        profile = baseline[name]
        base = profile["mean"]
        p05 = profile["p05"]
        p95 = profile["p95"]
        delta = round(current - base, 4)
        direction = _direction_for_feature(name)
        percentile_status = _percentile_status(current, p05, p95)
        regressed = (
            percentile_status == "below_band"
            if direction == "higher"
            else percentile_status == "above_band"
        )
        if regressed:
            regression_score += abs(delta)
        comparisons.append(
            {
                "feature": name,
                "current": current,
                "baseline": base,
                "baseline_mean": base,
                "baseline_p05": p05,
                "baseline_p95": p95,
                "delta": delta,
                "direction": direction,
                "percentile_status": percentile_status,
                "status": "regressed" if regressed else "ok",
            }
        )
    regressions = [item for item in comparisons if item["status"] == "regressed"]
    regressions.sort(key=lambda item: abs(float(item["delta"])), reverse=True)
    return {
        "status": "regressed" if regressions else "ok",
        "baseline": baseline_meta,
        "regression_score": round(regression_score, 4),
        "comparisons": comparisons,
        "largest_regressions": regressions[:5],
    }


def _percentile_status(current: float, p05: float, p95: float) -> str:
    if current < p05:
        return "below_band"
    if current > p95:
        return "above_band"
    return "within_band"


def _direction_for_feature(name: str) -> str:
    if any(token in name for token in ("true_peak", "clipping", "roughness", "dissonance")):
        return "lower"
    return "higher"


def _drift_monitor(features: dict[str, float], previous: dict[str, float] | None) -> dict[str, Any]:
    if not previous:
        return {"status": "baseline", "changes": []}
    changes = []
    for name, current in sorted(features.items()):
        if name not in previous:
            continue
        delta = round(current - previous[name], 4)
        if abs(delta) < _drift_threshold(name):
            continue
        changes.append({"feature": name, "delta": delta, "previous": previous[name], "current": current})
    return {"status": "drift" if changes else "stable", "changes": changes}


def _drift_threshold(name: str) -> float:
    if "lufs" in name or "snr" in name:
        return 3.0
    if "mos" in name or "p563" in name:
        return 0.35
    return 1.0


def _ci_checks(
    labels: list[dict[str, Any]],
    baseline_comparison: dict[str, Any],
    custom_ci_fail: bool,
) -> list[dict[str, Any]]:
    checks = []
    blocking_labels = [label for label in labels if label.get("severity") == "fail"]
    checks.append(
        {
            "id": "audio_quality_gate",
            "status": "fail" if blocking_labels else "pass",
            "message": (
                f"{len(blocking_labels)} blocking quality label(s) detected"
                if blocking_labels
                else "No blocking quality labels detected"
            ),
        }
    )
    if baseline_comparison:
        checks.append(
            {
                "id": "baseline_regression_gate",
                "status": "fail" if baseline_comparison.get("status") == "regressed" else "pass",
                "message": baseline_comparison.get("status", "no baseline"),
            }
        )
    if custom_ci_fail:
        checks.append(
            {
                "id": "custom_insight_rules_gate",
                "status": "fail",
                "message": "One or more custom insight rules requested CI failure.",
            }
        )
    return checks


def _attach_dataset_audit(results: list[FileResult]) -> None:
    signatures: dict[str, list[FileResult]] = defaultdict(list)
    for result in results:
        signature = result.insights.get("quality_fingerprint", {}).get("signature")
        if signature:
            signatures[signature].append(result)
    # Segments of one recording routinely share a fingerprint; only a match
    # against a different source recording counts as a duplicate.
    duplicate_paths = {
        result.path
        for grouped in signatures.values()
        if len({item.source_file or item.path for item in grouped}) > 1
        for result in grouped
    }
    for result in results:
        issues = []
        if result.path in duplicate_paths:
            issues.append(
                {
                    "id": "duplicate_quality_fingerprint",
                    "severity": "warn",
                    "message": "Another file has the same coarse quality fingerprint.",
                }
            )
        if result.duration_s and result.duration_s < 0.25:
            issues.append(
                {
                    "id": "too_short_for_review",
                    "severity": "warn",
                    "message": "File is shorter than 250 ms and may not be useful for QA.",
                }
            )
        if result.error:
            issues.append(
                {
                    "id": "unreadable_audio",
                    "severity": "error",
                    "message": result.error,
                }
            )
        result.insights["dataset_audit"] = issues


def _attach_segment_heatmaps(results: list[FileResult]) -> None:
    by_source: dict[str, list[FileResult]] = defaultdict(list)
    for result in results:
        if result.segment_index is not None:
            by_source[result.source_file or result.path].append(result)
    for source_file, segments in by_source.items():
        cells = []
        for segment in sorted(segments, key=lambda item: item.segment_index or 0):
            labels = segment.insights.get("defect_labels", [])
            cells.append(
                {
                    "segment_index": segment.segment_index,
                    "start_s": segment.segment_start_s,
                    "end_s": segment.segment_end_s,
                    "status": "warn" if labels else "ok",
                    "labels": [label["id"] for label in labels],
                    "details": [label["evidence"] for label in labels],
                }
            )
        heatmap = {"source_file": source_file, "cells": cells}
        for segment in segments:
            segment.insights["segment_heatmap"] = heatmap


def _attach_triage(results: list[FileResult]) -> None:
    scored = []
    for result in results:
        score = 0.0
        reasons = []
        for label in result.insights.get("defect_labels", []):
            severity = label.get("severity")
            confidence = float(label.get("confidence", 0.0))
            if severity == "fail":
                score += 50.0 * confidence
            elif severity == "warn":
                score += 20.0 * confidence
            else:
                score += 5.0 * confidence
            reasons.append(str(label.get("id", "label")))
        if result.insights.get("baseline_comparison", {}).get("status") == "regressed":
            score += 25.0
            reasons.append("baseline_regression")
        if result.insights.get("drift_monitor", {}).get("status") == "drift":
            score += 15.0
            reasons.append("drift")
        if any(check.get("status") == "fail" for check in result.insights.get("ci_checks", [])):
            score += 20.0
            reasons.append("ci_failure")
        scored.append((result, round(score, 3), reasons))
    ranked = sorted(scored, key=lambda item: (-item[1], item[0].path))
    ranks = {id(result): index for index, (result, _, _) in enumerate(ranked, start=1)}
    for result, score, reasons in scored:
        result.insights["triage"] = {
            "rank": ranks[id(result)],
            "score": score,
            "reasons": reasons[:8],
        }


def _snippet_name(result: FileResult, *, disambiguate: bool = False) -> str:
    source = result.source_file or result.path
    safe_source = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in Path(source).stem)
    if disambiguate:
        safe_source += "-" + hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:8]
    return f"{safe_source}-segment-{result.segment_index or 0}.wav"


def _write_wav(path: Path, audio: Any, sr: int) -> None:
    import wave

    import numpy as np

    data = np.asarray(audio, dtype=np.float32)
    channels = 1
    if data.ndim == 2:
        channels = int(data.shape[0])
        data = data.T.reshape(-1)
    # Same 2**15 scale AudioLoader uses, so 16-bit sources round-trip exactly.
    pcm = np.clip(np.round(data * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())

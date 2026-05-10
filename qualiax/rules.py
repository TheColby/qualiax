"""
Threshold-rule loading and evaluation for qualiax reports.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from .models import DiagnosticEntry, FileResult, MetricResult


@dataclass(frozen=True)
class ThresholdRule:
    metric: str
    min: Optional[float] = None
    max: Optional[float] = None
    equals: Optional[Any] = None
    group: Optional[str] = None
    message: Optional[str] = None
    require_present: bool = False


@dataclass(frozen=True)
class ThresholdViolation:
    file: str
    metric: str
    group: str
    actual: Any
    rule: ThresholdRule
    message: str


@dataclass(frozen=True)
class ThresholdRuleLint:
    level: str                      # warn | error
    message: str
    rule_index: Optional[int] = None


def load_threshold_rules(path: Path) -> list[ThresholdRule]:
    payload = _load_config(path)
    if isinstance(payload, list):
        raw_rules = payload
    elif isinstance(payload, dict) and isinstance(payload.get("rules"), list):
        raw_rules = payload["rules"]
    else:
        raise ValueError("Threshold rule config must be a list or an object with a 'rules' list.")

    rules: list[ThresholdRule] = []
    for index, raw in enumerate(raw_rules, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"Rule #{index} must be an object.")
        metric = raw.get("metric")
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError(f"Rule #{index} is missing a non-empty 'metric' field.")
        rule = ThresholdRule(
            metric=metric.strip(),
            min=_maybe_float(raw.get("min")),
            max=_maybe_float(raw.get("max")),
            equals=raw.get("equals"),
            group=str(raw["group"]).strip() if raw.get("group") is not None else None,
            message=str(raw["message"]).strip() if raw.get("message") else None,
            require_present=bool(raw.get("require_present", False)),
        )
        if rule.min is None and rule.max is None and rule.equals is None and not rule.require_present:
            raise ValueError(
                f"Rule #{index} for metric '{rule.metric}' must define min, max, equals, or require_present."
            )
        rules.append(rule)
    return rules


def apply_threshold_rules(
    results: list[FileResult],
    rules: list[ThresholdRule],
    *,
    unmatched_behavior: str = "ignore",
) -> list[ThresholdViolation]:
    violations: list[ThresholdViolation] = []
    if not rules:
        return violations

    matched_rules: set[int] = set()
    for result in results:
        result_violations = 0
        for idx, rule in enumerate(rules):
            matches = [
                metric
                for metric in result.metrics
                if _normalize_label(metric.name) == _normalize_label(rule.metric)
                and (rule.group is None or _normalize_label(metric.group) == _normalize_label(rule.group))
            ]
            if not matches:
                if rule.require_present:
                    violations.append(
                        ThresholdViolation(
                            file=result.path,
                            metric=rule.metric,
                            group=rule.group or "",
                            actual=None,
                            rule=rule,
                            message=rule.message or f"{rule.metric} is required but missing",
                        )
                    )
                    result_violations += 1
                continue
            matched_rules.add(idx)
            for metric in matches:
                if not _violates(metric, rule):
                    continue
                message = rule.message or _default_message(metric, rule)
                _append_warning(metric, message)
                violations.append(
                    ThresholdViolation(
                        file=result.path,
                        metric=metric.name,
                        group=metric.group,
                        actual=metric.value,
                        rule=rule,
                        message=message,
                    )
                )
                result_violations += 1
        if result_violations:
            note = f"threshold rules: {result_violations} violation(s)"
            if note not in result.notes:
                result.notes.append(note)
            _append_diagnostic(
                result,
                code="threshold_rules_violations",
                severity="warn",
                message=f"{result_violations} threshold rule violation(s)",
            )

    unmatched = [rule for idx, rule in enumerate(rules) if idx not in matched_rules]
    if unmatched_behavior not in {"ignore", "note", "violation", "error"}:
        raise ValueError("unmatched_behavior must be one of ignore, note, violation, or error.")
    if unmatched_behavior in {"note", "violation", "error"} and unmatched:
        message = (
            "threshold rules: unmatched metric names: "
            + ", ".join(sorted({rule.metric for rule in unmatched}))
        )
        for result in results:
            if message not in result.notes:
                result.notes.append(message)
            _append_diagnostic(
                result,
                code="threshold_rules_unmatched",
                severity="warn" if unmatched_behavior != "error" else "error",
                message=message,
            )
    if unmatched_behavior in {"violation", "error"} and unmatched:
        for rule in unmatched:
            violations.append(
                ThresholdViolation(
                    file="*",
                    metric=rule.metric,
                    group=rule.group or "",
                    actual=None,
                    rule=rule,
                    message=f"Rule metric did not match any analyzed metric: {rule.metric}",
                )
            )
    if unmatched_behavior == "error" and unmatched:
        raise ValueError(
            "Threshold rules referenced unknown metrics: "
            + ", ".join(sorted({rule.metric for rule in unmatched}))
        )
    return violations


def lint_threshold_rules(
    rules: list[ThresholdRule],
    *,
    valid_groups: Optional[set[str]] = None,
) -> list[ThresholdRuleLint]:
    issues: list[ThresholdRuleLint] = []
    seen_keys: set[tuple[str, str, Optional[float], Optional[float], Any, bool]] = set()
    for index, rule in enumerate(rules, start=1):
        if (
            rule.min is not None
            and rule.max is not None
            and rule.min > rule.max
        ):
            issues.append(
                ThresholdRuleLint(
                    level="error",
                    message=f"Rule #{index} has min > max for metric '{rule.metric}'.",
                    rule_index=index,
                )
            )
        if valid_groups is not None and rule.group is not None:
            normalized_group = _normalize_label(rule.group)
            normalized_valid = {_normalize_label(group) for group in valid_groups}
            if normalized_group not in normalized_valid:
                issues.append(
                    ThresholdRuleLint(
                        level="error",
                        message=f"Rule #{index} references unknown group '{rule.group}'.",
                        rule_index=index,
                    )
                )
        key = (
            _normalize_label(rule.metric),
            _normalize_label(rule.group or ""),
            rule.min,
            rule.max,
            rule.equals,
            rule.require_present,
        )
        if key in seen_keys:
            issues.append(
                ThresholdRuleLint(
                    level="warn",
                    message=f"Rule #{index} duplicates an earlier rule for '{rule.metric}'.",
                    rule_index=index,
                )
            )
        seen_keys.add(key)
    return issues


def _load_config(path: Path) -> Any:
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix == ".toml":
        try:
            import tomllib
        except ModuleNotFoundError:
            try:
                import tomli as tomllib
            except ModuleNotFoundError as exc:
                raise ValueError("TOML rule files require Python 3.11+ or the optional 'tomli' package.") from exc
        return tomllib.loads(text)
    raise ValueError("Unsupported rules format. Use .json or .toml.")


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _violates(metric: MetricResult, rule: ThresholdRule) -> bool:
    value = metric.value
    if value is None:
        return rule.require_present
    if rule.equals is not None and value != rule.equals:
        return True
    if isinstance(value, (int, float)):
        if rule.min is not None and float(value) < rule.min:
            return True
        if rule.max is not None and float(value) > rule.max:
            return True
    return False


def _default_message(metric: MetricResult, rule: ThresholdRule) -> str:
    checks = []
    if rule.min is not None:
        checks.append(f">= {rule.min:g}")
    if rule.max is not None:
        checks.append(f"<= {rule.max:g}")
    if rule.equals is not None:
        checks.append(f"== {rule.equals}")
    if not checks:
        return f"{metric.name} violated threshold rule"
    return f"{metric.name} should be {' and '.join(checks)}"


def _append_warning(metric: MetricResult, message: str) -> None:
    if metric.warning:
        parts = {part.strip() for part in metric.warning.split(";") if part.strip()}
        if message not in parts:
            metric.warning = metric.warning + f"; {message}"
    else:
        metric.warning = message


def _append_diagnostic(
    result: FileResult,
    *,
    code: str,
    severity: str,
    message: str,
) -> None:
    for diagnostic in result.diagnostics:
        if diagnostic.code == code and diagnostic.message == message:
            diagnostic.count += 1
            return
    result.diagnostics.append(
        DiagnosticEntry(
            code=code,
            severity=severity,
            source="rules",
            message=message,
        )
    )


def _normalize_label(value: str) -> str:
    return " ".join(str(value).strip().lower().split())

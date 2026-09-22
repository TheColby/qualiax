"""
Built-in validation helpers for qualiax reports and scorecards.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import INSIGHTS_SCHEMA, REPORT_SCHEMA, SCORECARD_SCHEMA


@dataclass(frozen=True)
class ValidationIssue:
    path: str
    message: str


def validate_report_payload(payload: Any) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not isinstance(payload, list):
        return [ValidationIssue(path="$", message="Report payload must be a list.")]
    for index, item in enumerate(payload):
        issues.extend(_validate_required(item, REPORT_SCHEMA["items"], path=f"$[{index}]"))
        insights = item.get("insights") if isinstance(item, dict) else None
        if isinstance(insights, dict) and insights:
            # Non-empty insights (from --insights) must satisfy the insight contract.
            issues.extend(validate_insight_payload(insights, path=f"$[{index}].insights"))
    return issues


def validate_insight_payload(payload: Any, *, path: str = "$") -> list[ValidationIssue]:
    return _validate_required(payload, INSIGHTS_SCHEMA, path=path)


def validate_scorecard_payload(payload: Any) -> list[ValidationIssue]:
    return _validate_required(payload, SCORECARD_SCHEMA, path="$")


def validate_report_file(path: Path) -> list[ValidationIssue]:
    payload = _load_payload(path)
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        if not isinstance(payload, list):
            return [ValidationIssue(path="$", message="JSONL payload must load as a list of objects.")]
        return validate_report_payload(payload)
    if isinstance(payload, list):
        return validate_report_payload(payload)
    return validate_scorecard_payload(payload)


def assert_valid_report_payload(payload: Any) -> None:
    issues = validate_report_payload(payload)
    if issues:
        raise ValueError(_format_issues(issues))


def assert_valid_scorecard_payload(payload: Any) -> None:
    issues = validate_scorecard_payload(payload)
    if issues:
        raise ValueError(_format_issues(issues))


def _load_payload(path: Path) -> Any:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _validate_required(payload: Any, schema: dict[str, Any], *, path: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    schema_type = schema.get("type")
    if schema_type == "object":
        if not isinstance(payload, dict):
            return [ValidationIssue(path=path, message="Expected object.")]
        required = schema.get("required", [])
        for key in required:
            if key not in payload:
                issues.append(ValidationIssue(path=f"{path}.{key}", message="Missing required field."))
        for key, subschema in schema.get("properties", {}).items():
            if key not in payload:
                continue
            if isinstance(subschema, dict):
                issues.extend(_validate_type(payload[key], subschema, path=f"{path}.{key}"))
        if schema.get("additionalProperties") is False:
            allowed = set(schema.get("properties", {}))
            for key in payload:
                if key not in allowed:
                    issues.append(ValidationIssue(path=f"{path}.{key}", message="Unexpected field."))
        return issues
    if schema_type == "array":
        if not isinstance(payload, list):
            return [ValidationIssue(path=path, message="Expected array.")]
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(payload):
                issues.extend(_validate_type(item, item_schema, path=f"{path}[{index}]"))
        return issues
    return _validate_type(payload, schema, path=path)


def _validate_type(value: Any, schema: dict[str, Any], *, path: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        if any(_matches_type(value, item_type) for item_type in schema_type):
            if isinstance(value, dict) and schema.get("properties"):
                issues.extend(_validate_required(value, {**schema, "type": "object"}, path=path))
            elif isinstance(value, list) and schema.get("items"):
                issues.extend(_validate_required(value, {**schema, "type": "array"}, path=path))
            return issues
        return [ValidationIssue(path=path, message=f"Expected one of {schema_type!r}.")]
    if schema_type in {"object", "array"}:
        return _validate_required(value, schema, path=path)
    if schema_type and not _matches_type(value, schema_type):
        return [ValidationIssue(path=path, message=f"Expected {schema_type}.")]
    if "const" in schema and value != schema["const"]:
        issues.append(ValidationIssue(path=path, message=f"Expected constant value {schema['const']!r}."))
    if "enum" in schema and value not in schema["enum"]:
        issues.append(ValidationIssue(path=path, message=f"Expected one of {schema['enum']!r}."))
    if schema_type == "integer" and "minimum" in schema and isinstance(value, int) and value < schema["minimum"]:
        issues.append(ValidationIssue(path=path, message=f"Value must be >= {schema['minimum']}."))
    if schema_type == "number" and "minimum" in schema and isinstance(value, (int, float)) and value < schema["minimum"]:
        issues.append(ValidationIssue(path=path, message=f"Value must be >= {schema['minimum']}."))
    if isinstance(value, list):
        min_items = schema.get("minItems")
        max_items = schema.get("maxItems")
        if min_items is not None and len(value) < min_items:
            issues.append(ValidationIssue(path=path, message=f"Array must contain at least {min_items} item(s)."))
        if max_items is not None and len(value) > max_items:
            issues.append(ValidationIssue(path=path, message=f"Array must contain at most {max_items} item(s)."))
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                issues.extend(_validate_type(item, item_schema, path=f"{path}[{index}]"))
    return issues


def _matches_type(value: Any, schema_type: str) -> bool:
    return {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
        "null": value is None,
    }.get(schema_type, True)


def _format_issues(issues: list[ValidationIssue]) -> str:
    return "; ".join(f"{issue.path}: {issue.message}" for issue in issues)

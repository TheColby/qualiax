"""
Published JSON Schema documents for qualiax outputs.
"""
from __future__ import annotations

from copy import deepcopy

from .insights import INSIGHT_SCHEMA_VERSION
from .version import OUTPUT_SCHEMA_VERSION


_DIAGNOSTIC_SCHEMA = {
    "type": "object",
    "required": ["code", "severity", "source", "message", "count", "context"],
    "properties": {
        "code": {"type": "string"},
        "severity": {"type": "string", "enum": ["info", "warn", "error"]},
        "source": {"type": "string"},
        "message": {"type": "string"},
        "group": {"type": ["string", "null"]},
        "metric": {"type": ["string", "null"]},
        "count": {"type": "integer", "minimum": 1},
        "context": {"type": "object"},
    },
    "additionalProperties": False,
}

_GROUP_HEALTH_SCHEMA = {
    "type": "object",
    "required": ["group", "status", "metric_count", "warning_count", "missing_count", "diagnostic_count"],
    "properties": {
        "group": {"type": "string"},
        "status": {"type": "string", "enum": ["ok", "warn", "partial", "missing", "error", "skipped"]},
        "metric_count": {"type": "integer", "minimum": 0},
        "warning_count": {"type": "integer", "minimum": 0},
        "missing_count": {"type": "integer", "minimum": 0},
        "diagnostic_count": {"type": "integer", "minimum": 0},
    },
    "additionalProperties": False,
}

_PROVENANCE_SCHEMA = {
    "type": ["object", "null"],
    "properties": {
        "compute_backend": {"type": ["string", "null"]},
        "python_version": {"type": ["string", "null"]},
        "platform": {"type": ["string", "null"]},
        "runtime_fingerprint": {"type": ["string", "null"]},
        "model_runtime": {"type": ["string", "null"]},
        "model_assets": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "bytes", "sha256"],
                "properties": {
                    "name": {"type": "string"},
                    "file": {"type": "string"},
                    "bytes": {"type": "integer", "minimum": 0},
                    "sha256": {"type": "string"},
                    "verified": {"type": "boolean"},
                    "source": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}

_METRIC_SCHEMA = {
    "type": "object",
    "required": ["name", "value", "unit", "description", "group", "higher_is_better", "warning", "reference_range", "confidence", "calibration_note"],
    "properties": {
        "name": {"type": "string"},
        "value": {},
        "unit": {"type": "string"},
        "description": {"type": "string"},
        "group": {"type": "string"},
        "higher_is_better": {"type": ["boolean", "null"]},
        "warning": {"type": ["string", "null"]},
        "reference_range": {
            "type": ["array", "null"],
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
        },
        "confidence": {"type": ["string", "null"]},
        "calibration_note": {"type": ["string", "null"]},
    },
    "additionalProperties": False,
}

INSIGHTS_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://qualiax.dev/schema/insights.json",
    "title": "qualiax insights payload",
    "type": "object",
    "required": [
        "version",
        "quality_fingerprint",
        "defect_labels",
        "repair_suggestions",
        "ci_checks",
        "triage",
    ],
    "properties": {
        "version": {"type": "string", "const": INSIGHT_SCHEMA_VERSION},
        "quality_fingerprint": {
            "type": "object",
            "required": ["version", "sensitivity", "signature", "features", "buckets"],
            "properties": {
                "version": {"type": "integer"},
                "sensitivity": {"type": "string", "enum": ["coarse", "balanced", "strict"]},
                "signature": {"type": "string"},
                "features": {"type": "object"},
                "buckets": {"type": "object"},
            },
        },
        "defect_labels": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "severity", "confidence", "evidence", "evidence_metrics"],
                "properties": {
                    "id": {"type": "string"},
                    "severity": {"type": "string", "enum": ["info", "warn", "fail"]},
                    "confidence": {"type": ["number", "null"]},
                    "evidence": {"type": "string"},
                    "evidence_metrics": {"type": "array", "items": {"type": "object"}},
                },
            },
        },
        "repair_suggestions": {"type": "array", "items": {"type": "string"}},
        "mos_explanation": {"type": "array", "items": {"type": "object"}},
        "baseline_comparison": {"type": "object"},
        "drift_monitor": {"type": "object"},
        "dataset_audit": {"type": "array", "items": {"type": "object"}},
        "ci_checks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "status", "message"],
                "properties": {
                    "id": {"type": "string"},
                    "status": {"type": "string", "enum": ["pass", "fail"]},
                    "message": {"type": "string"},
                },
            },
        },
        "segment_heatmap": {"type": "object"},
        "triage": {
            "type": "object",
            "required": ["rank", "score", "reasons"],
            "properties": {
                "rank": {"type": "integer", "minimum": 1},
                "score": {"type": "number"},
                "reasons": {"type": "array", "items": {"type": "string"}},
            },
        },
        "snippets": {"type": "array", "items": {"type": "object"}},
    },
    "additionalProperties": False,
}

REPORT_ITEM_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://qualiax.dev/schema/report-item.json",
    "title": "qualiax report item",
    "type": "object",
    "required": [
        "schema_version",
        "tool_version",
        "file",
        "duration_s",
        "sample_rate",
        "channels",
        "notes",
        "confidence_notes",
        "diagnostics",
        "group_health",
        "provenance",
        "insights",
        "error",
        "metrics",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": OUTPUT_SCHEMA_VERSION},
        "tool_version": {"type": "string"},
        "file": {"type": "string"},
        "source_file": {"type": ["string", "null"]},
        "segment_index": {"type": ["integer", "null"]},
        "total_segments": {"type": ["integer", "null"]},
        "segment_start_s": {"type": ["number", "null"]},
        "segment_end_s": {"type": ["number", "null"]},
        "duration_s": {"type": "number"},
        "sample_rate": {"type": "integer"},
        "channels": {"type": "integer"},
        "bit_depth": {"type": ["integer", "null"]},
        "content_type": {"type": ["string", "null"]},
        "speech_confidence": {"type": ["number", "null"]},
        "notes": {"type": "array", "items": {"type": "string"}},
        "confidence_notes": {"type": "array", "items": {"type": "string"}},
        "diagnostics": {"type": "array", "items": _DIAGNOSTIC_SCHEMA},
        "group_health": {"type": "array", "items": _GROUP_HEALTH_SCHEMA},
        "provenance": _PROVENANCE_SCHEMA,
        "insights": {"type": "object"},
        "error": {"type": ["string", "null"]},
        "metrics": {"type": "array", "items": _METRIC_SCHEMA},
    },
    "additionalProperties": False,
}

REPORT_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://qualiax.dev/schema/report.json",
    "title": "qualiax report",
    "type": "array",
    "items": REPORT_ITEM_SCHEMA,
}

SCORECARD_SCHEMA = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://qualiax.dev/schema/scorecard.json",
    "title": "qualiax scorecard",
    "type": "object",
    "required": ["schema_version", "tool_version", "file_count", "status_counts", "confidence_notes", "metric_rollups"],
    "properties": {
        "schema_version": {"type": "string", "const": OUTPUT_SCHEMA_VERSION},
        "tool_version": {"type": "string"},
        "file_count": {"type": "integer", "minimum": 0},
        "status_counts": {"type": "object"},
        "confidence_notes": {"type": "array", "items": {"type": "string"}},
        "metric_rollups": {"type": "array", "items": {"type": "object"}},
        "insight_summary": {"type": "object"},
    },
    "additionalProperties": False,
}


def get_json_schema(name: str) -> dict:
    schemas = {
        "report": REPORT_SCHEMA,
        "report_item": REPORT_ITEM_SCHEMA,
        "scorecard": SCORECARD_SCHEMA,
        "insights": INSIGHTS_SCHEMA,
    }
    key = name.strip().lower()
    if key not in schemas:
        raise ValueError(f"Unknown schema: {name}. Valid schemas: {', '.join(sorted(schemas))}")
    return deepcopy(schemas[key])

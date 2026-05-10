"""qualiax — Perceptual Speech & Audio Quality Analyzer"""

from .api import AnalysisResult, PathLike, analyze, analyze_async, analyze_many, analyze_one
from .contracts import get_json_schema
from .models import DiagnosticEntry, FileResult, GroupHealth, MetricResult, ProvenanceInfo
from .metrics import (
    available_metric_groups,
    register_metric_group,
    unregister_metric_group,
)
from .presets import TaskPreset, available_presets, get_preset, lint_preset
from .rules import ThresholdRuleLint, lint_threshold_rules
from .scorecards import Scorecard, MetricRollup, MetricOutlier, build_scorecard, render_scorecard
from .validation import (
    ValidationIssue,
    assert_valid_report_payload,
    assert_valid_scorecard_payload,
    validate_report_file,
    validate_report_payload,
    validate_scorecard_payload,
)
from .version import __version__, OUTPUT_SCHEMA_VERSION

from . import gpu

def device_info() -> str:
    """Return the active compute device string."""
    return gpu.get_device_str()


__all__ = [
    "__version__",
    "OUTPUT_SCHEMA_VERSION",
    "AnalysisResult",
    "PathLike",
    "DiagnosticEntry",
    "FileResult",
    "GroupHealth",
    "MetricResult",
    "ProvenanceInfo",
    "analyze",
    "analyze_async",
    "analyze_one",
    "analyze_many",
    "available_metric_groups",
    "register_metric_group",
    "unregister_metric_group",
    "TaskPreset",
    "available_presets",
    "get_preset",
    "lint_preset",
    "ThresholdRuleLint",
    "lint_threshold_rules",
    "Scorecard",
    "MetricRollup",
    "MetricOutlier",
    "build_scorecard",
    "render_scorecard",
    "get_json_schema",
    "ValidationIssue",
    "validate_report_payload",
    "validate_scorecard_payload",
    "validate_report_file",
    "assert_valid_report_payload",
    "assert_valid_scorecard_payload",
    "device_info",
]

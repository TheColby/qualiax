"""qualiax — Perceptual Speech & Audio Quality Analyzer"""

from .api import AnalysisResult, PathLike, analyze, analyze_async, analyze_many, analyze_one
from .calibration import (
    CalibrationCase,
    calibrate_labels,
    cases_from_results,
    confidence_interval,
    consistency_report,
    variant_consistency,
)
from .contracts import get_json_schema
from .dataset_intelligence import audit_dataset
from .insights import (
    INSIGHT_SCHEMA_VERSION,
    InsightPayload,
    InsightValidationIssue,
    build_insight_summary_payload,
    enrich_existing_report,
    enrich_results,
    export_flagged_segment_snippets,
    validate_insights_report,
)
from .models import DiagnosticEntry, FileResult, GroupHealth, MetricResult, ProvenanceInfo
from .migrations import (
    ExitCode,
    MigrationRegistry,
    QualiaxDeprecationWarning,
    build_asset_lock,
    load_asset_lock,
    migrate_report,
    verify_asset_lock,
    warn_deprecated,
    write_asset_lock,
)
from .metrics import (
    available_metric_groups,
    register_metric_group,
    unregister_metric_group,
)
from .presets import TaskPreset, available_presets, get_preset, lint_preset
from .operations import (
    AlertPolicy,
    DriftHistory,
    RollingMetricWindow,
    deliver_alert,
    prometheus_metrics,
    send_webhook,
)
from .performance import (
    AnalysisCache,
    LocalExecutorAdapter,
    benchmark_budget,
    incremental_sources,
    iter_audio_chunks,
    snapshot_signatures,
    source_signature,
)
from .plugins import PLUGIN_API_VERSION, PluginManager
from .provenance import model_asset_paths
from .release import dependency_inventory, release_readiness, reproducibility_manifest, write_reproducibility_manifest
from .repair import RepairPlan, apply_repair_plan, build_repair_plan, evaluate_repair
from .review import ReviewStore
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
    "CalibrationCase",
    "DiagnosticEntry",
    "FileResult",
    "GroupHealth",
    "MetricResult",
    "ProvenanceInfo",
    "analyze",
    "analyze_async",
    "analyze_one",
    "analyze_many",
    "calibrate_labels",
    "confidence_interval",
    "consistency_report",
    "cases_from_results",
    "variant_consistency",
    "audit_dataset",
    "available_metric_groups",
    "register_metric_group",
    "unregister_metric_group",
    "ExitCode",
    "MigrationRegistry",
    "migrate_report",
    "warn_deprecated",
    "build_asset_lock",
    "verify_asset_lock",
    "write_asset_lock",
    "load_asset_lock",
    "QualiaxDeprecationWarning",
    "model_asset_paths",
    "TaskPreset",
    "available_presets",
    "get_preset",
    "lint_preset",
    "ThresholdRuleLint",
    "lint_threshold_rules",
    "RollingMetricWindow",
    "DriftHistory",
    "AlertPolicy",
    "prometheus_metrics",
    "send_webhook",
    "deliver_alert",
    "AnalysisCache",
    "LocalExecutorAdapter",
    "source_signature",
    "incremental_sources",
    "iter_audio_chunks",
    "benchmark_budget",
    "snapshot_signatures",
    "PLUGIN_API_VERSION",
    "PluginManager",
    "RepairPlan",
    "build_repair_plan",
    "apply_repair_plan",
    "evaluate_repair",
    "ReviewStore",
    "release_readiness",
    "dependency_inventory",
    "reproducibility_manifest",
    "write_reproducibility_manifest",
    "Scorecard",
    "MetricRollup",
    "MetricOutlier",
    "build_scorecard",
    "render_scorecard",
    "enrich_results",
    "enrich_existing_report",
    "INSIGHT_SCHEMA_VERSION",
    "InsightPayload",
    "InsightValidationIssue",
    "build_insight_summary_payload",
    "export_flagged_segment_snippets",
    "validate_insights_report",
    "get_json_schema",
    "ValidationIssue",
    "validate_report_payload",
    "validate_scorecard_payload",
    "validate_report_file",
    "assert_valid_report_payload",
    "assert_valid_scorecard_payload",
    "device_info",
]


def __getattr__(name: str):
    if name == "DistributedAdapter":
        from .migrations import warn_deprecated
        from .performance import DistributedAdapter

        warn_deprecated(
            "qualiax.DistributedAdapter",
            removal_version="2.0.0",
            alternative="a typing.Protocol in your own code",
            stacklevel=3,
        )
        return DistributedAdapter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

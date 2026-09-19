"""Safe audio-repair planning and before/after evaluation."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .models import FileResult


@dataclass(frozen=True)
class RepairPlan:
    source: Path
    output: Path
    filters: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def ffmpeg_args(self) -> list[str]:
        return ["ffmpeg", "-y", "-i", str(self.source), "-af", ",".join(self.filters), str(self.output)]


def build_repair_plan(result: FileResult, *, output_path: str | Path) -> RepairPlan:
    source = Path(result.source_file or result.path)
    output = Path(output_path)
    if source.expanduser().resolve(strict=False) == output.expanduser().resolve(strict=False):
        raise ValueError("Repair output must not overwrite the source audio.")
    labels = {str(item.get("id")) for item in result.insights.get("defect_labels", [])}
    filters = []
    reasons = []
    if "noisy_floor" in labels:
        filters.append("afftdn=nf=-25")
        reasons.append("reduce broadband noise")
    if labels & {"clipping_risk", "hard_clipping"}:
        filters.append("alimiter=limit=0.89")
        reasons.append("restore true-peak headroom")
    if "under_loud" in labels or "over_loud" in labels:
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
        reasons.append("normalize program loudness")
    if not filters:
        filters.append("anull")
        reasons.append("no repair recommended; create a review copy")
    return RepairPlan(source=source, output=output, filters=tuple(filters), reasons=tuple(reasons))


def apply_repair_plan(plan: RepairPlan, *, dry_run: bool = True) -> dict:
    if dry_run:
        return {"executed": False, "command": plan.ffmpeg_args, "output": str(plan.output)}
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(plan.ffmpeg_args, capture_output=True, text=True, check=False)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg repair failed: {process.stderr.strip()}")
    return {"executed": True, "command": plan.ffmpeg_args, "output": str(plan.output)}


def evaluate_repair(before: FileResult, after: FileResult) -> dict:
    before_metrics = _numeric_metrics(before)
    after_metrics = _numeric_metrics(after)
    improvements = []
    regressions = []
    for name in sorted(before_metrics.keys() & after_metrics.keys()):
        delta = round(after_metrics[name] - before_metrics[name], 6)
        item = {"metric": name, "before": before_metrics[name], "after": after_metrics[name], "delta": delta}
        if _higher_is_better(name):
            (improvements if delta > 0 else regressions if delta < 0 else []).append(item)
        else:
            (improvements if delta < 0 else regressions if delta > 0 else []).append(item)
    return {"improvements": improvements, "regressions": regressions, "accepted": len(regressions) == 0}


def _numeric_metrics(result: FileResult) -> dict[str, float]:
    return {
        metric.name: float(metric.value)
        for metric in result.metrics
        if isinstance(metric.value, (int, float)) and not isinstance(metric.value, bool)
    }


def _higher_is_better(name: str) -> bool:
    lowered = name.lower()
    return not any(token in lowered for token in ("peak", "clipping", "noise floor", "jitter", "shimmer", "roughness"))

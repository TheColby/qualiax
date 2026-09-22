"""Safe audio-repair planning and before/after evaluation.

Repairs are planned from insight defect labels and executed with ffmpeg. Safety
rules: execution is dry-run by default, the output may never be the source file
(checked by resolved path and by inode), and an existing output file is not
replaced unless ``overwrite_output=True``.
"""
from __future__ import annotations

import math
import numbers
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .models import FileResult, MetricResult

# Name fragments used only when a metric does not declare higher_is_better itself.
_LOWER_IS_BETTER_TOKENS = ("peak", "clipping", "clipped", "noise floor", "jitter", "shimmer", "roughness", "dropout")
_HIGHER_IS_BETTER_TOKENS = ("snr", "mos", "pesq", "stoi", "si-sdr", "hnr", "harmonic-to-noise")


@dataclass(frozen=True)
class RepairPlan:
    source: Path
    output: Path
    filters: tuple[str, ...]
    reasons: tuple[str, ...]
    overwrite_output: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "source", Path(self.source))
        object.__setattr__(self, "output", Path(self.output))
        _ensure_distinct(self.source, self.output)
        if not self.filters:
            raise ValueError("RepairPlan requires at least one filter.")

    @property
    def ffmpeg_args(self) -> list[str]:
        return [
            "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-y" if self.overwrite_output else "-n",
            "-i",
            _path_arg(self.source),
            "-af",
            ",".join(self.filters),
            _path_arg(self.output),
        ]


def build_repair_plan(
    result: FileResult,
    *,
    output_path: str | Path,
    loudness_target_lufs: float = -16.0,
    true_peak_dbtp: float = -1.5,
    overwrite_output: bool = False,
) -> RepairPlan:
    """Plan an ffmpeg filter chain for the defect labels in ``result.insights``.

    Filter order matters: declip first (it needs the original clipped samples),
    then denoise, then peak limiting, then loudness normalization. ``loudnorm``
    resamples to 192 kHz internally, so the original sample rate is restored
    afterwards when it is known.
    """
    source = Path(result.source_file or result.path)
    output = Path(output_path)
    _ensure_distinct(source, output)
    labels = {
        str(item.get("id"))
        for item in result.insights.get("defect_labels", [])
        if isinstance(item, dict)
    }
    filters: list[str] = []
    reasons: list[str] = []
    if "hard_clipping" in labels:
        filters.append("adeclip")
        reasons.append("reconstruct clipped samples")
    if "noisy_floor" in labels:
        filters.append("afftdn=nf=-25")
        reasons.append("reduce broadband noise")
    if labels & {"clipping_risk", "hard_clipping"}:
        # level=disabled: alimiter's default auto-level would push peaks back to 0 dBFS.
        limit = min(1.0, max(0.0625, 10 ** (true_peak_dbtp / 20.0)))
        filters.append(f"alimiter=limit={limit:.3f}:level=disabled")
        reasons.append("restore true-peak headroom")
    if labels & {"under_loud", "over_loud"}:
        filters.append(f"loudnorm=I={loudness_target_lufs:g}:TP={true_peak_dbtp:g}:LRA=11")
        reasons.append(f"normalize program loudness to {loudness_target_lufs:g} LUFS")
        if result.sample_rate and result.sample_rate > 0:
            filters.append(f"aresample={int(result.sample_rate)}")
            reasons.append("restore the original sample rate after loudnorm")
    if not filters:
        filters.append("anull")
        reasons.append("no repair recommended; create a review copy")
    return RepairPlan(
        source=source,
        output=output,
        filters=tuple(filters),
        reasons=tuple(reasons),
        overwrite_output=overwrite_output,
    )


def apply_repair_plan(plan: RepairPlan, *, dry_run: bool = True, timeout: float | None = None) -> dict:
    """Run the plan with ffmpeg. Dry-run (the default) only returns the command."""
    if dry_run:
        return {"executed": False, "command": plan.ffmpeg_args, "output": str(plan.output)}
    _ensure_distinct(plan.source, plan.output)
    if not plan.source.exists():
        raise FileNotFoundError(f"Repair source not found: {plan.source}")
    if plan.output.exists() and not plan.overwrite_output:
        raise FileExistsError(f"Refusing to overwrite existing repair output: {plan.output}")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to apply repair plans but was not found on PATH.")
    plan.output.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.run(plan.ffmpeg_args, capture_output=True, text=True, check=False, timeout=timeout)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg repair failed: {process.stderr.strip()}")
    return {"executed": True, "command": plan.ffmpeg_args, "output": str(plan.output)}


def evaluate_repair(
    before: FileResult,
    after: FileResult,
    *,
    targets: Mapping[str, float] | None = None,
    tolerance: float = 1e-6,
) -> dict:
    """Compare metrics before and after a repair.

    Direction comes from each metric's ``higher_is_better`` flag, falling back to
    name heuristics (SNR/MOS up, peak/clipping/noise floor down). Metrics named
    in ``targets`` (for example ``{"Integrated Loudness (LUFS)": -16.0}``) are
    judged by distance to the target. Metrics with no known direction are listed
    as ``neutral`` and never block acceptance. Changes smaller than ``tolerance``
    count as unchanged.
    """
    targets = dict(targets or {})
    before_metrics = _numeric_metrics(before)
    after_metrics = _numeric_metrics(after)
    improvements = []
    regressions = []
    neutral = []
    for name in sorted(before_metrics.keys() & after_metrics.keys()):
        old_metric, old = before_metrics[name]
        new_metric, new = after_metrics[name]
        delta = round(new - old, 6)
        item = {"metric": name, "before": old, "after": new, "delta": delta}
        if name in targets:
            target = float(targets[name])
            gain = abs(old - target) - abs(new - target)
            item["target"] = target
        else:
            direction = _direction(old_metric, new_metric)
            if direction is None:
                if abs(new - old) > tolerance:
                    neutral.append(item)
                continue
            gain = (new - old) if direction else (old - new)
        if gain > tolerance:
            improvements.append(item)
        elif gain < -tolerance:
            regressions.append(item)
    return {
        "improvements": improvements,
        "regressions": regressions,
        "neutral": neutral,
        "missing_after": sorted(before_metrics.keys() - after_metrics.keys()),
        "accepted": not regressions,
    }


def _ensure_distinct(source: Path, output: Path) -> None:
    source_resolved = Path(source).expanduser().resolve(strict=False)
    output_resolved = Path(output).expanduser().resolve(strict=False)
    same = source_resolved == output_resolved
    if not same and source_resolved.exists() and output_resolved.exists():
        # Catches hard links and case-insensitive filesystems.
        same = os.path.samefile(source_resolved, output_resolved)
    if same:
        raise ValueError("Repair output must not overwrite the source audio.")


def _path_arg(path: Path) -> str:
    text = str(path)
    # A leading "-" would be parsed by ffmpeg as an option.
    return f".{os.sep}{text}" if text.startswith("-") else text


def _numeric_metrics(result: FileResult) -> dict[str, tuple[MetricResult, float]]:
    values = {}
    for metric in result.metrics:
        if isinstance(metric.value, numbers.Real) and not isinstance(metric.value, bool):
            number = float(metric.value)
            if math.isfinite(number):
                values[metric.name] = (metric, number)
    return values


def _direction(before: MetricResult, after: MetricResult) -> bool | None:
    for metric in (before, after):
        if metric.higher_is_better is not None:
            return bool(metric.higher_is_better)
    lowered = before.name.lower()
    if any(token in lowered for token in _LOWER_IS_BETTER_TOKENS):
        return False
    if any(token in lowered for token in _HIGHER_IS_BETTER_TOKENS):
        return True
    return None

"""
Built-in task presets for common qualiax workflows.
"""
from __future__ import annotations

from dataclasses import dataclass

from .rules import ThresholdRule, ThresholdRuleLint, lint_threshold_rules


@dataclass(frozen=True)
class TaskPreset:
    name: str
    description: str
    metric_groups: tuple[str, ...]
    rules: tuple[ThresholdRule, ...]


_PRESETS: dict[str, TaskPreset] = {
    "podcast": TaskPreset(
        name="podcast",
        description="Voice-first podcast and interview review.",
        metric_groups=("basic", "loudness", "speech", "perceptual"),
        rules=(
            ThresholdRule(
                metric="Integrated Loudness (LUFS)",
                group="loudness",
                min=-18.0,
                max=-14.0,
                message="Podcast loudness should usually land between -18 and -14 LUFS.",
            ),
            ThresholdRule(
                metric="True Peak",
                group="loudness",
                max=-1.0,
                message="Podcast true peak should stay at or below -1 dBTP.",
            ),
            ThresholdRule(
                metric="Active Speech Level (ASL)",
                group="speech",
                min=-30.0,
                message="Active speech level is very low for spoken-word material.",
            ),
        ),
    ),
    "call-center-qa": TaskPreset(
        name="call-center-qa",
        description="Call-review profile emphasizing speech clarity and background conditions.",
        metric_groups=("basic", "loudness", "temporal", "noise", "speech", "prosody", "speaker", "perceptual"),
        rules=(
            ThresholdRule(
                metric="Estimated SNR",
                group="noise",
                min=12.0,
                message="Estimated SNR is low for call-review workflows.",
            ),
            ThresholdRule(
                metric="Speech/Activity Ratio",
                group="temporal",
                min=25.0,
                message="Speech activity is unusually sparse for a call recording.",
            ),
            ThresholdRule(
                metric="True Peak",
                group="loudness",
                max=-1.0,
                message="Call recordings should avoid hard limiting above -1 dBTP.",
            ),
        ),
    ),
    "speech-enhancement": TaskPreset(
        name="speech-enhancement",
        description="Speech enhancement and denoising evaluation.",
        metric_groups=("basic", "loudness", "noise", "speech", "perceptual", "prosody"),
        rules=(
            ThresholdRule(
                metric="Estimated SNR",
                group="noise",
                min=15.0,
                message="Enhanced speech still has a low estimated SNR.",
            ),
            ThresholdRule(
                metric="Harmonic-to-Noise Ratio (HNR)",
                group="noise",
                min=5.0,
                message="HNR remains low after enhancement.",
            ),
            ThresholdRule(
                metric="P.563 Proxy (NB Quality Estimate)",
                group="perceptual",
                min=2.5,
                message="Single-ended perceptual quality remains weak after enhancement.",
            ),
        ),
    ),
    "music-mastering": TaskPreset(
        name="music-mastering",
        description="Music loudness, dynamics, and spectral-balance review.",
        metric_groups=("basic", "loudness", "spectral", "temporal", "psychoacoustic", "perceptual"),
        rules=(
            ThresholdRule(
                metric="Integrated Loudness (LUFS)",
                group="loudness",
                min=-16.0,
                max=-8.0,
                message="Integrated loudness sits outside the usual modern mastering band.",
            ),
            ThresholdRule(
                metric="True Peak",
                group="loudness",
                max=-1.0,
                message="Mastered deliverables should usually stay at or below -1 dBTP.",
            ),
            ThresholdRule(
                metric="Loudness Range (LRA)",
                group="loudness",
                max=12.0,
                message="Loudness range is unusually wide for a mastered deliverable.",
            ),
        ),
    ),
}


def available_presets() -> list[str]:
    """Return the built-in preset names."""
    return sorted(_PRESETS)


def get_preset(name: str) -> TaskPreset:
    """Return a built-in preset by name."""
    key = name.strip().lower()
    preset = _PRESETS.get(key)
    if preset is None:
        valid = ", ".join(available_presets())
        raise ValueError(f"Unknown preset: {name}. Valid presets: {valid}")
    return preset


def lint_preset(preset: TaskPreset) -> list[ThresholdRuleLint]:
    """Validate a preset's metric groups and embedded threshold rules."""
    from .metrics import available_metric_groups

    valid_groups = set(available_metric_groups())
    issues = lint_threshold_rules(list(preset.rules), valid_groups=valid_groups)
    invalid_groups = [group for group in preset.metric_groups if group not in valid_groups]
    for group in invalid_groups:
        issues.append(
            ThresholdRuleLint(
                level="error",
                message=f"Preset '{preset.name}' references unknown metric group '{group}'.",
            )
        )
    return issues

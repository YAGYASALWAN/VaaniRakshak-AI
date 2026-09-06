"""Validated audit/sampling settings, separate from audio transformation settings."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any

import yaml

from .adapters import normalize_language


@dataclass(frozen=True)
class BiasThresholds:
    categorical_medium: float = 0.20
    categorical_high: float = 0.50
    duration_ratio_medium: float = 1.25
    duration_ratio_high: float = 2.0
    speaker_share_medium: float = 0.10
    speaker_share_high: float = 0.25
    generator_share_medium: float = 0.60
    generator_share_high: float = 0.80
    rms_gap_medium: float = 0.03
    rms_gap_high: float = 0.10
    missing_medium: float = 0.20
    missing_high: float = 0.80

    def __post_init__(self) -> None:
        values = asdict(self)
        for key, value in values.items():
            if not isfinite(value) or value < 0:
                raise ValueError(f"Invalid threshold: {key}")
            if key.endswith("_medium") and value >= values[key.replace("_medium", "_high")]:
                raise ValueError(f"{key} must be less than its high threshold")
            if not key.startswith(("duration_ratio", "rms_gap")) and value > 1:
                raise ValueError(f"{key} must be <= 1")
        if self.duration_ratio_medium <= 1:
            raise ValueError("Duration ratio thresholds must exceed 1")


@dataclass(frozen=True)
class SamplingConfig:
    seed: int = 42
    target_total_hours: float | None = None
    target_hours_per_class: dict[str, float] = field(
        default_factory=lambda: {"bonafide": 1.0, "spoof": 1.0})
    languages: tuple[str, ...] = ()
    target_hours_per_language: dict[str, float] = field(default_factory=dict)
    max_speakers: int = 500
    min_speakers_per_language: int = 2
    max_clips_per_speaker: int = 20
    generator_balancing: bool = True
    max_generator_share: float = 0.60
    balance_tolerance: float = 0.05
    holdout_generators: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if set(self.target_hours_per_class) != {"bonafide", "spoof"}:
            raise ValueError("Provide exactly bonafide and spoof hour targets")
        values = [*self.target_hours_per_class.values(), *self.target_hours_per_language.values()]
        if self.target_total_hours is not None:
            values.append(self.target_total_hours)
        if any(not isfinite(x) or x <= 0 for x in values):
            raise ValueError("Hour targets must be finite and positive")
        for key in ("max_speakers", "min_speakers_per_language", "max_clips_per_speaker"):
            x = getattr(self, key)
            if isinstance(x, bool) or not isinstance(x, int) or x <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 < self.max_generator_share <= 1 or not 0 <= self.balance_tolerance < 1:
            raise ValueError("Invalid generator share or balance tolerance")
        if not isinstance(self.generator_balancing, bool):
            raise ValueError("generator_balancing must be boolean")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("seed must be integer")
        normalized = tuple(normalize_language(x) for x in self.languages)
        if None in normalized or len(set(normalized)) != len(normalized):
            raise ValueError("Languages must be nonempty and unique after normalization")
        object.__setattr__(self, "languages", normalized)
        targets = {normalize_language(k): v for k, v in self.target_hours_per_language.items()}
        if None in targets or len(targets) != len(self.target_hours_per_language):
            raise ValueError("Duplicate or missing language target")
        object.__setattr__(self, "target_hours_per_language", targets)

    def class_seconds(self) -> dict[str, float]:
        """Total-hours target takes precedence; divide it equally across classes."""
        if self.target_total_hours is not None:
            return {label: self.target_total_hours * 1800 for label in ("bonafide", "spoof")}
        return {k: v * 3600 for k, v in self.target_hours_per_class.items()}


def load_settings(path: str | Path) -> tuple[SamplingConfig, BiasThresholds, dict[str, Any]]:
    """Read YAML; reject typos and let dataclasses reject unsupported settings."""
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Settings must be a mapping")
    if set(raw) - {"sampling", "bias_thresholds", "inputs", "output"}:
        raise ValueError("Unknown top-level settings")
    return (SamplingConfig(**raw.get("sampling", {})),
            BiasThresholds(**raw.get("bias_thresholds", {})), raw)

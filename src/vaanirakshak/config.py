"""Load and validate the YAML data-engineering configuration.

Why this module exists
----------------------
Preprocessing constants (16 kHz, 4 s windows, pad vs skip) must live in one
place. If they are copied into scripts, train and evaluation can silently
diverge — which itself can become a dataset shortcut.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

ShortClipPolicy = Literal["pad", "skip"]
PadPosition = Literal["end", "start"]


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    mono: bool = True
    window_seconds: float = 4.0
    hop_seconds: float = 2.0
    output_subtype: str = "PCM_16"


@dataclass(frozen=True)
class SegmentationConfig:
    short_clip_policy: ShortClipPolicy = "pad"
    pad_position: PadPosition = "end"


@dataclass(frozen=True)
class ValidationConfig:
    min_duration_seconds: float = 0.05
    silence_rms_threshold: float = 1.0e-4
    min_non_silent_ratio: float = 0.01
    silence_frame_seconds: float = 0.02


@dataclass(frozen=True)
class SplitConfig:
    train_ratio: float = 0.8
    dev_ratio: float = 0.1
    test_ratio: float = 0.1
    seed: int = 42


@dataclass(frozen=True)
class PathConfig:
    raw_data: Path = Path("data/raw")
    processed_data: Path = Path("data/processed")
    manifests: Path = Path("data/manifests")


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    role: str
    default_label: str | None
    raw_subdir: str
    use_for_training: bool = True
    use_for_hyperparameter_tuning: bool = True


@dataclass(frozen=True)
class AppConfig:
    audio: AudioConfig
    segmentation: SegmentationConfig
    validation: ValidationConfig
    splits: SplitConfig
    paths: PathConfig
    datasets: dict[str, DatasetSpec]
    bona_fide_label: str = "bonafide"
    spoof_label: str = "spoof"

    def dataset_spec(self, name: str) -> DatasetSpec:
        if name not in self.datasets:
            raise KeyError(
                f"Unknown dataset '{name}'. Known datasets: {sorted(self.datasets)}"
            )
        return self.datasets[name]


def _require_mapping(value: Any, key: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"Config key '{key}' must be a mapping, got {type(value)!r}")
    return value


def load_config(path: str | Path) -> AppConfig:
    """Parse ``data_config.yaml`` into frozen dataclasses.

    Inputs
        path: filesystem path to the YAML file.
    Outputs
        AppConfig with typed nested configs.
    Edge cases
        Missing optional keys fall back to dataclass defaults.
        Ratios that do not sum to 1.0 raise ValueError.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("Top-level YAML document must be a mapping")

    audio_raw = _require_mapping(raw.get("audio", {}), "audio")
    seg_raw = _require_mapping(raw.get("segmentation", {}), "segmentation")
    val_raw = _require_mapping(raw.get("validation", {}), "validation")
    split_raw = _require_mapping(raw.get("splits", {}), "splits")
    path_raw = _require_mapping(raw.get("paths", {}), "paths")
    label_raw = _require_mapping(raw.get("labels", {}), "labels")
    datasets_raw = _require_mapping(raw.get("datasets", {}), "datasets")

    audio = AudioConfig(
        sample_rate=int(audio_raw.get("sample_rate", 16000)),
        mono=bool(audio_raw.get("mono", True)),
        window_seconds=float(audio_raw.get("window_seconds", 4.0)),
        hop_seconds=float(audio_raw.get("hop_seconds", 2.0)),
        output_subtype=str(audio_raw.get("output_subtype", "PCM_16")),
    )
    if audio.sample_rate <= 0:
        raise ValueError("audio.sample_rate must be positive")
    if audio.window_seconds <= 0 or audio.hop_seconds <= 0:
        raise ValueError("window_seconds and hop_seconds must be positive")

    policy = str(seg_raw.get("short_clip_policy", "pad"))
    if policy not in {"pad", "skip"}:
        raise ValueError("segmentation.short_clip_policy must be 'pad' or 'skip'")
    pad_position = str(seg_raw.get("pad_position", "end"))
    if pad_position not in {"end", "start"}:
        raise ValueError("segmentation.pad_position must be 'end' or 'start'")

    splits = SplitConfig(
        train_ratio=float(split_raw.get("train_ratio", 0.8)),
        dev_ratio=float(split_raw.get("dev_ratio", 0.1)),
        test_ratio=float(split_raw.get("test_ratio", 0.1)),
        seed=int(split_raw.get("seed", 42)),
    )
    ratio_sum = splits.train_ratio + splits.dev_ratio + splits.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    datasets: dict[str, DatasetSpec] = {}
    for name, spec in datasets_raw.items():
        spec_map = _require_mapping(spec, f"datasets.{name}")
        datasets[str(name)] = DatasetSpec(
            name=str(name),
            role=str(spec_map.get("role", "train_dev")),
            default_label=spec_map.get("default_label"),
            raw_subdir=str(spec_map.get("raw_subdir", name)),
            use_for_training=bool(spec_map.get("use_for_training", True)),
            use_for_hyperparameter_tuning=bool(
                spec_map.get("use_for_hyperparameter_tuning", True)
            ),
        )

    return AppConfig(
        audio=audio,
        segmentation=SegmentationConfig(
            short_clip_policy=policy,  # type: ignore[arg-type]
            pad_position=pad_position,  # type: ignore[arg-type]
        ),
        validation=ValidationConfig(
            min_duration_seconds=float(val_raw.get("min_duration_seconds", 0.05)),
            silence_rms_threshold=float(val_raw.get("silence_rms_threshold", 1.0e-4)),
            min_non_silent_ratio=float(val_raw.get("min_non_silent_ratio", 0.01)),
            silence_frame_seconds=float(val_raw.get("silence_frame_seconds", 0.02)),
        ),
        splits=splits,
        paths=PathConfig(
            raw_data=Path(str(path_raw.get("raw_data", "data/raw"))),
            processed_data=Path(str(path_raw.get("processed_data", "data/processed"))),
            manifests=Path(str(path_raw.get("manifests", "data/manifests"))),
        ),
        datasets=datasets,
        bona_fide_label=str(label_raw.get("bona_fide", "bonafide")),
        spoof_label=str(label_raw.get("spoof", "spoof")),
    )


def default_config_path() -> Path:
    """Return the repository's default YAML path relative to CWD / parents."""
    candidates = [
        Path("configs/data_config.yaml"),
        Path(__file__).resolve().parents[2] / "configs" / "data_config.yaml",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]

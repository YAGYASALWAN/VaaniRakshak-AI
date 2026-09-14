"""Detector adapters for the VaaniRakshak V2 streaming product.

The product engine consumes a small detector contract and does not need to know
how a checkpoint is implemented. This module is the boundary between trained
anti-spoofing models and the live-call system.

A legacy V1 EnglishCNN checkpoint can be connected explicitly for integration
validation, but it is deliberately labelled ``legacy-experimental``. Loading a
legacy checkpoint never upgrades its evidence quality or generalisation claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vaanirakshak.v2_audio import MODEL_SAMPLE_RATE


V2_CHECKPOINT_SCHEMA = "vaanirakshak-v2-detector-v1"
LEGACY_SCHEMA = "english-cnn-v1"
SUPPORTED_FRONTEND = "english-logmel-v1"
MODEL_SAMPLES = 4 * MODEL_SAMPLE_RATE


@dataclass(frozen=True)
class DetectorInfo:
    name: str
    mode: str
    schema: str
    threshold: float
    calibrated_probability: bool
    device: str
    notice: str
    checkpoint: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "schema": self.schema,
            "threshold": round(self.threshold, 6),
            "calibrated_probability": self.calibrated_probability,
            "device": self.device,
            "notice": self.notice,
            "checkpoint": self.checkpoint,
        }


def _validate_threshold(value: Any) -> float:
    threshold = float(value)
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("Checkpoint threshold must be finite and between 0 and 1")
    return threshold


class CheckpointDetector:
    """Adapter from a saved PyTorch anti-spoofing checkpoint to the V2 contract.

    Current adapter support intentionally starts with the existing log-mel + CNN
    architecture so the real inference path can be exercised end-to-end. A new V2
    model can later be added behind this same public interface.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cpu", *, allow_legacy: bool = False):
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Detector checkpoint not found: {path}")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")

        import torch
        from vaanirakshak.english_training import EnglishCNN, Frontend, CACHE_VERSION

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested for the detector but is unavailable")

        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("Detector checkpoint must contain a state dictionary")

        schema = state.get("schema")
        if schema == V2_CHECKPOINT_SCHEMA:
            architecture = state.get("architecture")
            if architecture != LEGACY_SCHEMA:
                raise ValueError(f"Unsupported V2 detector architecture: {architecture!r}")
            mode = "trained"
            calibrated = bool(state.get("calibrated_probability", False))
            notice = str(state.get("notice") or "V2 detector checkpoint. Risk aggregation is not proof of authenticity.")
        elif schema == LEGACY_SCHEMA:
            if not allow_legacy:
                raise ValueError(
                    "Legacy V1 checkpoint refused. Set allow_legacy=True only for product-integration validation."
                )
            architecture = LEGACY_SCHEMA
            mode = "legacy-experimental"
            calibrated = False
            notice = (
                "Legacy V1 EnglishCNN connected only to validate V2 product integration. "
                "Its scores are not a validated V2 authenticity assessment."
            )
        else:
            raise ValueError(f"Unsupported detector checkpoint schema: {schema!r}")

        if state.get("frontend") != CACHE_VERSION or CACHE_VERSION != SUPPORTED_FRONTEND:
            raise ValueError("Checkpoint frontend is incompatible with the V2 adapter")
        if "model" not in state:
            raise ValueError("Checkpoint is missing model weights")

        self.threshold = _validate_threshold(state.get("threshold"))
        self.calibrated_probability = calibrated
        self.mode = mode
        self.name = str(state.get("model_name") or path.parent.name or path.stem)
        self.schema = str(schema)
        self.device = device
        self.notice = notice
        self.checkpoint_path = path

        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))
        self._torch = torch
        self._model = EnglishCNN().to(device).eval()
        self._model.load_state_dict(state["model"])
        self._frontend = Frontend().to(device).eval()

    @property
    def info(self) -> DetectorInfo:
        return DetectorInfo(
            name=self.name,
            mode=self.mode,
            schema=self.schema,
            threshold=self.threshold,
            calibrated_probability=self.calibrated_probability,
            device=self.device,
            notice=self.notice,
            checkpoint=str(self.checkpoint_path),
        )

    def score(self, samples: np.ndarray, sample_rate: int) -> float:
        if sample_rate != MODEL_SAMPLE_RATE:
            raise ValueError(f"Checkpoint detector requires {MODEL_SAMPLE_RATE} Hz audio")
        wave = np.asarray(samples, dtype=np.float32)
        if wave.ndim != 1 or wave.size == 0 or not np.isfinite(wave).all():
            raise ValueError("Detector input must be a finite nonempty mono waveform")

        # Match the four-second representation used by the current checkpoint
        # family. Short final windows are zero-padded; long inputs are truncated.
        fixed = np.zeros(MODEL_SAMPLES, dtype=np.float32)
        count = min(MODEL_SAMPLES, wave.size)
        fixed[:count] = np.clip(wave[:count], -1.0, 1.0)

        tensor = self._torch.from_numpy(fixed).unsqueeze(0).to(self.device)
        with self._torch.inference_mode():
            features = self._frontend(tensor)
            logit = self._model(features)
            score = float(logit.sigmoid().item())
        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Detector produced an invalid score")
        return score

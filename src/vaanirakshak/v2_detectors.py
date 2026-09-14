"""Detector adapters for the VaaniRakshak V2 streaming product.

The product engine consumes one small detector contract and does not need to know
whether the active checkpoint is a legacy log-mel CNN or a V2 SSL model.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from vaanirakshak.v2_audio import MODEL_SAMPLE_RATE
from vaanirakshak.v2_ssl_model import ARCHITECTURE as WAVLM_ARCHITECTURE


V2_CHECKPOINT_SCHEMA = "vaanirakshak-v2-detector-v1"
LEGACY_SCHEMA = "english-cnn-v1"
SUPPORTED_FRONTEND = "english-logmel-v1"
RAW_FRONTEND = "raw-16khz-v1"
MODEL_SAMPLES = 4 * MODEL_SAMPLE_RATE


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class DetectorInfo:
    name: str
    mode: str
    schema: str
    architecture: str
    threshold: float
    calibrated_probability: bool
    device: str
    notice: str
    checkpoint: str | None = None
    checkpoint_sha256: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "schema": self.schema,
            "architecture": self.architecture,
            "threshold": round(self.threshold, 6),
            "calibrated_probability": self.calibrated_probability,
            "device": self.device,
            "notice": self.notice,
            "checkpoint": self.checkpoint,
            "checkpoint_sha256": self.checkpoint_sha256,
        }


def _validate_threshold(value: Any) -> float:
    try:
        threshold = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Checkpoint threshold must be numeric") from exc
    if not np.isfinite(threshold) or not 0.0 < threshold < 1.0:
        raise ValueError("Checkpoint threshold must be finite and between 0 and 1")
    return threshold


class CheckpointDetector:
    """Adapter from a saved PyTorch anti-spoofing checkpoint to the V2 contract."""

    def __init__(self, checkpoint: str | Path, device: str = "cpu", *, allow_legacy: bool = False):
        path = Path(checkpoint).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Detector checkpoint not found: {path}")
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")

        import torch

        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested for the detector but is unavailable")

        checkpoint_sha256 = _file_sha256(path)
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise ValueError("Detector checkpoint must contain a state dictionary")
        if "model" not in state:
            raise ValueError("Checkpoint is missing model weights")

        schema = state.get("schema")
        if schema == V2_CHECKPOINT_SCHEMA:
            architecture = str(state.get("architecture") or "")
            if architecture not in {LEGACY_SCHEMA, WAVLM_ARCHITECTURE}:
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

        self.threshold = _validate_threshold(state.get("threshold"))
        self.calibrated_probability = calibrated
        self.mode = mode
        self.name = str(state.get("model_name") or path.parent.name or path.stem)
        self.schema = str(schema)
        self.architecture = architecture
        self.device = device
        self.notice = notice
        self.checkpoint_path = path
        self.checkpoint_sha256 = checkpoint_sha256
        self._torch = torch
        self._frontend = None

        torch.set_num_threads(min(4, max(1, torch.get_num_threads())))

        if architecture == LEGACY_SCHEMA:
            from vaanirakshak.english_training import EnglishCNN, Frontend, CACHE_VERSION

            if state.get("frontend") != CACHE_VERSION or CACHE_VERSION != SUPPORTED_FRONTEND:
                raise ValueError("Checkpoint frontend is incompatible with the log-mel adapter")
            self._model = EnglishCNN().to(device).eval()
            self._model.load_state_dict(state["model"])
            self._frontend = Frontend().to(device).eval()

        elif architecture == WAVLM_ARCHITECTURE:
            if state.get("frontend") != RAW_FRONTEND:
                raise ValueError("WavLM checkpoint must declare raw-16khz-v1 frontend")
            backbone_config = state.get("backbone_config")
            model_spec = state.get("model_spec")
            if not isinstance(backbone_config, dict) or not isinstance(model_spec, dict):
                raise ValueError("WavLM checkpoint is missing backbone_config or model_spec")
            try:
                from vaanirakshak.v2_ssl_model import WavLMAntiSpoof
                self._model = WavLMAntiSpoof.from_exported_config(backbone_config, model_spec).to(device).eval()
            except ImportError as exc:
                raise RuntimeError(
                    "WavLM checkpoint requires the optional transformers dependency. "
                    "Install requirements-v2-train.txt or transformers."
                ) from exc
            self._model.load_state_dict(state["model"])
        else:  # pragma: no cover - guarded above
            raise AssertionError("unreachable detector architecture")

    @property
    def info(self) -> DetectorInfo:
        return DetectorInfo(
            name=self.name,
            mode=self.mode,
            schema=self.schema,
            architecture=self.architecture,
            threshold=self.threshold,
            calibrated_probability=self.calibrated_probability,
            device=self.device,
            notice=self.notice,
            checkpoint=str(self.checkpoint_path),
            checkpoint_sha256=self.checkpoint_sha256,
        )

    def score(self, samples: np.ndarray, sample_rate: int) -> float:
        if sample_rate != MODEL_SAMPLE_RATE:
            raise ValueError(f"Checkpoint detector requires {MODEL_SAMPLE_RATE} Hz audio")
        wave = np.asarray(samples, dtype=np.float32)
        if wave.ndim != 1 or wave.size == 0 or not np.isfinite(wave).all():
            raise ValueError("Detector input must be a finite nonempty mono waveform")

        fixed = np.zeros(MODEL_SAMPLES, dtype=np.float32)
        count = min(MODEL_SAMPLES, wave.size)
        fixed[:count] = np.clip(wave[:count], -1.0, 1.0)
        tensor = self._torch.from_numpy(fixed).unsqueeze(0).to(self.device)

        with self._torch.inference_mode():
            if self.architecture == LEGACY_SCHEMA:
                features = self._frontend(tensor)
                logit = self._model(features)
            else:
                attention_mask = self._torch.zeros_like(tensor, dtype=self._torch.long)
                attention_mask[:, :count] = 1
                logit = self._model(tensor, attention_mask=attention_mask)
            score = float(logit.sigmoid().item())

        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("Detector produced an invalid score")
        return score

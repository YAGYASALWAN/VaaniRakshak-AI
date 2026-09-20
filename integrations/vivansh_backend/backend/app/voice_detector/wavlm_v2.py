"""Drop-in VaaniRakshak V2 WavLM adapter for Vivansh backend.

This file implements backend.app.voice_detector.base.VoiceDetector and loads the
portable V2 checkpoint produced by YAGYASALWAN/VaaniRakshak-AI.

Expected checkpoint:
  schema       = vaanirakshak-v2-detector-v1
  architecture = wavlm-attention-v1
  frontend     = raw-16khz-v1

The model is reconstructed entirely from config stored inside best.pt; no
Hugging Face download is required at inference time.
"""
from __future__ import annotations

from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
import threading
from typing import Any
import wave

from app.contracts.events import WindowVerdict
from app.voice_detector.base import DetectorError, DetectorNotReady, VoiceDetector

RATE = 16_000
SAMPLES = 4 * RATE
SCHEMA = "vaanirakshak-v2-detector-v1"
ARCHITECTURE = "wavlm-attention-v1"
FRONTEND = "raw-16khz-v1"
CALIBRATION_METHOD = "temperature-scaling-v1"


@dataclass(frozen=True)
class _Spec:
    backbone_id: str = "microsoft/wavlm-base-plus"
    attention_hidden: int = 128
    classifier_hidden: int = 256
    dropout: float = 0.20


def _confidence(score: float, threshold: float, calibrated: bool) -> float:
    if calibrated:
        return round(min(1.0, abs(score - 0.5) * 2.0), 4)
    span = max(threshold, 1.0 - threshold)
    if span <= 0:
        return 0.0
    return round(min(1.0, abs(score - threshold) / span), 4)


def _wav_to_float_window(audio: bytes):
    try:
        with wave.open(io.BytesIO(audio), "rb") as handle:
            channels = handle.getnchannels()
            width = handle.getsampwidth()
            rate = handle.getframerate()
            raw = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError) as exc:
        raise DetectorError(f"window payload is not a readable WAV: {exc}") from None

    if channels != 1 or width != 2:
        raise DetectorError(
            "V2 window payload must be mono 16-bit PCM; "
            f"got {channels} channel(s), {width * 8}-bit"
        )
    if rate != RATE:
        raise DetectorError(f"V2 window payload must be {RATE} Hz; got {rate} Hz")

    import numpy as np

    samples = np.frombuffer(raw[: len(raw) - (len(raw) % 2)], dtype="<i2")
    wave_float = samples.astype(np.float32) / 32768.0
    fixed = np.zeros(SAMPLES, dtype=np.float32)
    count = min(SAMPLES, wave_float.size)
    fixed[:count] = np.clip(wave_float[:count], -1.0, 1.0)
    return fixed, count


class _WavLMAntiSpoof:
    """Lazy model factory kept private so importing the backend stays light."""

    @staticmethod
    def build(backbone_config: dict[str, Any], model_spec: dict[str, Any]):
        try:
            import torch
            from torch import nn
            from transformers import WavLMConfig, WavLMModel
        except ImportError as exc:
            raise DetectorNotReady(
                "V2 WavLM inference requires torch and transformers. "
                "Install transformers in the backend environment."
            ) from exc

        spec = _Spec(**model_spec)
        backbone = WavLMModel(WavLMConfig.from_dict(backbone_config))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.backbone = backbone
                hidden_size = int(backbone.config.hidden_size)
                self.attention = nn.Sequential(
                    nn.Linear(hidden_size, spec.attention_hidden),
                    nn.Tanh(),
                    nn.Linear(spec.attention_hidden, 1),
                )
                self.classifier = nn.Sequential(
                    nn.LayerNorm(hidden_size),
                    nn.Dropout(spec.dropout),
                    nn.Linear(hidden_size, spec.classifier_hidden),
                    nn.GELU(),
                    nn.Dropout(spec.dropout),
                    nn.Linear(spec.classifier_hidden, 1),
                )

            def forward(self, input_values, attention_mask=None):
                output = self.backbone(
                    input_values=input_values,
                    attention_mask=attention_mask,
                    return_dict=True,
                )
                hidden = output.last_hidden_state
                logits = self.attention(hidden).squeeze(-1)
                feature_mask = None
                if attention_mask is not None:
                    feature_mask = self.backbone._get_feature_vector_attention_mask(
                        hidden.shape[1], attention_mask
                    )
                    logits = logits.masked_fill(
                        ~feature_mask, torch.finfo(logits.dtype).min
                    )
                weights = torch.softmax(logits, dim=1)
                if feature_mask is not None:
                    weights = weights * feature_mask.to(weights.dtype)
                    weights = weights / weights.sum(
                        dim=1, keepdim=True
                    ).clamp_min(1e-8)
                pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
                return self.classifier(pooled).squeeze(-1)

        return Model()


class WavLMV2Detector(VoiceDetector):
    """Run the portable V2 WavLM checkpoint behind the backend detector seam.

    One instance can safely be shared across sessions: PyTorch inference is
    serialized with a lock. This avoids loading a ~379 MB checkpoint once per
    active call.
    """

    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        self.path = Path(checkpoint).expanduser().resolve()
        self.device = str(device).lower()
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        self._model = None
        self._torch = None
        self._threshold: float | None = None
        self._temperature = 1.0
        self._calibrated = False
        self._notice = ""
        self._model_name = self.path.parent.name or self.path.stem
        self._evaluation = self._load_evaluation()
        self._lock = threading.Lock()

    def status(self) -> dict[str, Any]:
        return {
            "ready": self.path.is_file(),
            "model": f"wavlm-v2/{self._model_name}" if self.path.is_file() else None,
            "kind": "trained_checkpoint",
            "architecture": ARCHITECTURE,
            "checkpoint": str(self.path),
            "device": self.device,
            "loaded": self._model is not None,
            "threshold": self._threshold,
            "calibrated": self._calibrated,
            "score_semantics": (
                "calibrated_probability"
                if self._calibrated
                else "uncalibrated_spoof_score"
            ),
            "evaluation": self._evaluation,
            "notice": self._notice
            or "Higher synthetic_score means more likely synthetic. "
            "Uncalibrated scores are not probabilities.",
            "scripted": False,
        }

    def analyze_window(
        self,
        start_s: float,
        end_s: float,
        audio: bytes | None = None,
    ) -> WindowVerdict:
        if not audio:
            raise DetectorError(
                "the V2 WavLM detector requires audio bytes for each window"
            )
        self._ensure_loaded()
        fixed, count = _wav_to_float_window(audio)

        torch = self._torch
        tensor = torch.from_numpy(fixed).unsqueeze(0).to(self.device)
        mask = torch.zeros_like(tensor, dtype=torch.long)
        mask[:, :count] = 1

        with self._lock, torch.inference_mode():
            logit = self._model(tensor, attention_mask=mask)
            if self._calibrated:
                logit = logit / self._temperature
            score = float(logit.sigmoid().item())

        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise DetectorError("V2 detector returned an invalid score")

        threshold = self._threshold if self._threshold is not None else 0.5
        return WindowVerdict(
            start_s=start_s,
            end_s=end_s,
            synthetic_score=score,
            confidence=_confidence(score, threshold, self._calibrated),
            # MeasuredQualityDetector replaces this with measured audio quality.
            audio_quality=0.5,
            model_id=f"wavlm-v2/{self._model_name}",
        )

    def close(self) -> None:
        # Shared instance: intentionally keep the model loaded for the process
        # lifetime. The prototype owns the detector, not individual sessions.
        return None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.path.is_file():
            raise DetectorNotReady(f"no V2 checkpoint at {self.path}")

        try:
            import torch
        except ImportError:
            raise DetectorNotReady("torch is not installed") from None

        if self.device == "cuda" and not torch.cuda.is_available():
            raise DetectorNotReady("CUDA was requested but torch cannot see a CUDA GPU")

        try:
            state = torch.load(self.path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise DetectorNotReady(f"could not load V2 checkpoint: {exc}") from exc

        if not isinstance(state, dict):
            raise DetectorError("V2 checkpoint must contain a state dictionary")
        if state.get("schema") != SCHEMA:
            raise DetectorError(
                f"checkpoint schema {state.get('schema')!r} is not {SCHEMA!r}"
            )
        if state.get("architecture") != ARCHITECTURE:
            raise DetectorError(
                "checkpoint is not the V2 WavLM attention architecture"
            )
        if state.get("frontend") != FRONTEND:
            raise DetectorError("V2 checkpoint frontend must be raw-16khz-v1")

        try:
            threshold = float(state["threshold"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DetectorError("V2 checkpoint has no valid threshold") from exc
        if not math.isfinite(threshold) or not 0.0 < threshold < 1.0:
            raise DetectorError("V2 checkpoint threshold must be in (0, 1)")

        backbone_config = state.get("backbone_config")
        model_spec = state.get("model_spec")
        if not isinstance(backbone_config, dict) or not isinstance(model_spec, dict):
            raise DetectorError(
                "V2 checkpoint is missing exported WavLM architecture metadata"
            )

        calibrated = bool(state.get("calibrated_probability", False))
        temperature = 1.0
        if calibrated:
            calibration = state.get("calibration")
            if not isinstance(calibration, dict):
                raise DetectorError("calibrated V2 checkpoint has no calibration metadata")
            if calibration.get("method") != CALIBRATION_METHOD:
                raise DetectorError(
                    f"unsupported V2 calibration method: {calibration.get('method')!r}"
                )
            try:
                temperature = float(calibration["temperature"])
            except (KeyError, TypeError, ValueError) as exc:
                raise DetectorError("invalid V2 calibration temperature") from exc
            if not math.isfinite(temperature) or temperature <= 0.0:
                raise DetectorError("V2 calibration temperature must be positive")

        model = _WavLMAntiSpoof.build(backbone_config, model_spec)
        model.load_state_dict(state["model"])
        model = model.to(self.device).eval()

        self._torch = torch
        self._model = model
        self._threshold = threshold
        self._calibrated = calibrated
        self._temperature = temperature
        self._notice = str(
            state.get("notice")
            or "VaaniRakshak V2 WavLM detector. Scores are not proof of authenticity."
        )
        self._model_name = str(
            state.get("model_name") or self.path.parent.name or self.path.stem
        )

    def _load_evaluation(self) -> dict[str, Any] | None:
        path = self.path.parent / "evaluation.json"
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

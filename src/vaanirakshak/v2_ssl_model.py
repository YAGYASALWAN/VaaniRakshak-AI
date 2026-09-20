"""Self-supervised speech backbone for VaaniRakshak V2.

This model uses WavLM hidden states plus learnable attention pooling and a compact
binary anti-spoofing head. Training may initialise the backbone from Hugging Face,
but exported V2 checkpoints contain the WavLM config and full state dict so live
inference does not need to download the backbone again.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


ARCHITECTURE = "wavlm-attention-v1"
DEFAULT_BACKBONE = "microsoft/wavlm-base-plus"


@dataclass(frozen=True)
class WavLMSpec:
    backbone_id: str = DEFAULT_BACKBONE
    attention_hidden: int = 128
    classifier_hidden: int = 256
    dropout: float = 0.20

    def as_dict(self) -> dict[str, Any]:
        return {
            "backbone_id": self.backbone_id,
            "attention_hidden": self.attention_hidden,
            "classifier_hidden": self.classifier_hidden,
            "dropout": self.dropout,
        }


class WavLMAntiSpoof(nn.Module):
    """WavLM encoder + attentive temporal pooling + binary logit head."""

    def __init__(self, backbone: nn.Module, spec: WavLMSpec):
        super().__init__()
        self.backbone = backbone
        self.spec = spec
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

    @classmethod
    def from_pretrained(
        cls,
        backbone_id: str = DEFAULT_BACKBONE,
        *,
        attention_hidden: int = 128,
        classifier_hidden: int = 256,
        dropout: float = 0.20,
        freeze_feature_encoder: bool = True,
    ) -> "WavLMAntiSpoof":
        from transformers import WavLMModel

        backbone = WavLMModel.from_pretrained(backbone_id)
        if freeze_feature_encoder:
            backbone.freeze_feature_encoder()
        return cls(
            backbone,
            WavLMSpec(
                backbone_id=backbone_id,
                attention_hidden=attention_hidden,
                classifier_hidden=classifier_hidden,
                dropout=dropout,
            ),
        )

    @classmethod
    def from_exported_config(cls, config: dict, spec: dict) -> "WavLMAntiSpoof":
        from transformers import WavLMConfig, WavLMModel

        backbone_config = WavLMConfig.from_dict(config)
        backbone = WavLMModel(backbone_config)
        return cls(backbone, WavLMSpec(**spec))

    def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if input_values.ndim != 2:
            raise ValueError("input_values must have shape [batch, samples]")

        output = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
            return_dict=True,
        )
        hidden = output.last_hidden_state
        logits = self.attention(hidden).squeeze(-1)

        feature_mask = None
        if attention_mask is not None:
            feature_mask = self.backbone._get_feature_vector_attention_mask(hidden.shape[1], attention_mask)
            logits = logits.masked_fill(~feature_mask, torch.finfo(logits.dtype).min)

        weights = torch.softmax(logits, dim=1)
        if feature_mask is not None:
            weights = weights * feature_mask.to(weights.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        return self.classifier(pooled).squeeze(-1)

    def export_model_config(self) -> dict[str, Any]:
        return {
            "architecture": ARCHITECTURE,
            "backbone_config": self.backbone.config.to_dict(),
            "model_spec": self.spec.as_dict(),
        }

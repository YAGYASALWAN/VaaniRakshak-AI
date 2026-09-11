"""Recording-level VaaniRakshak model built from 4-second acoustic windows.

Design goal
-----------
A source recording remains ONE training example even when it is split into many
4-second windows. This prevents a long synthetic recording from contributing
more labels/loss terms than a short human recording.

The model therefore performs:
    recording -> 4 s windows -> shared CNN encoder -> window embeddings
              -> masked mean pooling -> one recording logit

This module is intentionally parallel to the current centre-window MLAAD
pipeline so the ongoing feature-preparation run is not changed.
"""

from __future__ import annotations

import io
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from torch import nn
import torchaudio

from vaanirakshak.english_training import Frontend

RATE = 16_000
WINDOW_SECONDS = 4
WINDOW_SAMPLES = RATE * WINDOW_SECONDS
MAX_RECORDING_SECONDS = 120


def recording_windows(raw: bytes, *, max_seconds: int = MAX_RECORDING_SECONDS) -> np.ndarray:
    """Return up to 120 s of audio as consecutive 4-second mono 16 kHz windows.

    The return value has shape ``[num_windows, 64000]``. The final partial
    window is zero-padded. A 120-second recording therefore yields 30 windows,
    but it still remains one recording-level example for the classifier.
    """
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("Expected nonempty encoded audio bytes")

    with sf.SoundFile(io.BytesIO(raw)) as audio:
        rate = int(audio.samplerate)
        channels = int(audio.channels)
        frames = int(len(audio))
        if channels not in (1, 2):
            raise ValueError("Audio must be mono or stereo")
        if not 8_000 <= rate <= 96_000:
            raise ValueError("Audio sample rate must be between 8 and 96 kHz")
        if frames <= 0:
            raise ValueError("Audio must be nonempty")

        # Read at most the first max_seconds. Keeping a hard cap bounds memory
        # and gives every recording the same maximum temporal opportunity.
        read_frames = min(frames, rate * max_seconds)
        wave = audio.read(read_frames, dtype="float32", always_2d=True).mean(axis=1)

    if not np.isfinite(wave).all():
        raise ValueError("Audio contains non-finite samples")
    if np.sqrt(np.mean(wave**2)) < 1e-5:
        raise ValueError("Audio is effectively silent")

    tensor = torch.from_numpy(wave)
    if rate != RATE:
        tensor = torchaudio.functional.resample(tensor, rate, RATE)

    if tensor.numel() > RATE * max_seconds:
        tensor = tensor[: RATE * max_seconds]

    count = max(1, (tensor.numel() + WINDOW_SAMPLES - 1) // WINDOW_SAMPLES)
    padded = torch.nn.functional.pad(tensor, (0, count * WINDOW_SAMPLES - tensor.numel()))
    return padded.reshape(count, WINDOW_SAMPLES).numpy().astype(np.float32, copy=False)


def window_features(raw: bytes, device: str = "cuda") -> torch.Tensor:
    """Convert one recording into ``[windows, 64, 401]`` log-mel features."""
    waves = recording_windows(raw)
    frontend = Frontend().to(device).eval()
    with torch.no_grad():
        return frontend(torch.from_numpy(waves).to(device)).cpu()


class WindowEncoder(nn.Module):
    """Shared CNN that converts one 4-second log-mel window into a 128-D vector."""

    def __init__(self):
        super().__init__()
        layers: list[nn.Module] = []
        for a, b in ((1, 16), (16, 32), (32, 64), (64, 128)):
            layers += [
                nn.Conv2d(a, b, 3, padding=1),
                nn.BatchNorm2d(b),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ]
        self.network = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError("Expected [windows, mel_bins, frames]")
        return self.network(features.unsqueeze(1))


class RecordingLevelCNN(nn.Module):
    """Classify a whole recording while keeping one loss term per recording.

    Input shape: ``[batch, windows, 64, 401]``
    Mask shape:  ``[batch, windows]`` where True marks a real window.

    Windows are encoded independently with shared weights. Their embeddings are
    then pooled inside each recording, producing exactly one recording logit.
    """

    def __init__(self):
        super().__init__()
        self.encoder = WindowEncoder()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, 1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError("Expected [batch, windows, mel_bins, frames]")
        if mask.shape != features.shape[:2]:
            raise ValueError("Mask must match [batch, windows]")

        batch, windows, mel_bins, frames = features.shape
        flat = features.reshape(batch * windows, mel_bins, frames)
        embeddings = self.encoder(flat).reshape(batch, windows, 128)

        weights = mask.to(embeddings.dtype).unsqueeze(-1)
        denominator = weights.sum(dim=1).clamp_min(1.0)
        recording_embedding = (embeddings * weights).sum(dim=1) / denominator
        return self.classifier(self.dropout(recording_embedding)).squeeze(1)


def collate_recordings(batch: Iterable[tuple[torch.Tensor, float]]):
    """Pad a batch of variable-window recordings without changing label weight.

    Each item is ``(features, label)`` with features shaped ``[windows,64,401]``.
    The returned target contains one label per source recording, not per window.
    """
    batch = list(batch)
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    max_windows = max(features.shape[0] for features, _ in batch)
    mel_bins, frames = batch[0][0].shape[1:]
    values = torch.zeros(len(batch), max_windows, mel_bins, frames, dtype=torch.float32)
    mask = torch.zeros(len(batch), max_windows, dtype=torch.bool)
    labels = torch.empty(len(batch), dtype=torch.float32)

    for index, (features, label) in enumerate(batch):
        if features.ndim != 3 or features.shape[1:] != (mel_bins, frames):
            raise ValueError("All recordings must contain compatible window features")
        n = features.shape[0]
        values[index, :n] = features.float()
        mask[index, :n] = True
        labels[index] = float(label)

    return values, mask, labels

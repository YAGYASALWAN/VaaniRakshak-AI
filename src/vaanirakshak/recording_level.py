"""Recording-level VaaniRakshak model built from overlapping acoustic windows.

Design goals
------------
1. A source recording remains ONE training example even when it is split into many
   windows. This prevents a long synthetic recording from contributing more labels
   and loss terms than a short human one.
2. Every window still gets its own score. VaaniRakshak has to say *when* a call
   turned suspicious, not only whether it was, so the classifier is applied before
   the pooling step rather than after it. The recording-level loss is unchanged and
   the per-window series comes out for free.

The model therefore performs:
    recording -> windows -> shared CNN encoder -> window embeddings
              -> per-window logits -> masked pooling -> one recording logit

Window geometry comes from ``configs/data_config.yaml`` (4 s window, 2 s hop by
default) rather than being hard-coded here, so this module, the preprocessing
pipeline and any future streaming detector cut audio the same way.

This module is intentionally parallel to the current centre-window MLAAD pipeline
so the ongoing feature-preparation run is not changed.
"""

from __future__ import annotations

from functools import lru_cache
import io
import math
from typing import Iterable

import numpy as np
import soundfile as sf
import torch
from torch import nn
import torchaudio

from vaanirakshak.config import default_config_path, load_config
from vaanirakshak.english_training import Frontend

MAX_RECORDING_SECONDS = 120
#: Windows whose audio is mostly zero padding are dropped. A trailing window that is
#: 95% digital silence is not evidence about the speaker, and because the two classes
#: can have different duration distributions, the *number* of such windows is exactly
#: the kind of duration shortcut ``data/audit.py`` exists to catch.
MIN_COVERAGE = 0.5
POOLING_MODES = ("mean", "logsumexp")


@lru_cache(maxsize=1)
def windowing():
    """(sample_rate, window_samples, hop_samples) from the project configuration."""
    audio = load_config(default_config_path()).audio
    rate = int(audio.sample_rate)
    return rate, int(round(audio.window_seconds * rate)), int(round(audio.hop_seconds * rate))


def window_bounds(total_samples, window_samples, hop_samples):
    """Start offsets covering ``total_samples``, and each window's real-audio share.

    The final window is zero-padded rather than dropped, matching the configured
    ``short_clip_policy: pad``; its coverage says how much of it is real.
    """
    if total_samples <= 0 or window_samples <= 0 or hop_samples <= 0:
        raise ValueError("Window geometry requires positive lengths")
    count = 1 if total_samples <= window_samples else math.ceil((total_samples - window_samples) / hop_samples) + 1
    starts = [index * hop_samples for index in range(count)]
    coverage = [min(max(total_samples - start, 0), window_samples) / window_samples for start in starts]
    return starts, coverage


def recording_windows(raw: bytes, *, max_seconds: int = MAX_RECORDING_SECONDS,
                      min_coverage: float = MIN_COVERAGE) -> tuple[np.ndarray, np.ndarray]:
    """Return up to ``max_seconds`` of audio as mono 16 kHz windows plus coverage.

    The windows have shape ``[num_windows, window_samples]`` and coverage has shape
    ``[num_windows]``, giving each window's fraction of real (non-padded) audio.
    Windows below ``min_coverage`` are dropped, except that a recording always keeps
    at least one window. However many windows come back, the recording remains one
    example for the classifier.
    """
    if not isinstance(raw, bytes) or not raw:
        raise ValueError("Expected nonempty encoded audio bytes")
    rate_out, window_samples, hop_samples = windowing()

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
    if rate != rate_out:
        tensor = torchaudio.functional.resample(tensor, rate, rate_out)
    tensor = tensor[: rate_out * max_seconds]

    starts, coverage = window_bounds(tensor.numel(), window_samples, hop_samples)
    keep = [index for index, share in enumerate(coverage) if share >= min_coverage] or [0]
    padded = torch.nn.functional.pad(tensor, (0, max(starts) + window_samples - tensor.numel()))
    windows = torch.stack([padded[starts[index]: starts[index] + window_samples] for index in keep])
    return (windows.numpy().astype(np.float32, copy=False),
            np.array([coverage[index] for index in keep], dtype=np.float32))


def make_window_features(device: str = "cuda"):
    """Build one frontend and reuse it, mirroring ``mlaad_training.feature_transform``.

    Constructing a ``Frontend`` per recording costs a full module build and device
    transfer for every item of a corpus-scale preparation run.
    """
    frontend = Frontend().to(device).eval()

    def transform(raw: bytes) -> tuple[torch.Tensor, torch.Tensor]:
        """One recording as ``([windows, mel_bins, frames], [windows])`` coverage."""
        waves, coverage = recording_windows(raw)
        with torch.no_grad():
            features = frontend(torch.from_numpy(waves).to(device)).cpu()
        return features, torch.from_numpy(coverage)

    return transform


class WindowEncoder(nn.Module):
    """Shared CNN that converts one log-mel window into a 128-D vector.

    Deliberately layer-for-layer identical to ``EnglishCNN``'s convolutional trunk,
    including parameter names, so a trained checkpoint can be warm-started into it
    rather than retrained from scratch. See ``warm_start_encoder``.
    """

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


def masked_pool(window_logits: torch.Tensor, weights: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """Combine per-window logits into one recording logit, ignoring padded windows.

    ``mean`` treats every second of the call alike. ``logsumexp`` is a smooth
    maximum, so a short cloned passage inside an otherwise ordinary call can still
    carry the recording - which is the shape of the attack VaaniRakshak is built
    for, and the reason a plain mean is a poor default once calls get long.
    """
    if mode not in POOLING_MODES:
        raise ValueError(f"Unknown pooling mode: {mode!r}")
    weights = weights.to(window_logits.dtype)
    if window_logits.shape != weights.shape:
        raise ValueError("Weights must match [batch, windows]")
    denominator = weights.sum(dim=1).clamp_min(1e-6)
    if mode == "mean":
        return (window_logits * weights).sum(dim=1) / denominator
    masked = window_logits.masked_fill(weights <= 0, float("-inf"))
    return torch.logsumexp(masked, dim=1) - denominator.log()


class RecordingLevelCNN(nn.Module):
    """Classify a whole recording while keeping one loss term per recording.

    Input shape:  ``[batch, windows, mel_bins, frames]``
    Weight shape: ``[batch, windows]``, the real-audio share of each window; 0 marks
    a padded slot added by the collate function.

    Windows are encoded with shared weights and scored individually, then the window
    logits are pooled into exactly one recording logit. ``forward(..., per_window=True)``
    also returns the window logits, which are the rolling score and the suspicious
    timestamps the product needs.
    """

    def __init__(self, pooling: str = "mean"):
        super().__init__()
        if pooling not in POOLING_MODES:
            raise ValueError(f"Unknown pooling mode: {pooling!r}")
        self.pooling = pooling
        self.encoder = WindowEncoder()
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, 1)

    def forward(self, features: torch.Tensor, weights: torch.Tensor, per_window: bool = False):
        if features.ndim != 4:
            raise ValueError("Expected [batch, windows, mel_bins, frames]")
        if weights.shape != features.shape[:2]:
            raise ValueError("Weights must match [batch, windows]")

        batch, windows, mel_bins, frames = features.shape
        flat = features.reshape(batch * windows, mel_bins, frames)
        valid = weights.reshape(-1) > 0
        if not bool(valid.any()):
            raise ValueError("Every recording needs at least one real window")
        # Encode only real windows. Padded slots must not reach the encoder at all:
        # in training mode BatchNorm would otherwise fold their zeros into the batch
        # statistics, so how many recordings a batch happened to pad would change
        # the scores of the recordings it did not.
        if bool(valid.all()):
            embeddings = self.encoder(flat)
        else:
            embeddings = flat.new_zeros(batch * windows, 128)
            embeddings[valid] = self.encoder(flat[valid])
        embeddings = embeddings.reshape(batch, windows, 128)
        window_logits = self.classifier(self.dropout(embeddings)).squeeze(-1)
        pooled = masked_pool(window_logits, weights, self.pooling)
        return (pooled, window_logits) if per_window else pooled


def warm_start_encoder(encoder: WindowEncoder, checkpoint, map_location: str = "cpu") -> list[str]:
    """Copy a trained ``EnglishCNN`` convolutional trunk into a window encoder.

    The two trunks share parameter names, so the existing ASVspoof/MLAAD checkpoint
    is a free initialisation for the recording-level model instead of training from
    scratch. Returns the parameter names that were copied, and raises if none were,
    so a silent no-op cannot pass for a warm start.
    """
    state = torch.load(checkpoint, map_location=map_location, weights_only=True)
    if state.get("schema") != "english-cnn-v1":
        raise ValueError("Not an English CNN checkpoint")
    target = encoder.state_dict()
    shared = {key: value for key, value in state["model"].items()
              if key in target and target[key].shape == value.shape}
    if not shared:
        raise ValueError("Checkpoint shares no parameters with the window encoder")
    encoder.load_state_dict(shared, strict=False)
    return sorted(shared)


def collate_recordings(batch: Iterable[tuple[torch.Tensor, torch.Tensor, float]]):
    """Pad a batch of variable-window recordings without changing label weight.

    Each item is ``(features, coverage, label)`` with features shaped
    ``[windows, mel_bins, frames]`` and coverage shaped ``[windows]``. Padded slots
    get weight 0; real windows keep their coverage, so a window that is half silence
    counts half. The returned target holds one label per source recording, never one
    per window.
    """
    batch = list(batch)
    if not batch:
        raise ValueError("Cannot collate an empty batch")

    max_windows = max(features.shape[0] for features, _, _ in batch)
    mel_bins, frames = batch[0][0].shape[1:]
    values = torch.zeros(len(batch), max_windows, mel_bins, frames, dtype=torch.float32)
    weights = torch.zeros(len(batch), max_windows, dtype=torch.float32)
    labels = torch.empty(len(batch), dtype=torch.float32)

    for index, (features, coverage, label) in enumerate(batch):
        if features.ndim != 3 or features.shape[1:] != (mel_bins, frames):
            raise ValueError("All recordings must contain compatible window features")
        n = features.shape[0]
        if coverage.shape != (n,):
            raise ValueError("Coverage must give one value per window")
        if not torch.all((coverage > 0) & (coverage <= 1)):
            raise ValueError("Coverage must lie in (0, 1]")
        values[index, :n] = features.float()
        weights[index, :n] = coverage.float()
        labels[index] = float(label)

    return values, weights, labels

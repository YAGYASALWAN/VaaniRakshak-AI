"""Class-agnostic audio transforms used by both bona fide and spoof pipelines.

Nothing here should depend on the label. If a future step needs different
treatment for spoofed audio, that is a modelling choice — not a preprocess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torchaudio

from vaanirakshak.config import AudioConfig, SegmentationConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WaveformSegment:
    """One analysis window cut from a longer recording."""

    waveform: np.ndarray
    start_seconds: float
    end_seconds: float
    padded: bool


def convert_to_mono(waveform: np.ndarray) -> np.ndarray:
    """Average channels into a single 1-D float32 track.

    Stereo is two microphones / a stereo mix stored as two sample streams of
    equal length. Averaging is a simple, label-independent collapse. We do not
    pick the louder channel: that would correlate with recording setup.

    Inputs
        waveform: 1-D (already mono) or 2-D ``(samples, channels)``.
    Outputs
        1-D float32 array of length ``samples``.
    Edge cases
        0 channels is invalid. One channel is squeezed, not averaged twice.
    """
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 1:
        return array
    if array.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D array, got shape {array.shape}")
    if array.shape[1] == 1:
        return array[:, 0]
    logger.debug("Converting %s channels to mono by mean", array.shape[1])
    return np.mean(array, axis=1, dtype=np.float32)


def resample_waveform(
    waveform: np.ndarray,
    orig_sample_rate: int,
    target_sample_rate: int,
) -> np.ndarray:
    """Change the number of samples per second without changing duration.

    Downsampling (e.g. 48 kHz → 16 kHz) would alias high frequencies if we
    just kept every Nth sample. ``torchaudio.functional.resample`` applies a
    windowed-sinc low-pass filter, then interpolates onto the new time grid.

    Inputs
        waveform: 1-D mono float array.
        orig_sample_rate: Hz of the current array.
        target_sample_rate: desired Hz (16_000 in the default config).
    Outputs
        1-D float32 array whose length is approximately
        ``len(waveform) * target / orig``.
    Edge cases
        Equal rates return the input (copy-free when already float32).
        orig or target <= 0 raises ValueError.
    """
    if orig_sample_rate <= 0 or target_sample_rate <= 0:
        raise ValueError(
            f"Sample rates must be positive, got {orig_sample_rate} -> {target_sample_rate}"
        )
    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError("resample_waveform expects a mono 1-D array")
    if orig_sample_rate == target_sample_rate:
        return array

    tensor = torch.from_numpy(array).unsqueeze(0)
    resampled = torchaudio.functional.resample(
        tensor,
        orig_freq=int(orig_sample_rate),
        new_freq=int(target_sample_rate),
    )
    out = resampled.squeeze(0).contiguous().numpy().astype(np.float32, copy=False)
    logger.debug(
        "Resampled %s samples @ %s Hz -> %s samples @ %s Hz",
        array.shape[0],
        orig_sample_rate,
        out.shape[0],
        target_sample_rate,
    )
    return out


def _window_and_hop_samples(sample_rate: int, window_seconds: float, hop_seconds: float) -> tuple[int, int]:
    window_samples = int(round(window_seconds * sample_rate))
    hop_samples = int(round(hop_seconds * sample_rate))
    if window_samples <= 0 or hop_samples <= 0:
        raise ValueError(
            f"window/hop must map to >= 1 sample (window={window_samples}, hop={hop_samples})"
        )
    return window_samples, hop_samples


def _pad_to_length(
    waveform: np.ndarray,
    length: int,
    pad_position: str,
) -> np.ndarray:
    if waveform.shape[0] >= length:
        return waveform[:length]
    missing = length - waveform.shape[0]
    zeros = np.zeros(missing, dtype=np.float32)
    if pad_position == "start":
        return np.concatenate([zeros, waveform])
    return np.concatenate([waveform, zeros])


def segment_waveform(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    window_seconds: float | None = None,
    hop_seconds: float | None = None,
    short_clip_policy: str | None = None,
    pad_position: str | None = None,
    audio_config: AudioConfig | None = None,
    segmentation_config: SegmentationConfig | None = None,
) -> list[WaveformSegment]:
    """Cut a mono waveform into fixed-length, optionally overlapping windows.

    Inputs
        waveform: 1-D mono samples at ``sample_rate``.
        window_seconds / hop_seconds: from config if omitted.
        short_clip_policy: ``pad`` (default) or ``skip``.
        pad_position: ``end`` (default) or ``start``.
    Outputs
        List of WaveformSegment, each with exactly ``window_samples`` samples.
        Times are in seconds relative to the start of *this* waveform.
    Edge cases
        Clip shorter than one window: pad to one window, or return [].
        Trailing remainder after the last full hop: same policy.
        hop > window leaves uncovered gaps (allowed, logged).
    """
    audio_cfg = audio_config or AudioConfig()
    seg_cfg = segmentation_config or SegmentationConfig()
    window_seconds = audio_cfg.window_seconds if window_seconds is None else window_seconds
    hop_seconds = audio_cfg.hop_seconds if hop_seconds is None else hop_seconds
    short_clip_policy = seg_cfg.short_clip_policy if short_clip_policy is None else short_clip_policy
    pad_position = seg_cfg.pad_position if pad_position is None else pad_position

    if short_clip_policy not in {"pad", "skip"}:
        raise ValueError("short_clip_policy must be 'pad' or 'skip'")
    if waveform.ndim != 1:
        raise ValueError("segment_waveform expects a mono 1-D array")
    if hop_seconds > window_seconds:
        logger.warning(
            "hop_seconds (%.3f) > window_seconds (%.3f); windows will not overlap and may gap",
            hop_seconds,
            window_seconds,
        )

    window_samples, hop_samples = _window_and_hop_samples(
        sample_rate, window_seconds, hop_seconds
    )
    n_samples = int(waveform.shape[0])
    duration = n_samples / float(sample_rate)

    if n_samples < window_samples:
        if short_clip_policy == "skip":
            logger.info(
                "Skipping short clip (%.4fs < %.4fs window)",
                duration,
                window_seconds,
            )
            return []
        padded = _pad_to_length(waveform, window_samples, pad_position)
        return [
            WaveformSegment(
                waveform=padded,
                start_seconds=0.0,
                end_seconds=window_seconds,
                padded=True,
            )
        ]

    segments: list[WaveformSegment] = []
    start = 0
    while start + window_samples <= n_samples:
        end = start + window_samples
        segments.append(
            WaveformSegment(
                waveform=np.asarray(waveform[start:end], dtype=np.float32),
                start_seconds=start / float(sample_rate),
                end_seconds=end / float(sample_rate),
                padded=False,
            )
        )
        start += hop_samples

    if start < n_samples:
        if short_clip_policy == "skip":
            logger.debug("Dropping trailing %.4fs remainder", (n_samples - start) / sample_rate)
        else:
            tail = np.asarray(waveform[start:], dtype=np.float32)
            padded = _pad_to_length(tail, window_samples, pad_position)
            segments.append(
                WaveformSegment(
                    waveform=padded,
                    start_seconds=start / float(sample_rate),
                    end_seconds=(start + window_samples) / float(sample_rate),
                    padded=True,
                )
            )

    return segments

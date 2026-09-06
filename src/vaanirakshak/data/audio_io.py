"""Load and save audio without mutating the original files.

Conceptual model
----------------
A WAV/FLAC file stores a sequence of amplitude samples taken at a fixed
sample rate (how many samples represent one second of air-pressure change).

``soundfile`` (libsndfile) decodes that container into a NumPy array.
We never overwrite the source path; processed audio is written elsewhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from vaanirakshak.exceptions import AudioLoadError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AudioData:
    """In-memory waveform plus the properties of the file it came from."""

    waveform: np.ndarray
    sample_rate: int
    num_channels: int
    path: Path
    subtype: str | None = None
    format: str | None = None

    @property
    def duration_seconds(self) -> float:
        if self.sample_rate <= 0:
            return 0.0
        return float(self.waveform.shape[0]) / float(self.sample_rate)


def load_audio(path: str | Path) -> AudioData:
    """Decode an audio file into float32 samples in approximately ``[-1, 1]``.

    Inputs
        path: existing audio file (WAV/FLAC/OGG, depending on libsndfile).
    Outputs
        AudioData with shape ``(num_samples, num_channels)`` even for mono
        (the channel axis is always present so later code is uniform).
    Edge cases
        Missing file, unsupported codec, truncated containers, and zero-length
        payloads raise AudioLoadError instead of returning empty arrays quietly.
    """
    audio_path = Path(path)
    if not audio_path.is_file():
        raise AudioLoadError(f"Audio file does not exist: {audio_path}")

    try:
        waveform, sample_rate = sf.read(str(audio_path), always_2d=True, dtype="float32")
        info = sf.info(str(audio_path))
    except Exception as exc:  # soundfile raises a variety of RuntimeError subclasses
        raise AudioLoadError(f"Unreadable or corrupt audio file: {audio_path}") from exc

    if sample_rate is None or int(sample_rate) <= 0:
        raise AudioLoadError(f"Invalid sample rate {sample_rate} in {audio_path}")

    waveform = np.asarray(waveform, dtype=np.float32)
    num_channels = int(waveform.shape[1]) if waveform.ndim == 2 else 1
    logger.debug(
        "Loaded %s: sr=%s channels=%s samples=%s subtype=%s",
        audio_path,
        sample_rate,
        num_channels,
        waveform.shape[0],
        getattr(info, "subtype", None),
    )
    return AudioData(
        waveform=waveform,
        sample_rate=int(sample_rate),
        num_channels=num_channels,
        path=audio_path,
        subtype=getattr(info, "subtype", None),
        format=getattr(info, "format", None),
    )


def save_audio(
    path: str | Path,
    waveform: np.ndarray,
    sample_rate: int,
    *,
    subtype: str = "PCM_16",
) -> Path:
    """Write a waveform to a new file. Never used on raw dataset paths.

    Inputs
        path: destination (parent directories are created).
        waveform: 1-D mono or 2-D ``(samples, channels)`` float array.
        sample_rate: Hz of the array you are writing (should already be 16 kHz).
        subtype: PCM encoding. Default 16-bit integer WAV.
    Outputs
        The Path that was written.
    Edge cases
        Non-finite samples are rejected. Values outside [-1, 1] are clipped
        with a warning — we do not peak-normalize, because loudness statistics
        differ across datasets and must stay inspectable.
    """
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)

    array = np.asarray(waveform, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.ndim != 2:
        raise ValueError(f"Expected 1-D or 2-D waveform, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"Refusing to write non-finite samples to {dest}")

    peak = float(np.max(np.abs(array))) if array.size else 0.0
    if peak > 1.0:
        logger.warning(
            "Clipping %s: peak amplitude %.4f exceeds 1.0 (no peak-normalization)",
            dest,
            peak,
        )
        array = np.clip(array, -1.0, 1.0)

    sf.write(str(dest), array, int(sample_rate), subtype=subtype)
    logger.debug("Wrote %s (%s samples, sr=%s)", dest, array.shape[0], sample_rate)
    return dest

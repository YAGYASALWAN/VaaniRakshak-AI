"""Deterministic waveform stress transforms for frozen VaaniRakshak V2 evaluation.

These transforms are evaluation tools, not training augmentation. They operate on
mono float32 16 kHz waveforms and never modify the model or its saved threshold.
"""
from __future__ import annotations

import hashlib
import math
import warnings

import numpy as np
import torch
import torchaudio

from vaanirakshak.v2_audio import MODEL_SAMPLE_RATE


def _wave(value: np.ndarray) -> np.ndarray:
    wave = np.asarray(value, dtype=np.float32)
    if wave.ndim != 1 or wave.size == 0 or not np.isfinite(wave).all():
        raise ValueError("Expected a finite nonempty mono waveform")
    return np.clip(wave, -1.0, 1.0).astype(np.float32, copy=False)


def center_duration(wave: np.ndarray, seconds: float, sample_rate: int = MODEL_SAMPLE_RATE) -> np.ndarray:
    wave = _wave(wave)
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    wanted = max(1, int(round(seconds * sample_rate)))
    if len(wave) <= wanted:
        return wave.copy()
    start = (len(wave) - wanted) // 2
    return wave[start : start + wanted].copy()


def narrowband_8khz(wave: np.ndarray) -> np.ndarray:
    """Round-trip 16 kHz speech through an 8 kHz sampling channel."""
    wave = _wave(wave)
    tensor = torch.from_numpy(wave)
    eight = torchaudio.functional.resample(tensor, MODEL_SAMPLE_RATE, 8_000)
    sixteen = torchaudio.functional.resample(eight, 8_000, MODEL_SAMPLE_RATE)
    return sixteen.float().clamp(-1.0, 1.0).cpu().numpy()


def _g711_roundtrip(wave: np.ndarray, law: str) -> np.ndarray:
    """Round-trip through 8 kHz G.711 μ-law or A-law companding.

    Python 3.11's stdlib audioop implements the codec exactly enough for this
    evaluation utility. The import is local because audioop is deprecated and may
    need replacement before moving the project to Python versions where it is gone.
    """
    wave = _wave(wave)
    eight = torchaudio.functional.resample(torch.from_numpy(wave), MODEL_SAMPLE_RATE, 8_000)
    pcm = np.clip(np.rint(eight.cpu().numpy() * 32767.0), -32768, 32767).astype("<i2")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import audioop
    except ImportError as exc:  # pragma: no cover - Python 3.11 CI provides audioop
        raise RuntimeError("G.711 stress transforms currently require Python 3.11 audioop") from exc

    if law == "mulaw":
        encoded = audioop.lin2ulaw(pcm.tobytes(), 2)
        decoded = audioop.ulaw2lin(encoded, 2)
    elif law == "alaw":
        encoded = audioop.lin2alaw(pcm.tobytes(), 2)
        decoded = audioop.alaw2lin(encoded, 2)
    else:
        raise ValueError("law must be 'mulaw' or 'alaw'")

    restored_8k = np.frombuffer(decoded, dtype="<i2").astype(np.float32) / 32768.0
    restored = torchaudio.functional.resample(torch.from_numpy(restored_8k), 8_000, MODEL_SAMPLE_RATE)
    return restored.float().clamp(-1.0, 1.0).cpu().numpy()


def g711_mulaw(wave: np.ndarray) -> np.ndarray:
    return _g711_roundtrip(wave, "mulaw")


def g711_alaw(wave: np.ndarray) -> np.ndarray:
    return _g711_roundtrip(wave, "alaw")


def white_noise_at_snr(wave: np.ndarray, snr_db: float, *, key: str) -> np.ndarray:
    """Add deterministic white-noise stress at a requested signal/noise ratio.

    This is a controlled synthetic stress test, not a substitute for later
    real-background-noise evaluation.
    """
    wave = _wave(wave)
    if not math.isfinite(snr_db):
        raise ValueError("snr_db must be finite")
    signal_power = float(np.mean(np.square(wave), dtype=np.float64))
    if signal_power <= 1e-12:
        raise ValueError("Cannot define SNR for effectively silent audio")

    seed_bytes = hashlib.blake2s(f"{key}:{snr_db:.6f}".encode("utf-8"), digest_size=8).digest()
    seed = int.from_bytes(seed_bytes, "little")
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(wave)).astype(np.float32)
    noise_power = float(np.mean(np.square(noise), dtype=np.float64))
    wanted_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise *= math.sqrt(wanted_noise_power / max(noise_power, 1e-12))
    return np.clip(wave + noise, -1.0, 1.0).astype(np.float32)


def measured_snr_db(clean: np.ndarray, noisy: np.ndarray) -> float:
    clean = _wave(clean)
    noisy = _wave(noisy)
    if len(clean) != len(noisy):
        raise ValueError("clean and noisy waveforms must have equal length")
    noise = noisy.astype(np.float64) - clean.astype(np.float64)
    signal_power = float(np.mean(np.square(clean), dtype=np.float64))
    noise_power = float(np.mean(np.square(noise), dtype=np.float64))
    if noise_power <= 1e-15:
        return float("inf")
    return 10.0 * math.log10(signal_power / noise_power)

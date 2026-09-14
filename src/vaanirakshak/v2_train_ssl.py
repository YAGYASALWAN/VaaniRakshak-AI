"""Train the first VaaniRakshak V2 SSL anti-spoofing detector.

This script trains ONLY on manifest split=train and selects model/threshold ONLY
from split=dev. It never evaluates or tunes against split=test. Final test and
cross-dataset evaluation belong in a separate frozen-checkpoint command.

Example:
    python -m vaanirakshak.v2_train_ssl \
        --manifest data/v2/train_manifest.jsonl \
        --output models/v2_wavlm_run \
        --device cuda
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchaudio

from vaanirakshak.v2_audio import MODEL_SAMPLE_RATE
from vaanirakshak.v2_data_contract import audit_manifest, load_jsonl, manifest_fingerprint
from vaanirakshak.v2_detectors import RAW_FRONTEND, V2_CHECKPOINT_SCHEMA
from vaanirakshak.v2_metrics import eer, metrics_at_threshold, threshold_for_max_fpr
from vaanirakshak.v2_ssl_model import ARCHITECTURE, DEFAULT_BACKBONE, WavLMAntiSpoof


WINDOW_SAMPLES = 4 * MODEL_SAMPLE_RATE


def _resolve_audio_ref(audio_ref: str, manifest_dir: Path) -> Path:
    path = Path(audio_ref).expanduser()
    if not path.is_absolute():
        path = manifest_dir / path
    return path.resolve()


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    with sf.SoundFile(path) as stream:
        if stream.channels < 1 or stream.channels > 8:
            raise ValueError(f"Unsupported channel count in {path}: {stream.channels}")
        if not 4_000 <= stream.samplerate <= 192_000:
            raise ValueError(f"Unsupported sample rate in {path}: {stream.samplerate}")
        wave = stream.read(dtype="float32", always_2d=True).mean(axis=1)
        rate = int(stream.samplerate)
    if wave.size == 0 or not np.isfinite(wave).all():
        raise ValueError(f"Invalid audio waveform: {path}")
    return wave.astype(np.float32, copy=False), rate


def _fixed_window(wave: np.ndarray, rate: int, *, training: bool, rng: random.Random) -> tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.from_numpy(wave)
    if rate != MODEL_SAMPLE_RATE:
        tensor = torchaudio.functional.resample(tensor, rate, MODEL_SAMPLE_RATE)
    tensor = tensor.float().clamp(-1.0, 1.0)

    if len(tensor) > WINDOW_SAMPLES:
        if training:
            start = rng.randint(0, len(tensor) - WINDOW_SAMPLES)
        else:
            start = (len(tensor) - WINDOW_SAMPLES) // 2
        tensor = tensor[start : start + WINDOW_SAMPLES]
        valid = WINDOW_SAMPLES
    else:
        valid = len(tensor)

    output = torch.zeros(WINDOW_SAMPLES, dtype=torch.float32)
    output[:valid] = tensor[:valid]
    mask = torch.zeros(WINDOW_SAMPLES, dtype=torch.long)
    mask[:valid] = 1
    return output, mask


class ManifestAudioDataset(Dataset):
    def __init__(self, records, manifest_dir: Path, split: str, *, seed: int = 42):
        self.records = [record for record in records if record.split == split]
        self.manifest_dir = manifest_dir
        self.training = split == "train"
        self.seed = seed
        self.epoch = 0
        if not self.records:
            raise ValueError(f"Manifest contains no {split!r} records")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        wave, rate = _read_audio(_resolve_audio_ref(record.audio_ref, self.manifest_dir))
        # Python's built-in hash() is intentionally process-randomized. Derive a
        # stable per-record/per-epoch seed instead so crops vary across epochs but
        # remain reproducible across independent runs with the same global seed.
        material = f"{self.seed}:{self.epoch}:{record.record_id}".encode("utf-8")
        crop_seed = int.from_bytes(hashlib.blake2s(material, digest_size=8).digest(), "little")
        rng = random.Random(crop_seed)
        values, mask = _fixed_window(wave, rate, training=self.training, rng=rng)
        label = torch.tensor(1.0 if record.label == "spoof" else 0.0, dtype=torch.float32)
        return values, mask, label


def _balanced_sampler(dataset: ManifestAudioDataset, seed: int) -> WeightedRandomSampler:
    groups = Counter((record.dataset, record.label) for record in dataset.records)
    weights = [1.0 / groups[(record.dataset, record.label)] for record in dataset.records]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)


def _set_backbone_trainable(model: WavLMAntiSpoof, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable
    # Keep the raw convolutional feature encoder frozen even when transformer
    # layers are fine-tuned; this reduces memory/overfitting for the first V2 run.
    if hasattr(model.backbone, "freeze_feature_encoder"):
        model.backbone.freeze_feature_encoder()


def _parameter_groups(model: WavLMAntiSpoof, backbone_lr: float, head_lr: float):
    backbone_ids = {id(parameter) for parameter in model.backbone.parameters()}
    backbone = [parameter for parameter in model.parameters() if id(parameter) in backbone_ids]
    head = [parameter for parameter in model.parameters() if id(parameter) not in backbone_ids]
    return [
        {"params": backbone, "lr": backbone_lr},
        {"params": head, "lr": head_lr},
    ]


@torch.no_grad()
def evaluate(model: WavLMAntiSpoof, loader: DataLoader, device: str) -> tuple[float, list[int], list[float]]:
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    total_loss = 0.0
    total_items = 0
    for values, mask, target in loader:
        values = values.to(device)
        mask = mask.to(device)
        target = target.to(device)
        logits = model(values, attention_mask=mask)
        loss = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="sum")
        total_loss += float(loss.item())
        total_items += len(target)
        labels.extend(target.int().cpu().tolist())
        scores.extend(logits.sigmoid().cpu().tolist())
    return total_loss / max(1, total_items), labels, scores


def _atomic_torch_save(value: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def run(
    manifest: Path,
    output: Path,
    *,
    backbone_id: str = DEFAULT_BACKBONE,
    epochs: int = 8,
    batch_size: int = 4,
    device: str = "cuda",
    head_only_epochs: int = 1,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    max_dev_fpr: float = 0.05,
    seed: int = 42,
) -> Path:
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    manifest = manifest.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(manifest)
    audit = audit_manifest(records)
    audit.raise_for_errors()
    if not any(record.split == "train" for record in records) or not any(record.split == "dev" for record in records):
        raise ValueError("Training requires both train and dev manifest splits")

    train_data = ManifestAudioDataset(records, manifest.parent, "train", seed=seed)
    dev_data = ManifestAudioDataset(records, manifest.parent, "dev", seed=seed)
    sampler = _balanced_sampler(train_data, seed)
    train_loader = DataLoader(train_data, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=device == "cuda")
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device == "cuda")

    model = WavLMAntiSpoof.from_pretrained(backbone_id).to(device)
    optimizer = torch.optim.AdamW(
        _parameter_groups(model, backbone_lr, head_lr),
        weight_decay=weight_decay,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    run_name = f"wavlm_v2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    fingerprint = manifest_fingerprint(records)
    history: list[dict] = []
    best_eer = math.inf
    best_path = output / "best.pt"

    _write_json(
        output / "run_config.json",
        {
            "run_name": run_name,
            "architecture": ARCHITECTURE,
            "backbone_id": backbone_id,
            "epochs": epochs,
            "batch_size": batch_size,
            "head_only_epochs": head_only_epochs,
            "backbone_lr": backbone_lr,
            "head_lr": head_lr,
            "weight_decay": weight_decay,
            "max_dev_fpr": max_dev_fpr,
            "seed": seed,
            "manifest": str(manifest),
            "manifest_fingerprint": fingerprint,
            "audit_counts": audit.counts,
            "threshold_policy": "maximize spoof recall subject to development-set FPR budget",
            "test_data_used_during_training": False,
        },
    )

    for epoch in range(1, epochs + 1):
        train_data.set_epoch(epoch)
        _set_backbone_trainable(model, epoch > head_only_epochs)
        model.train()
        started = time.monotonic()
        running_loss = 0.0
        seen = 0

        for values, mask, target in train_loader:
            values = values.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits = model(values, attention_mask=mask)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * len(target)
            seen += len(target)

        dev_loss, dev_labels, dev_scores = evaluate(model, dev_loader, device)
        dev_eer, dev_eer_threshold = eer(dev_labels, dev_scores)
        threshold = threshold_for_max_fpr(dev_labels, dev_scores, max_dev_fpr)
        dev_metrics = metrics_at_threshold(dev_labels, dev_scores, threshold)
        entry = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, seen),
            "dev_loss": dev_loss,
            "dev_eer": dev_eer,
            "dev_eer_threshold": dev_eer_threshold,
            "selected_threshold": threshold,
            "dev_metrics": dev_metrics,
            "backbone_trainable": epoch > head_only_epochs,
            "seconds": time.monotonic() - started,
        }
        history.append(entry)
        _write_json(output / "history.json", history)
        print(
            f"Epoch {epoch}/{epochs}: loss={entry['train_loss']:.4f} "
            f"dev_eer={dev_eer:.4f} dev_f1={dev_metrics['f1']:.4f} "
            f"threshold={threshold:.4f}",
            flush=True,
        )

        if dev_eer < best_eer:
            best_eer = dev_eer
            export = model.export_model_config()
            checkpoint = {
                "schema": V2_CHECKPOINT_SCHEMA,
                "architecture": ARCHITECTURE,
                "frontend": RAW_FRONTEND,
                "model": model.state_dict(),
                "backbone_config": export["backbone_config"],
                "model_spec": export["model_spec"],
                "threshold": float(threshold),
                "calibrated_probability": False,
                "model_name": run_name,
                "notice": (
                    "VaaniRakshak V2 SSL detector. Score threshold was selected on development data; "
                    "scores are not calibrated probabilities."
                ),
                "training": {
                    "epoch": epoch,
                    "manifest_fingerprint": fingerprint,
                    "dev_metrics": dev_metrics,
                    "dev_eer": dev_eer,
                    "max_dev_fpr": max_dev_fpr,
                    "test_data_used_during_training": False,
                },
            }
            _atomic_torch_save(checkpoint, best_path)

    if not best_path.is_file():
        raise RuntimeError("Training completed without producing a best checkpoint")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--head-only-epochs", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-dev-fpr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    path = run(
        args.manifest,
        args.output,
        backbone_id=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        head_only_epochs=args.head_only_epochs,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        max_dev_fpr=args.max_dev_fpr,
        seed=args.seed,
    )
    print(f"Best V2 checkpoint: {path}", flush=True)


if __name__ == "__main__":
    main()

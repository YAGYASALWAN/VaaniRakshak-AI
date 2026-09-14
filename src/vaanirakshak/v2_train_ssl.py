"""Train the first VaaniRakshak V2 SSL anti-spoofing detector.

This script trains ONLY on manifest split=train and selects model/threshold ONLY
from split=dev. It never evaluates or tunes against split=test. Final test and
cross-dataset evaluation belong in a separate frozen-checkpoint command.

A run writes ``last.pt`` after each completed epoch. Use ``--resume`` with the
same output directory to restore model, optimizer, AMP, sampler and RNG state.
Resume is refused if the manifest or training contract changed. The resumable
state also stores the exported WavLM architecture config so recovery does not
need to redownload/re-resolve the pretrained backbone.
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
TRAINING_STATE_SCHEMA = "vaanirakshak-v2-training-state-v2"


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
        material = f"{self.seed}:{self.epoch}:{record.record_id}".encode("utf-8")
        crop_seed = int.from_bytes(hashlib.blake2s(material, digest_size=8).digest(), "little")
        rng = random.Random(crop_seed)
        values, mask = _fixed_window(wave, rate, training=self.training, rng=rng)
        label = torch.tensor(1.0 if record.label == "spoof" else 0.0, dtype=torch.float32)
        return values, mask, label


def _balanced_sampler(dataset: ManifestAudioDataset, seed: int) -> tuple[WeightedRandomSampler, torch.Generator]:
    groups = Counter((record.dataset, record.label) for record in dataset.records)
    weights = [1.0 / groups[(record.dataset, record.label)] for record in dataset.records]
    generator = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True, generator=generator)
    return sampler, generator


def _set_backbone_trainable(model: WavLMAntiSpoof, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable
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


def _training_contract(
    *,
    backbone_id: str,
    epochs: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    gradient_checkpointing: bool,
    head_only_epochs: int,
    backbone_lr: float,
    head_lr: float,
    weight_decay: float,
    max_dev_fpr: float,
    seed: int,
) -> dict:
    return {
        "architecture": ARCHITECTURE,
        "backbone_id": backbone_id,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "gradient_accumulation_steps": int(gradient_accumulation_steps),
        "gradient_checkpointing": bool(gradient_checkpointing),
        "head_only_epochs": int(head_only_epochs),
        "backbone_lr": float(backbone_lr),
        "head_lr": float(head_lr),
        "weight_decay": float(weight_decay),
        "max_dev_fpr": float(max_dev_fpr),
        "seed": int(seed),
    }


def _validate_resume_state(state: dict, *, contract: dict, fingerprint: str) -> None:
    if not isinstance(state, dict) or state.get("schema") != TRAINING_STATE_SCHEMA:
        raise ValueError("last.pt is not a VaaniRakshak V2 resumable training state")
    if state.get("manifest_fingerprint") != fingerprint:
        raise ValueError("Resume manifest differs from the saved training run")
    if state.get("training_contract") != contract:
        raise ValueError("Resume hyperparameters/backbone differ from the saved training run")

    required = {
        "model",
        "optimizer",
        "scaler",
        "epoch",
        "history",
        "best_eer",
        "torch_rng",
        "sampler_rng",
        "backbone_config",
        "model_spec",
    }
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"Resume state is missing required fields: {missing}")

    epoch = state.get("epoch")
    max_epochs = int(contract.get("epochs", 0))
    if isinstance(epoch, bool) or not isinstance(epoch, int) or not 1 <= epoch <= max_epochs:
        raise ValueError("Resume state has an invalid completed epoch")

    history = state.get("history")
    if not isinstance(history, list) or len(history) != epoch:
        raise ValueError("Resume history does not match the completed epoch")
    history_epochs = []
    for entry in history:
        if not isinstance(entry, dict):
            raise ValueError("Resume history contains a non-object entry")
        value = entry.get("epoch")
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError("Resume history contains an invalid epoch")
        history_epochs.append(value)
    if history_epochs != list(range(1, epoch + 1)):
        raise ValueError("Resume history epochs are not contiguous from epoch 1")

    try:
        best_eer = float(state.get("best_eer"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Resume state has an invalid best_eer") from exc
    if not math.isfinite(best_eer) or not 0.0 <= best_eer <= 1.0:
        raise ValueError("Resume state has an invalid best_eer")

    if not isinstance(state.get("backbone_config"), dict) or not isinstance(state.get("model_spec"), dict):
        raise ValueError("Resume state is missing a valid exported WavLM model configuration")


def _load_training_state(path: Path) -> dict:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError("Could not load last.pt; the resumable training state may be incomplete or corrupt") from exc
    if not isinstance(state, dict):
        raise ValueError("last.pt did not contain a training-state dictionary")
    return state


def run(
    manifest: Path,
    output: Path,
    *,
    backbone_id: str = DEFAULT_BACKBONE,
    epochs: int = 8,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    gradient_checkpointing: bool = True,
    device: str = "cuda",
    head_only_epochs: int = 1,
    backbone_lr: float = 1e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 1e-4,
    max_dev_fpr: float = 0.05,
    seed: int = 42,
    resume: bool = False,
) -> Path:
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be cpu or cuda")
    if epochs < 1 or batch_size < 1 or gradient_accumulation_steps < 1:
        raise ValueError("epochs, batch_size and gradient_accumulation_steps must be positive")
    if not 0 <= head_only_epochs < epochs + 1:
        raise ValueError("head_only_epochs must be between 0 and epochs")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    manifest = manifest.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    run_config_path = output / "run_config.json"
    last_path = output / "last.pt"
    best_path = output / "best.pt"

    if resume:
        if not run_config_path.is_file() or not last_path.is_file() or not best_path.is_file():
            raise ValueError("--resume requires existing run_config.json, last.pt and best.pt in the output directory")
    elif run_config_path.exists() or last_path.exists() or best_path.exists():
        raise ValueError("Output directory already contains a V2 training run; use a new directory or --resume")

    records = load_jsonl(manifest)
    audit = audit_manifest(records)
    audit.raise_for_errors()
    train_records = [record for record in records if record.split == "train"]
    dev_records = [record for record in records if record.split == "dev"]
    if not train_records or not dev_records:
        raise ValueError("Training requires both train and dev manifest splits")
    if {record.label for record in train_records} != {"bonafide", "spoof"}:
        raise ValueError("Training split must contain bonafide and spoof records")
    if {record.label for record in dev_records} != {"bonafide", "spoof"}:
        raise ValueError("Development split must contain bonafide and spoof records")

    fingerprint = manifest_fingerprint(records)
    contract = _training_contract(
        backbone_id=backbone_id,
        epochs=epochs,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=gradient_checkpointing,
        head_only_epochs=head_only_epochs,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        max_dev_fpr=max_dev_fpr,
        seed=seed,
    )

    train_data = ManifestAudioDataset(records, manifest.parent, "train", seed=seed)
    dev_data = ManifestAudioDataset(records, manifest.parent, "dev", seed=seed)
    sampler, sampler_generator = _balanced_sampler(train_data, seed)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=device == "cuda",
    )
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=device == "cuda")

    resume_state: dict | None = None
    history: list[dict] = []
    best_eer = math.inf
    start_epoch = 1

    if resume:
        try:
            saved_config = json.loads(run_config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Could not read run_config.json for resume") from exc
        if saved_config.get("manifest_fingerprint") != fingerprint:
            raise ValueError("run_config manifest fingerprint differs from the requested manifest")
        if saved_config.get("training_contract") != contract:
            raise ValueError("run_config training contract differs from the requested resume configuration")
        run_name = str(saved_config.get("run_name") or "")
        if not run_name:
            raise ValueError("run_config is missing run_name")

        resume_state = _load_training_state(last_path)
        _validate_resume_state(resume_state, contract=contract, fingerprint=fingerprint)
        try:
            model = WavLMAntiSpoof.from_exported_config(
                resume_state["backbone_config"], resume_state["model_spec"]
            ).to(device)
        except Exception as exc:
            raise ValueError("Could not reconstruct WavLM from the saved resume configuration") from exc
    else:
        run_name = f"wavlm_v2_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        model = WavLMAntiSpoof.from_pretrained(backbone_id).to(device)
        run_config = {
            "run_name": run_name,
            **contract,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "manifest": str(manifest),
            "manifest_fingerprint": fingerprint,
            "audit_counts": audit.counts,
            "training_contract": contract,
            "threshold_policy": "maximize spoof recall subject to development-set FPR budget",
            "test_data_used_during_training": False,
            "resume_granularity": "last completed epoch",
            "resume_model_reconstruction": "offline from last.pt exported config",
        }
        _write_json(run_config_path, run_config)

    if gradient_checkpointing:
        enable = getattr(model.backbone, "gradient_checkpointing_enable", None)
        if callable(enable):
            enable()
        else:
            raise ValueError("Selected WavLM backbone does not support gradient checkpointing")

    optimizer = torch.optim.AdamW(_parameter_groups(model, backbone_lr, head_lr), weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    exported_model_config = model.export_model_config()

    if resume_state is not None:
        try:
            model.load_state_dict(resume_state["model"])
            optimizer.load_state_dict(resume_state["optimizer"])
            scaler.load_state_dict(resume_state["scaler"])
            history = list(resume_state["history"])
            best_eer = float(resume_state["best_eer"])
            start_epoch = int(resume_state["epoch"]) + 1
            torch.set_rng_state(resume_state["torch_rng"])
            sampler_generator.set_state(resume_state["sampler_rng"])
            if device == "cuda" and resume_state.get("cuda_rng"):
                torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
        except Exception as exc:
            raise ValueError("Could not restore model/optimizer/RNG state from last.pt") from exc
        print(f"Resuming {run_name} from completed epoch {start_epoch - 1}/{epochs}", flush=True)

    if start_epoch > epochs:
        print(f"Training already completed through epoch {epochs}; using {best_path}", flush=True)
        return best_path

    for epoch in range(start_epoch, epochs + 1):
        train_data.set_epoch(epoch)
        _set_backbone_trainable(model, epoch > head_only_epochs)
        model.train()
        started = time.monotonic()
        running_loss = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        loader_steps = len(train_loader)

        for step, (values, mask, target) in enumerate(train_loader, 1):
            values = values.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits = model(values, attention_mask=mask)
                raw_loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
                loss = raw_loss / gradient_accumulation_steps
            if not torch.isfinite(raw_loss):
                raise ValueError("Non-finite training loss")
            scaler.scale(loss).backward()
            running_loss += float(raw_loss.item()) * len(target)
            seen += len(target)

            should_step = step % gradient_accumulation_steps == 0 or step == loader_steps
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

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
            checkpoint = {
                "schema": V2_CHECKPOINT_SCHEMA,
                "architecture": ARCHITECTURE,
                "frontend": RAW_FRONTEND,
                "model": model.state_dict(),
                "backbone_config": exported_model_config["backbone_config"],
                "model_spec": exported_model_config["model_spec"],
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

        # Resume starts from the next epoch. We intentionally checkpoint only at
        # epoch boundaries so a partially completed epoch is rerun consistently.
        training_state = {
            "schema": TRAINING_STATE_SCHEMA,
            "manifest_fingerprint": fingerprint,
            "training_contract": contract,
            "backbone_config": exported_model_config["backbone_config"],
            "model_spec": exported_model_config["model_spec"],
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "history": history,
            "best_eer": best_eer,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
            "sampler_rng": sampler_generator.get_state(),
        }
        _atomic_torch_save(training_state, last_path)

    if not best_path.is_file():
        raise RuntimeError("Training completed without producing a best checkpoint")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--head-only-epochs", type=int, default=1)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-dev-fpr", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume the last completed epoch in --output.")
    args = parser.parse_args()
    path = run(
        args.manifest,
        args.output,
        backbone_id=args.backbone,
        epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        device=args.device,
        head_only_epochs=args.head_only_epochs,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
        max_dev_fpr=args.max_dev_fpr,
        seed=args.seed,
        resume=args.resume,
    )
    print(f"Best V2 checkpoint: {path}", flush=True)


if __name__ == "__main__":
    main()

"""Stream public English ASVspoof audio once into reusable log-mel features."""
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import sqlite3
import time

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchaudio

from vaanirakshak.baseline_download import file_sha, save_json
from vaanirakshak.baseline_rules import binary_metrics
from vaanirakshak.english_rules import COUNTS, NOTICE, REPOSITORY, REVISION, check_disjoint, record_metadata, validate_partition

RATE, SAMPLES, FEATURE_SHAPE = 16000, 64000, (64, 401)
CACHE_VERSION = "english-logmel-v1"


def waveform_window(raw):
    """Read at most four seconds; pad short clips without dropping official records."""
    if not isinstance(raw, bytes) or not 0 < len(raw) <= 32 * 1024**2:
        raise ValueError("Expected bounded embedded original audio bytes")
    with sf.SoundFile(io.BytesIO(raw)) as audio:
        if audio.samplerate != RATE or audio.channels != 1 or not 0 < len(audio) <= RATE * 120:
            raise ValueError("Expected nonempty mono 16 kHz ASVspoof audio under 120 seconds")
        original_frames = len(audio)
        start = max(0, (original_frames - SAMPLES) // 2)
        audio.seek(start)
        wave = audio.read(SAMPLES, dtype="float32")
    if not np.isfinite(wave).all():
        raise ValueError("Non-finite audio samples")
    result = np.zeros(SAMPLES, dtype=np.float32)
    result[:len(wave)] = wave
    return result, {"original_frames": original_frames, "sample_rate": RATE,
                    "window_start_frame": start, "padded": original_frames < SAMPLES}


class Frontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=RATE, n_fft=512,
                   win_length=400, hop_length=160, n_mels=64, f_min=20, f_max=7600)

    def forward(self, waves):
        mel = self.mel(waves.float()).clamp_min(1e-8).log()
        return (mel - mel.mean((1, 2), keepdim=True)) / mel.std((1, 2), keepdim=True).clamp_min(1e-5)


class EnglishCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a, b in ((1, 16), (16, 32), (32, 64), (64, 128)):
            layers += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)]
        self.network = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                                     nn.Dropout(.3), nn.Linear(128, 1))

    def forward(self, features):
        return self.network(features.unsqueeze(1)).squeeze(1)


def score_features(frontend, model, waves):
    """Score a batch of four-second waveforms. The one feature-to-score path here.

    Returns ``(scores, usable)``, both lists of length ``len(waves)``. ``usable`` is
    False wherever the frontend or the model produced a non-finite value, which
    happens on out-of-domain audio; callers decide whether that is fatal.

    Precision: the feature cache is stored as float16, so the model was fitted on
    float16-rounded features and inference rounds the same way. The forward pass
    itself stays in float32. Autocast is what can overflow on unfamiliar audio, not
    this cast - normalized log-mel sits near +/-5 and float16 reaches 65504 - so the
    two concerns are separate and both are handled here.
    """
    with torch.no_grad():
        features = frontend(waves.float()).half().float()
        finite_features = torch.isfinite(features).flatten(1).all(1)
        logits = model(torch.nan_to_num(features)).float()
        usable = finite_features & torch.isfinite(logits)
        scores = logits.sigmoid()
    return scores.cpu().tolist(), usable.cpu().tolist()


def stream_split(split):
    from datasets import Audio, load_dataset
    stream = load_dataset(REPOSITORY, "default", split=split, revision=REVISION,
                          streaming=True, token=False)
    if getattr((stream.features or {}).get("key"), "names", None) != ["bonafide", "spoof"]:
        raise ValueError("Dataset label mapping differs from the verified source")
    # SoundFile handles the embedded FLAC bytes; no TorchCodec/FFmpeg dependency.
    return stream.cast_column("audio", Audio(sampling_rate=RATE, decode=False))


def cache_split(root, split, device, stream_factory=None):
    """Complete caches need no network. Partial caches resume computation, but may reread network bytes."""
    root = Path(root) / split
    root.mkdir(parents=True, exist_ok=True)
    size = sum(COUNTS[split].values())
    identity = {"schema": CACHE_VERSION, "repository": REPOSITORY, "revision": REVISION, "split": split,
                "rows": size, "feature_shape": list(FEATURE_SHAPE)}
    contract = root / "contract.json"
    if contract.exists() and json.loads(contract.read_text()) != identity:
        raise ValueError("Feature cache configuration changed; use a new --data directory")
    save_json(contract, identity)
    path = root / "features.npy"
    receipt = root / "complete.json"
    with closing(sqlite3.connect(root / "records.sqlite")) as db:
        db.execute("CREATE TABLE IF NOT EXISTS records (idx INTEGER PRIMARY KEY, metadata TEXT NOT NULL)")
        saved = db.execute("SELECT idx, metadata FROM records ORDER BY idx").fetchall()
        if [idx for idx, _ in saved] != list(range(len(saved))) or len(saved) > size:
            raise ValueError("Cache index is not contiguous")
        rows = [json.loads(value) for _, value in saved]
        if receipt.exists():
            info = json.loads(receipt.read_text())
            validate_partition(rows, split)
            if info.get("contract") != identity or file_sha(path) != info.get("feature_sha256"):
                raise ValueError("Completed feature cache checksum mismatch")
            features = np.load(path, mmap_mode="r", allow_pickle=False)
            if features.shape != (size, *FEATURE_SHAPE) or features.dtype != np.float16:
                raise ValueError("Feature cache shape or dtype mismatch")
            print(f"Using cached {split}: {size:,} recordings", flush=True)
            return features, rows
        required = size * np.prod(FEATURE_SHAPE) * 2
        if not path.exists() and shutil.disk_usage(root).free < required + 1024**3:
            raise ValueError(f"Insufficient disk space for {split} feature cache")
        if path.exists():
            features = np.load(path, mmap_mode="r+", allow_pickle=False)
            if features.shape != (size, *FEATURE_SHAPE) or features.dtype != np.float16:
                raise ValueError("Incomplete cache has wrong shape/dtype")
        else:
            if rows:
                raise ValueError("Feature file missing for existing cache records")
            features = np.lib.format.open_memmap(path, mode="w+", dtype="float16", shape=(size, *FEATURE_SHAPE))
        print(f"Loading English {split}: {len(rows):,}/{size:,} cached. Connecting to Hugging Face...", flush=True)
        factory = stream_factory or stream_split
        stream = factory(split)
        start = len(rows)
        if start:
            print("Resuming feature computation; the stream may reread earlier remote rows.", flush=True)
            stream = stream.skip(start)
        frontend = Frontend().to(device).eval()
        pending_waves, pending_rows = [], []
        last = time.monotonic()

        def flush():
            nonlocal last
            if not pending_rows:
                return
            with torch.no_grad():
                values = frontend(torch.from_numpy(np.stack(pending_waves)).to(device)).cpu().numpy().astype(np.float16)
            if values.shape[1:] != FEATURE_SHAPE or not np.isfinite(values).all():
                raise ValueError("Invalid computed features")
            offset = len(rows)
            features[offset:offset + len(pending_rows)] = values
            features.flush()  # Commit audio-derived features before advancing the resume cursor.
            db.executemany("INSERT INTO records VALUES (?, ?)",
                           [(offset + i, json.dumps(r)) for i, r in enumerate(pending_rows)])
            db.commit()
            rows.extend(pending_rows)
            pending_rows.clear()
            pending_waves.clear()
            now = time.monotonic()
            if now - last >= 5 or len(rows) == size:
                print(f"  {split}: {len(rows):,}/{size:,} ({len(rows)/size:.1%})", flush=True)
                last = now

        for source in stream:
            if len(rows) + len(pending_rows) >= size:
                raise ValueError("Source contains more rows than the pinned dataset contract")
            metadata = record_metadata(source, split)
            audio = source.get("audio")
            raw = audio.get("bytes") if isinstance(audio, dict) else None
            wave, properties = waveform_window(raw)
            metadata.update(properties)
            metadata["audio_sha256"] = hashlib.sha256(raw).hexdigest()
            pending_waves.append(wave)
            pending_rows.append(metadata)
            if len(pending_rows) >= 64:
                flush()
        flush()
        validate_partition(rows, split)
        save_json(receipt, {"contract": identity, "feature_sha256": file_sha(path),
                            "counts": dict(Counter(r["label"] for r in rows))})
        print(f"Completed {split} features: {len(rows):,} recordings", flush=True)
        return features, rows


class FeatureDataset(Dataset):
    def __init__(self, features, rows):
        self.features = features
        self.labels = [r["label"] for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.from_numpy(self.features[idx].astype(np.float32)), torch.tensor(self.labels[idx], dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, scores = [], []
    loss = 0.0
    for features, target in loader:
        output = model(features.to(device))
        loss += nn.functional.binary_cross_entropy_with_logits(output, target.to(device), reduction="sum").item()
        labels.extend(target.int().tolist())
        scores.extend(output.sigmoid().cpu().tolist())
    return loss / len(labels), labels, scores


def atomic_torch_save(value, path):
    temporary = path.with_suffix(".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def run(root, epochs=20, batch_size=32, resume=None, device="cuda", profile=None):
    profile = profile or {}
    repository = profile.get("repository", REPOSITORY)
    revision = profile.get("revision", REVISION)
    notice = profile.get("notice", NOTICE)
    counts = profile.get("counts", COUNTS)
    load_cache = profile.get("cache_split", cache_split)
    check_partitions = profile.get("check_disjoint", check_disjoint)
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable. Run with the clean GPU Python environment.")
    torch.set_num_threads(4)
    torch.manual_seed(42)
    root = Path(root).resolve()
    print(notice, flush=True)
    print("Recordings:", {s: sum(n.values()) for s, n in counts.items()}, flush=True)
    train_x, train_rows = load_cache(root, "train", device)
    dev_x, dev_rows = load_cache(root, "validation", device)
    check_partitions({"train": train_rows, "validation": dev_rows})
    data_identity = hashlib.sha256(json.dumps({"schema": CACHE_VERSION, "repository": repository,
                       "revision": revision, "train": train_rows, "validation": dev_rows}, sort_keys=True).encode()).hexdigest()
    run_dir = Path(resume).resolve() if resume else Path("models") / (
        profile.get("run_prefix", "english_") + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    run_dir.mkdir(parents=True, exist_ok=True)
    model = EnglishCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    generator = torch.Generator().manual_seed(42)
    class_counts = Counter(r["label"] for r in train_rows)
    weights = [1 / class_counts[r["label"]] for r in train_rows]
    sampler = WeightedRandomSampler(weights, num_samples=len(train_rows), replacement=True, generator=generator)
    train_loader = DataLoader(FeatureDataset(train_x, train_rows), batch_size=batch_size, sampler=sampler,
                              num_workers=0, pin_memory=device == "cuda")
    dev_loader = DataLoader(FeatureDataset(dev_x, dev_rows), batch_size=batch_size, num_workers=0)
    history, best, stale, start_epoch = [], float("inf"), 0, 1
    if resume:
        state = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
        if state["data_identity"] != data_identity or state["batch_size"] != batch_size:
            raise ValueError("Resume data or batch size differs from the saved run")
        if (run_dir / "evaluation.json").exists():
            raise ValueError("This run has already evaluated its test set; start a new experiment explicitly")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scaler.load_state_dict(state["scaler"])
        torch.set_rng_state(state["rng"])
        generator.set_state(state["sampler_rng"])
        if device == "cuda" and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        history, best, stale = state["history"], state["best"], state["stale"]
        start_epoch = state["epoch"] + 1
    save_json(run_dir / "run_config.json", {"repository": repository, "revision": revision,
              "epochs_requested": epochs, "batch_size": batch_size, "data_identity": data_identity,
              "data_directory": str(root), "notice": notice, "torch_version": str(torch.__version__),
              "sampling": "All training data eligible; class-balanced replacement sampling per epoch"})
    print("Training on", torch.cuda.get_device_name(0) if device == "cuda" else "CPU", flush=True)
    print("Checkpoints:", run_dir, flush=True)
    for epoch in range(start_epoch, epochs + 1):
        if stale >= 5:
            break
        model.train()
        total, started = 0.0, time.monotonic()
        for step, (features, target) in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                output = model(features.to(device))
                loss = nn.functional.binary_cross_entropy_with_logits(output, target.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(target)
            if step % 100 == 0 or step == len(train_loader):
                print(f"Epoch {epoch}/{epochs}, batch {step}/{len(train_loader)}, loss {loss.item():.4f}", flush=True)
        dev_loss, labels, scores = evaluate(model, dev_loader, device)
        metrics = binary_metrics(labels, scores)
        history.append({"epoch": epoch, "train_loss": total / len(train_rows), "dev_loss": dev_loss,
                        "dev_metrics": metrics, "seconds": time.monotonic() - started})
        print(f"Epoch {epoch}: dev F1={metrics['f1']:.4f}; AUC={metrics['roc_auc']:.4f}; loss={dev_loss:.4f}", flush=True)
        if dev_loss < best:
            best, stale = dev_loss, 0
            atomic_torch_save({"schema": "english-cnn-v1", "model": model.state_dict(),
                              "epoch": epoch, "threshold": .5, "data_identity": data_identity,
                              "frontend": CACHE_VERSION, "notice": notice}, run_dir / "best.pt")
        else:
            stale += 1
        atomic_torch_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                          "scaler": scaler.state_dict(), "rng": torch.get_rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else [],
                          "sampler_rng": generator.get_state(), "history": history, "epoch": epoch,
                          "best": best, "stale": stale, "batch_size": batch_size,
                          "data_identity": data_identity}, run_dir / "last.pt")
        save_json(run_dir / "history.json", history)
    if not (run_dir / "best.pt").exists():
        raise ValueError("No trained checkpoint available")
    print("Training finished. Loading held-out test data for final evaluation...", flush=True)
    test_x, test_rows = load_cache(root, "test", device)
    check_partitions({"train": train_rows, "validation": dev_rows, "test": test_rows})
    state = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    test_loader = DataLoader(FeatureDataset(test_x, test_rows), batch_size=batch_size, num_workers=0)
    test_loss, labels, scores = evaluate(model, test_loader, device)
    report = {"status": "english_experiment_complete", "notice": notice, "repository": repository,
              "revision": revision, "counts": {s: {str(k): v for k, v in n.items()} for s, n in counts.items()},
              "best_epoch": state["epoch"], "checkpoint": str(run_dir / "best.pt"), "test_loss": test_loss,
              "test_metrics": binary_metrics(labels, scores), "threshold_policy": "Fixed 0.5; epoch chosen only by dev loss"}
    save_json(run_dir / "evaluation.json", report)
    save_json(run_dir / "test_predictions.json", [{"id": r["id"], "label": y, "synthetic_score": p}
              for r, y, p in zip(test_rows, labels, scores)])
    print(json.dumps(report, indent=2), flush=True)
    return run_dir


def predict(checkpoint, path, device="cuda"):
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state.get("schema") != "english-cnn-v1" or state.get("frontend") != CACHE_VERSION:
        raise ValueError("Not an English CNN checkpoint")
    if Path(path).stat().st_size > 32 * 1024**2:
        raise ValueError("Prediction input exceeds 32 MiB")
    wave, details = waveform_window(Path(path).read_bytes())
    frontend, model = Frontend().to(device).eval(), EnglishCNN().to(device).eval()
    model.load_state_dict(state["model"])
    (score,), (usable,) = score_features(frontend, model, torch.from_numpy(wave).unsqueeze(0).to(device))
    if not usable:
        raise ValueError("Model returned an invalid score for this recording")
    result = {"prediction": "synthetic" if score >= state["threshold"] else "bonafide",
              "synthetic_score": score, "calibrated_probability": False, "window": details, "notice": state["notice"]}
    print(json.dumps(result, indent=2))
    return result

"""Small log-mel CNN baseline. Run through scripts/run_baseline.py."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import random

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchaudio

from vaanirakshak.baseline_download import Downloader, file_sha, make_plan, save_json
from vaanirakshak.baseline_rules import (
    LANGUAGES, WARNING, assigned_split, balanced_sample, binary_metrics, references,
    remove_reference_conflicts, speaker_keys, validate_rows,
)
from vaanirakshak.data.preprocessing import convert_to_mono, resample_waveform

RATE = 16000
LENGTH = 4 * RATE


def audio_window(file):
    """Identical bounded centre-window preprocessing for acquisition and prediction."""
    with sf.SoundFile(file) as stream:
        rate, channels, frames = stream.samplerate, stream.channels, len(stream)
        duration = frames / rate
        if not 1 <= duration <= 30 or not 8000 <= rate <= 192000 or not 1 <= channels <= 8:
            raise ValueError("Audio must be 1–30 seconds, 8–192 kHz, and 1–8 channels")
        start = max(0, (frames - 4 * rate) // 2)
        stream.seek(start)
        waveform = stream.read(min(frames, 4 * rate), dtype="float32", always_2d=True)
        info = {"original_duration_seconds": duration, "original_sample_rate": rate,
                "original_channels": channels, "original_format": stream.format,
                "original_subtype": stream.subtype, "window_start_seconds": start / rate,
                "padded": frames < 4 * rate}
    if not np.isfinite(waveform).all():
        raise ValueError("Non-finite audio")
    mono = resample_waveform(convert_to_mono(waveform), rate, RATE)
    if float(np.sqrt(np.mean(mono**2))) < 1e-5:
        raise ValueError("Silent window")
    result = np.zeros(LENGTH, dtype=np.float32)
    result[:min(LENGTH, len(mono))] = mono[:LENGTH]
    return np.clip(result, -1, 1), info


def prepare(root, token):
    import pyarrow.parquet as pq

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        validate_rows(manifest["rows"])
        print("Using existing prepared manifest:", manifest_path, flush=True)
        return manifest_path
    client = Downloader(token)
    plan = make_plan(client)
    plan_path = root / "download_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing download plan differs; use a new output directory")
    save_json(plan_path, plan)
    print(f'Original shard download plan: {plan["planned_bytes"] / 1024**3:.2f} GiB; '
          'hard limit 8 GiB. Completed shards are reused.', flush=True)
    all_rows, rejected = [], Counter()
    seen_audio, seen_windows = set(), set()
    (root / "processed").mkdir(exist_ok=True)
    report_path = root / "preparation_report.json"
    try:
        for item in plan["files"]:
            shard = client.download(item, root / "shards")
            source, language = item["source"], item["language"]
            label = int(source == "indicsynth")
            candidates = {}
            with pq.ParquetFile(shard) as parquet:
                # Read speaker metadata before any audio decoding.
                fields = (['speaker_id'] if label == 0 else ["Source Speaker_ID", "Target Speaker ID",
                          "Generative Model", "Source Reference Audio", "Target Reference Audio"])
                if not set(fields + ["audio"]).issubset(parquet.schema_arrow.names):
                    raise ValueError(f"Original schema differs for {source}/{language}")
                if parquet.metadata.num_rows > 200000:
                    raise ValueError("Shard row count exceeds baseline limit")
                metadata = parquet.read(columns=fields).to_pylist()
                indices = list(range(len(metadata)))
                random.Random(item["path"] + source).shuffle(indices)
                per_split = Counter()
                for index in indices:
                    row = metadata[index]
                    try:
                        speakers = speaker_keys(row, source)
                    except ValueError:
                        rejected["missing_identity_or_generator"] += 1
                        continue
                    split = assigned_split(speakers)
                    if split is None:
                        rejected["source_target_cross_partition"] += 1
                        continue
                    if per_split[split] >= (115 if split == "train" else 40):
                        continue
                    per_split[split] += 1
                    candidates[index] = (row, speakers, split)
                offset = 0
                for batch in parquet.iter_batches(batch_size=4, columns=["audio"], use_threads=False):
                    if not candidates:
                        break
                    for local, audio in enumerate(batch.column(0).to_pylist()):
                        index = offset + local
                        selected = candidates.pop(index, None)
                        if selected is None:
                            continue
                        row, speakers, split = selected
                        raw = audio.get("bytes") if isinstance(audio, dict) else None
                        if not isinstance(raw, bytes) or not 0 < len(raw) <= 32 * 1024**2:
                            rejected["no_bounded_embedded_original"] += 1
                            continue
                        digest = hashlib.sha256(raw).hexdigest()
                        if digest in seen_audio:
                            rejected["duplicate_original"] += 1
                            continue
                        try:
                            waveform, info = audio_window(io.BytesIO(raw))
                        except (ValueError, RuntimeError):
                            rejected["audio_validation"] += 1
                            continue
                        # Hash the actual PCM16 representation used by the model.
                        buffer = io.BytesIO()
                        sf.write(buffer, waveform, RATE, format="WAV", subtype="PCM_16")
                        encoded = buffer.getvalue()
                        window_sha = hashlib.sha256(encoded).hexdigest()
                        if window_sha in seen_windows:
                            rejected["duplicate_processed_window"] += 1
                            continue
                        rid = hashlib.sha256(f'{source}:{item["revision"]}:{item["path"]}:{index}'.encode()).hexdigest()
                        relative = f"processed/{rid}.wav"
                        (root / relative).write_bytes(encoded)
                        seen_audio.add(digest)
                        seen_windows.add(window_sha)
                        all_rows.append({"id": rid, "path": relative, "label": label, "split": split,
                                         "language": language, "speakers": speakers,
                                         "references": references(row, source),
                                         "generator": row.get("Generative Model", "bona_fide"),
                                         "repository": item["repository"], "revision": item["revision"],
                                         "source_split": "train", "source_shard": item["path"],
                                         "source_row": index, "original_path_metadata": audio.get("path"),
                                         "audio_sha256": digest, "window_sha256": window_sha, **info})
                    offset += batch.num_rows
            print(f'Prepared {source}/{language} shard; {len(all_rows)} valid windows so far', flush=True)
            save_json(report_path, {"status": "preparing", "valid_windows": len(all_rows),
                                    "rejections": dict(rejected), "downloaded_bytes": client.bytes_read})
        filtered = remove_reference_conflicts(all_rows)
        rejected["reference_cross_partition"] = len(all_rows) - len(filtered)
        save_json(report_path, {"status": "checking_splits", "rejections": dict(rejected),
                               "available_cells": dict(Counter(f'{r["split"]}/{r["language"]}/{r["label"]}' for r in filtered)),
                               "downloaded_bytes": client.bytes_read, "warning": WARNING})
        selected = balanced_sample(filtered)
        manifest = {"schema": "baseline-manifest-v1", "seed": 42, "sample_rate": RATE,
                    "window_samples": LENGTH, "window_policy": "centre_crop_or_end_pad",
                    "labels": {"0": "bona_fide", "1": "synthetic"}, "warning": WARNING, "rows": selected}
        save_json(manifest_path, manifest)
        save_json(report_path, {"status": "prepared", "recordings": len(selected),
                               "counts": validate_rows(selected), "rejections": dict(rejected),
                               "generators": dict(Counter(str(r["generator"]) for r in selected)),
                               "downloaded_bytes": client.bytes_read, "warning": WARNING})
        print("Prepared", len(selected), "recordings:", manifest_path, flush=True)
        return manifest_path
    except Exception:
        if not report_path.exists():
            save_json(report_path, {"status": "incomplete", "valid_windows": len(all_rows),
                                    "rejections": dict(rejected), "downloaded_bytes": client.bytes_read})
        raise


class AudioDataset(Dataset):
    def __init__(self, root, rows):
        self.root, self.rows = Path(root), rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        waveform, rate = sf.read(self.root / row["path"], dtype="float32")
        if rate != RATE or waveform.shape != (LENGTH,) or not np.isfinite(waveform).all():
            raise ValueError("Prepared waveform is invalid")
        return torch.from_numpy(waveform), torch.tensor(row["label"], dtype=torch.float32)


class AudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=RATE, n_fft=512,
                   win_length=400, hop_length=160, n_mels=64, f_min=20, f_max=7600)
        layers = []
        for incoming, outgoing in ((1, 16), (16, 32), (32, 64)):
            layers.extend([nn.Conv2d(incoming, outgoing, 3, padding=1), nn.BatchNorm2d(outgoing),
                           nn.ReLU(), nn.MaxPool2d(2)])
        self.encoder = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
                                     nn.Dropout(0.25), nn.Linear(64, 1))

    def forward(self, waveform):
        # FFT and logarithms stay FP32 even under mixed precision.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            mel = self.mel(waveform.float()).clamp_min(1e-8).log()
            mel = (mel - mel.mean(dim=(1, 2), keepdim=True)) / mel.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        return self.encoder(mel.unsqueeze(1)).squeeze(1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    labels, scores = [], []
    total_loss = 0.0
    for wave, target in loader:
        logits = model(wave.to(device))
        total_loss += nn.functional.binary_cross_entropy_with_logits(logits, target.to(device), reduction="sum").item()
        labels.extend(target.int().tolist())
        scores.extend(logits.sigmoid().cpu().tolist())
    return total_loss / len(labels), labels, scores


def train(manifest_path, epochs=15, batch_size=16, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; install the GPU-enabled PyTorch pair first")
    torch.set_num_threads(4)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed_all(42)
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "baseline-manifest-v1" or manifest.get("window_samples") != LENGTH:
        raise ValueError("Unsupported baseline manifest")
    rows = manifest["rows"]
    counts = validate_rows(rows)
    root = manifest_path.parent
    for row in rows:
        path = (root / row["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or file_sha(path) != row["window_sha256"]:
            raise ValueError("Processed file changed or escapes dataset directory")
    subsets = {split: [r for r in rows if r["split"] == split] for split in ("train", "dev", "test")}
    loaders = {split: DataLoader(AudioDataset(root, group), batch_size=batch_size,
               shuffle=split == "train", num_workers=0, pin_memory=device == "cuda") for split, group in subsets.items()}
    run = Path("models") / ("baseline_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    run.mkdir(parents=True)
    model = AudioCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best, stale, history = float("inf"), 0, []
    print(WARNING, flush=True)
    print("Training on", torch.cuda.get_device_name(0) if device == "cuda" else "CPU", flush=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for wave, target in loaders["train"]:
            wave, target = wave.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device, enabled=device == "cuda"):
                logits = model(wave)
                loss = nn.functional.binary_cross_entropy_with_logits(logits, target)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is non-finite")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(target)
        dev_loss, labels, scores = evaluate(model, loaders["dev"], device)
        entry = {"epoch": epoch, "train_loss": total_loss / len(subsets["train"]),
                 "dev_loss": dev_loss, "dev_metrics": binary_metrics(labels, scores)}
        history.append(entry)
        save_json(run / "history.json", history)
        print(f'Epoch {epoch}/{epochs}: train loss {entry["train_loss"]:.4f}; '
              f'dev loss {dev_loss:.4f}; dev F1 {entry["dev_metrics"]["f1"]:.3f}', flush=True)
        if dev_loss < best:
            best, stale = dev_loss, 0
            checkpoint = {"schema": "baseline-cnn-v1", "model": model.state_dict(), "epoch": epoch,
                          "manifest_sha256": file_sha(manifest_path), "threshold": 0.5,
                          "sample_rate": RATE, "window_samples": LENGTH, "warning": WARNING,
                          "torch_version": str(torch.__version__)}
            temporary = run / "best.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(run / "best.pt")
        else:
            stale += 1
        if stale >= 4:
            print("Early stopping after four epochs without lower validation loss", flush=True)
            break
    checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    _, labels, scores = evaluate(model, loaders["test"], device)
    report = {"status": "experimental_baseline_complete", "warning": WARNING,
              "checkpoint": str(run / "best.pt"), "best_epoch": checkpoint["epoch"],
              "manifest_sha256": checkpoint["manifest_sha256"], "counts": counts,
              "threshold_policy": "fixed 0.5; best epoch selected only by dev loss",
              "test": binary_metrics(labels, scores), "test_by_language": {}}
    for language in LANGUAGES:
        indexes = [i for i, row in enumerate(subsets["test"]) if row["language"] == language]
        report["test_by_language"][language] = binary_metrics([labels[i] for i in indexes], [scores[i] for i in indexes])
    save_json(run / "evaluation.json", report)
    save_json(run / "test_predictions.json", [{"id": row["id"], "label": y, "synthetic_score": p}
              for row, y, p in zip(subsets["test"], labels, scores)])
    print(json.dumps(report, indent=2), flush=True)
    return run


def predict(checkpoint_path, audio_path, device="cuda"):
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; pass --device cpu for prediction")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("schema") != "baseline-cnn-v1":
        raise ValueError("Unsupported checkpoint")
    waveform, info = audio_window(str(audio_path))
    # Match acquisition's PCM16 conversion before inference.
    buffer = io.BytesIO()
    sf.write(buffer, waveform, RATE, format="WAV", subtype="PCM_16")
    buffer.seek(0)
    waveform, _ = sf.read(buffer, dtype="float32")
    model = AudioCNN().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        score = model(torch.from_numpy(waveform).unsqueeze(0).to(device)).sigmoid().item()
    result = {"prediction": "synthetic" if score >= checkpoint["threshold"] else "bona_fide",
              "synthetic_score": score, "score_is_calibrated_probability": False,
              "window": info, "warning": WARNING}
    print(json.dumps(result, indent=2))
    return result

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vaanirakshak.models.aasist import AASIST

# Reuse the SEA-Spoof loader we already proved works
from smoke_seaspoof import SEASpoofEnglishDataset


TRAIN_SAMPLES = 500
VAL_SAMPLES = 200

BATCH_SIZE = 2
ACCUMULATION_STEPS = 4

EPOCHS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

CHECKPOINT_DIR = Path("artifacts/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = model(audio)
                loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)

            predictions = logits.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)

    return avg_loss, accuracy


def main():

    torch.manual_seed(42)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU not available."
        )

    device = torch.device("cuda")

    print()
    print("================================")
    print("VaaniRakshak AASIST Smoke Train")
    print("================================")
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
            2,
        ),
        "GB",
    )
    print()

    # -----------------------------
    # DATASETS
    # -----------------------------

    train_dataset = SEASpoofEnglishDataset(
        split="train",
        training=True,
        max_samples=TRAIN_SAMPLES,
    )

    val_dataset = SEASpoofEnglishDataset(
        split="validation",
        training=False,
        max_samples=VAL_SAMPLES,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
    )

    # -----------------------------
    # MODEL
    # -----------------------------

    model = AASIST().to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # -----------------------------
    # OPTIMIZER
    # -----------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scaler = torch.amp.GradScaler("cuda")

    # -----------------------------
    # TRAIN
    # -----------------------------

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        optimizer.zero_grad(set_to_none=True)

        running_loss = 0.0
        samples_seen = 0

        bonafide_seen = 0
        spoof_seen = 0

        for step, batch in enumerate(
            train_loader,
            start=1,
        ):

            audio = batch["audio"].to(
                device,
                non_blocking=True,
            )

            labels = batch["label"].to(
                device,
                non_blocking=True,
            )

            bonafide_seen += (
                labels == 0
            ).sum().item()

            spoof_seen += (
                labels == 1
            ).sum().item()

            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
            ):
                logits = model(audio)

                loss = criterion(
                    logits,
                    labels,
                )

                loss = (
                    loss
                    / ACCUMULATION_STEPS
                )

            scaler.scale(loss).backward()

            if (
                step % ACCUMULATION_STEPS == 0
            ):
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad(
                    set_to_none=True
                )

            actual_loss = (
                loss.item()
                * ACCUMULATION_STEPS
            )

            running_loss += (
                actual_loss
                * labels.size(0)
            )

            samples_seen += labels.size(0)

            if step % 25 == 0:

                avg = (
                    running_loss
                    / samples_seen
                )

                allocated = (
                    torch.cuda.memory_allocated()
                    / 1024**3
                )

                reserved = (
                    torch.cuda.memory_reserved()
                    / 1024**3
                )

                print(
                    f"Epoch {epoch} | "
                    f"Step {step} | "
                    f"Samples {samples_seen} | "
                    f"Loss {avg:.4f} | "
                    f"GPU allocated "
                    f"{allocated:.2f} GB | "
                    f"reserved "
                    f"{reserved:.2f} GB"
                )

        # Finish remaining gradients
        if (
            step % ACCUMULATION_STEPS
            != 0
        ):
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(
                set_to_none=True
            )

        train_loss = (
            running_loss
            / max(samples_seen, 1)
        )

        print()
        print(
            f"Epoch {epoch} training done"
        )
        print(
            f"Train loss: "
            f"{train_loss:.4f}"
        )
        print(
            f"Bonafide samples: "
            f"{bonafide_seen}"
        )
        print(
            f"Spoof samples: "
            f"{spoof_seen}"
        )

        # -------------------------
        # VALIDATE
        # -------------------------

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Validation loss: "
            f"{val_loss:.4f}"
        )

        print(
            f"Validation accuracy: "
            f"{val_accuracy:.4%}"
        )

        # -------------------------
        # SAVE
        # -------------------------

        checkpoint = {
            "epoch": epoch,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy":
                val_accuracy,
        }

        torch.save(
            checkpoint,
            CHECKPOINT_DIR / "smoke_last.pt",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                checkpoint,
                CHECKPOINT_DIR
                / "smoke_best.pt",
            )

            print(
                "Saved new best checkpoint."
            )

        print()
        print("----------------------------")
        print()

    print(
        "SMOKE TRAINING COMPLETE"
    )

    print(
        "Best validation loss:",
        best_val_loss,
    )


if __name__ == "__main__":
    main()
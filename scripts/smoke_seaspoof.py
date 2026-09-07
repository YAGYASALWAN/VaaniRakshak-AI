import io

import numpy as np
import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader


SAMPLE_RATE = 16000
TARGET_SECONDS = 4
TARGET_SAMPLES = 64600

LABEL_MAP = {
    "bonafide": 0,
    "spoof": 1,
}


class SEASpoofEnglishDataset(IterableDataset):
    def __init__(self, split="train", training=True, max_samples=None):
        self.split = split
        self.training = training
        self.max_samples = max_samples

    def _load_stream(self):
        ds = load_dataset(
            "Jack-ppkdczgx/SEA-Spoof",
            split=self.split,
            streaming=True,

            # IMPORTANT:
            # filter English at parquet level
            filters=[("language", "==", "en")],

            columns=[
                "audio",
                "row_id",
                "language",
                "label",
                "spoof_type",
                "sampling_rate",
            ],
        )

        # Prevent TorchCodec from decoding.
        # We decode FLAC ourselves with soundfile.
        ds = ds.decode(False)

        if self.training:
            ds = ds.shuffle(
                seed=42,
                buffer_size=2000,
            )

        return ds

    def _decode_audio(self, example):
        audio_bytes = example["audio"]["bytes"]

        waveform, sr = sf.read(
            io.BytesIO(audio_bytes),
            dtype="float32",
        )

        # stereo -> mono
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)

        # resample only if necessary
        if sr != SAMPLE_RATE:
            waveform_tensor = torch.from_numpy(waveform).unsqueeze(0)

            waveform_tensor = torchaudio.functional.resample(
                waveform_tensor,
                sr,
                SAMPLE_RATE,
            )

            waveform = waveform_tensor.squeeze(0).numpy()

        return waveform.astype(np.float32)

    def _crop_or_pad(self, waveform):
        n = len(waveform)

        if n > TARGET_SAMPLES:

            if self.training:
                start = np.random.randint(
                    0,
                    n - TARGET_SAMPLES + 1,
                )
            else:
                # deterministic center crop
                start = (n - TARGET_SAMPLES) // 2

            waveform = waveform[
                start:start + TARGET_SAMPLES
            ]

        elif n < TARGET_SAMPLES:
            pad = TARGET_SAMPLES - n

            waveform = np.pad(
                waveform,
                (0, pad),
            )

        return waveform

    def __iter__(self):
        ds = self._load_stream()

        count = 0

        for example in ds:
            waveform = self._decode_audio(example)
            waveform = self._crop_or_pad(waveform)

            label_name = example["label"]

            if label_name not in LABEL_MAP:
                continue

            yield {
                "audio": torch.tensor(
                    waveform,
                    dtype=torch.float32,
                ),

                "label": torch.tensor(
                    LABEL_MAP[label_name],
                    dtype=torch.long,
                ),

                "row_id": example["row_id"],
                "label_name": label_name,
            }

            count += 1

            if (
                self.max_samples is not None
                and count >= self.max_samples
            ):
                break


if __name__ == "__main__":

    print("Creating SEA-Spoof English stream...")

    dataset = SEASpoofEnglishDataset(
        split="train",
        training=True,
        max_samples=32,
    )

    loader = DataLoader(
        dataset,
        batch_size=8,
        num_workers=0,
    )

    print("Loading first batch...")

    batch = next(iter(loader))

    print()
    print("SUCCESS")
    print("---------------------------")
    print("Audio shape:", batch["audio"].shape)
    print("Label shape:", batch["label"].shape)
    print("Labels:", batch["label"])
    print("Label names:", batch["label_name"])
    print("Audio dtype:", batch["audio"].dtype)
    print("Min amplitude:", batch["audio"].min().item())
    print("Max amplitude:", batch["audio"].max().item())
    print("---------------------------")

    assert batch["audio"].shape == (8, 64000)

    print()
    print("SEA-Spoof -> PyTorch pipeline WORKS.")
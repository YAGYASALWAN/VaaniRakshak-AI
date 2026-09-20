import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import Dataset

import vaanirakshak.v2_train_ssl as train_ssl


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def freeze_feature_encoder(self):
        return None

    def gradient_checkpointing_enable(self):
        return None


class TinyAntiSpoof(nn.Module):
    pretrained_calls = 0
    exported_calls = 0

    def __init__(self):
        super().__init__()
        self.backbone = TinyBackbone()
        self.head = nn.Linear(1, 1)

    @classmethod
    def from_pretrained(cls, backbone_id):
        cls.pretrained_calls += 1
        return cls()

    @classmethod
    def from_exported_config(cls, backbone_config, model_spec):
        cls.exported_calls += 1
        if backbone_config.get("kind") != "tiny" or model_spec.get("kind") != "tiny-head":
            raise ValueError("unexpected tiny model config")
        return cls()

    def export_model_config(self):
        return {
            "backbone_config": {"kind": "tiny"},
            "model_spec": {"kind": "tiny-head"},
        }

    def forward(self, values, attention_mask=None):
        feature = values.mean(dim=1, keepdim=True) * self.backbone.scale
        return self.head(feature).squeeze(-1)


class TinyDataset(Dataset):
    def __init__(self, records, manifest_dir, split, *, seed=42):
        self.records = [record for record in records if record.split == split]
        self.epoch = 0
        if not self.records:
            raise ValueError("empty tiny split")

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        label = 1.0 if record.label == "spoof" else 0.0
        # Distinct class means make development ranking deterministic enough for
        # the metrics path while keeping the tensors tiny and fast.
        value = 0.8 if label else -0.8
        values = torch.full((64,), value, dtype=torch.float32)
        mask = torch.ones(64, dtype=torch.long)
        return values, mask, torch.tensor(label, dtype=torch.float32)


def records():
    return [
        SimpleNamespace(record_id="tr-real", dataset="tiny", split="train", label="bonafide"),
        SimpleNamespace(record_id="tr-fake", dataset="tiny", split="train", label="spoof"),
        SimpleNamespace(record_id="dev-real", dataset="tiny", split="dev", label="bonafide"),
        SimpleNamespace(record_id="dev-fake", dataset="tiny", split="dev", label="spoof"),
    ]


class AuditOK:
    counts = {"records": 4}

    def raise_for_errors(self):
        return None


class TrainingResumeSmokeTests(unittest.TestCase):
    def setUp(self):
        TinyAntiSpoof.pretrained_calls = 0
        TinyAntiSpoof.exported_calls = 0

    def test_interrupted_epoch_resumes_from_last_completed_epoch_offline(self):
        items = records()
        real_evaluate = train_ssl.evaluate
        evaluation_calls = {"count": 0}

        def interrupt_second_evaluation(model, loader, device):
            evaluation_calls["count"] += 1
            if evaluation_calls["count"] == 2:
                raise RuntimeError("simulated interruption")
            return real_evaluate(model, loader, device)

        common_patches = [
            patch.object(train_ssl, "WavLMAntiSpoof", TinyAntiSpoof),
            patch.object(train_ssl, "ManifestAudioDataset", TinyDataset),
            patch.object(train_ssl, "load_jsonl", return_value=items),
            patch.object(train_ssl, "audit_manifest", return_value=AuditOK()),
            patch.object(train_ssl, "manifest_fingerprint", return_value="tiny-manifest-v1"),
        ]

        with tempfile.TemporaryDirectory() as root:
            manifest = Path(root) / "manifest.jsonl"
            manifest.write_text("test fixture\n", encoding="utf-8")
            output = Path(root) / "run"

            for context in common_patches:
                context.start()
            self.addCleanup(lambda: [context.stop() for context in reversed(common_patches)])

            with patch.object(train_ssl, "evaluate", side_effect=interrupt_second_evaluation):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    train_ssl.run(
                        manifest,
                        output,
                        backbone_id="tiny/backbone",
                        epochs=2,
                        batch_size=2,
                        gradient_accumulation_steps=1,
                        gradient_checkpointing=True,
                        device="cpu",
                        head_only_epochs=1,
                        seed=7,
                    )

            self.assertTrue((output / "best.pt").is_file())
            self.assertTrue((output / "last.pt").is_file())
            first_state = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(first_state["epoch"], 1)
            self.assertEqual(len(first_state["history"]), 1)
            self.assertEqual(first_state["backbone_config"]["kind"], "tiny")
            self.assertEqual(TinyAntiSpoof.pretrained_calls, 1)

            best = train_ssl.run(
                manifest,
                output,
                backbone_id="tiny/backbone",
                epochs=2,
                batch_size=2,
                gradient_accumulation_steps=1,
                gradient_checkpointing=True,
                device="cpu",
                head_only_epochs=1,
                seed=7,
                resume=True,
            )

            self.assertEqual(best, output.resolve() / "best.pt")
            resumed_state = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            self.assertEqual(resumed_state["epoch"], 2)
            self.assertEqual([entry["epoch"] for entry in resumed_state["history"]], [1, 2])
            self.assertEqual(TinyAntiSpoof.pretrained_calls, 1, "resume must not call from_pretrained again")
            self.assertGreaterEqual(TinyAntiSpoof.exported_calls, 1, "resume must reconstruct from saved config")


if __name__ == "__main__":
    unittest.main()

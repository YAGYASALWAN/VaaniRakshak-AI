import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

import vaanirakshak.v2_calibrate as calibrate_module
from vaanirakshak.v2_detectors import CALIBRATION_METHOD, V2_CHECKPOINT_SCHEMA


class AuditOK:
    counts = {"records": 4}

    def raise_for_errors(self):
        return None


class TinyDevDataset:
    def __init__(self, records, manifest_dir, split, *, seed=0):
        self.records = [record for record in records if record.split == split]
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        label = 1.0 if record.label == "spoof" else 0.0
        value = 0.8 if label else -0.8
        return torch.full((64,), value), torch.ones(64, dtype=torch.long), torch.tensor(label)


class FakeDetector:
    def __init__(self, checkpoint, device="cpu"):
        self.threshold = 0.65

    def score(self, values, sample_rate):
        return 0.8 if float(values.mean()) > 0 else 0.2


def fake_records():
    return [
        SimpleNamespace(record_id="tr-real", split="train", label="bonafide"),
        SimpleNamespace(record_id="tr-fake", split="train", label="spoof"),
        SimpleNamespace(record_id="dev-real", split="dev", label="bonafide"),
        SimpleNamespace(record_id="dev-fake", split="dev", label="spoof"),
    ]


class V2CalibrationCommandTests(unittest.TestCase):
    def write_checkpoint(self, root, *, fingerprint="fixture-fingerprint", calibrated=False):
        path = Path(root) / "best.pt"
        state = {
            "schema": V2_CHECKPOINT_SCHEMA,
            "architecture": "wavlm-base-plus-attentive-v1",
            "frontend": "raw-16khz-v1",
            "model": {"dummy": torch.tensor([1.0])},
            "backbone_config": {"hidden_size": 4},
            "model_spec": {"hidden_size": 4},
            "threshold": 0.65,
            "calibrated_probability": calibrated,
            "model_name": "fixture-model",
            "training": {"manifest_fingerprint": fingerprint},
        }
        if calibrated:
            state["calibration"] = {"method": CALIBRATION_METHOD, "temperature": 1.5}
        torch.save(state, path)
        return path

    def common_patches(self, *, fingerprint="fixture-fingerprint"):
        return [
            patch.object(calibrate_module, "load_jsonl", return_value=fake_records()),
            patch.object(calibrate_module, "audit_manifest", return_value=AuditOK()),
            patch.object(calibrate_module, "manifest_fingerprint", return_value=fingerprint),
            patch.object(calibrate_module, "ManifestAudioDataset", TinyDevDataset),
            patch.object(calibrate_module, "CheckpointDetector", FakeDetector),
        ]

    def test_writes_calibrated_checkpoint_without_overwriting_source(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = Path(root) / "manifest.jsonl"
            manifest.write_text("fixture\n", encoding="utf-8")
            source = self.write_checkpoint(root)
            output = Path(root) / "best_calibrated.pt"

            contexts = self.common_patches()
            for context in contexts:
                context.start()
            try:
                report = calibrate_module.calibrate(manifest, source, output, device="cpu")
            finally:
                for context in reversed(contexts):
                    context.stop()

            self.assertTrue(source.is_file())
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".pt.calibration.json").is_file())
            state = torch.load(output, map_location="cpu", weights_only=True)
            self.assertTrue(state["calibrated_probability"])
            self.assertEqual(state["calibration"]["method"], CALIBRATION_METHOD)
            self.assertEqual(state["calibration"]["fit_split"], "dev")
            self.assertFalse(state["calibration"]["test_data_used"])
            self.assertTrue(state["calibration"]["decision_preserving_threshold_transform"])
            self.assertNotEqual(state["threshold"], 0.65)
            self.assertEqual(report["calibration"]["records"], 2)
            self.assertLessEqual(
                report["calibration"]["metrics_after"]["nll"],
                report["calibration"]["metrics_before"]["nll"] + 1e-8,
            )

    def test_manifest_fingerprint_must_match_training_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = Path(root) / "manifest.jsonl"
            manifest.write_text("fixture\n", encoding="utf-8")
            source = self.write_checkpoint(root, fingerprint="training-fingerprint")
            output = Path(root) / "calibrated.pt"

            contexts = self.common_patches(fingerprint="different-manifest")
            for context in contexts:
                context.start()
            try:
                with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                    calibrate_module.calibrate(manifest, source, output)
            finally:
                for context in reversed(contexts):
                    context.stop()

    def test_already_calibrated_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = Path(root) / "manifest.jsonl"
            manifest.write_text("fixture\n", encoding="utf-8")
            source = self.write_checkpoint(root, calibrated=True)
            output = Path(root) / "calibrated-again.pt"

            contexts = self.common_patches()
            for context in contexts:
                context.start()
            try:
                with self.assertRaisesRegex(ValueError, "already marked calibrated"):
                    calibrate_module.calibrate(manifest, source, output)
            finally:
                for context in reversed(contexts):
                    context.stop()

    def test_source_checkpoint_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = Path(root) / "manifest.jsonl"
            manifest.write_text("fixture\n", encoding="utf-8")
            source = self.write_checkpoint(root)
            with self.assertRaisesRegex(ValueError, "new checkpoint"):
                calibrate_module.calibrate(manifest, source, source)


if __name__ == "__main__":
    unittest.main()

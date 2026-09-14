import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import vaanirakshak.v2_calibrate as calibrate_module
from vaanirakshak.v2_detectors import CALIBRATION_METHOD, V2_CHECKPOINT_SCHEMA


RATE = 16_000


class AuditOK:
    counts = {"records": 4}

    def raise_for_errors(self):
        return None


class FakeDetector:
    def __init__(self, checkpoint, device="cpu"):
        self.threshold = 0.65

    def score(self, values, sample_rate):
        return 0.8 if float(values.mean()) > 0 else 0.2


def fake_records():
    return [
        SimpleNamespace(record_id="tr-real", split="train", label="bonafide", audio_ref="tr-real.wav"),
        SimpleNamespace(record_id="tr-fake", split="train", label="spoof", audio_ref="tr-fake.wav"),
        SimpleNamespace(record_id="dev-real", split="dev", label="bonafide", audio_ref="dev-real.wav"),
        SimpleNamespace(record_id="dev-fake", split="dev", label="spoof", audio_ref="dev-fake.wav"),
    ]


def fake_read(path):
    value = 0.6 if "fake" in str(path) else -0.6
    # Ten seconds creates several overlapping four-second windows, forcing the
    # calibration command to exercise deterministic capped window selection.
    return np.full(10 * RATE, value, dtype=np.float32)


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
            patch.object(calibrate_module, "CheckpointDetector", FakeDetector),
            patch.object(calibrate_module, "_read_16khz", side_effect=fake_read),
        ]

    def test_window_selector_spreads_capped_windows_across_recording(self):
        wave = np.arange(10 * RATE, dtype=np.float32)
        all_windows = calibrate_module._windows(wave)
        selected = calibrate_module._select_calibration_windows(wave, maximum=4)
        self.assertEqual(len(selected), 4)
        np.testing.assert_array_equal(selected[0], all_windows[0])
        np.testing.assert_array_equal(selected[-1], all_windows[-1])

    def test_writes_recording_balanced_calibrated_checkpoint_without_overwriting_source(self):
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
            metadata = state["calibration"]
            self.assertTrue(state["calibrated_probability"])
            self.assertEqual(metadata["method"], CALIBRATION_METHOD)
            self.assertEqual(metadata["fit_split"], "dev")
            self.assertEqual(metadata["calibration_unit"], calibrate_module.CALIBRATION_UNIT)
            self.assertTrue(metadata["recording_balanced_fit"])
            self.assertEqual(metadata["development_recordings"], 2)
            self.assertEqual(metadata["calibration_windows"], 8)
            self.assertEqual(metadata["min_windows_per_recording"], 4)
            self.assertEqual(metadata["max_windows_observed_per_recording"], 4)
            self.assertFalse(metadata["test_data_used"])
            self.assertTrue(metadata["decision_preserving_threshold_transform"])
            self.assertNotEqual(state["threshold"], 0.65)
            self.assertLessEqual(
                metadata["recording_balanced_nll_after"],
                metadata["recording_balanced_nll_before"] + 1e-8,
            )
            self.assertEqual(report["calibration"]["calibration_windows"], 8)

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

    def test_invalid_window_cap_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = Path(root) / "manifest.jsonl"
            manifest.write_text("fixture\n", encoding="utf-8")
            source = self.write_checkpoint(root)
            output = Path(root) / "calibrated.pt"
            with self.assertRaisesRegex(ValueError, "max_windows_per_recording"):
                calibrate_module.calibrate(
                    manifest,
                    source,
                    output,
                    max_windows_per_recording=0,
                )


if __name__ == "__main__":
    unittest.main()

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from vaanirakshak.english_training import CACHE_VERSION, EnglishCNN
from vaanirakshak.v2_detectors import (
    CALIBRATION_METHOD,
    CheckpointDetector,
    V2_CHECKPOINT_SCHEMA,
)


class CheckpointDetectorTests(unittest.TestCase):
    def _write_checkpoint(
        self,
        root,
        *,
        schema="english-cnn-v1",
        threshold=0.57,
        calibrated=False,
        temperature=2.0,
    ):
        path = Path(root) / "best.pt"
        state = {
            "schema": schema,
            "architecture": "english-cnn-v1" if schema == V2_CHECKPOINT_SCHEMA else None,
            "model": EnglishCNN().state_dict(),
            "threshold": threshold,
            "frontend": CACHE_VERSION,
            "notice": "unit-test checkpoint",
            "calibrated_probability": calibrated,
            "model_name": "unit-test-model",
        }
        if calibrated:
            state["calibration"] = {
                "method": CALIBRATION_METHOD,
                "temperature": temperature,
                "fit_split": "dev",
            }
        torch.save(state, path)
        return path

    def test_legacy_checkpoint_is_refused_by_default(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write_checkpoint(root)
            with self.assertRaisesRegex(ValueError, "Legacy V1 checkpoint refused"):
                CheckpointDetector(path)

    def test_legacy_checkpoint_can_be_opted_in_for_integration_only(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write_checkpoint(root, threshold=0.61)
            detector = CheckpointDetector(path, allow_legacy=True)
            self.assertEqual(detector.mode, "legacy-experimental")
            self.assertEqual(detector.threshold, 0.61)
            self.assertFalse(detector.calibrated_probability)
            self.assertIn("Legacy V1", detector.notice)

    def test_v2_checkpoint_metadata_and_real_inference_path(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write_checkpoint(
                root,
                schema=V2_CHECKPOINT_SCHEMA,
                threshold=0.73,
                calibrated=True,
                temperature=1.7,
            )
            detector = CheckpointDetector(path)
            self.assertEqual(detector.mode, "trained")
            self.assertEqual(detector.name, "unit-test-model")
            self.assertEqual(detector.threshold, 0.73)
            self.assertTrue(detector.calibrated_probability)
            self.assertAlmostEqual(detector.temperature, 1.7)
            self.assertEqual(detector.info.calibration["method"], CALIBRATION_METHOD)

            expected_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(detector.checkpoint_sha256, expected_digest)
            self.assertEqual(detector.info.checkpoint_sha256, expected_digest)
            self.assertEqual(detector.info.as_dict()["checkpoint_sha256"], expected_digest)

            t = np.arange(64_000, dtype=np.float32) / 16_000.0
            wave = (0.05 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
            score = detector.score(wave, 16_000)
            self.assertTrue(np.isfinite(score))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_calibrated_checkpoint_requires_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write_checkpoint(root, schema=V2_CHECKPOINT_SCHEMA, calibrated=True)
            state = torch.load(path, map_location="cpu", weights_only=True)
            del state["calibration"]
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "missing calibration metadata"):
                CheckpointDetector(path)

    def test_invalid_calibration_temperature_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write_checkpoint(
                root,
                schema=V2_CHECKPOINT_SCHEMA,
                calibrated=True,
                temperature=0.0,
            )
            with self.assertRaisesRegex(ValueError, "temperature"):
                CheckpointDetector(path)

    def test_uncalibrated_checkpoint_cannot_carry_active_calibration_metadata(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write_checkpoint(root, schema=V2_CHECKPOINT_SCHEMA, calibrated=False)
            state = torch.load(path, map_location="cpu", weights_only=True)
            state["calibration"] = {"method": CALIBRATION_METHOD, "temperature": 2.0}
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "calibration metadata"):
                CheckpointDetector(path)

    def test_invalid_threshold_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            path = self._write_checkpoint(root, schema=V2_CHECKPOINT_SCHEMA, threshold=1.2)
            with self.assertRaisesRegex(ValueError, "threshold"):
                CheckpointDetector(path)

    def test_missing_checkpoint_is_not_silently_ignored(self):
        with self.assertRaises(FileNotFoundError):
            CheckpointDetector("definitely-not-a-real-checkpoint.pt")


if __name__ == "__main__":
    unittest.main()

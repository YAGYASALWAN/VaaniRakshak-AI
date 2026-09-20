import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import vaanirakshak.v2_evaluate as evaluate_module
from vaanirakshak.v2_calibration import apply_temperature, transform_threshold


class AuditOK:
    counts = {"records": 2}
    warnings = ()

    def raise_for_errors(self):
        return None


class Info:
    def as_dict(self):
        return {"name": "fake", "calibrated_probability": True}


class FakeDetector:
    threshold = 0.5
    calibrated_probability = True
    info = Info()

    def __init__(self, checkpoint, device="cpu"):
        return None

    def score(self, wave, sample_rate):
        return 0.8 if float(np.mean(wave)) > 0 else 0.2


class SequenceDetector:
    def __init__(self, scores):
        self.scores = iter(scores)

    def score(self, wave, sample_rate):
        return float(next(self.scores))


def records():
    return [
        SimpleNamespace(
            record_id="real-1",
            dataset="tiny",
            split="test",
            label="bonafide",
            generator_id=None,
            audio_ref="real.wav",
        ),
        SimpleNamespace(
            record_id="fake-1",
            dataset="tiny",
            split="test",
            label="spoof",
            generator_id="tiny-generator",
            audio_ref="fake.wav",
        ),
    ]


class V2EvaluateReportTests(unittest.TestCase):
    def test_frozen_report_includes_probability_quality_and_score_semantics(self):
        with tempfile.TemporaryDirectory() as root:
            manifest = Path(root) / "manifest.jsonl"
            manifest.write_text("fixture\n", encoding="utf-8")
            checkpoint = Path(root) / "best_calibrated.pt"
            checkpoint.write_bytes(b"fixture")
            output = Path(root) / "report"

            def fake_read(path):
                value = -0.5 if "real" in str(path) else 0.5
                return np.full(16_000, value, dtype=np.float32)

            with patch.object(evaluate_module, "load_jsonl", return_value=records()), patch.object(
                evaluate_module, "audit_manifest", return_value=AuditOK()
            ), patch.object(
                evaluate_module, "manifest_fingerprint", return_value="fixture-fingerprint"
            ), patch.object(
                evaluate_module, "CheckpointDetector", FakeDetector
            ), patch.object(
                evaluate_module, "_read_16khz", side_effect=fake_read
            ):
                report = evaluate_module.evaluate(manifest, checkpoint, output, scenario="standard")

            self.assertTrue(report["probability_quality"]["declared_calibrated_probability"])
            self.assertEqual(report["probability_quality"]["n"], 2)
            self.assertEqual(report["recording_aggregation"], "median_logit")
            self.assertIn("brier", report["probability_quality"])
            self.assertIn("ece", report["probability_quality"])
            self.assertIn("nll", report["probability_quality"])
            self.assertIn("probability_quality", report["by_dataset"]["tiny"])

            rows = [json.loads(line) for line in (output / "scores.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual({row["score_semantics"] for row in rows}, {"calibrated_probability"})
            self.assertEqual({row["recording_aggregation"] for row in rows}, {"median_logit"})
            self.assertEqual([row["predicted_spoof"] for row in rows], [False, True])

    def test_even_window_recording_preserves_operating_point_after_temperature_scaling(self):
        # Six seconds yields two overlapping 4-second windows (0-4 and 2-6).
        wave = np.zeros(6 * 16_000, dtype=np.float32)
        raw_window_scores = [0.35, 0.85]
        raw_threshold = 0.65
        temperature = 2.4

        raw_score, _raw_windows, _ = evaluate_module._score_recording(
            SequenceDetector(raw_window_scores), wave
        )
        calibrated_windows = apply_temperature(raw_window_scores, temperature).tolist()
        calibrated_threshold = transform_threshold(raw_threshold, temperature)
        calibrated_score, _cal_windows, _ = evaluate_module._score_recording(
            SequenceDetector(calibrated_windows), wave
        )

        self.assertEqual(raw_score >= raw_threshold, calibrated_score >= calibrated_threshold)


if __name__ == "__main__":
    unittest.main()

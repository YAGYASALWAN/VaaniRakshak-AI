import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from vaanirakshak.v2_data_contract import AudioRecord, write_jsonl
from vaanirakshak.v2_robustness import (
    center_duration,
    g711_alaw,
    g711_mulaw,
    measured_snr_db,
    narrowband_8khz,
    white_noise_at_snr,
)
from vaanirakshak.v2_robustness_eval import default_conditions, evaluate_robustness


RATE = 16_000


def sine(seconds=4.0, amplitude=0.05, frequency=440.0):
    t = np.arange(int(RATE * seconds), dtype=np.float32) / RATE
    return (amplitude * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


class RobustnessTransformTests(unittest.TestCase):
    def test_center_duration_crops_without_padding(self):
        wave = sine(seconds=5.0)
        cropped = center_duration(wave, 2.0)
        self.assertEqual(len(cropped), RATE * 2)
        short = sine(seconds=1.0)
        self.assertEqual(len(center_duration(short, 2.0)), len(short))

    def test_narrowband_roundtrip_returns_finite_16khz_length(self):
        wave = sine(seconds=2.0)
        result = narrowband_8khz(wave)
        self.assertTrue(np.isfinite(result).all())
        self.assertAlmostEqual(len(result), len(wave), delta=4)

    def test_g711_roundtrips_are_finite(self):
        wave = sine(seconds=1.0)
        for transform in (g711_mulaw, g711_alaw):
            result = transform(wave)
            self.assertTrue(np.isfinite(result).all())
            self.assertAlmostEqual(len(result), len(wave), delta=4)
            self.assertLessEqual(float(np.max(np.abs(result))), 1.0)

    def test_noise_is_deterministic_and_near_requested_snr(self):
        wave = sine(seconds=3.0, amplitude=0.03)
        first = white_noise_at_snr(wave, 10.0, key="record-1")
        second = white_noise_at_snr(wave, 10.0, key="record-1")
        different = white_noise_at_snr(wave, 10.0, key="record-2")
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, different))
        self.assertAlmostEqual(measured_snr_db(wave, first), 10.0, delta=0.2)

    def test_default_suite_contains_release_gate_conditions(self):
        self.assertEqual(
            list(default_conditions()),
            [
                "clean",
                "duration_2s",
                "duration_4s",
                "duration_8s",
                "narrowband_8khz",
                "g711_mulaw",
                "g711_alaw",
                "noise_20db",
                "noise_10db",
                "noise_5db",
            ],
        )


class _Info:
    def as_dict(self):
        return {
            "name": "fake-test-detector",
            "mode": "test",
            "schema": "test",
            "threshold": 0.5,
            "calibrated_probability": False,
            "device": "cpu",
            "notice": "test only",
            "checkpoint": None,
        }


class _FakeDetector:
    threshold = 0.5
    calibrated_probability = False
    info = _Info()

    def __init__(self, checkpoint, device="cpu"):
        self.checkpoint = checkpoint
        self.device = device

    def score(self, samples, sample_rate):
        self.assert_rate = sample_rate
        rms = float(np.sqrt(np.mean(np.square(samples), dtype=np.float64)))
        return 0.9 if rms > 0.05 else 0.1


class RobustnessEvaluatorTests(unittest.TestCase):
    def test_evaluator_uses_one_frozen_threshold_and_reports_calibration_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            audio = root / "audio"
            audio.mkdir()
            bonafide = sine(seconds=4.0, amplitude=0.02)
            spoof = sine(seconds=4.0, amplitude=0.10)
            sf.write(audio / "real.flac", bonafide, RATE, format="FLAC")
            sf.write(audio / "spoof.flac", spoof, RATE, format="FLAC")

            records = [
                AudioRecord(
                    record_id="real-1",
                    label="bonafide",
                    dataset="unit",
                    split="test",
                    audio_ref="audio/real.flac",
                    content_sha256=hashlib.sha256((audio / "real.flac").read_bytes()).hexdigest(),
                    speaker_id="human-a",
                    sample_rate=RATE,
                    codec="flac",
                ),
                AudioRecord(
                    record_id="spoof-1",
                    label="spoof",
                    dataset="unit",
                    split="test",
                    audio_ref="audio/spoof.flac",
                    content_sha256=hashlib.sha256((audio / "spoof.flac").read_bytes()).hexdigest(),
                    generator_id="generator-a",
                    speaker_id="voice-a",
                    sample_rate=RATE,
                    codec="flac",
                ),
            ]
            manifest = root / "manifest.jsonl"
            write_jsonl(manifest, records)
            checkpoint = root / "fake.pt"
            checkpoint.write_bytes(b"test")

            conditions = {
                "clean": lambda wave, key: wave,
                "quietened": lambda wave, key: wave * 0.1,
            }
            with patch("vaanirakshak.v2_robustness_eval.CheckpointDetector", _FakeDetector):
                report = evaluate_robustness(
                    manifest,
                    checkpoint,
                    root / "report",
                    conditions=conditions,
                    save_scores=True,
                )

            self.assertEqual(report["threshold_source"], "frozen checkpoint; unchanged for every robustness condition")
            self.assertEqual(report["recording_aggregation"], "median_logit")
            self.assertEqual(report["score_semantics"], "uncalibrated_detector_score")
            self.assertEqual(report["conditions"]["clean"]["threshold"], 0.5)
            self.assertEqual(report["conditions"]["quietened"]["threshold"], 0.5)
            self.assertEqual(report["conditions"]["clean"]["f1"], 1.0)
            self.assertLess(report["conditions"]["quietened"]["f1"], 1.0)

            clean_probability = report["conditions"]["clean"]["probability_quality"]
            quiet_probability = report["conditions"]["quietened"]["probability_quality"]
            for metric in ("nll", "brier", "ece"):
                self.assertIn(metric, clean_probability)
                self.assertIn(metric, quiet_probability)
                self.assertIn(f"delta_{metric}", report["degradation_vs_clean"]["quietened"])
            self.assertFalse(clean_probability["declared_calibrated_probability"])

            self.assertTrue((root / "report" / "robustness.json").is_file())
            self.assertTrue((root / "report" / "robustness_scores.jsonl").is_file())


if __name__ == "__main__":
    unittest.main()

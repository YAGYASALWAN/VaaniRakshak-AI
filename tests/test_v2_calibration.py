import unittest

import numpy as np

from vaanirakshak.v2_calibration import (
    apply_temperature,
    binary_nll,
    brier_score,
    calibration_metrics,
    expected_calibration_error,
    fit_temperature,
    transform_threshold,
)


class V2CalibrationTests(unittest.TestCase):
    def test_temperature_transform_preserves_threshold_decisions(self):
        scores = np.asarray([0.05, 0.30, 0.64, 0.65, 0.80, 0.97], dtype=np.float64)
        threshold = 0.65
        temperature = 2.3
        calibrated = apply_temperature(scores, temperature)
        calibrated_threshold = transform_threshold(threshold, temperature)
        np.testing.assert_array_equal(scores >= threshold, calibrated >= calibrated_threshold)

    def test_fit_temperature_reduces_nll_for_miscalibrated_scores(self):
        labels = [0, 0, 1, 1]
        # Two extremely confident mistakes make temperature > 1 preferable.
        scores = [0.01, 0.99, 0.01, 0.99]
        before = binary_nll(labels, scores)
        temperature = fit_temperature(labels, scores)
        after_scores = apply_temperature(scores, temperature)
        after = binary_nll(labels, after_scores)
        self.assertGreater(temperature, 1.0)
        self.assertLess(after, before)

    def test_calibration_metrics_are_finite(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        report = calibration_metrics(labels, scores)
        self.assertEqual(report["n"], 4)
        self.assertGreaterEqual(report["nll"], 0.0)
        self.assertGreaterEqual(report["brier"], 0.0)
        self.assertGreaterEqual(report["ece"], 0.0)
        self.assertLessEqual(report["ece"], 1.0)

    def test_brier_score_matches_manual_value(self):
        labels = [0, 1]
        scores = [0.25, 0.75]
        self.assertAlmostEqual(brier_score(labels, scores), 0.0625)

    def test_ece_rejects_invalid_bin_count(self):
        with self.assertRaisesRegex(ValueError, "bins"):
            expected_calibration_error([0, 1], [0.2, 0.8], bins=1)

    def test_calibration_requires_both_classes(self):
        with self.assertRaisesRegex(ValueError, "both"):
            fit_temperature([1, 1], [0.6, 0.7])

    def test_invalid_temperature_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "temperature"):
            apply_temperature([0.2, 0.8], 0.0)


if __name__ == "__main__":
    unittest.main()

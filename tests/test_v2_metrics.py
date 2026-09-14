import unittest

from vaanirakshak.v2_metrics import (
    average_precision,
    eer,
    metrics_at_threshold,
    roc_auc,
    threshold_for_max_fpr,
)


class V2MetricsTests(unittest.TestCase):
    def test_perfect_ranking_has_unit_auc(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        self.assertAlmostEqual(roc_auc(labels, scores), 1.0)
        self.assertAlmostEqual(average_precision(labels, scores), 1.0)

    def test_reversed_ranking_has_zero_roc_auc(self):
        labels = [0, 0, 1, 1]
        scores = [0.9, 0.8, 0.2, 0.1]
        self.assertAlmostEqual(roc_auc(labels, scores), 0.0)

    def test_max_fpr_threshold_respects_bonafide_budget(self):
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        scores = [0.10, 0.20, 0.30, 0.40, 0.35, 0.60, 0.80, 0.90]
        threshold = threshold_for_max_fpr(labels, scores, max_fpr=0.25)
        metrics = metrics_at_threshold(labels, scores, threshold)
        self.assertLessEqual(metrics["false_positive_rate"], 0.25)
        self.assertGreaterEqual(metrics["recall"], 0.75)

    def test_zero_fpr_budget_has_finite_reject_all_fallback(self):
        labels = [0, 1]
        scores = [0.9, 0.8]
        threshold = threshold_for_max_fpr(labels, scores, max_fpr=0.0)
        self.assertGreater(threshold, max(scores))
        metrics = metrics_at_threshold(labels, scores, threshold)
        self.assertEqual(metrics["false_positive_rate"], 0.0)

    def test_eer_is_low_for_separable_scores(self):
        value, threshold = eer([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertLessEqual(value, 0.01)
        self.assertTrue(0.2 < threshold <= 0.8)

    def test_metrics_report_confusion_counts(self):
        result = metrics_at_threshold([0, 0, 1, 1], [0.1, 0.7, 0.6, 0.9], 0.65)
        self.assertEqual(result["tp"], 1)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["tn"], 1)
        self.assertEqual(result["fn"], 1)

    def test_single_class_metrics_are_rejected(self):
        with self.assertRaises(ValueError):
            roc_auc([1, 1], [0.2, 0.8])


if __name__ == "__main__":
    unittest.main()

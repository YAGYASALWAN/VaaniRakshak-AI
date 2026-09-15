import unittest

from vaanirakshak.sea_training import select_groups


class SeaGroupSelectionRegressionTests(unittest.TestCase):
    def test_large_holdout_groups_can_use_global_budget(self):
        candidates = [
            {
                "unit": "train-bonafide",
                "split": "train",
                "english_rows": 150,
                "labels": {"bonafide": 150},
                "estimated_bytes": 100,
            },
            {
                "unit": "train-spoof",
                "split": "train",
                "english_rows": 150,
                "labels": {"spoof": 150},
                "estimated_bytes": 100,
            },
            {
                "unit": "validation-mixed-large",
                "split": "validation",
                "english_rows": 300,
                "labels": {"bonafide": 150, "spoof": 150},
                "estimated_bytes": 1_800,
            },
            {
                "unit": "evaluation-mixed-large",
                "split": "evaluation",
                "english_rows": 300,
                "labels": {"bonafide": 150, "spoof": 150},
                "estimated_bytes": 1_800,
            },
        ]

        # Under the old rigid 80/10/10 allocation, each holdout received only
        # 500 bytes and both 1,800-byte row groups were rejected even though the
        # complete selection fits comfortably inside the 5,000-byte global cap.
        selected = select_groups(candidates, budget=5_000)

        self.assertEqual({group["unit"] for group in selected}, {group["unit"] for group in candidates})
        self.assertLessEqual(sum(group["estimated_bytes"] for group in selected), 5_000)

    def test_global_budget_still_fails_closed_when_minimum_coverage_cannot_fit(self):
        candidates = [
            {
                "unit": f"{split}-mixed",
                "split": split,
                "english_rows": 300,
                "labels": {"bonafide": 150, "spoof": 150},
                "estimated_bytes": 2_000,
            }
            for split in ("train", "validation", "evaluation")
        ]

        with self.assertRaisesRegex(ValueError, "global byte budget"):
            select_groups(candidates, budget=5_000)


if __name__ == "__main__":
    unittest.main()

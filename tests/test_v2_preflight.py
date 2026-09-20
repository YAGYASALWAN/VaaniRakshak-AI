import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from scripts.v2_preflight import run


GIB = 1024 ** 3


class V2PreflightTests(unittest.TestCase):
    def test_rejects_budget_below_one_gb(self):
        with self.assertRaisesRegex(ValueError, "budget_gb"):
            run(data_root=Path("data/v2_sea_en"), budget_gb=0.5, require_cuda=False, offline=True)

    def test_offline_cpu_preflight_can_succeed_with_disk_headroom(self):
        usage = type("Usage", (), {"total": 200 * GIB, "used": 20 * GIB, "free": 180 * GIB})()
        with patch("scripts.v2_preflight.shutil.disk_usage", return_value=usage), patch.object(
            torch.cuda, "is_available", return_value=False
        ):
            report = run(
                data_root=Path("data/v2_sea_en"),
                budget_gb=1.0,
                require_cuda=False,
                offline=True,
            )
        self.assertTrue(report["ok"])
        self.assertFalse(report["cuda"]["available"])
        self.assertFalse(report["huggingface"]["checked"])
        self.assertIsNone(report["huggingface"]["pinned_sea_access"])

    def test_required_cuda_is_not_silently_relaxed(self):
        usage = type("Usage", (), {"total": 200 * GIB, "used": 20 * GIB, "free": 180 * GIB})()
        with patch("scripts.v2_preflight.shutil.disk_usage", return_value=usage), patch.object(
            torch.cuda, "is_available", return_value=False
        ):
            with self.assertRaisesRegex(ValueError, "CUDA is required"):
                run(
                    data_root=Path("data/v2_sea_en"),
                    budget_gb=1.0,
                    require_cuda=True,
                    offline=True,
                )

    def test_insufficient_disk_is_a_hard_failure(self):
        usage = type("Usage", (), {"total": 12 * GIB, "used": 10 * GIB, "free": 2 * GIB})()
        with patch("scripts.v2_preflight.shutil.disk_usage", return_value=usage), patch.object(
            torch.cuda, "is_available", return_value=False
        ):
            with self.assertRaisesRegex(ValueError, "Insufficient free disk"):
                run(
                    data_root=Path("data/v2_sea_en"),
                    budget_gb=1.0,
                    require_cuda=False,
                    offline=True,
                )


if __name__ == "__main__":
    unittest.main()

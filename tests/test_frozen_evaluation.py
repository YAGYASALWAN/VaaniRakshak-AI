"""Sampling rules for the frozen cross-corpus evaluator."""
import importlib.util
from pathlib import Path
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "torchaudio", "soundfile", "numpy"))
if AVAILABLE:
    spec = importlib.util.spec_from_file_location(
        "frozen", Path(__file__).resolve().parents[1] / "scripts/evaluate_frozen_asvspoof.py")
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)


@unittest.skipUnless(AVAILABLE, "Requires GPU environment packages (CPU tests supported)")
class FrozenEvaluationTests(unittest.TestCase):
    def test_official_attack_families_match_the_partitions(self):
        self.assertEqual(len(frozen.ATTACKS_BY_SPLIT["validation"]), 6)
        self.assertEqual(len(frozen.ATTACKS_BY_SPLIT["test"]), 13)
        self.assertEqual(frozen.ATTACKS_BY_SPLIT["validation"][0], "A01")
        self.assertEqual(frozen.ATTACKS_BY_SPLIT["test"][-1], "A19")
        self.assertFalse(set(frozen.ATTACKS_BY_SPLIT["validation"]) & set(frozen.ATTACKS_BY_SPLIT["test"]))

    def test_stride_leaves_headroom_over_the_quota(self):
        # 4,914 recordings per attack, 100 wanted: roughly 150 should pass the gate.
        stride = frozen.sampling_stride(4914, 100)
        self.assertEqual(stride, 32)
        self.assertGreater(4914 / stride, 100)
        # Asking for everything, or for nothing, must never skip recordings.
        self.assertEqual(frozen.sampling_stride(100, 100), 1)
        self.assertEqual(frozen.sampling_stride(4914, 0), 1)

    def test_gate_is_deterministic_seeded_and_spread_across_the_stream(self):
        ids = [f"LA_E_{i:07d}" for i in range(6000)]
        chosen = [i for i in ids if frozen.gate(i, 32, 42)]
        self.assertEqual(chosen, [i for i in ids if frozen.gate(i, 32, 42)])
        self.assertNotEqual(chosen, [i for i in ids if frozen.gate(i, 32, 7)])
        self.assertAlmostEqual(len(chosen) / len(ids), 1 / 32, delta=0.01)
        # The point of the gate: selections must not cluster at the head of the
        # stream the way an ordered cap does.
        first_half = sum(1 for i in chosen if int(i[-7:]) < 3000)
        self.assertAlmostEqual(first_half / len(chosen), 0.5, delta=0.1)
        self.assertTrue(all(frozen.gate(i, 1, 42) for i in ids[:50]))


if __name__ == "__main__":
    unittest.main()

import unittest

from vaanirakshak.v2_data_contract import AudioRecord, audit_unseen_generator
from vaanirakshak.v2_manifest_tools import build_unseen_generator_scenario, identity_components


def rec(record_id, label, generator, source, speaker, digest):
    return AudioRecord(
        record_id=record_id,
        label=label,
        dataset="POOL",
        split="train",
        audio_ref=f"audio/{record_id}.flac",
        speaker_id=speaker,
        generator_id=generator,
        source_utterance_id=source,
        content_sha256=digest * 64,
        codec="FLAC/PCM_16",
        sample_rate=16_000,
    )


class V2ManifestToolsTests(unittest.TestCase):
    def _pool(self):
        records = []
        digests = iter("abcdefghijklmnopqrstuvwxyz")
        for index in range(1, 7):
            source = f"utt-{index}"
            speaker = f"speaker-{index}"
            records.append(rec(f"real-{index}", "bonafide", None, source, speaker, next(digests)))
            generator = "GEN-HOLD" if index == 1 else "GEN-TRAIN"
            records.append(rec(f"fake-{index}", "spoof", generator, source, speaker, next(digests)))
        # Same test utterance synthesized by a seen generator: this must be dropped,
        # not sent to training and not counted as unseen-generator test evidence.
        records.append(rec("fake-conflict", "spoof", "GEN-TRAIN", "utt-1", "speaker-1", next(digests)))
        return records

    def test_identity_components_join_source_and_speaker_variants(self):
        components = identity_components(self._pool())
        first = next(group for group in components if any(item.record_id == "real-1" for item in group))
        self.assertEqual({item.record_id for item in first}, {"real-1", "fake-1", "fake-conflict"})

    def test_unseen_generator_scenario_is_strictly_disjoint(self):
        scenario, report = build_unseen_generator_scenario(
            self._pool(),
            heldout_generators={"GEN-HOLD"},
            dev_fraction=0.2,
            seed=7,
        )
        audit = audit_unseen_generator(scenario)
        self.assertTrue(audit.ok, audit.errors)

        test = [item for item in scenario if item.split == "test"]
        train_dev = [item for item in scenario if item.split in {"train", "dev"}]
        self.assertEqual({item.generator_id for item in test if item.label == "spoof"}, {"GEN-HOLD"})
        self.assertNotIn("GEN-HOLD", {item.generator_id for item in train_dev if item.generator_id})
        self.assertNotIn("fake-conflict", {item.record_id for item in scenario})
        self.assertGreaterEqual(report["dropped_records"], 1)
        self.assertEqual({item.label for item in test}, {"bonafide", "spoof"})

    def test_heldout_generator_must_exist(self):
        with self.assertRaisesRegex(ValueError, "not present"):
            build_unseen_generator_scenario(self._pool(), heldout_generators={"DOES-NOT-EXIST"})

    def test_scenario_is_reproducible_for_same_seed(self):
        first, _ = build_unseen_generator_scenario(self._pool(), heldout_generators={"GEN-HOLD"}, seed=99)
        second, _ = build_unseen_generator_scenario(self._pool(), heldout_generators={"GEN-HOLD"}, seed=99)
        mapping_a = [(item.record_id, item.split) for item in first]
        mapping_b = [(item.record_id, item.split) for item in second]
        self.assertEqual(mapping_a, mapping_b)


if __name__ == "__main__":
    unittest.main()

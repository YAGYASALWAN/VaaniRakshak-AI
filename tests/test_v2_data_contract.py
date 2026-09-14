import unittest

from vaanirakshak.v2_data_contract import (
    AudioRecord,
    audit_cross_dataset,
    audit_manifest,
    audit_unseen_generator,
    manifest_fingerprint,
)


def record(
    record_id,
    label,
    dataset,
    split,
    digest,
    *,
    generator=None,
    source=None,
):
    return AudioRecord(
        record_id=record_id,
        label=label,
        dataset=dataset,
        split=split,
        audio_ref=f"audio/{record_id}.wav",
        generator_id=generator,
        source_utterance_id=source,
        content_sha256=digest,
        sample_rate=16_000,
    )


def digest(char):
    return char * 64


class ManifestAuditTests(unittest.TestCase):
    def test_clean_manifest_passes(self):
        records = [
            record("tr-real", "bonafide", "A", "train", digest("a")),
            record("tr-fake", "spoof", "A", "train", digest("b"), generator="g1"),
            record("dev-real", "bonafide", "A", "dev", digest("c")),
            record("dev-fake", "spoof", "A", "dev", digest("d"), generator="g1"),
            record("te-real", "bonafide", "B", "test", digest("e")),
            record("te-fake", "spoof", "B", "test", digest("f"), generator="g2"),
        ]
        audit = audit_manifest(records)
        self.assertTrue(audit.ok, audit.errors)
        self.assertEqual(audit.counts["records"], 6)

    def test_duplicate_audio_hash_across_splits_is_rejected(self):
        shared = digest("a")
        records = [
            record("train", "bonafide", "A", "train", shared),
            record("test", "bonafide", "B", "test", shared),
        ]
        audit = audit_manifest(records, require_generator_for_spoof=False)
        self.assertFalse(audit.ok)
        self.assertTrue(any("appears across splits" in error for error in audit.errors))

    def test_related_source_utterance_cannot_cross_splits(self):
        records = [
            record("human", "bonafide", "A", "train", digest("a"), source="utt-1"),
            record("fake", "spoof", "A", "test", digest("b"), generator="g9", source="utt-1"),
        ]
        audit = audit_manifest(records)
        self.assertFalse(audit.ok)
        self.assertTrue(any("source_utterance_id" in error for error in audit.errors))

    def test_spoof_generator_metadata_is_required(self):
        records = [record("fake", "spoof", "A", "train", digest("a"))]
        audit = audit_manifest(records)
        self.assertFalse(audit.ok)
        self.assertTrue(any("lacks generator_id" in error for error in audit.errors))

    def test_unseen_generator_detects_contamination(self):
        records = [
            record("tr-real", "bonafide", "A", "train", digest("a")),
            record("tr-fake", "spoof", "A", "train", digest("b"), generator="shared"),
            record("te-real", "bonafide", "A", "test", digest("c")),
            record("te-fake", "spoof", "A", "test", digest("d"), generator="shared"),
        ]
        audit = audit_unseen_generator(records)
        self.assertFalse(audit.ok)
        self.assertTrue(any("contaminated" in error for error in audit.errors))

    def test_cross_dataset_detects_train_test_dataset_overlap(self):
        records = [
            record("tr-real", "bonafide", "A", "train", digest("a")),
            record("tr-fake", "spoof", "A", "train", digest("b"), generator="g1"),
            record("te-real", "bonafide", "A", "test", digest("c")),
            record("te-fake", "spoof", "A", "test", digest("d"), generator="g2"),
        ]
        audit = audit_cross_dataset(records, evaluation_datasets={"A"})
        self.assertFalse(audit.ok)
        self.assertTrue(any("also appear in train/dev" in error for error in audit.errors))

    def test_manifest_fingerprint_is_order_independent(self):
        a = record("a", "bonafide", "A", "train", digest("a"))
        b = record("b", "spoof", "A", "train", digest("b"), generator="g1")
        self.assertEqual(manifest_fingerprint([a, b]), manifest_fingerprint([b, a]))


if __name__ == "__main__":
    unittest.main()

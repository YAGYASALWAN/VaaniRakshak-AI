"""Offline guards against incorrect labels and train/test leakage."""
import unittest
from unittest.mock import patch

from vaanirakshak.english_rules import COUNTS, check_disjoint, record_metadata, validate_partition


def source(label=0, number=1):
    return dict(key=label, audio_file_name=f"LA_T_{number:07d}",
                speaker_id="LA_0001", system_id="-" if label == 0 else "A01")


class EnglishRulesTests(unittest.TestCase):
    def test_label_mapping(self):
        self.assertEqual(record_metadata(source(0), "train")["label"], 0)
        self.assertEqual(record_metadata(source(1), "train")["label"], 1)

    def test_bad_label_or_attack_rejected(self):
        for changes in ({"key": True}, {"key": "spoof"}, {"system_id": "A01"},
                        {"speaker_id": None}, {"audio_file_name": "LA_E_0000001"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                record_metadata(source() | changes, "train")

    def test_complete_partition_and_missing_class(self):
        rows = [record_metadata(source(0), "train"), record_metadata(source(1, 2), "train")]
        with patch.dict(COUNTS, {"train": {0: 1, 1: 1}}):
            validate_partition(rows, "train")
            with self.assertRaises(ValueError):
                validate_partition(rows[:1], "train")

    def test_duplicate_recording_rejected(self):
        rows = [record_metadata(source(0), "train"), record_metadata(source(1), "train")]
        with patch.dict(COUNTS, {"train": {0: 1, 1: 1}}), self.assertRaises(ValueError):
            validate_partition(rows, "train")

    def test_each_cross_split_identity_rejected(self):
        a = dict(id="train1", speaker="speaker1", audio_sha256="hash1")
        b = dict(id="test1", speaker="speaker2", audio_sha256="hash2")
        check_disjoint({"train": [a, a], "test": [b]})
        for field in a:
            with self.subTest(field=field), self.assertRaises(ValueError):
                check_disjoint({"train": [a], "test": [b | {field: a[field]}]})


if __name__ == "__main__":
    unittest.main()

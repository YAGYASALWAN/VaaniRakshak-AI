import unittest
from vaanirakshak.seaspoof_audit import audit


def fixture():
    return [dict(row_id=f"{split}-{label}", utterance_id=f"{split}-{label}", language="en",
                 split=split, label=label) for split in ("train", "validation", "evaluation") for label in ("bonafide", "spoof")]


class SEAAuditTests(unittest.TestCase):
    def test_valid_metadata_does_not_invent_speaker_evidence(self):
        result = audit(fixture())
        self.assertEqual(result["status"], "metadata_checks_passed")
        self.assertEqual(result["speaker_separation"], "unverified")
        self.assertEqual(result["label_mapping"], {"bonafide": 0, "spoof": 1})

    def test_missing_class_and_invalid_numeric_labels(self):
        self.assertEqual(audit(fixture()[:-1])["status"], "issues_found")
        rows = fixture()
        rows[0]["label"] = 0
        self.assertEqual(audit(rows)["status"], "issues_found")

    def test_leakage_rejected(self):
        for field, value in (("row_id", "duplicate"), ("speaker_id", "same-speaker"), ("audio_sha256", "a" * 64)):
            rows = fixture()
            rows[0][field] = rows[2][field] = value
            with self.subTest(field=field):
                self.assertEqual(audit(rows)["status"], "issues_found")


if __name__ == "__main__":
    unittest.main()

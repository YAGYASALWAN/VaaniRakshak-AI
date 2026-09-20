import unittest

from vaanirakshak.v2_data_contract import AudioRecord, audit_manifest


class SpeakerLeakageContractTests(unittest.TestCase):
    def test_known_speaker_cannot_cross_train_and_test(self):
        train = AudioRecord(
            record_id="train-real",
            label="bonafide",
            dataset="A",
            split="train",
            audio_ref="train.flac",
            speaker_id="speaker-1",
            content_sha256="a" * 64,
        )
        test = AudioRecord(
            record_id="test-real",
            label="bonafide",
            dataset="A",
            split="test",
            audio_ref="test.flac",
            speaker_id="speaker-1",
            content_sha256="b" * 64,
        )
        audit = audit_manifest([train, test], require_generator_for_spoof=False)
        self.assertFalse(audit.ok)
        self.assertTrue(any("speaker_id" in error and "crosses splits" in error for error in audit.errors))

    def test_missing_speaker_ids_are_not_invented_or_rejected(self):
        train = AudioRecord(
            record_id="train-real",
            label="bonafide",
            dataset="A",
            split="train",
            audio_ref="train.flac",
            content_sha256="a" * 64,
        )
        test = AudioRecord(
            record_id="test-real",
            label="bonafide",
            dataset="A",
            split="test",
            audio_ref="test.flac",
            content_sha256="b" * 64,
        )
        audit = audit_manifest([train, test], require_generator_for_spoof=False)
        self.assertTrue(audit.ok, audit.errors)
        self.assertEqual(audit.counts["speaker_ids_present"], 0)


if __name__ == "__main__":
    unittest.main()

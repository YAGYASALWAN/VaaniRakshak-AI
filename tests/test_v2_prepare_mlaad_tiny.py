import tempfile
import unittest
from pathlib import Path

from vaanirakshak.v2_data_contract import AudioRecord, audit_manifest
from vaanirakshak.v2_prepare_mlaad_tiny import (
    _clean_relative,
    _component_bucket,
    _generator,
    _normalize_original_reference,
    _source_id,
    _speaker,
    split_records,
)


class MLAADTinyPreparationTests(unittest.TestCase):
    def test_metadata_provenance_helpers(self):
        self.assertEqual(_clean_relative("./original/en/en_sample_000477.wav"), "original/en/en_sample_000477.wav")
        self.assertEqual(
            _normalize_original_reference("./original/en/en_US/by_book/sample.wav"),
            "en_US/by_book/sample.wav",
        )
        self.assertEqual(
            _normalize_original_reference("en_US/by_book/sample.wav"),
            "en_US/by_book/sample.wav",
        )
        self.assertEqual(
            _source_id("./original/en/en_US/by_book/sample.wav"),
            "MLAAD-tiny:en_US/by_book/sample.wav",
        )
        self.assertEqual(
            _source_id("en_US/by_book/sample.wav"),
            "MLAAD-tiny:en_US/by_book/sample.wav",
        )
        with tempfile.TemporaryDirectory() as folder:
            meta = Path(folder) / "Cartesia.ai (Sonic-3)" / "meta.csv"
            meta.parent.mkdir()
            row = {"model_name": "Cartesia.ai (Sonic-3)", "reference_speaker": "speaker-1"}
            self.assertEqual(_generator(row, meta), "MLAAD:Cartesia.ai (Sonic-3)")
            self.assertEqual(_speaker(row), "MLAAD:reference:speaker-1")
            self.assertIsNone(_speaker({"reference_speaker": "unknown"}))

    def test_split_keeps_source_pairs_together_and_all_labels_present(self):
        records = []
        for index in range(500):
            source = f"MLAAD-tiny:en_US/sample-{index}.wav"
            records.extend(
                [
                    AudioRecord(
                        record_id=f"real-{index}",
                        label="bonafide",
                        dataset="MLAAD-tiny",
                        split="train",
                        audio_ref=f"real-{index}.wav",
                        language="en",
                        source_utterance_id=source,
                        content_sha256=f"{index + 1:064x}"[-64:],
                        codec="WAV",
                    ),
                    AudioRecord(
                        record_id=f"fake-{index}",
                        label="spoof",
                        dataset="MLAAD-tiny",
                        split="train",
                        audio_ref=f"fake-{index}.wav",
                        language="en",
                        generator_id=f"MLAAD:generator-{index % 7}",
                        source_utterance_id=source,
                        content_sha256=f"{index + 1001:064x}"[-64:],
                        codec="WAV",
                    ),
                ]
            )

        result = split_records(records, seed=42)
        audit = audit_manifest(result)
        self.assertTrue(audit.ok, audit.errors)
        self.assertEqual({item.split for item in result}, {"train", "dev", "test"})
        for split in ("train", "dev", "test"):
            self.assertEqual({item.label for item in result if item.split == split}, {"bonafide", "spoof"})

        owner = {}
        for item in result:
            previous = owner.setdefault(item.source_utterance_id, item.split)
            self.assertEqual(previous, item.split)

    def test_component_bucket_is_deterministic(self):
        record = AudioRecord(
            record_id="fixture",
            label="bonafide",
            dataset="MLAAD-tiny",
            split="train",
            audio_ref="fixture.wav",
            source_utterance_id="source",
            content_sha256="0" * 64,
        )
        self.assertEqual(_component_bucket([record], 42), _component_bucket([record], 42))


if __name__ == "__main__":
    unittest.main()

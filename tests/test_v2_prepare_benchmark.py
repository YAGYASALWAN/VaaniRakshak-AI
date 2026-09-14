import json
import unittest

from vaanirakshak.v2_prepare_benchmark import (
    _generator_id,
    _label,
    _notes,
    _safe_filename,
    _source_identity,
    _speaker_identity,
)


class V2BenchmarkPreparationTests(unittest.TestCase):
    def test_class_label_indices_map_to_v2_labels(self):
        self.assertEqual(_label(0), "bonafide")
        self.assertEqual(_label(1), "spoof")
        self.assertEqual(_label("bonafide"), "bonafide")
        self.assertEqual(_label("spoof"), "spoof")

    def test_notes_accept_json_string(self):
        value = {"speaker_id": "S1", "attack_id": "A07"}
        self.assertEqual(_notes(json.dumps(value)), value)

    def test_attack_id_becomes_generator_identity_only_for_spoof(self):
        notes = {"attack_id": "A07"}
        self.assertEqual(_generator_id("ASV", "spoof", notes), "ASV:A07")
        self.assertIsNone(_generator_id("ASV", "bonafide", notes))

    def test_missing_spoof_attack_identity_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "attack/generator"):
            _generator_id("ASV", "spoof", {})

    def test_source_and_speaker_are_namespaced(self):
        notes = {"source_id": "SRC42", "speaker_id": "SPK9"}
        self.assertEqual(_source_identity("ASV5", notes), "ASV5:SRC42")
        self.assertEqual(_speaker_identity("ASV5", notes), "ASV5:SPK9")

    def test_filename_is_filesystem_safe(self):
        value = _safe_filename("speaker/a:b?c")
        self.assertNotIn("/", value)
        self.assertNotIn(":", value)
        self.assertNotIn("?", value)


if __name__ == "__main__":
    unittest.main()

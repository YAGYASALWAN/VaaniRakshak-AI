import io
import unittest

import numpy as np
import soundfile as sf

from vaanirakshak.v2_prepare_sea import (
    _canonical_flac,
    _evaluation_first,
    _generator,
    _source_utterance,
    _speaker,
)


class V2SEAPreparationTests(unittest.TestCase):
    def _wav_bytes(self, rate=48_000, channels=2):
        t = np.arange(rate, dtype=np.float32) / rate
        mono = 0.1 * np.sin(2 * np.pi * 440.0 * t)
        wave = np.stack([mono] * channels, axis=1) if channels > 1 else mono
        buffer = io.BytesIO()
        sf.write(buffer, wave, rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    def test_canonical_audio_is_mono_16khz_flac(self):
        encoded, meta = _canonical_flac(self._wav_bytes())
        with sf.SoundFile(io.BytesIO(encoded)) as stream:
            self.assertEqual(stream.format, "FLAC")
            self.assertEqual(stream.samplerate, 16_000)
            self.assertEqual(stream.channels, 1)
            self.assertGreater(len(stream), 15_900)
            self.assertLess(len(stream), 16_100)
        self.assertEqual(meta["source_sample_rate"], 48_000)
        self.assertEqual(meta["source_channels"], 2)
        self.assertEqual(meta["canonical_sample_rate"], 16_000)

    def test_spoof_generator_comes_from_source_model(self):
        row = {"source_model": "voicebox-x"}
        self.assertEqual(_generator(row, "spoof"), "SEA:voicebox-x")
        self.assertIsNone(_generator(row, "bonafide"))

    def test_missing_spoof_generator_is_not_invented(self):
        self.assertIsNone(_generator({"source_model": ""}, "spoof"))
        self.assertIsNone(_generator({}, "spoof"))

    def test_source_utterance_is_namespaced_by_source_dataset(self):
        row = {"source_dataset": "corpus-a", "utterance_id": "utt-42"}
        self.assertEqual(_source_utterance(row), "SEA:corpus-a:utt-42")

    def test_speaker_metadata_is_namespaced(self):
        row = {"speaker_id": "spk-7", "speaker_or_voice": "fallback"}
        self.assertEqual(_speaker(row, "bonafide"), "SEA:human:spk-7")
        self.assertEqual(_speaker(row, "spoof"), "SEA:voice:spk-7")

    def test_materialization_orders_evaluation_before_training(self):
        groups = [
            {"split": "train", "path": "z", "row_group": 0},
            {"split": "validation", "path": "b", "row_group": 0},
            {"split": "evaluation", "path": "c", "row_group": 1},
            {"split": "evaluation", "path": "a", "row_group": 0},
        ]
        ordered = _evaluation_first(groups)
        self.assertEqual([group["split"] for group in ordered], ["evaluation", "evaluation", "validation", "train"])
        self.assertEqual([group["path"] for group in ordered[:2]], ["a", "c"])


if __name__ == "__main__":
    unittest.main()

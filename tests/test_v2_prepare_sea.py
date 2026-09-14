import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import soundfile as sf

from vaanirakshak.sea_transfer import REPOSITORY, REVISION
from vaanirakshak.v2_prepare_sea import (
    MIN_PREP_FREE_BYTES,
    _canonical_flac,
    _evaluation_first,
    _generator,
    _require_disk_headroom,
    _scan_groups,
    _source_utterance,
    _speaker,
    prepare,
)


class _ScopeClient:
    used = 0

    def __init__(self):
        self.events = []

    def begin_scope(self, scope_id):
        self.events.append(("begin", scope_id))

    def end_scope(self, scope_id, *, clear):
        self.events.append(("end", scope_id, clear))

    def clear_scope(self, scope_id):
        self.events.append(("clear", scope_id))


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

    def test_disk_floor_fails_before_exhaustion(self):
        with tempfile.TemporaryDirectory() as folder, patch(
            "vaanirakshak.v2_prepare_sea.shutil.disk_usage",
            return_value=SimpleNamespace(free=MIN_PREP_FREE_BYTES - 1),
        ):
            with self.assertRaisesRegex(ValueError, "free disk space"):
                _require_disk_headroom(Path(folder))

    def test_failed_metadata_scan_keeps_retry_scope_for_next_run(self):
        item = {"path": "data/train/fake.parquet", "size": 4096, "split": "train"}
        source = {"repository": REPOSITORY, "revision": REVISION, "files": [item]}
        client = _ScopeClient()

        with tempfile.TemporaryDirectory() as folder, patch(
            "pyarrow.parquet.ParquetFile", side_effect=RuntimeError("simulated metadata crash")
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated metadata crash"):
                _scan_groups(Path(folder), source, client, budget=1_000_000_000)

        self.assertEqual(len(client.events), 2)
        self.assertEqual(client.events[0][0], "begin")
        self.assertTrue(client.events[0][1].startswith("v2-sea-metadata:"))
        self.assertEqual(client.events[1], ("end", client.events[0][1], False))

    def test_failed_row_group_keeps_retry_scope_for_next_run(self):
        group = {
            "unit": "unit-a",
            "path": "data/train/fake.parquet",
            "split": "train",
            "row_group": 0,
            "english_rows": 1,
            "labels": {"bonafide": 1},
            "estimated_bytes": 1024,
        }
        source = {
            "repository": REPOSITORY,
            "revision": REVISION,
            "files": [{"path": group["path"], "size": 4096, "split": "train"}],
        }
        client = _ScopeClient()
        plan = {"groups": [group]}

        with tempfile.TemporaryDirectory() as folder, patch(
            "vaanirakshak.v2_prepare_sea._scan_groups", return_value=plan
        ), patch("pyarrow.parquet.ParquetFile", side_effect=RuntimeError("simulated row-group crash")):
            with self.assertRaisesRegex(RuntimeError, "simulated row-group crash"):
                prepare(Path(folder), source, client, budget=1_000_000_000)

        scope = "v2-sea-row-group:unit-a"
        self.assertEqual(client.events, [("begin", scope), ("end", scope, False)])


if __name__ == "__main__":
    unittest.main()

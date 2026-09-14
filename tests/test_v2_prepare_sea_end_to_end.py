import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

from vaanirakshak.sea_transfer import REPOSITORY, REVISION
from vaanirakshak.v2_prepare_sea import prepare


RATE = 16_000


def _wav_bytes(frequency: float) -> bytes:
    t = np.arange(RATE, dtype=np.float32) / RATE
    wave = (0.05 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, wave, RATE, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _tiny_parquet() -> bytes:
    audio_type = pa.struct([pa.field("bytes", pa.binary())])
    table = pa.table(
        {
            "language": pa.array(["en", "en"]),
            "label": pa.array(["bonafide", "spoof"]),
            "audio": pa.array(
                [{"bytes": _wav_bytes(220.0)}, {"bytes": _wav_bytes(440.0)}],
                type=audio_type,
            ),
            "row_id": pa.array(["real-1", "spoof-1"]),
            "split": pa.array(["train", "train"]),
            "source_model": pa.array([None, "tiny-tts"]),
            "source_dataset": pa.array(["tiny-source", "tiny-source"]),
            "speaker_id": pa.array(["human-1", "voice-1"]),
            "speaker_or_voice": pa.array(["human-1", "voice-1"]),
            "utterance_id": pa.array(["utt-real", "utt-spoof"]),
        }
    )
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink, row_group_size=2)
    return sink.getvalue().to_pybytes()


class _LocalRangeClient:
    def __init__(self, blobs):
        self.blobs = blobs
        self.used = 0
        self.events = []
        self.active = None

    def fetch(self, item, offset, size):
        data = self.blobs[item["path"]]
        return data[offset : offset + size]

    def begin_scope(self, scope_id):
        if self.active is not None:
            raise RuntimeError("nested local test scope")
        self.active = scope_id
        self.events.append(("begin", scope_id))

    def end_scope(self, scope_id, *, clear):
        if self.active != scope_id:
            raise RuntimeError("wrong local test scope")
        self.events.append(("end", scope_id, clear))
        self.active = None

    def clear_scope(self, scope_id):
        self.events.append(("clear", scope_id))


class V2SEAEndToEndPreparationTests(unittest.TestCase):
    def test_tiny_parquet_materializes_receipt_manifest_and_reuses_receipt(self):
        blob = _tiny_parquet()
        source_path = "data/train/tiny.parquet"
        item = {"path": source_path, "size": len(blob), "split": "train", "sha256": "fixture"}
        source = {"repository": REPOSITORY, "revision": REVISION, "files": [item]}

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = _LocalRangeClient({source_path: blob})
            manifest = prepare(root, source, first, budget=1_000_000_000)

            self.assertTrue(manifest.is_file())
            rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["label"] for row in rows}, {"bonafide", "spoof"})
            self.assertEqual({row["sample_rate"] for row in rows}, {RATE})
            self.assertEqual({row["codec"] for row in rows}, {"FLAC/PCM_16"})
            self.assertEqual(
                {row["generator_id"] for row in rows if row["label"] == "spoof"},
                {"SEA:tiny-tts"},
            )
            for row in rows:
                self.assertTrue((root / row["audio_ref"]).is_file())

            receipts = list((root / "receipts").glob("*.json"))
            self.assertEqual(len(receipts), 1)
            audit = json.loads((root / "v2_sea_audit.json").read_text(encoding="utf-8"))
            self.assertEqual(audit["records"], 2)
            self.assertTrue(audit["retry_cache"]["enabled"])
            self.assertTrue(any(event[0] == "begin" and event[1].startswith("v2-sea-metadata:") for event in first.events))
            self.assertTrue(any(event[0] == "begin" and event[1].startswith("v2-sea-row-group:") for event in first.events))
            self.assertTrue(all(event[2] is True for event in first.events if event[0] == "end"))

            # Rerun with a fresh client. The existing plan/receipt should be enough;
            # no Parquet range read is needed for the completed row group.
            second = _LocalRangeClient({source_path: b""})
            second_manifest = prepare(root, source, second, budget=1_000_000_000)
            self.assertEqual(second_manifest.read_bytes(), manifest.read_bytes())
            self.assertEqual([event[0] for event in second.events], ["clear"])
            self.assertTrue(second.events[0][1].startswith("v2-sea-row-group:"))


if __name__ == "__main__":
    unittest.main()

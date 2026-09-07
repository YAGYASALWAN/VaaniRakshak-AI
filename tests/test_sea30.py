import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vaanirakshak.sea_transfer import MAX_BYTES, RangeFile, Transfer, SafeRedirect
from vaanirakshak.sea_training import select_groups, filter_duplicates

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "torchaudio", "soundfile", "numpy", "pyarrow"))


class SeaBudgetTests(unittest.TestCase):
    def test_selection_reserves_all_splits_and_classes(self):
        candidates = [dict(unit=f"{split}-{label}", split=split, english_rows=120,
                           labels={label: 120}, estimated_bytes=10)
                      for split in ("train", "validation", "evaluation") for label in ("bonafide", "spoof")]
        chosen = select_groups(candidates, budget=1000)
        self.assertEqual(len(chosen), 6)
        self.assertLessEqual(sum(r["estimated_bytes"] for r in chosen), 1000)
        with self.assertRaises(ValueError):
            select_groups(candidates[:4], budget=1000)

    def test_duplicates_keep_holdout_and_remove_training_copy(self):
        rows = [dict(id="train-a", split="train", audio_sha256="same", window_sha256="same-window"),
                dict(id="test-a", split="test", audio_sha256="same", window_sha256="same-window")]
        kept, rejected = filter_duplicates(rows)
        self.assertEqual([r["id"] for r in kept], ["test-a"])
        self.assertEqual(rejected, {"train": 1})

    def test_hard_budget_before_network(self):
        with tempfile.TemporaryDirectory() as temp:
            client = Transfer(temp, "unit-test-not-a-real-token")
            client.ledger["reserved_bytes"] = MAX_BYTES - 1
            with patch.object(client.opener, "open", side_effect=AssertionError("Must not request")):
                with self.assertRaisesRegex(ValueError, "ceiling"):
                    client.fetch(dict(path="data/train/x.parquet", size=20), 0, 2)

    def test_redirect_strips_authorization(self):
        from urllib.request import Request
        request = Request("https://huggingface.co/file", headers={"Authorization": "Bearer fixture", "Range": "bytes=1-2"})
        redirected = SafeRedirect().redirect_request(request, None, 302, "", {}, "https://cdn.hf.co/file")
        self.assertIsNone(redirected.get_header("Authorization"))
        self.assertEqual(redirected.get_header("Range"), "bytes=1-2")
        with self.assertRaises(ValueError):
            SafeRedirect().redirect_request(request, None, 302, "", {}, "https://unrelated.example/file")

    def test_verified_range_and_cache_accounting(self):
        class Response(io.BytesIO):
            status = 206
            headers = {"Content-Range": "bytes 2-4/10"}

        with tempfile.TemporaryDirectory() as temp:
            client = Transfer(temp, "unit-test-not-a-real-token")
            item = dict(path="data/train/fixture.parquet", size=10)
            with patch.object(client.opener, "open", return_value=Response(b"234")) as opened:
                self.assertEqual(client.fetch(item, 2, 3), b"234")
                self.assertEqual(client.fetch(item, 2, 3), b"234")
                self.assertEqual(opened.call_count, 1)
            self.assertEqual(client.used, 3)
            self.assertEqual(json.loads(client.ledger_path.read_text())["reserved_bytes"], 3)

    def test_no_full_file_fallback(self):
        class Response(io.BytesIO):
            status = 200
            headers = {}

            def read(self, *args):
                raise AssertionError("Must reject full-file response without consuming it")

        with tempfile.TemporaryDirectory() as temp:
            client = Transfer(temp, "unit-test-not-a-real-token")
            with patch.object(client.opener, "open", return_value=Response()):
                with self.assertRaisesRegex(ValueError, "full-file"):
                    client.fetch(dict(path="data/train/fixture.parquet", size=10), 2, 3)

    def test_range_file_matches_seek_read_contract(self):
        payload = b"0123456789"

        class Client:
            def fetch(self, item, offset, size):
                return payload[offset:offset+size]

        with RangeFile(Client(), dict(size=10, path="fixture")) as stream:
            self.assertEqual(stream.read(2), b"01")
            stream.seek(-3, 2)
            self.assertEqual(stream.read(), b"789")
            self.assertEqual(stream.read(), b"")


@unittest.skipUnless(AVAILABLE, "Requires audio/PyTorch/Parquet runtime")
class SeaPreparationTests(unittest.TestCase):
    def test_english_filter_cache_reuse_and_training(self):
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        import soundfile as sf
        import torch
        from vaanirakshak.sea_training import prepare, profile
        from vaanirakshak.english_training import run
        from vaanirakshak.demo_server import Detector
        torch.set_num_threads(2)
        payloads, files = {}, []
        for split_index, split in enumerate(("train", "validation", "evaluation")):
            records = []
            for i in range(201):
                buffer = io.BytesIO()
                rng = np.random.default_rng(split_index * 1000 + i)
                sf.write(buffer, rng.normal(0, .05, 16000).astype(np.float32), 16000, format="FLAC")
                records.append(dict(row_id=f"{split}-{i}", utterance_id=str(i), split=split,
                                    language="en" if i < 200 else "hi", label="spoof" if i % 2 else "bonafide",
                                    audio={"bytes": buffer.getvalue(), "path": f"{i}.flac"},
                                    source_model="fixture", source_dataset="fixture", speaker_or_voice=f"{split}-{i}"))
            sink = io.BytesIO()
            pq.write_table(pa.Table.from_pylist(records), sink, row_group_size=201)
            name = f"data/{split}/fixture.parquet"
            payloads[name] = sink.getvalue()
            files.append(dict(path=name, split=split, size=len(payloads[name])))

        class Client:
            used = 0
            forbid = False

            def fetch(self, item, offset, size):
                if self.forbid:
                    raise AssertionError("Completed preparation must work without network")
                self.used += size
                return payloads[item["path"]][offset:offset+size]

        client = Client()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows, counts = prepare(root, dict(files=files), client, "cpu")
            self.assertEqual(len(rows), 600)
            self.assertEqual({r["language"] for r in rows}, {"en"})
            self.assertEqual(counts, {s: {0: 100, 1: 100} for s in ("train", "validation", "test")})
            client.forbid = True
            cached, _ = prepare(root, dict(files=files), client, "cpu")
            self.assertEqual(rows, cached)
            # Run a small end-to-end model/checkpoint/inference check from the prepared features.
            subset = [r for s in ("train", "validation", "test") for r in [x for x in rows if x["split"] == s][:4]]
            small_counts = {s: {0: 2, 1: 2} for s in ("train", "validation", "test")}
            settings = profile(root, subset, small_counts)
            settings["run_prefix"] = str(root / "fixture_")
            result = run(root, epochs=1, batch_size=2, device="cpu", profile=settings,
                         resume=None)
            # The absolute fixture prefix keeps test checkpoints inside the temporary directory.
            self.assertTrue((result / "evaluation.json").exists())
            detector = Detector(result / "best.pt", "cpu")
            prediction = detector.analyze(records[0]["audio"]["bytes"])
            self.assertTrue(0 <= prediction["synthetic_score"] <= 1)
            import shutil
            shutil.rmtree(result)


if __name__ == "__main__":
    unittest.main()

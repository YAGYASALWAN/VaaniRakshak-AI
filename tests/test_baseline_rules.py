import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.error import HTTPError

from vaanirakshak.baseline_download import Downloader, MAX_DOWNLOAD, make_plan
from vaanirakshak.baseline_rules import (assigned_split, balanced_sample, binary_metrics,
                                       references, remove_reference_conflicts, speaker_keys, validate_rows)


def fixture_rows():
    rows = []
    for split in ("train", "dev", "test"):
        for language in ("Hindi", "Punjabi"):
            for label in (0, 1):
                for index in range(35 if split == "train" else 12):
                    rid = f"{split}:{language}:{label}:{index}"
                    rows.append({"id": rid, "split": split, "language": language, "label": label,
                                 "speakers": [f"{split}:{language}:{label}:speaker:{index % 3}"],
                                 "references": [], "audio_sha256": rid, "window_sha256": rid})
    return rows


class BaselineRulesTests(unittest.TestCase):
    def test_source_target_namespace_and_numeric_identity(self):
        row = {"Source Speaker_ID": 42.0, "Target Speaker ID": 42, "Generative Model": "freevc24"}
        self.assertEqual(speaker_keys(row, "indicsynth"), ["indicsynth:speaker:42"])

    def test_missing_vc_source_rejected(self):
        with self.assertRaises(ValueError):
            speaker_keys({"Target Speaker ID": 5, "Generative Model": "freevc24"}, "indicsynth")

    def test_cross_partition_pair_discarded(self):
        keys = [f"speaker:{i}" for i in range(100)]
        train = next(k for k in keys if assigned_split([k]) == "train")
        test = next(k for k in keys if assigned_split([k]) == "test")
        self.assertIsNone(assigned_split([train, test]))
        self.assertEqual(assigned_split([train, train]), "train")

    def test_reference_normalization(self):
        self.assertEqual(references({"Source Reference Audio": r"dir\clip.wav"}, "indicsynth"),
                         ["indicsynth:reference:clip.wav"])

    def test_reference_conflicts_drop_both_sides(self):
        rows = [{"split": "train", "references": ["a"]}, {"split": "test", "references": ["a"]},
                {"split": "dev", "references": ["b"]}]
        self.assertEqual(remove_reference_conflicts(rows), rows[2:])

    def test_balanced_reproducible_sample(self):
        rows = fixture_rows()
        result = balanced_sample(rows)
        self.assertEqual(result, balanced_sample(rows))
        self.assertEqual(len(result), 236)
        self.assertEqual(len(validate_rows(result)), 12)

    def test_missing_class_cell_rejected(self):
        with self.assertRaises(ValueError):
            balanced_sample([r for r in fixture_rows() if not (r["split"] == "test" and r["label"] == 1)])

    def test_speaker_leak_rejected(self):
        rows = fixture_rows()
        rows[-1]["speakers"] = rows[0]["speakers"]
        with self.assertRaises(ValueError):
            validate_rows(rows)

    def test_duplicate_audio_leak_rejected(self):
        for key in ("audio_sha256", "window_sha256"):
            rows = fixture_rows()
            rows[-1][key] = rows[0][key]
            with self.assertRaises(ValueError):
                validate_rows(rows)

    def test_metrics_ties_and_confusion(self):
        report = binary_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])
        self.assertEqual(report["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertEqual(report["roc_auc"], 0.75)
        self.assertEqual(report["f1"], 0.5)
        self.assertEqual(binary_metrics([0, 1], [0.5, 0.5])["roc_auc"], 0.5)

    def test_one_class_metric_rejected(self):
        with self.assertRaises(ValueError):
            binary_metrics([1, 1], [0.2, 0.9])

    def test_token_stripped_on_cdn_redirect(self):
        client = Downloader("hf_fake_test_token")
        error = HTTPError("https://huggingface.co", 302, "redirect",
                          {"Location": "https://cas-bridge.xethub.hf.co/object"}, io.BytesIO())
        client.opener = Mock()
        client.opener.open.side_effect = [error, Mock()]
        client.open("https://huggingface.co/datasets/test")
        requests = [call.args[0] for call in client.opener.open.call_args_list]
        self.assertEqual(requests[0].get_header("Authorization"), "Bearer hf_fake_test_token")
        self.assertIsNone(requests[1].get_header("Authorization"))

    def test_unapproved_redirect_rejected(self):
        client = Downloader("hf_fake_test_token")
        client.opener = Mock()
        client.opener.open.side_effect = HTTPError("https://huggingface.co", 302, "redirect",
                           {"Location": "https://example.invalid/object"}, io.BytesIO())
        with self.assertRaises(ValueError):
            client.open("https://huggingface.co/datasets/test")
        self.assertEqual(client.opener.open.call_count, 1)

    def test_download_budget_rejected_before_network(self):
        client = Downloader("hf_fake_test_token")
        client.bytes_read = MAX_DOWNLOAD
        client.open = Mock()
        item = {"source": "fixture", "repository": "fixture/repo", "revision": "revision", "path": "Hindi/train.parquet", "size": 100}
        with tempfile.TemporaryDirectory() as folder, self.assertRaises(ValueError):
            client.download(item, folder)
        client.open.assert_not_called()

    def test_size_mismatch_rejected_before_body(self):
        client = Downloader("hf_fake_test_token")
        response = Mock(status=200, headers={"Content-Length": "101"})
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        client.open = Mock(return_value=response)
        item = {"source": "fixture", "repository": "fixture/repo", "revision": "revision", "path": "Hindi/train.parquet", "size": 100}
        with tempfile.TemporaryDirectory() as folder, patch("shutil.disk_usage") as disk:
            disk.return_value.free = 10 * 1024**3
            with self.assertRaises(ValueError):
                client.download(item, folder)
            self.assertFalse(list(Path(folder).glob("*.partial")))
        response.read.assert_not_called()

    def test_download_checksum_and_cache(self):
        import hashlib
        content = b"original fixture"
        client = Downloader("hf_fake_test_token")
        response = Mock(status=200, headers={"Content-Length": str(len(content))})
        response.read.return_value = content
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        client.open = Mock(return_value=response)
        item = {"source": "fixture", "repository": "fixture/repo", "revision": "revision", "path": "Hindi/train.parquet",
                "size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
        with tempfile.TemporaryDirectory() as folder, patch("shutil.disk_usage") as disk:
            disk.return_value.free = 10 * 1024**3
            first = client.download(item, folder)
            self.assertEqual(first.read_bytes(), content)
            self.assertEqual(client.download(item, folder), first)
            self.assertEqual(client.open.call_count, 1)
            first.write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                client.download(item, folder)

    def test_plan_rejects_large_total_before_acquisition(self):
        client = Mock()
        client.listing.return_value = [{"path": f"train-{i}.parquet", "size": 600 * 1024**2} for i in range(4)]
        with self.assertRaises(ValueError):
            make_plan(client)
        client.download.assert_not_called()


if __name__ == "__main__":
    unittest.main()

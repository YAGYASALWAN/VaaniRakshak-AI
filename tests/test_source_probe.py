"""Offline discovery tests. Network responses are synthetic, never real downloads."""
import tempfile
import unittest
from unittest.mock import Mock
from email.message import Message
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from vaanirakshak.data.source_probe import (
    LANGUAGES, SOURCES, MetadataClient, inspect_rows, run_probe, select_subsets,
)


class FakeClient:
    bytes_read = 0

    def __init__(self):
        self.urls = []

    def get(self, url):
        self.urls.append(url)
        parsed = urlsplit(url)
        query = parse_qs(parsed.query)
        if parsed.path.startswith("/api/datasets/"):
            return {"sha": "fixture-revision", "gated": False}
        if parsed.path == "/splits":
            return {"splits": [{"dataset": query["dataset"][0], "config": language,
                                "split": "train"} for language in LANGUAGES]}
        if parsed.path == "/rows":
            return {"features": [{"name": "audio"}, {"name": "duration"}],
                    "rows": [{"row_idx": 0, "row": {
                        "audio": [{"src": "https://example.invalid/never-fetch.wav"}],
                        "duration": None}, "truncated_cells": []}], "num_rows_total": 1000}
        raise AssertionError("Unexpected request")


class SourceProbeTests(unittest.TestCase):
    def test_selects_actual_case(self):
        payload = {"splits": [{"dataset": SOURCES["indicvoices"], "config": "hindi", "split": "train"}]}
        selected = select_subsets(payload, SOURCES["indicvoices"])
        self.assertEqual(selected["Hindi"], "hindi")
        self.assertIsNone(selected["Punjabi"])

    def test_does_not_substitute_test_split(self):
        payload = {"splits": [{"dataset": SOURCES["indicvoices"], "config": "Hindi", "split": "test"}]}
        self.assertIsNone(select_subsets(payload, SOURCES["indicvoices"])["Hindi"])

    def test_missing_fields_reported(self):
        payload = FakeClient().get("https://datasets-server.huggingface.co/rows")
        report = inspect_rows(payload, 3)
        self.assertEqual(report["nonnull_in_preview"]["duration"], 0)
        self.assertFalse(report["stable_audio_locator_verified"])

    def test_truncation_rejected(self):
        payload = FakeClient().get("https://datasets-server.huggingface.co/rows")
        payload["rows"][0]["truncated_cells"] = ["audio"]
        with self.assertRaises(ValueError):
            inspect_rows(payload, 3)

    def test_audio_urls_never_requested(self):
        client = FakeClient()
        with tempfile.TemporaryDirectory() as directory:
            report = run_probe(Path(directory) / "probe", client)
            self.assertFalse(report["audio_downloaded"])
            self.assertEqual(len(client.urls), 16)
            self.assertTrue(all("example.invalid" not in url for url in client.urls))
            self.assertTrue((Path(directory) / "probe/probe_summary.json").exists())

    def test_output_must_be_new(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                run_probe(Path(directory), FakeClient())

    def test_audio_and_arbitrary_endpoints_rejected(self):
        for url in ("https://datasets-server.huggingface.co/cached-assets/audio.wav",
                    "https://huggingface.co/resolve/main/audio.parquet",
                    "http://datasets-server.huggingface.co/rows"):
            with self.assertRaises(ValueError):
                MetadataClient().get(url)

    def test_request_failure_preserved(self):
        class FailingClient:
            bytes_read = 0
            def get(self, url):
                raise OSError("fixture unavailable")
        with tempfile.TemporaryDirectory() as directory:
            report = run_probe(Path(directory) / "probe", FailingClient())
        self.assertEqual(len(report["sources"]["indicvoices"]["errors"]), 2)

    def test_row_limit(self):
        with self.assertRaises(ValueError):
            run_probe(Path("unused"), FakeClient(), rows_per_subset=100)

    def test_empty_preview_rejected(self):
        with self.assertRaises(ValueError):
            inspect_rows({"rows": [], "features": [{"name": "audio"}]}, 3)

    def test_non_json_response_rejected_before_body(self):
        client = MetadataClient()
        response = Mock()
        response.headers = Message()
        response.headers["Content-Type"] = "audio/wav"
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        client.opener = Mock()
        client.opener.open.return_value = context
        with self.assertRaises(ValueError):
            client.get("https://datasets-server.huggingface.co/rows")
        response.read.assert_not_called()

    def test_oversized_response_rejected_before_body(self):
        client = MetadataClient()
        response = Mock()
        response.headers = Message()
        response.headers["Content-Type"] = "application/json"
        response.headers["Content-Length"] = "99999999"
        context = Mock()
        context.__enter__ = Mock(return_value=response)
        context.__exit__ = Mock(return_value=False)
        client.opener = Mock()
        client.opener.open.return_value = context
        with self.assertRaises(ValueError):
            client.get("https://datasets-server.huggingface.co/rows")
        response.read.assert_not_called()


if __name__ == "__main__":
    unittest.main()

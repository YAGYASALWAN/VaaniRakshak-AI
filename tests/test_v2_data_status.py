import json
import tempfile
import unittest
from pathlib import Path

from scripts.v2_data_status import inspect
from vaanirakshak.v2_data_contract import AudioRecord, write_jsonl


class V2DataStatusTests(unittest.TestCase):
    def test_not_started_directory_is_safe(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "future-run"
            report = inspect(root)
            self.assertEqual(report["state"], "not_started")
            self.assertTrue(report["safe_to_resume"])
            self.assertFalse(report["network_used"])
            self.assertEqual(report["transfer"]["reserved_bytes"], 0)

    def test_partial_plan_and_receipt_are_reported_as_materializing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "receipts").mkdir()
            (root / "metadata").mkdir()
            (root / "metadata" / "source.json").write_text("{}\n", encoding="utf-8")
            (root / "transfer.json").write_text(json.dumps({"reserved_bytes": 1234}), encoding="utf-8")
            plan = {
                "groups": [
                    {"unit": "g1", "estimated_bytes": 100},
                    {"unit": "g2", "estimated_bytes": 200},
                ],
                "estimated_payload_bytes": 300,
                "english_rows_before_audit": 20,
            }
            (root / "v2_sea_plan.json").write_text(json.dumps(plan), encoding="utf-8")
            receipt = {"group": {"unit": "g1"}, "records": [{"fixture": 1}, {"fixture": 2}]}
            (root / "receipts" / "g1.json").write_text(json.dumps(receipt), encoding="utf-8")

            report = inspect(root)
            self.assertEqual(report["state"], "materializing")
            self.assertTrue(report["safe_to_resume"])
            self.assertEqual(report["receipts"]["completed_groups"], 1)
            self.assertEqual(report["receipts"]["planned_groups"], 2)
            self.assertEqual(report["receipts"]["receipt_records"], 2)
            self.assertEqual(report["transfer"]["reserved_bytes"], 1234)
            self.assertIn("same v2_prepare_sea", report["next_action"])

    def test_malformed_transfer_ledger_blocks_automatic_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "transfer.json").write_text(json.dumps({"reserved_bytes": -5}), encoding="utf-8")
            report = inspect(root)
            self.assertEqual(report["state"], "blocked")
            self.assertFalse(report["safe_to_resume"])
            self.assertTrue(any("reserved_bytes" in message for message in report["issues"]))

    def test_audited_manifest_is_complete_even_when_empty_splits_are_warnings(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            records = [
                AudioRecord(
                    record_id="real-1",
                    label="bonafide",
                    dataset="fixture",
                    split="train",
                    audio_ref="audio/real.flac",
                    content_sha256="a" * 64,
                    speaker_id="speaker-a",
                    sample_rate=16_000,
                    codec="FLAC/PCM_16",
                ),
                AudioRecord(
                    record_id="fake-1",
                    label="spoof",
                    dataset="fixture",
                    split="train",
                    audio_ref="audio/fake.flac",
                    content_sha256="b" * 64,
                    generator_id="fixture-generator",
                    speaker_id="speaker-b",
                    sample_rate=16_000,
                    codec="FLAC/PCM_16",
                ),
            ]
            write_jsonl(root / "manifest.jsonl", records)
            report = inspect(root)
            self.assertEqual(report["state"], "complete")
            self.assertTrue(report["safe_to_resume"])
            self.assertTrue(report["manifest"]["audit_ok"])
            self.assertEqual(report["manifest"]["records"], 2)
            self.assertTrue(report["manifest"]["audit_warnings"])

    def test_retry_cache_usage_is_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scope = root / "range_cache" / "scope-a"
            scope.mkdir(parents=True)
            (scope / "range.bin").write_bytes(b"123456")
            (scope / "range.json").write_text("{}", encoding="utf-8")
            report = inspect(root)
            self.assertEqual(report["retry_cache"]["scopes"], 1)
            self.assertEqual(report["retry_cache"]["files"], 2)
            self.assertGreaterEqual(report["retry_cache"]["bytes"], 8)


if __name__ == "__main__":
    unittest.main()

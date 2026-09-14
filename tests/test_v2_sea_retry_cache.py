import json
import tempfile
import unittest
from pathlib import Path

from vaanirakshak.sea_transfer import MAX_BYTES, Transfer


class _Response:
    def __init__(self, data: bytes, content_range: str):
        self._data = data
        self._position = 0
        self.status = 206
        self.headers = {"Content-Range": content_range}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size=-1):
        if size < 0:
            size = len(self._data) - self._position
        start = self._position
        end = min(len(self._data), start + size)
        self._position = end
        return self._data[start:end]


class _Opener:
    def __init__(self, data: bytes, content_range: str):
        self.data = data
        self.content_range = content_range
        self.calls = 0

    def open(self, request, timeout=90):
        del request, timeout
        self.calls += 1
        return _Response(self.data, self.content_range)


class _FailOpener:
    def open(self, request, timeout=90):
        del request, timeout
        raise AssertionError("network must not be used when the persistent retry range is valid")


class V2SEARetryCacheTests(unittest.TestCase):
    def test_invalid_transfer_ledgers_are_rejected_before_network_setup(self):
        fixtures = [
            ("{not-json", "invalid JSON"),
            (json.dumps([]), "JSON object"),
            (json.dumps({"reserved_bytes": True}), "integer"),
            (json.dumps({"reserved_bytes": -1}), "hard ceiling"),
            (json.dumps({"reserved_bytes": MAX_BYTES + 1}), "hard ceiling"),
        ]
        for content, message in fixtures:
            with self.subTest(content=content), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                (root / "transfer.json").write_text(content, encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    Transfer(root, "unit-test-token")

    def test_exact_transfer_ceiling_loads_but_refuses_new_remote_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "transfer.json").write_text(
                json.dumps({"reserved_bytes": MAX_BYTES}), encoding="utf-8"
            )
            transfer = Transfer(root, "unit-test-token")
            transfer.opener = _FailOpener()
            with self.assertRaisesRegex(ValueError, "30 GB transfer ceiling"):
                transfer.fetch({"path": "data/train/example.parquet", "size": 10}, 0, 1)

    def test_persistent_range_survives_new_transfer_without_double_reservation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            item = {"path": "data/train/example.parquet", "size": 10}
            scope = "v2-sea-row-group:test-unit"

            first = Transfer(root, "unit-test-token")
            opener = _Opener(b"abcd", "bytes 0-3/10")
            first.opener = opener
            first.begin_scope(scope)
            self.assertEqual(first.fetch(item, 0, 4), b"abcd")
            self.assertEqual(first.used, 4)
            self.assertEqual(opener.calls, 1)
            first.end_scope(scope, clear=False)

            second = Transfer(root, "unit-test-token")
            second.opener = _FailOpener()
            second.begin_scope(scope)
            before = second.used
            self.assertEqual(second.fetch(item, 0, 4), b"abcd")
            self.assertEqual(second.used, before, "cached retry must not reserve the same remote bytes twice")
            second.end_scope(scope, clear=True)
            self.assertFalse((root / "range_cache").exists())

    def test_corrupt_persistent_range_is_deleted_and_refetched_fail_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            item = {"path": "data/train/example.parquet", "size": 10}
            scope = "v2-sea-row-group:test-corrupt"

            first = Transfer(root, "unit-test-token")
            first.opener = _Opener(b"abcd", "bytes 0-3/10")
            first.begin_scope(scope)
            self.assertEqual(first.fetch(item, 0, 4), b"abcd")
            first.end_scope(scope, clear=False)
            self.assertEqual(first.used, 4)

            cache_dir = first._scope_directory(scope)
            data_files = list(cache_dir.glob("*.bin"))
            self.assertEqual(len(data_files), 1)
            data_files[0].write_bytes(b"wxyz")

            second = Transfer(root, "unit-test-token")
            opener = _Opener(b"abcd", "bytes 0-3/10")
            second.opener = opener
            second.begin_scope(scope)
            self.assertEqual(second.fetch(item, 0, 4), b"abcd")
            self.assertEqual(opener.calls, 1, "corrupt cached bytes must never be trusted")
            self.assertEqual(second.used, 8, "refetch remains charged by the fail-closed transfer ledger")
            second.end_scope(scope, clear=True)

    def test_incomplete_cache_pair_is_dropped_before_refetch(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            item = {"path": "data/train/example.parquet", "size": 10}
            scope = "v2-sea-row-group:test-incomplete"

            first = Transfer(root, "unit-test-token")
            first.opener = _Opener(b"abcd", "bytes 0-3/10")
            first.begin_scope(scope)
            self.assertEqual(first.fetch(item, 0, 4), b"abcd")
            first.end_scope(scope, clear=False)

            cache_dir = first._scope_directory(scope)
            metadata_files = [path for path in cache_dir.glob("*.json") if path.name != "scope.json"]
            self.assertEqual(len(metadata_files), 1)
            metadata_files[0].unlink()

            second = Transfer(root, "unit-test-token")
            opener = _Opener(b"abcd", "bytes 0-3/10")
            second.opener = opener
            second.begin_scope(scope)
            self.assertEqual(second.fetch(item, 0, 4), b"abcd")
            self.assertEqual(opener.calls, 1)
            self.assertEqual(second.used, 8)
            second.end_scope(scope, clear=True)

    def test_scope_reopen_removes_stale_temporary_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            scope = "v2-sea-row-group:test-temp"
            first = Transfer(root, "unit-test-token")
            first.begin_scope(scope)
            cache_dir = first._scope_directory(scope)
            first.end_scope(scope, clear=False)
            stale = cache_dir / "orphan.bin.tmp"
            stale.write_bytes(b"partial")
            self.assertTrue(stale.exists())

            second = Transfer(root, "unit-test-token")
            second.begin_scope(scope)
            self.assertFalse(stale.exists())
            second.end_scope(scope, clear=True)

    def test_nested_retry_scopes_are_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            transfer = Transfer(Path(folder), "unit-test-token")
            transfer.begin_scope("one")
            with self.assertRaisesRegex(RuntimeError, "already active"):
                transfer.begin_scope("two")
            transfer.end_scope("one", clear=True)


if __name__ == "__main__":
    unittest.main()

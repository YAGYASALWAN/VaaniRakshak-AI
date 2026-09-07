import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from vaanirakshak.baseline_download import save_json


class AtomicJSONTests(unittest.TestCase):
    def test_temporary_windows_lock_retries_without_losing_old_value(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "transfer.json"
            target.write_text('{"reserved_bytes": 10}')
            replace = Path.replace
            attempts = []

            def transient(source, dest):
                attempts.append(source)
                self.assertEqual(json.loads(target.read_text())["reserved_bytes"], 10)
                if len(attempts) < 3:
                    raise PermissionError("Simulated Windows lock")
                return replace(source, dest)

            with patch.object(Path, "replace", transient), patch("vaanirakshak.baseline_download.time.sleep"):
                save_json(target, {"reserved_bytes": 20})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(json.loads(target.read_text())["reserved_bytes"], 20)
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_permanent_lock_preserves_both_ledger_and_pending_update(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "transfer.json"
            target.write_text('{"reserved_bytes": 10}')
            with patch.object(Path, "replace", side_effect=PermissionError("Locked")), patch("vaanirakshak.baseline_download.time.sleep"):
                with self.assertRaisesRegex(PermissionError, "outside OneDrive"):
                    save_json(target, {"reserved_bytes": 20})
            self.assertEqual(json.loads(target.read_text())["reserved_bytes"], 10)
            pending = list(Path(folder).glob("*.tmp"))
            self.assertEqual(len(pending), 1)
            self.assertEqual(json.loads(pending[0].read_text())["reserved_bytes"], 20)


if __name__ == "__main__":
    unittest.main()

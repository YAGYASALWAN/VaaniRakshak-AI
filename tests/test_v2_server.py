from array import array
import threading
import unittest

from vaanirakshak.v2_server import (
    REALTIME_BUDGET_MS,
    _ingest_without_blocking_event_loop,
    _with_realtime_budget,
)


class FakeSession:
    def __init__(self, buffered_samples=0, window_samples=100):
        self.buffer = array("h", [0] * buffered_samples)
        self.window_samples = window_samples
        self.thread_ids = []
        self.payloads = []

    def ingest_pcm16le(self, payload):
        self.thread_ids.append(threading.get_ident())
        self.payloads.append(payload)
        return ["emitted"] if len(self.buffer) + len(payload) // 2 >= self.window_samples else []


class RealtimeBudgetTests(unittest.TestCase):
    def test_budget_matches_two_second_hop(self):
        self.assertEqual(REALTIME_BUDGET_MS, 2000.0)

    def test_summary_reports_positive_realtime_margin(self):
        result = _with_realtime_budget({"mean_total_window_ms": 1250.0})
        self.assertTrue(result["observed_mean_within_hop_budget"])
        self.assertEqual(result["realtime_margin_ms"], 750.0)
        self.assertEqual(result["realtime_budget_ms"], 2000.0)

    def test_summary_reports_overrun(self):
        result = _with_realtime_budget({"mean_total_window_ms": 2250.0})
        self.assertFalse(result["observed_mean_within_hop_budget"])
        self.assertEqual(result["realtime_margin_ms"], -250.0)

    def test_summary_handles_no_windows_yet(self):
        result = _with_realtime_budget({"mean_total_window_ms": None})
        self.assertIsNone(result["observed_mean_within_hop_budget"])
        self.assertIsNone(result["realtime_margin_ms"])


class AsyncWindowExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_cheap_packet_buffering_stays_on_event_loop_thread(self):
        session = FakeSession(buffered_samples=0, window_samples=100)
        caller_thread = threading.get_ident()
        emitted = await _ingest_without_blocking_event_loop(session, b"\x00\x00" * 10)
        self.assertEqual(emitted, [])
        self.assertEqual(session.thread_ids, [caller_thread])

    async def test_packet_triggering_window_analysis_moves_to_worker_thread(self):
        session = FakeSession(buffered_samples=90, window_samples=100)
        caller_thread = threading.get_ident()
        emitted = await _ingest_without_blocking_event_loop(session, b"\x00\x00" * 10)
        self.assertEqual(emitted, ["emitted"])
        self.assertEqual(len(session.thread_ids), 1)
        self.assertNotEqual(session.thread_ids[0], caller_thread)


if __name__ == "__main__":
    unittest.main()

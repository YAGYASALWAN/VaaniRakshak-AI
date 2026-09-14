import struct
import unittest

from vaanirakshak.v2_engine import StreamingSession, aggregate_call, WindowResult


class FixedDetector:
    name = "fixed-test-detector"
    mode = "test"

    def __init__(self, scores):
        self.scores = iter(scores)

    def score(self, samples, sample_rate):
        return next(self.scores)


def pcm16(samples):
    return struct.pack("<" + "h" * len(samples), *samples)


class StreamingSessionTests(unittest.TestCase):
    def test_emits_four_second_window(self):
        sample_rate = 8_000
        session = StreamingSession(FixedDetector([0.8]), sample_rate=sample_rate)
        payload = pcm16([1000] * (sample_rate * 4))
        emitted = session.ingest_pcm16le(payload)
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].start_seconds, 0.0)
        self.assertEqual(emitted[0].end_seconds, 4.0)
        self.assertAlmostEqual(emitted[0].synthetic_score, 0.8)

    def test_partial_audio_is_buffered(self):
        sample_rate = 8_000
        session = StreamingSession(FixedDetector([0.4]), sample_rate=sample_rate)
        self.assertEqual(session.ingest_pcm16le(pcm16([200] * sample_rate)), [])
        self.assertEqual(session.live_summary()["segments_analyzed"], 0)

    def test_finalize_analyzes_two_second_tail(self):
        sample_rate = 8_000
        session = StreamingSession(FixedDetector([0.7]), sample_rate=sample_rate)
        session.ingest_pcm16le(pcm16([500] * (sample_rate * 2)))
        final = session.finalize()
        self.assertEqual(final["segments_analyzed"], 1)
        self.assertEqual(final["suspicious_segments"], 1)

    def test_short_tail_does_not_fake_evidence(self):
        sample_rate = 8_000
        session = StreamingSession(FixedDetector([]), sample_rate=sample_rate)
        session.ingest_pcm16le(pcm16([500] * sample_rate))
        final = session.finalize()
        self.assertEqual(final["segments_analyzed"], 0)
        self.assertEqual(final["verdict"], "Insufficient audio")

    def test_rejects_odd_pcm_payload(self):
        session = StreamingSession(FixedDetector([]), sample_rate=16_000)
        with self.assertRaises(ValueError):
            session.ingest_pcm16le(b"\x00")


class AggregationTests(unittest.TestCase):
    def test_high_scores_raise_call_risk(self):
        detector = FixedDetector([])
        windows = [
            WindowResult(0, 0.0, 4.0, 0.90),
            WindowResult(1, 4.0, 8.0, 0.88),
            WindowResult(2, 8.0, 12.0, 0.92),
        ]
        result = aggregate_call(windows, 12.0, detector)
        self.assertGreaterEqual(result["risk_score"], 81)
        self.assertEqual(result["verdict"], "Likely synthetic")
        self.assertEqual(result["suspicious_segments"], 3)

    def test_low_scores_remain_low_risk(self):
        detector = FixedDetector([])
        windows = [
            WindowResult(0, 0.0, 4.0, 0.10),
            WindowResult(1, 4.0, 8.0, 0.18),
            WindowResult(2, 8.0, 12.0, 0.20),
        ]
        result = aggregate_call(windows, 12.0, detector)
        self.assertLessEqual(result["risk_score"], 30)
        self.assertEqual(result["verdict"], "Likely genuine")


if __name__ == "__main__":
    unittest.main()

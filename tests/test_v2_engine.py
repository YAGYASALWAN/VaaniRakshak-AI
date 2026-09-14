import struct
import unittest

from vaanirakshak.v2_audio import AudioQuality, MODEL_SAMPLE_RATE
from vaanirakshak.v2_engine import StreamingSession, aggregate_call, WindowResult


class FixedDetector:
    name = "fixed-test-detector"
    mode = "test"
    calibrated_probability = False
    notice = "test detector"

    def __init__(self, scores, threshold=0.65):
        self.scores = iter(scores)
        self.calls = []
        self.threshold = threshold

    def score(self, samples, sample_rate):
        self.calls.append((len(samples), sample_rate))
        return next(self.scores)


def pcm16(samples):
    return struct.pack("<" + "h" * len(samples), *samples)


def quality(*, speech_ratio=1.0, usable=True, reason=None):
    return AudioQuality(
        rms_dbfs=-25.0 if usable else -120.0,
        peak=0.2 if usable else 0.0,
        clipping_ratio=0.0,
        speech_ratio=speech_ratio,
        usable=usable,
        reason=reason,
    )


def window(index, start, end, score, q, threshold=0.65):
    return WindowResult(index, start, end, score, q, threshold)


class StreamingSessionTests(unittest.TestCase):
    def test_emits_four_second_window_normalized_to_16khz(self):
        sample_rate = 48_000
        detector = FixedDetector([0.8])
        session = StreamingSession(detector, sample_rate=sample_rate)
        payload = pcm16([1000] * (sample_rate * 4))
        emitted = session.ingest_pcm16le(payload)

        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0].start_seconds, 0.0)
        self.assertEqual(emitted[0].end_seconds, 4.0)
        self.assertAlmostEqual(emitted[0].synthetic_score, 0.8)
        self.assertEqual(emitted[0].threshold, 0.65)
        self.assertEqual(detector.calls[0][1], MODEL_SAMPLE_RATE)
        self.assertAlmostEqual(detector.calls[0][0], MODEL_SAMPLE_RATE * 4, delta=2)
        self.assertIsNotNone(emitted[0].preprocessing_ms)
        self.assertIsNotNone(emitted[0].inference_ms)
        self.assertIsNotNone(emitted[0].total_analysis_ms)
        self.assertGreaterEqual(emitted[0].preprocessing_ms, 0.0)
        self.assertGreaterEqual(emitted[0].inference_ms, 0.0)
        self.assertGreaterEqual(emitted[0].total_analysis_ms, emitted[0].preprocessing_ms)
        summary = session.live_summary()
        self.assertIsNotNone(summary["mean_preprocessing_ms"])
        self.assertIsNotNone(summary["mean_inference_ms"])
        self.assertIsNotNone(summary["mean_total_window_ms"])

    def test_uses_two_second_hop_for_overlapping_windows(self):
        sample_rate = 8_000
        detector = FixedDetector([0.4, 0.6])
        session = StreamingSession(detector, sample_rate=sample_rate)
        emitted = session.ingest_pcm16le(pcm16([1000] * (sample_rate * 6)))

        self.assertEqual(len(emitted), 2)
        self.assertEqual((emitted[0].start_seconds, emitted[0].end_seconds), (0.0, 4.0))
        self.assertEqual((emitted[1].start_seconds, emitted[1].end_seconds), (2.0, 6.0))

        final = session.finalize()
        self.assertEqual(final["windows_seen"], 2, "finalization must not duplicate the retained overlap")

    def test_partial_audio_is_buffered(self):
        sample_rate = 8_000
        session = StreamingSession(FixedDetector([0.4]), sample_rate=sample_rate)
        self.assertEqual(session.ingest_pcm16le(pcm16([200] * sample_rate)), [])
        self.assertEqual(session.live_summary()["segments_analyzed"], 0)

    def test_finalize_analyzes_two_second_tail_when_it_is_new_audio(self):
        sample_rate = 8_000
        session = StreamingSession(FixedDetector([0.7]), sample_rate=sample_rate)
        session.ingest_pcm16le(pcm16([500] * (sample_rate * 2)))
        final = session.finalize()
        self.assertEqual(final["segments_analyzed"], 1)
        self.assertEqual(final["suspicious_segments"], 1)
        self.assertEqual(final["verdict"], "Insufficient evidence")

    def test_silence_is_quality_skipped_without_calling_detector(self):
        sample_rate = 16_000
        detector = FixedDetector([])
        session = StreamingSession(detector, sample_rate=sample_rate)
        emitted = session.ingest_pcm16le(pcm16([0] * (sample_rate * 4)))

        self.assertEqual(len(emitted), 1)
        self.assertFalse(emitted[0].analyzed)
        self.assertEqual(emitted[0].quality.reason, "too_quiet")
        self.assertEqual(detector.calls, [])
        self.assertIsNotNone(emitted[0].preprocessing_ms)
        self.assertIsNone(emitted[0].inference_ms)
        self.assertIsNotNone(emitted[0].total_analysis_ms)
        summary = session.live_summary()
        self.assertEqual(summary["segments_skipped"], 1)
        self.assertEqual(summary["segments_analyzed"], 0)
        self.assertFalse(summary["enough_evidence"])
        self.assertIsNotNone(summary["mean_preprocessing_ms"])
        self.assertIsNone(summary["mean_inference_ms"])
        self.assertIsNotNone(summary["mean_total_window_ms"])

    def test_short_tail_does_not_fake_evidence(self):
        sample_rate = 8_000
        session = StreamingSession(FixedDetector([]), sample_rate=sample_rate)
        session.ingest_pcm16le(pcm16([500] * sample_rate))
        final = session.finalize()
        self.assertEqual(final["segments_analyzed"], 0)
        self.assertEqual(final["verdict"], "Insufficient evidence")

    def test_rejects_odd_pcm_payload(self):
        session = StreamingSession(FixedDetector([]), sample_rate=16_000)
        with self.assertRaises(ValueError):
            session.ingest_pcm16le(b"\x00")

    def test_rejects_more_audio_after_finalize(self):
        session = StreamingSession(FixedDetector([]), sample_rate=16_000)
        session.finalize()
        with self.assertRaises(RuntimeError):
            session.ingest_pcm16le(pcm16([500] * 100))

    def test_detector_threshold_is_used_instead_of_product_constant(self):
        sample_rate = 16_000
        detector = FixedDetector([0.72], threshold=0.80)
        session = StreamingSession(detector, sample_rate=sample_rate)
        emitted = session.ingest_pcm16le(pcm16([1000] * (sample_rate * 4)))
        self.assertEqual(len(emitted), 1)
        self.assertFalse(emitted[0].suspicious)
        self.assertEqual(emitted[0].threshold, 0.80)
        self.assertEqual(session.live_summary()["threshold"], 0.80)


class AggregationTests(unittest.TestCase):
    def test_high_scores_raise_call_risk_when_evidence_is_sufficient(self):
        detector = FixedDetector([])
        q = quality()
        windows = [
            window(0, 0.0, 4.0, 0.90, q),
            window(1, 2.0, 6.0, 0.88, q),
            window(2, 4.0, 8.0, 0.92, q),
        ]
        result = aggregate_call(windows, 8.0, detector)
        self.assertTrue(result["enough_evidence"])
        self.assertGreaterEqual(result["risk_score"], 81)
        self.assertEqual(result["verdict"], "Likely synthetic")
        self.assertEqual(result["suspicious_segments"], 3)
        self.assertAlmostEqual(result["usable_speech_seconds"], 8.0)

    def test_low_scores_remain_low_risk(self):
        detector = FixedDetector([])
        q = quality()
        windows = [
            window(0, 0.0, 4.0, 0.10, q),
            window(1, 2.0, 6.0, 0.18, q),
            window(2, 4.0, 8.0, 0.20, q),
        ]
        result = aggregate_call(windows, 8.0, detector)
        self.assertTrue(result["enough_evidence"])
        self.assertLessEqual(result["risk_score"], 30)
        self.assertEqual(result["verdict"], "Likely genuine")

    def test_one_analyzed_window_cannot_force_verdict(self):
        detector = FixedDetector([])
        result = aggregate_call(
            [window(0, 0.0, 4.0, 0.99, quality())],
            4.0,
            detector,
        )
        self.assertFalse(result["enough_evidence"])
        self.assertEqual(result["verdict"], "Insufficient evidence")

    def test_skipped_window_is_not_counted_as_model_evidence(self):
        detector = FixedDetector([])
        windows = [
            window(0, 0.0, 4.0, None, quality(speech_ratio=0.0, usable=False, reason="too_quiet")),
            window(1, 2.0, 6.0, 0.9, quality()),
        ]
        result = aggregate_call(windows, 6.0, detector)
        self.assertEqual(result["windows_seen"], 2)
        self.assertEqual(result["segments_skipped"], 1)
        self.assertEqual(result["segments_analyzed"], 1)
        self.assertFalse(result["enough_evidence"])

    def test_suspicious_count_respects_detector_operating_threshold(self):
        detector = FixedDetector([], threshold=0.80)
        q = quality()
        windows = [
            window(0, 0.0, 4.0, 0.75, q, threshold=0.80),
            window(1, 2.0, 6.0, 0.82, q, threshold=0.80),
        ]
        result = aggregate_call(windows, 6.0, detector)
        self.assertEqual(result["suspicious_segments"], 1)
        self.assertEqual(result["threshold"], 0.80)


if __name__ == "__main__":
    unittest.main()

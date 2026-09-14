import unittest

import numpy as np

from vaanirakshak.v2_audio import (
    EnergySpeechGate,
    WebRTCSpeechGate,
    assess_audio_quality,
    build_speech_gate,
)


class FixedGate:
    def __init__(self, ratio):
        self.ratio = ratio
        self.name = "fixed-gate"

    def speech_ratio(self, wave, sample_rate):
        return self.ratio


class V2AudioGateTests(unittest.TestCase):
    def test_energy_gate_keeps_explicit_fallback_name(self):
        gate = build_speech_gate("energy")
        self.assertIsInstance(gate, EnergySpeechGate)
        self.assertEqual(gate.name, "energy-v1")

    def test_webrtc_gate_accepts_browser_rate_and_rejects_silence(self):
        gate = WebRTCSpeechGate(aggressiveness=2)
        silence = np.zeros(44_100 * 2, dtype=np.float32)
        ratio = gate.speech_ratio(silence, 44_100)
        self.assertEqual(ratio, 0.0)
        self.assertEqual(gate.name, "webrtc-vad-m2")

    def test_webrtc_aggressiveness_is_validated(self):
        with self.assertRaises(ValueError):
            WebRTCSpeechGate(aggressiveness=4)

    def test_unknown_gate_is_rejected(self):
        with self.assertRaises(ValueError):
            build_speech_gate("mystery-vad")

    def test_quality_records_gate_and_uses_gate_ratio(self):
        t = np.arange(16_000 * 2, dtype=np.float32) / 16_000
        wave = 0.10 * np.sin(2 * np.pi * 220.0 * t)
        quality = assess_audio_quality(wave, 16_000, speech_gate=FixedGate(0.8))
        self.assertTrue(quality.usable)
        self.assertAlmostEqual(quality.speech_ratio, 0.8)
        self.assertEqual(quality.speech_gate, "fixed-gate")
        self.assertEqual(quality.as_dict()["speech_gate"], "fixed-gate")

    def test_low_speech_ratio_refuses_otherwise_loud_audio(self):
        wave = np.full(16_000 * 2, 0.10, dtype=np.float32)
        quality = assess_audio_quality(wave, 16_000, speech_gate=FixedGate(0.1))
        self.assertFalse(quality.usable)
        self.assertEqual(quality.reason, "insufficient_speech")


if __name__ == "__main__":
    unittest.main()

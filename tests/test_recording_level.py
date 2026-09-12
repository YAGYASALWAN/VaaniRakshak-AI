"""Recording-level windowing, pooling and warm-start behaviour."""
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "torchaudio", "soundfile", "numpy"))
if AVAILABLE:
    import numpy as np
    import soundfile as sf
    import torch
    from vaanirakshak import recording_level as rl
    from vaanirakshak.english_training import EnglishCNN


@unittest.skipUnless(AVAILABLE, "Requires GPU environment packages (CPU tests supported)")
class RecordingLevelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(0)

    def audio(self, seconds=10.0, rate=16000):
        buffer = io.BytesIO()
        frames = int(seconds * rate)
        wave = .1 * np.sin(np.arange(frames) * 440 * 2 * np.pi / rate)
        sf.write(buffer, wave, rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    def features(self, windows, coverage=None):
        coverage = torch.ones(windows) if coverage is None else coverage
        return torch.randn(windows, 64, 401), coverage

    def test_window_geometry_follows_the_project_config(self):
        rate, window, hop = rl.windowing()
        # configs/data_config.yaml: 16 kHz, 4 s windows, 2 s hop (50% overlap).
        self.assertEqual((rate, window, hop), (16000, 64000, 32000))

    def test_window_bounds_overlap_and_report_coverage(self):
        starts, coverage = rl.window_bounds(10 * 16000, 64000, 32000)
        self.assertEqual(starts, [0, 32000, 64000, 96000])
        # 10 s at a 2 s hop: the fourth window starts at 6 s and ends exactly on the
        # end of the audio, so nothing here is padded.
        self.assertEqual(coverage, [1.0, 1.0, 1.0, 1.0])
        # A clip shorter than one window still yields exactly one, partly padded.
        starts, coverage = rl.window_bounds(16000, 64000, 32000)
        self.assertEqual(starts, [0])
        self.assertAlmostEqual(coverage[0], 0.25)
        with self.assertRaises(ValueError):
            rl.window_bounds(0, 64000, 32000)

    def test_low_coverage_windows_are_dropped_but_never_all_of_them(self):
        # 9 s of audio: windows at 0/2/4/6/8 s, the last holding only 1 s of audio.
        windows, coverage = rl.recording_windows(self.audio(9.0))
        self.assertEqual(windows.shape[1], 64000)
        self.assertTrue((coverage >= rl.MIN_COVERAGE).all())
        self.assertEqual(windows.shape[0], coverage.shape[0])
        # A 1-second recording is mostly padding, but dropping it entirely would
        # silently delete short clips and make duration a class shortcut.
        windows, coverage = rl.recording_windows(self.audio(1.0))
        self.assertEqual(windows.shape[0], 1)
        self.assertAlmostEqual(float(coverage[0]), 0.25, places=3)

    def test_recording_windows_rejects_unusable_audio(self):
        for bad in (b"", "not bytes", None):
            with self.assertRaises(ValueError):
                rl.recording_windows(bad)
        silent = io.BytesIO()
        sf.write(silent, np.zeros(32000), 16000, format="WAV", subtype="PCM_16")
        with self.assertRaises(ValueError):
            rl.recording_windows(silent.getvalue())

    def test_long_audio_is_capped(self):
        windows, _ = rl.recording_windows(self.audio(200.0), max_seconds=20)
        self.assertLessEqual(windows.shape[0] * 32000, 20 * 16000 + 64000)

    def test_collate_keeps_one_label_per_recording(self):
        batch = [self.features(2) + (1.0,), self.features(5) + (0.0,)]
        values, weights, labels = rl.collate_recordings(batch)
        self.assertEqual(tuple(values.shape), (2, 5, 64, 401))
        self.assertEqual(tuple(labels.shape), (2,))
        # The short recording contributes one label, not two, and its padded slots
        # carry no weight however many windows the long one has.
        self.assertEqual(weights[0].tolist(), [1., 1., 0., 0., 0.])
        self.assertEqual(weights[1].tolist(), [1.] * 5)
        with self.assertRaises(ValueError):
            rl.collate_recordings([])
        with self.assertRaises(ValueError):
            rl.collate_recordings([(torch.randn(2, 64, 401), torch.ones(3), 1.0)])
        with self.assertRaises(ValueError):
            rl.collate_recordings([(torch.randn(2, 64, 401), torch.zeros(2), 1.0)])

    def test_model_emits_one_logit_per_window_and_per_recording(self):
        model = rl.RecordingLevelCNN().eval()
        values, weights, _ = rl.collate_recordings([self.features(3) + (1.0,), self.features(1) + (0.0,)])
        pooled, window_logits = model(values, weights, per_window=True)
        self.assertEqual(tuple(pooled.shape), (2,))
        # The per-window series is what the rolling risk score and the suspicious
        # timestamps are built from; without it the product has nothing to plot.
        self.assertEqual(tuple(window_logits.shape), (2, 3))
        self.assertTrue(torch.isfinite(pooled).all())

    def test_padded_slots_never_reach_the_encoder(self):
        # Zero-weight slots hold arbitrary values. They must change nothing - in
        # training mode too, where BatchNorm would otherwise fold their statistics
        # into every real window in the batch.
        for training in (False, True):
            model = rl.RecordingLevelCNN()
            model.train(training)
            model.dropout.eval()
            features, coverage = self.features(3)
            with torch.no_grad():
                alone = model(features.unsqueeze(0), coverage.unsqueeze(0))
                stuffed = torch.cat([features, torch.randn(4, 64, 401) * 50]).unsqueeze(0)
                together = model(stuffed, torch.cat([coverage, torch.zeros(4)]).unsqueeze(0))
            self.assertAlmostEqual(float(alone[0]), float(together[0]), places=5,
                                   msg=f"padding leaked into the score (training={training})")
        with self.assertRaises(ValueError):
            model(torch.randn(1, 2, 64, 401), torch.zeros(1, 2))

    def test_recordings_in_a_batch_score_independently_at_inference(self):
        model = rl.RecordingLevelCNN().eval()
        features, coverage = self.features(3)
        with torch.no_grad():
            alone = model(features.unsqueeze(0), coverage.unsqueeze(0))
            values, weights, _ = rl.collate_recordings(
                [(features, coverage, 1.0), self.features(7) + (0.0,)])
            together = model(values, weights)
        self.assertAlmostEqual(float(alone[0]), float(together[0]), places=4)

    def test_coverage_weights_the_pool(self):
        logits = torch.tensor([[2.0, 0.0]])
        self.assertAlmostEqual(float(rl.masked_pool(logits, torch.tensor([[1.0, 1.0]]))), 1.0)
        # A window that is only a quarter real audio counts a quarter.
        self.assertAlmostEqual(float(rl.masked_pool(logits, torch.tensor([[1.0, 0.25]]))), 1.6)
        self.assertAlmostEqual(float(rl.masked_pool(logits, torch.tensor([[1.0, 0.0]]))), 2.0)

    def test_logsumexp_pool_lets_one_window_carry_the_recording(self):
        # A long ordinary call with a single strongly synthetic passage: the mean
        # dilutes it, the smooth maximum does not.
        logits = torch.tensor([[-4.0] * 19 + [6.0]])
        weights = torch.ones(1, 20)
        mean = float(rl.masked_pool(logits, weights, "mean"))
        smooth = float(rl.masked_pool(logits, weights, "logsumexp"))
        self.assertLess(mean, 0)
        self.assertGreater(smooth, mean)
        self.assertGreater(smooth, 2.0)
        with self.assertRaises(ValueError):
            rl.masked_pool(logits, weights, "median")

    def test_warm_start_copies_the_trained_convolutional_trunk(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            trained = EnglishCNN()
            torch.save({"schema": "english-cnn-v1", "model": trained.state_dict()}, path)
            encoder = rl.WindowEncoder()
            copied = rl.warm_start_encoder(encoder, path)
            self.assertTrue(copied)
            # Same trunk, same parameter names: the existing checkpoint is a free
            # initialisation instead of a fresh run.
            for key in copied:
                self.assertTrue(torch.equal(encoder.state_dict()[key], trained.state_dict()[key]))
            self.assertNotIn("network.19.weight", copied)

            torch.save({"schema": "something-else", "model": trained.state_dict()}, path)
            with self.assertRaises(ValueError):
                rl.warm_start_encoder(rl.WindowEncoder(), path)

    def test_end_to_end_from_audio_bytes(self):
        transform = rl.make_window_features(device="cpu")
        features, coverage = transform(self.audio(9.0))
        self.assertEqual(features.shape[0], coverage.shape[0])
        self.assertEqual(tuple(features.shape[1:]), (64, 401))
        model = rl.RecordingLevelCNN(pooling="logsumexp").eval()
        values, weights, labels = rl.collate_recordings([(features, coverage, 1.0)])
        pooled, window_logits = model(values, weights, per_window=True)
        self.assertEqual(tuple(labels.shape), (1,))
        self.assertEqual(window_logits.shape[1], features.shape[0])
        self.assertTrue(torch.isfinite(pooled).all())


if __name__ == "__main__":
    unittest.main()

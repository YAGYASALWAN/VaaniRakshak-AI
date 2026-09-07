"""Small CPU integration checks; never download benchmark audio."""
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "torchaudio", "soundfile", "numpy"))
if AVAILABLE:
    import numpy as np
    import soundfile as sf
    import torch
    from vaanirakshak import english_training as training


@unittest.skipUnless(AVAILABLE, "Requires GPU environment packages (CPU tests supported)")
class EnglishRuntimeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def audio(self, index=0, frames=8000, rate=16000):
        buffer = io.BytesIO()
        wave = .1 * np.sin(np.arange(frames) * (index + 100) / rate)
        sf.write(buffer, wave, rate, format="WAV", subtype="PCM_16")
        return buffer.getvalue()

    def test_short_audio_and_frontend_gradient(self):
        wave, metadata = training.waveform_window(self.audio())
        self.assertEqual(wave.shape, (64000,))
        self.assertTrue(metadata["padded"])
        self.assertTrue(np.all(wave[8000:] == 0))
        features = training.Frontend()(torch.from_numpy(np.stack([wave, wave])))
        self.assertEqual(tuple(features.shape), (2, 64, 401))
        model = training.EnglishCNN()
        loss = model(features).square().mean()
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaises(ValueError):
            training.waveform_window(self.audio(rate=8000))

    def test_cache_interrupt_resume_and_offline_reuse(self):
        records = [dict(key=i % 2, audio_file_name=f"LA_T_{i:07d}", speaker_id="LA_0001",
                        system_id="-" if i % 2 == 0 else "A01", audio={"bytes": self.audio(i)})
                   for i in range(66)]
        skipped = []

        class Stream:
            def __init__(self, fail=False, start=0):
                self.fail, self.start = fail, start

            def skip(self, n):
                skipped.append(n)
                return Stream(start=n)

            def __iter__(self):
                for i in range(self.start, len(records)):
                    if self.fail and i == 65:
                        raise ConnectionError("simulated interrupted transfer")
                    yield records[i]

        with tempfile.TemporaryDirectory() as temporary, patch.dict(training.COUNTS, {"train": {0: 33, 1: 33}}):
            root = Path(temporary)
            with self.assertRaises(ConnectionError):
                training.cache_split(root, "train", "cpu", lambda _: Stream(fail=True))
            features, rows = training.cache_split(root, "train", "cpu", lambda _: Stream())
            self.assertEqual(skipped, [64])
            self.assertEqual(len(rows), 66)
            expected = features.copy()
            del features

            def no_network(_):
                self.fail("Completed cache must not access the dataset")

            cached, cached_rows = training.cache_split(root, "train", "cpu", no_network)
            np.testing.assert_array_equal(cached, expected)
            self.assertEqual(rows, cached_rows)
            del cached
            path = root / "train" / "features.npy"
            with path.open("r+b") as handle:
                handle.seek(-1, 2)
                value = handle.read(1)
                handle.seek(-1, 2)
                handle.write(bytes([value[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, "checksum"):
                training.cache_split(root, "train", "cpu", no_network)


if __name__ == "__main__":
    unittest.main()

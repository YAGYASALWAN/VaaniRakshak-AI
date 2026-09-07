"""Real tensor/audio checks when the optional training dependencies are installed."""
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "torchaudio", "soundfile"))


@unittest.skipUnless(AVAILABLE, "Training dependencies not installed in this environment")
class BaselineRuntimeTests(unittest.TestCase):
    def test_preprocessing_model_gradient_and_checkpoint(self):
        import numpy as np
        import soundfile as sf
        import torch
        from vaanirakshak.baseline import AudioCNN, audio_window, predict
        torch.set_num_threads(2)
        t = np.arange(32000) / 16000
        waveform = (0.2 * np.sin(2 * np.pi * 440 * t)).astype("float32")
        source = io.BytesIO()
        sf.write(source, waveform, 16000, format="WAV")
        source.seek(0)
        processed, info = audio_window(source)
        self.assertEqual(processed.shape, (64000,))
        self.assertTrue(info["padded"])
        model = AudioCNN()
        output = model(torch.from_numpy(processed).repeat(2, 1))
        self.assertEqual(tuple(output.shape), (2,))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, torch.tensor([0., 1.]))
        loss.backward()
        self.assertTrue(torch.isfinite(model.encoder[-1].weight.grad).all())
        with tempfile.TemporaryDirectory() as folder:
            audio = Path(folder) / "sample.wav"
            checkpoint = Path(folder) / "model.pt"
            sf.write(audio, waveform, 16000)
            torch.save({"schema": "baseline-cnn-v1", "model": model.state_dict(), "threshold": .5}, checkpoint)
            result = predict(checkpoint, audio, "cpu")
            self.assertTrue(0 <= result["synthetic_score"] <= 1)

    def test_parquet_original_audio_representation(self):
        if not importlib.util.find_spec("pyarrow"):
            self.skipTest("PyArrow not installed")
        import numpy as np
        import pyarrow as pa
        import pyarrow.parquet as pq
        import soundfile as sf
        from vaanirakshak.baseline import audio_window
        buffer = io.BytesIO()
        sf.write(buffer, np.sin(np.arange(16000) * .1).astype("float32") * .1, 16000, format="WAV")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fixture.parquet"
            pq.write_table(pa.Table.from_pylist([{"speaker_id": "s1", "audio": {"bytes": buffer.getvalue(), "path": "original.wav"}}]), path)
            with pq.ParquetFile(path) as file:
                self.assertEqual(file.read(columns=["speaker_id"]).to_pylist(), [{"speaker_id": "s1"}])
                batch = next(file.iter_batches(batch_size=4, columns=["audio"], use_threads=False))
                raw = batch.column(0).to_pylist()[0]["bytes"]
                self.assertEqual(audio_window(io.BytesIO(raw))[0].shape, (64000,))


if __name__ == "__main__":
    unittest.main()

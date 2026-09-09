"""Small CPU integration tests; no remote audio or credentials required."""
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

AVAILABLE = all(importlib.util.find_spec(m) for m in ('torch', 'torchaudio', 'soundfile'))


@unittest.skipUnless(AVAILABLE, 'requires torch, torchaudio and soundfile')
class RuntimeTests(unittest.TestCase):
    def test_feature_and_demo_preprocessing_agree(self):
        import numpy as np
        import soundfile as sf
        import torch
        from vaanirakshak.mlaad_training import feature_transform
        from vaanirakshak.english_training import EnglishCNN, CACHE_VERSION
        from vaanirakshak.demo_server import Detector
        torch.set_num_threads(2)
        rate = 22050
        t = np.arange(rate * 3)/rate
        wave = np.stack([.2*np.sin(2*np.pi*230*t), .1*np.sin(2*np.pi*410*t)], axis=1)
        raw = io.BytesIO()
        sf.write(raw, wave, rate, format='WAV')
        feature, props = feature_transform('cpu')(raw.getvalue())
        self.assertTrue(props['padded'])
        model = EnglishCNN().eval()
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'best.pt'
            torch.save({'schema':'english-cnn-v1', 'frontend': CACHE_VERSION,
                        'model':model.state_dict(), 'threshold':.5, 'notice':'fixture'}, path)
            actual = Detector(path).analyze(raw.getvalue())['synthetic_score']
            with torch.no_grad():
                expected = model(torch.from_numpy(feature.astype(np.float32)).unsqueeze(0)).sigmoid().item()
            self.assertAlmostEqual(actual, expected, places=6)

    def test_offline_training_checkpoint_and_demo(self):
        import numpy as np
        import torch
        from vaanirakshak.mlaad_cache import FeatureCache, SHAPE, atomic_json, partition
        from vaanirakshak.mlaad_training import identity, train
        from vaanirakshak.demo_server import Detector
        from vaanirakshak import english_training
        source = {'repository':'fixture', 'revision':'fixture', 'rows':[], 'originals':[]}
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)/'cache'
            c = FeatureCache(root, identity(source))
            rng = np.random.default_rng(42)
            for i in range(100):
                for label in (0, 1):
                    row = {'id':f'{i}:{label}', 'original_file':f'en_US/speaker/book/wavs/{i}.wav',
                           'label':label, 'window_sha256':f'{i}:{label}'}
                    c.put(row, rng.normal(0, 1, SHAPE))
            rows, audit = partition(c.rows(), minimum=1)
            c.close()
            atomic_json(root/'plan.json', source)
            atomic_json(root/'manifest.json', {'identity':identity(source), 'rows':rows, 'audit':audit})
            cwd = os.getcwd()
            os.chdir(d)
            try:
                # Only the minimum per-class count is reduced for this tiny fixture.
                with patch('vaanirakshak.mlaad_training.partition', side_effect=lambda r: partition(r, minimum=1)), \
                     patch('vaanirakshak.mlaad_training.Network', side_effect=AssertionError('offline training made a network client')):
                    run = train(root, epochs=1, batch_size=16, device='cpu')
                    self.assertTrue((run/'best.pt').is_file())
                    self.assertTrue((run/'last.pt').is_file())
                    report = json.loads((run/'evaluation.json').read_text())
                    self.assertEqual(report['best_epoch'], 1)
                    self.assertTrue(Detector(run/'best.pt').status()['ready'])
                    with self.assertRaisesRegex(ValueError, 'already evaluated'):
                        train(root, epochs=2, batch_size=16, device='cpu', resume=run)
            finally:
                os.chdir(cwd)


if __name__ == '__main__':
    unittest.main()

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from vaanirakshak.v2_detectors import CheckpointDetector, RAW_FRONTEND, V2_CHECKPOINT_SCHEMA
from vaanirakshak.v2_ssl_model import ARCHITECTURE, WavLMAntiSpoof, WavLMSpec


TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None


@unittest.skipUnless(TRANSFORMERS_AVAILABLE, "transformers optional dependency not installed")
class WavLMPortableCheckpointTests(unittest.TestCase):
    def _tiny_model(self):
        from transformers import WavLMConfig, WavLMModel

        config = WavLMConfig(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            conv_dim=(8, 8, 8, 8, 8, 8, 8),
            num_conv_pos_embedding_groups=2,
            num_conv_pos_embeddings=16,
        )
        backbone = WavLMModel(config)
        spec = WavLMSpec(backbone_id="tiny-unit-test", attention_hidden=8, classifier_hidden=16, dropout=0.0)
        return WavLMAntiSpoof(backbone, spec)

    def test_exported_config_reconstructs_same_shapes(self):
        model = self._tiny_model().eval()
        exported = model.export_model_config()
        restored = WavLMAntiSpoof.from_exported_config(exported["backbone_config"], exported["model_spec"])
        restored.load_state_dict(model.state_dict())
        values = torch.zeros(2, 16_000)
        mask = torch.ones_like(values, dtype=torch.long)
        with torch.no_grad():
            output = restored(values, attention_mask=mask)
        self.assertEqual(tuple(output.shape), (2,))

    def test_product_adapter_loads_portable_wavlm_checkpoint(self):
        model = self._tiny_model().eval()
        exported = model.export_model_config()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "best.pt"
            torch.save(
                {
                    "schema": V2_CHECKPOINT_SCHEMA,
                    "architecture": ARCHITECTURE,
                    "frontend": RAW_FRONTEND,
                    "model": model.state_dict(),
                    "backbone_config": exported["backbone_config"],
                    "model_spec": exported["model_spec"],
                    "threshold": 0.6,
                    "calibrated_probability": False,
                    "model_name": "tiny-wavlm-test",
                    "notice": "unit test",
                },
                path,
            )
            detector = CheckpointDetector(path)
            self.assertEqual(detector.architecture, ARCHITECTURE)
            score = detector.score(np.zeros(16_000, dtype=np.float32), 16_000)
            self.assertTrue(np.isfinite(score))
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()

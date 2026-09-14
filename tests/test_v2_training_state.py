import unittest

from vaanirakshak.v2_train_ssl import (
    TRAINING_STATE_SCHEMA,
    _training_contract,
    _validate_resume_state,
)


class V2TrainingResumeContractTests(unittest.TestCase):
    def contract(self):
        return _training_contract(
            backbone_id="microsoft/wavlm-base-plus",
            epochs=8,
            batch_size=2,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            head_only_epochs=1,
            backbone_lr=1e-5,
            head_lr=1e-4,
            weight_decay=1e-4,
            max_dev_fpr=0.05,
            seed=42,
        )

    def state(self):
        return {
            "schema": TRAINING_STATE_SCHEMA,
            "manifest_fingerprint": "abc123",
            "training_contract": self.contract(),
            "backbone_config": {"hidden_size": 768},
            "model_spec": {"architecture": "wavlm"},
            "model": {},
            "optimizer": {},
            "scaler": {},
            "epoch": 3,
            "history": [{"epoch": 1}, {"epoch": 2}, {"epoch": 3}],
            "best_eer": 0.12,
            "torch_rng": "placeholder",
            "sampler_rng": "placeholder",
        }

    def test_matching_state_is_accepted(self):
        _validate_resume_state(self.state(), contract=self.contract(), fingerprint="abc123")

    def test_changed_manifest_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            _validate_resume_state(self.state(), contract=self.contract(), fingerprint="different")

    def test_changed_hyperparameter_is_rejected(self):
        changed = dict(self.contract())
        changed["batch_size"] = 4
        with self.assertRaisesRegex(ValueError, "hyperparameters/backbone"):
            _validate_resume_state(self.state(), contract=changed, fingerprint="abc123")

    def test_wrong_state_schema_is_rejected(self):
        state = self.state()
        state["schema"] = "not-v2-state"
        with self.assertRaisesRegex(ValueError, "resumable training state"):
            _validate_resume_state(state, contract=self.contract(), fingerprint="abc123")

    def test_missing_optimizer_state_is_rejected(self):
        state = self.state()
        del state["optimizer"]
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            _validate_resume_state(state, contract=self.contract(), fingerprint="abc123")

    def test_missing_exported_model_config_is_rejected(self):
        state = self.state()
        del state["backbone_config"]
        with self.assertRaisesRegex(ValueError, "missing required fields"):
            _validate_resume_state(state, contract=self.contract(), fingerprint="abc123")

    def test_completed_epoch_must_be_within_contract(self):
        state = self.state()
        state["epoch"] = 9
        with self.assertRaisesRegex(ValueError, "invalid completed epoch"):
            _validate_resume_state(state, contract=self.contract(), fingerprint="abc123")

    def test_history_must_match_completed_epoch(self):
        state = self.state()
        state["history"] = [{"epoch": 1}, {"epoch": 2}]
        with self.assertRaisesRegex(ValueError, "history does not match"):
            _validate_resume_state(state, contract=self.contract(), fingerprint="abc123")

    def test_history_epochs_must_be_contiguous(self):
        state = self.state()
        state["history"] = [{"epoch": 1}, {"epoch": 3}, {"epoch": 2}]
        with self.assertRaisesRegex(ValueError, "not contiguous"):
            _validate_resume_state(state, contract=self.contract(), fingerprint="abc123")

    def test_best_eer_must_be_finite_probability(self):
        state = self.state()
        state["best_eer"] = float("nan")
        with self.assertRaisesRegex(ValueError, "invalid best_eer"):
            _validate_resume_state(state, contract=self.contract(), fingerprint="abc123")


if __name__ == "__main__":
    unittest.main()

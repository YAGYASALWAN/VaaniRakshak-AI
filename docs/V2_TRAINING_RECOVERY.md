# VaaniRakshak V2 Training Recovery

This document defines how to start, interrupt, and resume the first WavLM V2 training run without silently changing the experiment.

The recovery unit is the **last completed epoch**. VaaniRakshak does not attempt to resume in the middle of a partially completed epoch. If the process stops during epoch N, the next `--resume` starts epoch N again from the checkpoint written after epoch N-1.

## 1. Preflight before a large run

From the repository root in PowerShell:

```powershell
python scripts/v2_preflight.py `
  --data-root data/v2_sea_en `
  --budget-gb 26 `
  --output reports/v2_preflight.json
```

A normal training preflight requires CUDA and checks the current Hugging Face account against the pinned SEA-Spoof repository/revision without downloading dataset audio.

For local code/debug checks that must not contact Hugging Face:

```powershell
python scripts/v2_preflight.py `
  --data-root data/v2_sea_en `
  --budget-gb 1 `
  --allow-cpu `
  --offline
```

The offline/CPU form is not evidence that the planned GPU training environment is ready.

## 2. Start a new training run

Use a new output directory for a new experiment:

```powershell
python -m vaanirakshak.v2_train_ssl `
  --manifest data/v2_sea_en/manifest.jsonl `
  --output models/v2_wavlm_sea `
  --device cuda `
  --epochs 8 `
  --batch-size 2 `
  --gradient-accumulation 4
```

The trainer refuses to start a new run if that output directory already contains V2 training artifacts. This prevents accidentally overwriting an earlier experiment.

## 3. Recovery artifacts

A running experiment writes:

```text
models/v2_wavlm_sea/
  run_config.json
  history.json
  best.pt
  last.pt
```

`run_config.json` is the immutable experiment contract. It includes the manifest fingerprint, backbone, epoch count, batch configuration, learning rates, threshold policy, seed, and other training settings.

`best.pt` is the current deployable detector candidate selected by development-set EER. Its detector threshold is also selected from development data only.

`last.pt` is the resumable training state written atomically after each completed epoch. It contains:

- the full current model state;
- optimizer state;
- AMP scaler state;
- completed epoch number;
- complete epoch history;
- best development EER so far;
- Torch RNG state;
- CUDA RNG state when training on CUDA;
- weighted-sampler RNG state;
- manifest fingerprint and training contract;
- exported WavLM backbone configuration and VaaniRakshak model specification.

Because the exported model configuration is stored in `last.pt`, a resumed run reconstructs WavLM from the saved configuration rather than calling `from_pretrained()` again. The pretrained backbone must be available when a **new** run starts, but an already-created V2 resume state is designed to reconstruct the architecture offline.

## 4. Resume after interruption

Use the exact same manifest, output directory, and training arguments, plus `--resume`:

```powershell
python -m vaanirakshak.v2_train_ssl `
  --manifest data/v2_sea_en/manifest.jsonl `
  --output models/v2_wavlm_sea `
  --device cuda `
  --epochs 8 `
  --batch-size 2 `
  --gradient-accumulation 4 `
  --resume
```

If epochs 1-3 completed and the process stopped during epoch 4, the trainer restores the state saved after epoch 3 and reruns epoch 4.

## 5. Conditions that intentionally block resume

Resume is refused when any of these conditions is detected:

- `run_config.json`, `last.pt`, or `best.pt` is missing;
- the manifest fingerprint changed;
- the backbone or any protected training hyperparameter changed;
- the resume-state schema is not the expected V2 schema;
- required model/optimizer/scaler/RNG fields are missing;
- exported WavLM architecture metadata is missing or invalid;
- the stored completed epoch is outside the configured epoch range;
- training history does not exactly match the completed epoch;
- history epochs are non-contiguous;
- the stored best EER is non-finite or outside `[0, 1]`;
- `last.pt` cannot be decoded;
- model, optimizer, scaler, or RNG state cannot be restored.

These are hard failures by design. The trainer must not guess how to repair a scientifically different or corrupted experiment.

## 6. Do not change these arguments during resume

The training contract currently protects:

```text
architecture
backbone_id
epochs
batch_size
gradient_accumulation_steps
gradient_checkpointing
head_only_epochs
backbone_lr
head_lr
weight_decay
max_dev_fpr
seed
```

If a different value is desired, create a new output directory and treat it as a new experiment.

## 7. Safe interruption behavior

`last.pt` is written through a temporary file and then atomically renamed. A process failure while writing should therefore leave either the previous complete `last.pt` or the new complete one, rather than intentionally replacing it with a partially written file.

A failure during an epoch does not update `last.pt`. This is deliberate: partial optimizer progress is discarded and the entire incomplete epoch is rerun.

## 8. After training finishes

Do not tune the stored threshold using the test set. Run the frozen evaluator:

```powershell
python -m vaanirakshak.v2_evaluate `
  --manifest data/v2_sea_en/manifest.jsonl `
  --checkpoint models/v2_wavlm_sea/best.pt `
  --output reports/v2_sea_in_domain `
  --scenario standard `
  --device cuda
```

Then continue with the strict unseen-generator, external ASVspoof, and robustness evaluations defined in `docs/VAANIRAKSHAK_V2_RUNBOOK.md`.

## 9. What recovery does not guarantee

Resume protects experiment identity and completed-epoch state. It does not guarantee bit-for-bit equality across different GPUs, CUDA/cuDNN versions, PyTorch versions, or nondeterministic GPU kernels. The run configuration and preflight/runtime metadata should therefore be retained with experimental results.

It also does not make model-quality claims. Accuracy, EER, false-positive behavior, generalization, telephony robustness, and live latency remain empirical release gates that must be measured after training.

# VaaniRakshak V2 Experiment Log

## 2026-09-16 — First Real V2 Training Pipeline

### Objective

Build the first real VaaniRakshak V2 synthetic-speech detector using a self-supervised speech backbone and a leakage-aware dataset pipeline.

## Dataset Investigation

### SEA-Spoof

SEA-Spoof was initially selected as the primary V2 training dataset.

During metadata inspection of the pinned release, the English subset was found to contain only bona-fide speech:

- Train English: 115,992 bona-fide, 0 spoof
- Validation English: 7,593 bona-fide, 0 spoof
- Evaluation English: 7,191 bona-fide, 0 spoof

Therefore, SEA-Spoof could not be used alone for binary English synthetic-speech training.

This was detected before the planned large audio transfer, preventing unnecessary dataset download and training.

## Dataset Pivot

The first V2 training experiment was moved to MLAAD-tiny.

MLAAD-tiny provides:

- Genuine English speech
- Synthetic English speech
- 64 TTS generator families
- Metadata describing synthetic-generation provenance

The local Git LFS clone is approximately 3.3 GiB.

## MLAAD-tiny Preparation

A dedicated V2 preparation pipeline was implemented.

The pipeline:

1. Reads bona-fide English audio.
2. Reads spoof-generator metadata.
3. Extracts generator identity when available.
4. Preserves original source-utterance provenance.
5. Computes SHA-256 hashes for audio.
6. Keeps identity-connected records in the same split where provenance permits.
7. Creates deterministic train/dev/test splits.
8. Produces a VaaniRakshak V2 JSONL manifest.
9. Runs the V2 dataset-contract audit.

Prepared dataset:

- Train: 9,808 recordings
- Dev: 1,348 recordings
- Test: 1,314 recordings
- Total: 12,470 recordings
- Spoof recordings: 6,400

Generated manifest:

`C:\VaaniRakshakData\MLAAD-tiny-v2\manifest.jsonl`

Raw audio is referenced in place instead of being duplicated.

MLAAD-tiny is a sampled subset, so spoof metadata may reference original M-AILABS utterances whose genuine WAV is not present locally. Those references are still retained as source-utterance identities for leakage control.

## Model Architecture

The first V2 detector uses:

- Microsoft WavLM Base+
- Raw waveform input
- 16 kHz model sample rate
- 4-second training windows
- Learnable temporal attention pooling
- Binary synthetic-speech classification head

The WavLM backbone starts from pretrained speech representations rather than training a speech encoder from scratch.

## Training Configuration

Initial experiment:

- Epochs: 8
- Micro-batch size: 2
- Gradient accumulation: 4
- Effective batch size: 8
- Head-only training during epoch 1
- Backbone learning rate: 1e-5
- Classification-head learning rate: 1e-4
- Weight decay: 1e-4
- Gradient checkpointing enabled
- Maximum development FPR target: 5%
- Device: CUDA

Training creates:

- `last.pt` — crash-safe resumable checkpoint
- `best.pt` — checkpoint selected by development EER
- `history.json` — epoch-level training history
- `run_config.json` — reproducibility metadata

## Evaluation Policy

The test split is not used during model training or threshold selection.

Model selection uses development EER.

The decision threshold is selected on development data while enforcing the configured false-positive-rate budget.

Final evaluation will include:

- Accuracy
- Precision
- Recall
- F1
- ROC-AUC
- PR-AUC
- EER
- False-positive rate
- False-negative rate
- Confusion matrix
- Inference latency

Later experiments will additionally evaluate:

- Unseen TTS generators
- Cross-dataset generalization
- Telephone bandwidth
- G.711 codecs
- Noise
- Short speech segments

## Current Status

- Dataset preparation: COMPLETE
- Leakage-aware manifest: COMPLETE
- WavLM backbone download: IN PROGRESS / COMPLETE ON FIRST RUN
- First real V2 training run: STARTING

No V2 accuracy or generalization claims are being made until frozen-checkpoint evaluation is complete.

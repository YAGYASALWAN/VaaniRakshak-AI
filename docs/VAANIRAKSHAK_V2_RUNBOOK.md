# VaaniRakshak V2 Runbook

This document is the executable path from raw data to the live VaaniRakshak V2 product.

V2 is deliberately product-first:

`microphone -> streaming audio -> quality gate -> overlapping windows -> detector -> temporal evidence -> risk engine -> final report`

The detector is replaceable. A model does not become the V2 product detector merely because it trains successfully.

## 0. Branch and environment

Work on the V2 branch, not V1/main:

```powershell
git fetch origin
git checkout v2-product-skeleton
git pull origin v2-product-skeleton
```

Install the project and the relevant V2 dependency groups:

```powershell
python -m pip install -e .
python -m pip install -r requirements-v2.txt
python -m pip install -r requirements-v2-data.txt
python -m pip install -r requirements-v2-train.txt
```

Keep your existing CUDA-compatible PyTorch/torchaudio pair if it is already working. Do not blindly replace it with a CPU build.

## 1. Product-only smoke test

No trained model is required for this step. The server deliberately uses the mock detector when no checkpoint is configured.

```powershell
python -m vaanirakshak.v2_server
```

Open:

`http://127.0.0.1:8766`

Allow microphone access and speak for at least 10-20 seconds. This verifies the product path only. MOCK scores are not authenticity evidence.

## 2. Prepare the primary SEA-Spoof raw-audio corpus

SEA-Spoof is gated. Use it only if your account has been approved under the dataset's access terms.

The V2 preparation path reuses the pinned repository/revision and bounded range reader already present in V1, but does not create log-mel features. English audio is materialized as mono 16 kHz FLAC and described by a V2 JSONL manifest.

From the repository root:

```powershell
python -m vaanirakshak.v2_prepare_sea `
  --output data/v2_sea_en `
  --budget-gb 26
```

Why 26 GB rather than 30 GB: the existing SEA transfer layer has a hard cumulative 30 GB ceiling and the V2 planner deliberately leaves safety headroom for metadata/range overhead.

Expected important outputs:

```text
data/v2_sea_en/
  manifest.jsonl
  v2_sea_plan.json
  v2_sea_audit.json
  transfer_state.json
  audio/SEA-Spoof/...
  metadata/...
  receipts/...
```

The builder:

- keeps the upstream source pinned;
- preserves official train/validation/evaluation roles as train/dev/test;
- processes evaluation before training when filtering duplicates;
- removes exact/canonical duplicate audio;
- requires spoof generator provenance (`source_model`);
- keeps known speaker/source-utterance provenance;
- refuses the completed manifest if V2 leakage checks fail.

## 3. Audit the ordinary SEA manifest

```powershell
python scripts/v2_manifest.py audit `
  --manifest data/v2_sea_en/manifest.jsonl `
  --scenario standard
```

Do not train if `ok` is false.

The audit checks, among other things:

- duplicate record IDs;
- reused audio paths;
- identical audio hashes crossing splits;
- related source utterances crossing splits;
- known speaker identities crossing splits;
- missing spoof generator IDs.

## 4. Train the first V2 WavLM detector

The initial V2 model family is:

`WavLM Base+ -> attentive temporal pooling -> compact binary head`

Default training is intentionally memory-conscious: micro-batch 2, gradient accumulation 4 (effective batch 8), gradient checkpointing enabled, and one head-only warm-up epoch before backbone fine-tuning.

```powershell
python -m vaanirakshak.v2_train_ssl `
  --manifest data/v2_sea_en/manifest.jsonl `
  --output models/v2_wavlm_sea `
  --device cuda `
  --epochs 8 `
  --batch-size 2 `
  --gradient-accumulation 4
```

Training uses only `split=train`.

Checkpoint selection and the operating threshold use only `split=dev`.

The training command never constructs or reads the test dataset.

Important outputs:

```text
models/v2_wavlm_sea/
  best.pt
  run_config.json
  history.json
```

`best.pt` is already in the checkpoint schema understood by the live V2 server. No conversion step is required.

## 5. Frozen in-domain evaluation

After training, freeze the checkpoint. Do not change its threshold after inspecting test performance.

```powershell
python -m vaanirakshak.v2_evaluate `
  --manifest data/v2_sea_en/manifest.jsonl `
  --checkpoint models/v2_wavlm_sea/best.pt `
  --output reports/v2_sea_in_domain `
  --scenario standard `
  --device cuda
```

The evaluator uses the threshold stored in the checkpoint. It does not choose a new test threshold.

Outputs include:

```text
reports/v2_sea_in_domain/
  evaluation.json
  scores.jsonl
```

Report ROC-AUC, PR-AUC, EER, precision, recall, F1, false-positive rate and false-negative rate. Do not report accuracy alone.

## 6. Build a strict unseen-generator experiment

First audit/summarize the SEA manifest and inspect the generator IDs under `spoof_generators`.

Then choose one or more generator families to hold out. Example placeholder:

```powershell
python scripts/v2_manifest.py unseen-generator `
  --manifest data/v2_sea_en/manifest.jsonl `
  --output data/v2_scenarios/sea_unseen_generator.jsonl `
  --holdout-generator "SEA:GENERATOR_NAME" `
  --dev-fraction 0.10 `
  --seed 42
```

This builder does more than generator filtering. It treats known speaker/source-utterance relationships as leakage components. If a held-out utterance also has a synthetic version from a training generator, that conflicting version is dropped instead of being allowed into train/dev.

Audit the derived scenario:

```powershell
python scripts/v2_manifest.py audit `
  --manifest data/v2_scenarios/sea_unseen_generator.jsonl `
  --scenario unseen-generator
```

### Important

The unseen-generator scenario requires a **new checkpoint trained on this generator-held-out manifest**:

```powershell
python -m vaanirakshak.v2_train_ssl `
  --manifest data/v2_scenarios/sea_unseen_generator.jsonl `
  --output models/v2_wavlm_unseen_generator `
  --device cuda
```

Then evaluate that checkpoint on the same scenario's frozen test split:

```powershell
python -m vaanirakshak.v2_evaluate `
  --manifest data/v2_scenarios/sea_unseen_generator.jsonl `
  --checkpoint models/v2_wavlm_unseen_generator/best.pt `
  --output reports/v2_unseen_generator `
  --scenario unseen-generator `
  --device cuda
```

Do not train on all generators and then describe one of those same generators as unseen.

## 7. External cross-dataset smoke tests

Two evaluation-only datasets are pinned in `configs/v2_external_benchmarks.json`:

- `ASVspoof2021_LA`: cross-dataset + telephony/transmission stress;
- `ASVspoof5`: newer large-scale Track 1 stress test.

For engineering smoke tests, materialize a balanced subset. Example: 200 bonafide + 200 spoof:

```powershell
python -m vaanirakshak.v2_prepare_benchmark `
  --benchmark ASVspoof2021_LA `
  --output data/v2_eval/asv2021_la_smoke `
  --max-per-label 200
```

Evaluate:

```powershell
python -m vaanirakshak.v2_evaluate `
  --manifest data/v2_eval/asv2021_la_smoke/manifest.jsonl `
  --checkpoint models/v2_wavlm_sea/best.pt `
  --output reports/asv2021_la_smoke `
  --scenario cross-dataset `
  --evaluation-dataset ASVspoof2021_LA `
  --device cuda
```

A smoke subset is for pipeline/debugging only. It must not be presented as full benchmark performance.

## 8. Full ASVspoof 2021 LA evaluation

Omit `--max-per-label`:

```powershell
python -m vaanirakshak.v2_prepare_benchmark `
  --benchmark ASVspoof2021_LA `
  --output data/v2_eval/asv2021_la_full
```

Then run the frozen evaluator using the SEA checkpoint as above.

Do not tune the model or threshold after reading this result if the result is intended to remain a clean external generalization test.

## 9. ASVspoof 5 smoke and full evaluation

Smoke:

```powershell
python -m vaanirakshak.v2_prepare_benchmark `
  --benchmark ASVspoof5 `
  --output data/v2_eval/asvspoof5_smoke `
  --max-per-label 200
```

Full benchmark: omit `--max-per-label`.

ASVspoof 5 is large. Plan storage and evaluation compute before materializing the full test package.

## 10. Plug the trained V2 checkpoint into the live product

PowerShell:

```powershell
$env:VAANIRAKSHAK_V2_CHECKPOINT = "models/v2_wavlm_sea/best.pt"
$env:VAANIRAKSHAK_V2_DEVICE = "cuda"
python -m vaanirakshak.v2_server
```

Open:

`http://127.0.0.1:8766`

The status panel should now report a trained detector rather than MOCK mode.

If checkpoint loading fails, the V2 server deliberately enters a configuration-error state. It does **not** silently substitute the mock detector.

## 11. Legacy V1 checkpoint integration (debug only)

V1 checkpoints are refused by default.

If we explicitly need to verify old-model/product wiring:

```powershell
$env:VAANIRAKSHAK_V2_CHECKPOINT = "path/to/v1/best.pt"
$env:VAANIRAKSHAK_ALLOW_LEGACY = "1"
python -m vaanirakshak.v2_server
```

The UI/server labels this mode `legacy-experimental`. This is not a V2 model-quality claim.

## 12. Release gate before calling the detector V2-ready

A checkpoint should not be promoted to the SIH-facing detector until we have, at minimum:

1. leakage-clean in-domain SEA evaluation;
2. strict unseen-generator evaluation;
3. frozen ASVspoof 2021 LA cross-dataset evaluation;
4. frozen ASVspoof 5 cross-dataset evaluation;
5. telephony robustness results;
6. duration robustness results;
7. noise robustness results;
8. false-positive analysis;
9. live inference latency measurements;
10. a checkpoint whose threshold was selected on development data only.

## Current limitations

- The live product's speech gate is currently an energy-based gate, not a production neural VAD.
- The call-level risk formula is a transparent product heuristic, not a calibrated probability model.
- WavLM V2 code and checkpoint plumbing exist, but model quality is unknown until real training/evaluation runs are completed.
- Detector scores are explicitly marked uncalibrated unless a future checkpoint provides validated calibration.
- SEA-Spoof use is subject to its approved non-commercial academic research terms.
- GitHub Actions status should be checked separately; the existence of CI configuration is not itself evidence that a remote run passed.

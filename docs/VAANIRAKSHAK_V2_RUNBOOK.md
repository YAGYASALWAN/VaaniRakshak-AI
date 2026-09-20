# VaaniRakshak V2 Runbook

This document is the executable path from raw data to the live VaaniRakshak V2 product.

V2 is deliberately product-first:

`microphone -> streaming audio -> WebRTC speech gate -> quality gate -> overlapping windows -> detector -> temporal evidence -> risk engine -> final report`

The detector is replaceable. A model does not become the V2 product detector merely because it trains successfully.

Detailed companion documents:

- `docs/V2_TRAINING_RECOVERY.md` — interruption/resume contract for WavLM training;
- `docs/V2_CALIBRATION.md` — development-only temperature calibration and probability interpretation;
- `docs/V2_CALL_AUDIT_SCHEMA.md` — privacy-preserving live call audit format.

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

### Run preflight before an expensive data/training run

```powershell
python scripts/v2_preflight.py `
  --data-root data/v2_sea_en `
  --budget-gb 26 `
  --output reports/v2_preflight.json
```

The preflight downloads no dataset audio. It checks:

- the pinned SEA-Spoof repository/revision configuration;
- conservative disk headroom for the planned payload;
- CUDA availability and visible GPU information;
- Python, PyTorch, torchaudio and Transformers versions;
- the current Hugging Face login;
- access to the exact pinned gated SEA-Spoof revision.

For data-preparation/debugging on a machine without CUDA, add `--allow-cpu`. To check only local state without contacting Hugging Face, add `--offline`.

Do not begin the large preparation run if preflight fails.

## 1. Product-only smoke test

No trained model is required for this step. The server deliberately uses the mock detector when no checkpoint is configured, but the live speech gate is real WebRTC VAD by default.

```powershell
python -m vaanirakshak.v2_server
```

Open:

`http://127.0.0.1:8766`

Allow microphone access and speak for at least 10-20 seconds. The dashboard should report `webrtc-vad-m2` as the active speech gate. This verifies the product path only. MOCK detector scores are not authenticity evidence.

The WebRTC gate is independently configurable:

```powershell
$env:VAANIRAKSHAK_V2_VAD = "webrtc"
$env:VAANIRAKSHAK_V2_VAD_AGGRESSIVENESS = "2"
python -m vaanirakshak.v2_server
```

Aggressiveness may be 0, 1, 2, or 3. Higher modes are more aggressive about rejecting non-speech.

The old energy gate is retained only as an explicit fallback/debug path:

```powershell
$env:VAANIRAKSHAK_V2_VAD = "energy"
python -m vaanirakshak.v2_server
```

A bad VAD configuration is treated as a visible configuration error; the server does not silently fall back to another gate.

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
  transfer.json
  audio/SEA-Spoof/...
  metadata/...
  receipts/...
```

`transfer.json` is the persistent byte-reservation ledger used by the bounded SEA reader. Do not delete it to bypass the transfer ceiling during the same preparation run.

The builder:

- keeps the upstream source pinned;
- preserves official train/validation/evaluation roles as train/dev/test;
- processes evaluation before training when filtering duplicates;
- removes exact/canonical duplicate audio;
- requires spoof generator provenance (`source_model`);
- keeps known speaker/source-utterance provenance;
- refuses the completed manifest if V2 leakage checks fail.

Completed row groups have receipts and are reusable on rerun. An interruption inside a row group can still require that incomplete group to be read again; the byte ledger remains fail-closed and may therefore consume additional reserved transfer budget. Do not delete the ledger to hide repeated transfer.

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

The threshold policy maximizes spoof recall subject to the configured development-set false-positive budget. If that budget cannot be achieved by any deployable threshold in `(0,1)`, training fails explicitly rather than silently relaxing the policy or writing an unloadable checkpoint.

The training command never constructs or reads the test dataset.

Important outputs:

```text
models/v2_wavlm_sea/
  best.pt
  last.pt
  run_config.json
  history.json
```

`best.pt` is the current detector candidate. `last.pt` is the resumable completed-epoch training state. It contains model/optimizer/AMP/RNG/sampler state plus the exported WavLM architecture configuration, allowing reconstruction from saved configuration instead of calling `from_pretrained()` again during resume.

If a run is interrupted, use the same command and arguments plus `--resume`:

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

Resume occurs from the last completed epoch. A partially completed epoch is deliberately rerun. Changed manifests or protected training hyperparameters are treated as a different experiment and are refused. See `docs/V2_TRAINING_RECOVERY.md` for the full recovery contract.

## 5. Optional development-only probability calibration

The raw WavLM sigmoid output is a detector score, not automatically a calibrated probability. If VaaniRakshak will display or export a probability interpretation, fit temperature scaling on development data only and create a separate checkpoint:

```powershell
python -m vaanirakshak.v2_calibrate `
  --manifest data/v2_sea_en/manifest.jsonl `
  --checkpoint models/v2_wavlm_sea/best.pt `
  --output-checkpoint models/v2_wavlm_sea/best_calibrated.pt `
  --device cuda
```

The calibration command:

- reads only `split=dev` for fitting;
- verifies the manifest fingerprint against checkpoint training provenance;
- never overwrites the source checkpoint;
- fits one positive temperature by development-set NLL;
- transforms the existing threshold through the same monotonic mapping;
- verifies the operating-point decisions are preserved;
- records source-checkpoint SHA-256 and calibration provenance;
- records `test_data_used = false`.

Frozen recording evaluation aggregates overlapping-window evidence with `median_logit`, not arithmetic median probability. That makes the recording aggregation commute with temperature scaling and preserves the transformed operating point even for recordings with an even number of windows.

See `docs/V2_CALIBRATION.md` for the complete interpretation and validation rules.

## 6. Frozen in-domain evaluation

After training, freeze the checkpoint. Do not change its threshold after inspecting test performance.

Evaluate the raw detector first:

```powershell
python -m vaanirakshak.v2_evaluate `
  --manifest data/v2_sea_en/manifest.jsonl `
  --checkpoint models/v2_wavlm_sea/best.pt `
  --output reports/v2_sea_in_domain_raw `
  --scenario standard `
  --device cuda
```

If a calibrated checkpoint was created, evaluate it separately on the same untouched test split:

```powershell
python -m vaanirakshak.v2_evaluate `
  --manifest data/v2_sea_en/manifest.jsonl `
  --checkpoint models/v2_wavlm_sea/best_calibrated.pt `
  --output reports/v2_sea_in_domain_calibrated `
  --scenario standard `
  --device cuda
```

The evaluator uses the threshold stored in the selected checkpoint. It does not choose a new test threshold.

Outputs include:

```text
reports/.../
  evaluation.json
  scores.jsonl
```

Report ROC-AUC, PR-AUC, EER, precision, recall, F1, false-positive rate and false-negative rate. Do not report accuracy alone.

The evaluator also reports NLL, Brier score and ECE. For an uncalibrated checkpoint these are diagnostic properties of a detector score and must not be described as probability quality. For a development-calibrated checkpoint they are untouched-test evidence of whether probability calibration generalized.

## 7. Build a strict unseen-generator experiment

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

If probability calibration is required for this separately trained unseen-generator experiment, fit it on that scenario's own development split. Do not reuse a temperature merely because it was fitted for a different training experiment.

## 8. External cross-dataset smoke tests

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

A SEA-development-calibrated checkpoint may also be evaluated unchanged on external datasets to measure calibration transfer. Do not fit a new temperature on an external test benchmark and then call that benchmark a clean held-out test.

## 9. Full ASVspoof 2021 LA evaluation

Omit `--max-per-label`:

```powershell
python -m vaanirakshak.v2_prepare_benchmark `
  --benchmark ASVspoof2021_LA `
  --output data/v2_eval/asv2021_la_full
```

Then run the frozen evaluator using the SEA checkpoint as above.

Do not tune the model, threshold or calibration temperature after reading this result if the result is intended to remain a clean external generalization test.

## 10. ASVspoof 5 smoke and full evaluation

Smoke:

```powershell
python -m vaanirakshak.v2_prepare_benchmark `
  --benchmark ASVspoof5 `
  --output data/v2_eval/asvspoof5_smoke `
  --max-per-label 200
```

Full benchmark: omit `--max-per-label`.

ASVspoof 5 is large. Plan storage and evaluation compute before materializing the full test package.

## 11. Frozen duration / telephony / noise robustness suite

Run the robustness suite against a **frozen checkpoint**. The threshold saved in the checkpoint is reused unchanged for every condition.

Example on the ordinary SEA test split:

```powershell
python -m vaanirakshak.v2_robustness_eval `
  --manifest data/v2_sea_en/manifest.jsonl `
  --checkpoint models/v2_wavlm_sea/best.pt `
  --output reports/v2_sea_robustness `
  --scenario standard `
  --device cuda
```

The default suite evaluates:

```text
clean
duration_2s
duration_4s
duration_8s
narrowband_8khz
g711_mulaw
g711_alaw
noise_20db
noise_10db
noise_5db
```

Output:

```text
reports/v2_sea_robustness/
  robustness.json
```

Add `--save-scores` only when per-record stress scores are useful for error analysis. On a very large benchmark this can create a substantial file.

The report contains absolute metrics for each condition plus degradation relative to clean audio. In particular, inspect changes in:

- EER;
- F1;
- ROC-AUC / PR-AUC;
- false-positive rate;
- false-negative rate.

For a calibrated checkpoint, probability quality under stress should also be inspected rather than assuming clean-development calibration remains valid after codec/noise/channel shifts.

The noise tests use deterministic synthetic white noise. They are a controlled engineering stress test, not a substitute for a later real-background-noise corpus.

The same evaluator can be run on an external benchmark. Example:

```powershell
python -m vaanirakshak.v2_robustness_eval `
  --manifest data/v2_eval/asv2021_la_smoke/manifest.jsonl `
  --checkpoint models/v2_wavlm_sea/best.pt `
  --output reports/asv2021_la_robustness_smoke `
  --scenario cross-dataset `
  --evaluation-dataset ASVspoof2021_LA `
  --device cuda
```

Smoke results remain non-reportable as full-benchmark performance.

## 12. Plug the trained V2 checkpoint into the live product

PowerShell using the raw checkpoint:

```powershell
$env:VAANIRAKSHAK_V2_CHECKPOINT = "models/v2_wavlm_sea/best.pt"
$env:VAANIRAKSHAK_V2_DEVICE = "cuda"
$env:VAANIRAKSHAK_V2_VAD = "webrtc"
$env:VAANIRAKSHAK_V2_VAD_AGGRESSIVENESS = "2"
python -m vaanirakshak.v2_server
```

Or, after dev-only calibration has been created and validated:

```powershell
$env:VAANIRAKSHAK_V2_CHECKPOINT = "models/v2_wavlm_sea/best_calibrated.pt"
$env:VAANIRAKSHAK_V2_DEVICE = "cuda"
$env:VAANIRAKSHAK_V2_VAD = "webrtc"
python -m vaanirakshak.v2_server
```

Open:

`http://127.0.0.1:8766`

The status panel should now report a trained detector and `webrtc-vad-m2` rather than MOCK/energy mode. It also states whether detector scores are calibrated probabilities.

A trained detector exposes a SHA-256 fingerprint of the exact loaded checkpoint. The browser audit export preserves that fingerprint, detector/calibration metadata, per-window evidence and latency, while excluding raw microphone audio.

Because V2 advances by a 2-second hop, the live dashboard also compares observed mean per-window processing time against a 2000 ms hop budget. This is an observed scheduling signal, not a worst-case concurrency guarantee.

If checkpoint or speech-gate loading fails, the V2 server deliberately enters a configuration-error state. It does **not** silently substitute the mock detector or another speech gate.

## 13. Legacy V1 checkpoint integration (debug only)

V1 checkpoints are refused by default.

If we explicitly need to verify old-model/product wiring:

```powershell
$env:VAANIRAKSHAK_V2_CHECKPOINT = "path/to/v1/best.pt"
$env:VAANIRAKSHAK_ALLOW_LEGACY = "1"
python -m vaanirakshak.v2_server
```

The UI/server labels this mode `legacy-experimental`. This is not a V2 model-quality claim.

## 14. Release gate before calling the detector V2-ready

A checkpoint should not be promoted to the SIH-facing detector until we have, at minimum:

1. leakage-clean in-domain SEA evaluation;
2. strict unseen-generator evaluation;
3. frozen ASVspoof 2021 LA cross-dataset evaluation;
4. frozen ASVspoof 5 cross-dataset evaluation;
5. telephony robustness results;
6. duration robustness results;
7. noise robustness results;
8. false-positive analysis;
9. live inference latency measurements showing whether observed processing stays within the 2-second hop budget;
10. a checkpoint whose threshold was selected on development data only;
11. reproducible training provenance and a tested recovery state;
12. if the product claims probability semantics: dev-only calibration provenance plus untouched-test NLL, Brier score and ECE, including cross-domain/calibration-shift inspection.

## Current limitations

- WebRTC VAD is the default live speech gate, but VAD only identifies likely speech/non-speech; it is not evidence that speech is human or synthetic.
- The energy-v1 gate remains available only for fallback/debugging.
- The call-level risk formula is a transparent product heuristic, not a calibrated probability of fraud or fakery.
- WavLM V2 code, recovery, calibration and checkpoint plumbing exist, but model quality is unknown until real training/evaluation runs are completed.
- Raw detector scores are explicitly marked uncalibrated. A checkpoint is marked calibrated only when it carries validated temperature-scaling metadata fitted on development data.
- Development-set calibration may degrade under external datasets, telephony codecs, noise or other distribution shift; this must be measured.
- Controlled white-noise robustness is not equivalent to real environmental-noise validation.
- SEA-Spoof use is subject to its approved non-commercial academic research terms.
- Completed SEA row groups are resumable, but an interrupted row group can consume additional transfer budget when repeated.
- A green CI run validates code/tests on the CI environment; it does not substitute for gated-dataset preparation or GPU model training.

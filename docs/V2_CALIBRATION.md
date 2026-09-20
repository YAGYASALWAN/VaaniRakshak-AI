# VaaniRakshak V2 Probability Calibration

VaaniRakshak separates **detector discrimination**, **probability calibration**, and **call-level security risk**.

A newly trained WavLM checkpoint produces a detector score in `[0, 1]`, but that score is explicitly **not** treated as a calibrated probability. Temperature scaling is an optional post-training step fitted on the development split only.

The test split is never used to fit the temperature or choose the operating threshold.

## 1. Train the uncalibrated detector

```powershell
python -m vaanirakshak.v2_train_ssl `
  --manifest data/v2_sea_en/manifest.jsonl `
  --output models/v2_wavlm_sea `
  --device cuda `
  --epochs 8 `
  --batch-size 2 `
  --gradient-accumulation 4
```

The resulting `best.pt` has:

```text
calibrated_probability = false
```

Its threshold was already selected on development data under the configured false-positive-rate budget.

## 2. Fit temperature on development data only

Create a **new** checkpoint rather than overwriting the original:

```powershell
python -m vaanirakshak.v2_calibrate `
  --manifest data/v2_sea_en/manifest.jsonl `
  --checkpoint models/v2_wavlm_sea/best.pt `
  --output-checkpoint models/v2_wavlm_sea/best_calibrated.pt `
  --device cuda
```

The calibrator:

- audits the manifest;
- requires both bonafide and spoof development examples;
- verifies the manifest fingerprint matches the checkpoint's training provenance;
- refuses an already-calibrated source checkpoint;
- scores only `split=dev`;
- fits a single positive temperature by minimizing development binary negative log-likelihood;
- transforms the existing detector threshold through the same temperature function;
- verifies that threshold decisions are preserved;
- writes a new checkpoint atomically;
- records the source-checkpoint SHA-256, temperature, original/transformed thresholds, fit split and calibration metrics;
- records `test_data_used = false`.

A sidecar report is also written:

```text
best_calibrated.pt.calibration.json
```

## 3. Why the threshold is transformed

Temperature scaling acts on detector logits:

```text
p = sigmoid(z)
calibrated_p = sigmoid(z / T)
```

If the original threshold is `t`, VaaniRakshak transforms it as:

```text
calibrated_t = sigmoid(logit(t) / T)
```

Because the transform is monotonic, every individual detector decision is preserved:

```text
p >= t
```

has the same truth value as:

```text
calibrated_p >= calibrated_t
```

Calibration therefore changes the interpretation of the score without silently moving the detector's security operating point.

## 4. Recording-level aggregation also stays invariant

Frozen evaluation aggregates multiple overlapping windows using **median logit**:

```text
window probability
      ↓ logit
window logit
      ↓ median
recording logit
      ↓ sigmoid
recording score
```

This is deliberate. Temperature scaling divides every window logit by the same positive scalar, so median-logit aggregation commutes with calibration, including recordings with an even number of windows.

An arithmetic median of probabilities would not have this guarantee for an even number of windows.

## 5. Live risk evidence also stays invariant

The live product displays detector scores and also derives a separate threshold-relative evidence signal for temporal risk aggregation.

The risk engine must not measure raw distance in calibrated-probability space, because temperature scaling changes that distance even when it preserves the detector decision.

V2 therefore reconstructs the detector's pre-calibration logit margin:

```text
visible_margin = logit(score) - logit(threshold)
raw_margin = visible_margin * evidence_temperature
evidence_signal = sigmoid(raw_margin)
```

For an uncalibrated checkpoint:

```text
evidence_temperature = 1
```

For a temperature-calibrated checkpoint:

```text
evidence_temperature = T
```

Since both the visible score and threshold logits were divided by `T`, multiplying their difference by `T` reconstructs the original margin.

The consequence is intentional: a raw checkpoint and its correctly calibrated counterpart should preserve all of the following for the same call audio:

- threshold-crossing decisions;
- suspicious-window count;
- per-window evidence signals;
- median/mean evidence statistics;
- consistency statistic;
- call risk score;
- risk label;
- final heuristic verdict.

The audit records this transform as:

```text
evidence_semantics = temperature-normalized-logit-margin
```

Calibration is allowed to change probability semantics. It is not allowed to silently change the current product security policy.

## 6. Evaluate the raw checkpoint

```powershell
python -m vaanirakshak.v2_evaluate `
  --manifest data/v2_sea_en/manifest.jsonl `
  --checkpoint models/v2_wavlm_sea/best.pt `
  --output reports/v2_sea_raw `
  --scenario standard `
  --device cuda
```

The report labels the score semantics as:

```text
uncalibrated_detector_score
```

NLL, Brier score and ECE are still reported as diagnostics, but the score must not be described as a probability.

## 7. Evaluate the calibrated checkpoint on untouched test data

```powershell
python -m vaanirakshak.v2_evaluate `
  --manifest data/v2_sea_en/manifest.jsonl `
  --checkpoint models/v2_wavlm_sea/best_calibrated.pt `
  --output reports/v2_sea_calibrated `
  --scenario standard `
  --device cuda
```

The report now labels the score semantics as:

```text
calibrated_probability
```

and reports test-set:

```text
NLL
Brier score
ECE
```

These test metrics evaluate whether development calibration generalized. They are **not** used to refit the temperature.

## 8. Expected raw-vs-calibrated invariants

For the same audio and the decision-preserving transformed threshold:

- window-level spoof decisions should be unchanged;
- recording-level decisions under median-logit aggregation should be unchanged;
- confusion counts, accuracy, precision, recall, F1, FPR and FNR should therefore be unchanged apart from numerical edge cases at an exact threshold tie;
- ROC-AUC and PR-AUC should be unchanged because temperature scaling is monotonic;
- EER value should be unchanged while its numerical score threshold may transform;
- live risk evidence and heuristic verdict should remain unchanged because risk uses the reconstructed pre-calibration logit margin;
- NLL, Brier score and ECE may change and are the main quantities calibration is intended to improve.

If classification metrics or live heuristic risk move materially after calibration, treat that as an implementation defect and investigate before promotion.

## 9. Cross-dataset and robustness calibration are separate release questions

A temperature fitted on SEA-Spoof development data may not remain well calibrated on a different corpus/channel/generator distribution.

Evaluate the same calibrated checkpoint without refitting on external benchmarks:

```powershell
python -m vaanirakshak.v2_evaluate `
  --manifest data/v2_eval/asv2021_la_full/manifest.jsonl `
  --checkpoint models/v2_wavlm_sea/best_calibrated.pt `
  --output reports/asv2021_la_calibrated `
  --scenario cross-dataset `
  --evaluation-dataset ASVspoof2021_LA `
  --device cuda
```

Repeat for ASVspoof 5 and run the robustness suite on the calibrated checkpoint. The robustness report compares NLL, Brier and ECE under duration, telephony and noise stress against the clean baseline without refitting calibration.

Do not fit a new temperature on an external **test** benchmark or stressed test condition and then continue calling that result a clean held-out evaluation.

## 10. Use the calibrated checkpoint in the live product

```powershell
$env:VAANIRAKSHAK_V2_CHECKPOINT = "models/v2_wavlm_sea/best_calibrated.pt"
$env:VAANIRAKSHAK_V2_DEVICE = "cuda"
$env:VAANIRAKSHAK_V2_VAD = "webrtc"
python -m vaanirakshak.v2_server
```

The V2 status/dashboard will expose:

```text
calibrated_probability = true
```

and the detector metadata includes the calibration record and calibrated-checkpoint SHA-256. The call audit separately records threshold-relative evidence semantics so probability display and security-risk evidence remain distinguishable.

## 11. Important interpretation boundary

A calibrated detector probability means approximately:

> under the calibration distribution and calibration procedure, the detector score has been adjusted to better match empirical synthetic-speech frequency.

It does **not** mean:

> probability that the caller is committing fraud.

VaaniRakshak's call-level risk score remains a separate temporal/security heuristic. Fraud risk depends on more than acoustic synthetic-speech evidence.

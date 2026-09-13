# VaaniRakshak-AI Version 2 — Evaluation Contract

**Status:** Draft v0.1 — not frozen

This document defines the experimental rules that Version 2 must follow before model training begins.

The purpose of this contract is to prevent a repeat of Version 1, where strong in-domain performance did not provide enough evidence of generator-independent and corpus-independent synthetic-speech detection.

Version 2 will therefore be evaluated as a **generalization problem**, not simply a binary-classification benchmark.

---

## 1. Core research question

Version 2 must answer:

> Can a detector learn synthetic-speech evidence that remains useful when speaker, generator, corpus and communication channel differ from the training distribution?

A model is not accepted merely because it performs well on a random held-out split.

---

## 2. Non-negotiable rules

1. **No test-set tuning.** Final holdouts cannot influence architecture, preprocessing, augmentation, feature selection, threshold selection or hyperparameters.
2. **No random-row fallback.** If speaker/source/generator grouping makes a requested split infeasible, the experiment must fail visibly rather than silently weaken the split.
3. **Identical preprocessing for both classes.** No class-specific denoising, trimming, normalization or resampling.
4. **Every reported metric must have provenance.** Dataset revision, split definition, checkpoint, threshold and prediction artifact must be traceable.
5. **External benchmarks are evaluation assets, not training assets.**
6. **Aggregate accuracy alone is insufficient.** Per-class, per-generator, per-attack and per-corpus behavior must be visible.
7. **Unseen-generator testing is mandatory before any strong generalization claim.**
8. **Cross-corpus testing is mandatory before any deployment-oriented claim.**
9. **A sigmoid score is not a calibrated probability unless calibration is explicitly established.**
10. **A call-level security verdict must remain separate from the raw model score.**

---

## 3. Evidence hierarchy

V2 will be judged through progressively harder evaluation gates:

```text
Gate 1: in-domain holdout
        ↓
Gate 2: speaker-disjoint holdout
        ↓
Gate 3: generator-disjoint holdout
        ↓
Gate 4: attack-family breakdown
        ↓
Gate 5: cross-corpus holdout
        ↓
Gate 6: locked external benchmark
        ↓
Gate 7: codec / channel stress tests
        ↓
Gate 8: realistic phone / VoIP evaluation
        ↓
Gate 9: multilingual evaluation
        ↓
Gate 10: modern unseen-generator evaluation
```

Passing an easier gate does not imply passing a harder gate.

---

## 4. Canonical provenance schema

Every sample should retain the following fields where the source permits them:

```text
sample_id
dataset
source_revision
label
speaker_id
source_speaker_id
target_speaker_id
generator
attack_family
language
accent_or_region
codec
container
sample_rate
channels
duration
source_recording_id
source_file
original_split
recording_condition
preprocessing_version
processed_hash
```

Unknown metadata must remain unknown rather than being guessed.

---

## 5. Dataset roles

The following are current candidates, not automatic approvals for V2 training.

| Dataset / workflow | Candidate role | Main risk to audit |
| --- | --- | --- |
| IndicVoices-R | Genuine Indian speech | corpus/processing identity, language/speaker balance |
| IndicSynth | Synthetic / converted Indian speech | source-corpus confounding, generator structure |
| MLAAD | Synthetic English speech | generator and corpus fingerprints |
| M-AILABS | Genuine English originals | genuine-corpus identity, source pairing |
| ASVspoof 2019 | Historical / external benchmark | benchmark-specific overfitting |
| ASVspoof 2021 DF | Locked external deepfake benchmark candidate | must remain external to tuning |
| SEA-Spoof workflow | Additional spoofing evaluation candidate | attack/source provenance must be verified |

Before any dataset is approved for V2 training, its schema, license/access, speaker structure, generator structure and class/corpus relationship must be documented.

---

## 6. Shortcut-learning audit

Before training an authenticity detector, V2 must test whether nuisance variables are easy to infer from the proposed features.

Diagnostic targets include:

- dataset identity,
- codec/container,
- language,
- speaker/source group,
- generator,
- recording condition,
- clip duration bucket.

The purpose is not to remove all such information. The purpose is to identify whether the authenticity label is nearly equivalent to one of these variables.

A particularly important test is:

```text
Can a small classifier predict dataset identity from the same representation?
```

If dataset identity is trivially separable and dataset is strongly correlated with class, the experiment must be treated as high shortcut risk.

---

## 7. Split rules

### 7.1 Speaker separation

Where reliable speaker identity exists, a speaker must not appear in both training and evaluation partitions.

### 7.2 Source-recording grouping

All windows and synthetic variants derived from the same source recording must remain in one partition unless a specific controlled paired experiment requires otherwise.

### 7.3 Generator separation

At least one primary V2 evaluation must hold out entire generators that are absent from training and development.

Example:

```text
TRAIN / DEV
Generator A
Generator B
Generator C
Generator D

LOCKED TEST
Generator X
Generator Y
Generator Z
```

### 7.4 Attack-family sampling

Evaluation samples must be stratified or otherwise audited so that dataset ordering cannot accidentally make one attack family dominate the test.

### 7.5 External holdouts

Locked external corpora must not influence model selection or threshold selection.

---

## 8. Baseline strategy

The first V2 model will deliberately remain simple.

**Baseline-0:** log-mel representation + compact CNN, with a capacity similar enough to V1 that the effect of improved experimental design can be studied.

The purpose of Baseline-0 is not to produce the final detector. It is to answer:

> Does the redesigned dataset and split protocol improve out-of-domain behavior before model complexity is increased?

If Baseline-0 again reaches near-perfect development metrics but fails generator-disjoint or cross-corpus tests, the response is to inspect the data/protocol again, not immediately scale the network.

---

## 9. Architecture escalation rule

Only after Baseline-0 survives the major data/evaluation gates should V2 compare stronger representations or anti-spoofing architectures.

Candidate directions may include:

- AASIST-style anti-spoofing,
- raw-waveform models,
- pretrained speech encoders,
- multi-view spectral representations.

A larger model is accepted only when it improves **generalization**, not merely in-domain accuracy.

---

## 10. Required metrics

Each important evaluation should report, where mathematically valid:

- sample count,
- class prevalence,
- confusion matrix,
- accuracy,
- balanced accuracy,
- precision / recall / F1 per class,
- specificity,
- false-positive rate on genuine speech,
- false-negative rate on spoof speech,
- ROC-AUC,
- PR-AUC / average precision,
- MCC,
- EER where appropriate,
- calibration diagnostics,
- invalid / rejected sample count.

Breakdowns must additionally be reported by:

- corpus,
- generator,
- attack family,
- language where applicable,
- major channel condition where applicable.

---

## 11. Threshold and calibration policy

The final test set may not be used to search for an operating threshold.

The intended sequence is:

```text
train model
    ↓
development scores
    ↓
select threshold / calibration on development data only
    ↓
freeze decision rule
    ↓
run final holdouts once
```

The UI must never present an uncalibrated sigmoid score as a literal probability of fraud or synthetic speech.

---

## 12. Channel robustness plan

After generalization is demonstrated on clean evaluation audio, V2 will test controlled communication distortions.

Candidate transformations include:

- resampling,
- telephone bandwidth limitation,
- codec compression,
- moderate background noise,
- clipping,
- reverberation,
- microphone response variation,
- packet-loss-like degradation where simulation is defensible.

Whenever possible, the same underlying utterance should be evaluated before and after transformation so score stability can be measured directly.

---

## 13. Call-level decision architecture

The detector output is evidence, not the final cybersecurity decision.

Target architecture:

```text
call audio
  -> short overlapping windows
  -> anti-spoofing inference
  -> audio-quality / validity checks
  -> temporal score aggregation
  -> uncertainty handling
  -> call-level risk state
  -> likely genuine / inconclusive / likely synthetic
  -> warning or step-up verification
```

This separation is mandatory:

```text
raw model score
!= calibrated probability
!= call-level risk
!= final security verdict
```

---

## 14. Engineering reliability requirements inherited from V1

V2 must include:

- record-level decode fault isolation,
- explicit NaN/Inf rejection,
- safe float32 inference fallback,
- framework-level CUDA verification,
- reproducible dependency definitions,
- immutable source revisions where practical,
- saved per-recording predictions,
- checkpoint provenance,
- evaluation reports that can be recomputed without rerunning inference where saved predictions exist.

---

## 15. Provisional acceptance philosophy

V2 is not considered successful because it exceeds a single accuracy target.

A lower but stable result on unseen generators and corpora is more valuable than near-perfect performance on a familiar development distribution.

The central acceptance condition is:

> **The detector must retain useful discrimination when speaker, generator, corpus and channel conditions move away from the training distribution.**

Exact numerical pass/fail thresholds are intentionally **not frozen in Draft v0.1**. They should be set only after dataset roles, class prevalence, operating risk and available evaluation sizes are finalized, and they must be fixed before final test results are inspected.

---

## 16. Milestone 0 exit criteria

Milestone 0 is complete only when the following are committed and reviewed:

- approved dataset roles,
- provenance schema,
- speaker/source/generator grouping rules,
- train/dev/test split rules,
- locked external holdouts,
- shortcut-leakage diagnostics,
- required metrics,
- threshold/calibration policy,
- artifact/provenance requirements,
- acceptance-gate definitions.

**Until those items are frozen, Version 2 training remains blocked.**

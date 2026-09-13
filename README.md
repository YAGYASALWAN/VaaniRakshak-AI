# VaaniRakshak-AI

AI-powered cybersecurity prototype for detecting voice-cloning and synthetic-speech impersonation attacks.

VaaniRakshak is being developed for SIH 2026 as a system that can analyse speech during or after a call, estimate whether the voice is likely bona fide or synthetic, and turn model evidence into a security decision rather than relying only on caller identity.

---

## Project status

| Version / milestone | Status | Meaning |
| --- | --- | --- |
| **Version 1 anti-spoofing baseline** | **Concluded / frozen** | Working CNN baseline and evaluation pipeline, but not accepted as deployment-ready because generalization was not proven strongly enough. |
| Dataset inspection and bias audit | Active engineering direction | Audit corpus/class confounding, speaker balance, generator balance, codec, duration, language and other shortcut risks before training again. |
| Source verification / acquisition planning | In progress | Verify real dataset schemas, provenance and acquisition constraints before selecting large audio subsets. |
| **Version 2 — Milestone 0** | **Next** | Freeze the V2 data/evaluation contract before training another detector. |
| **Version 2 detector training** | **Blocked until Milestone 0 passes** | No V2 model training until provenance, split rules, holdouts, metrics and acceptance gates are defined. |

> **Version 1 is not presented as production-ready.** Its strongest contribution is the engineering and experimental evidence that shaped the V2 design.

For the long-form analysis, see **[Version 1 Final Postmortem](docs/version1_postmortem.md)**.

---

# Version 1 — what happened and why it ended

Version 1 was our first end-to-end neural anti-spoofing baseline.

It proved that we could build the full technical loop:

```text
speech
  -> decoding / preprocessing
  -> fixed-duration audio segment
  -> log-mel style time-frequency features
  -> compact CNN classifier
  -> genuine / synthetic score
  -> thresholded decision
  -> evaluation report
  -> local demo
```

The model was **not** frozen because it was unable to learn. It was frozen because the evidence increasingly suggested that it could learn the development distributions extremely well without proving that it had learned a portable, generator-independent and corpus-independent notion of speech authenticity.

That distinction is the central lesson of V1.

---

## Evidence provenance: do not mix different V1 experiments

V1 went through multiple experiments. Their metrics must not be presented as if they came from one identical train/test setup.

### Repository-backed benchmark record

The repository contains a preserved ASVspoof 2019 LA evaluation under:

`records/performance/asv2019_20260909T183107466708Z/metrics.json`

Checkpoint:

`models/english_20260907T044807696536Z/best.pt`

At threshold `0.5`:

| Metric | Preserved result |
| --- | ---: |
| Test recordings | 71,237 |
| Accuracy | **87.77%** |
| Synthetic precision | **99.62%** |
| Synthetic recall | **86.70%** |
| Synthetic F1 | **92.71%** |
| Genuine specificity | **97.09%** |
| Balanced accuracy | **91.89%** |
| ROC-AUC | **0.9789** |
| Average precision | **0.9974** |

Confusion matrix:

```text
                predicted genuine   predicted synthetic
true genuine          7141                 214
true synthetic        8497               55385
```

These numbers show that V1 could function as a meaningful anti-spoofing classifier on that recorded benchmark distribution.

They do **not** establish reliable performance on modern unseen generators, live telephone channels, VoIP distortion, multilingual speech, new microphones, new corpora or adversarially selected attacks.

### Historical debugging observations from the V1 failure investigation

During the separate V1 debugging/generalization investigation, we also observed a much more dramatic pattern: development performance appeared to be around **99.78% accuracy with approximately 0.99995 ROC-AUC**, while a corrected cross-domain / attack-stratified evaluation later fell to roughly **48.5% accuracy**. In one investigated configuration, the genuine class was misclassified at approximately **99.33%**.

Attack-family analysis also exposed cases such as **A05/A06** with very poor separability, including observations below `0.5` ROC-AUC, which indicated that some unseen attack families could invert the ranking learned by the model instead of merely reducing confidence.

These figures are retained here as **historical debugging observations from the project investigation**. Unlike the ASVspoof 2019 record above, the repository does not yet contain the complete raw machine-readable artifact set for every one of these debugging figures. They must therefore not be mixed with the preserved benchmark record or presented as independently reproducible headline claims until their original prediction artifacts are committed.

This distinction matters because the failure story is not "87.77% became 48.5% on the same experiment." It is that **different experiments exposed very different levels of generalization, which forced us to stop treating in-domain success as proof of a deployable detector.**

---

# Chronology of the Version 1 failure

## Stage 1 — the first CNN pipeline worked

We built a compact CNN using fixed-duration speech windows and log-mel style spectral features. Training ran, checkpoints were produced, inference worked, and the local demo could return a genuine/synthetic score.

At this point the main question looked like a normal model-improvement problem: improve accuracy, gather more data and continue training.

That interpretation later proved incomplete.

## Stage 2 — internal metrics looked extremely strong

The development experiment produced near-perfect-looking internal metrics.

This was initially encouraging because it suggested that the CNN had learned a strong discriminative representation.

The mistake was treating a strong internal score as evidence that the network had learned **speech authenticity itself**.

A validation set can be held out and still share the same dataset fingerprints, speaker structure, recording chain, generator families, silence patterns and preprocessing artifacts as the training set.

So the model can generalize to the validation split while failing to generalize to the actual world.

## Stage 3 — the data design was inspected more critically

A central V1 workflow used **MLAAD synthetic speech** while genuine speech came from the corresponding **M-AILABS originals**.

That pairing is useful experimentally, but it creates a major confounding risk:

```text
label = synthetic  -> MLAAD-derived audio
label = genuine    -> M-AILABS-derived audio
```

The intended task is:

```text
human speech vs synthetic speech
```

but the easier task available to the network can become:

```text
MLAAD distribution vs M-AILABS distribution
```

The model may therefore exploit differences in:

- recording chain,
- microphone characteristics,
- codec or container history,
- sample-rate / bandwidth residue,
- loudness distributions,
- silence and padding,
- room acoustics,
- preprocessing history,
- speaker population,
- utterance duration,
- language/accent distribution,
- generator-specific spectral fingerprints.

This is **dataset shortcut learning**.

## Stage 4 — the first external test was itself flawed

When we began external attack testing, the first sampling approach did not provide a representative view of all spoofing attacks.

Because of dataset ordering, the initial sample was heavily dominated by **attack A01**.

That meant the apparent question was effectively:

```text
Can the model recognize this attack family?
```

rather than:

```text
Can the model generalize across unseen spoofing families?
```

This was an evaluation-design failure, not merely a model failure.

## Stage 5 — sampling was corrected

The evaluation was changed so that multiple attack families were represented rather than allowing one attack type to dominate.

Once the attack distribution was stratified, performance deteriorated dramatically in the debugging experiment.

This was the turning point: the model's earlier success could no longer be interpreted as broad deepfake detection ability.

## Stage 6 — genuine-call behavior became unacceptable

A security detector is unusable if it labels ordinary genuine speech as synthetic too often.

In one critical external configuration, the genuine class was misclassified at approximately **99.33%**.

For VaaniRakshak, that failure is more serious than a simple reduction in aggregate accuracy because a system that constantly warns users about legitimate callers would rapidly lose trust.

The product requirement therefore changed from maximizing headline accuracy to controlling both:

```text
false negatives: spoof accepted as genuine
false positives: genuine caller flagged as spoof
```

## Stage 7 — we tested the threshold hypothesis

A natural explanation was that the model might still rank genuine and spoof audio correctly but use the wrong decision threshold.

We therefore considered whether moving the binary threshold away from `0.5` could recover performance.

That would help if the problem were mainly calibration.

But attack-family analysis showed that the deeper issue was representation/generalization. A threshold can move a decision boundary; it cannot repair a feature space in which unseen genuine and spoof samples are not correctly ordered.

## Stage 8 — per-attack analysis exposed representation failure

Looking only at aggregate accuracy was not enough.

Per-attack inspection exposed attack families with extremely poor separability. Historical analysis included **A05/A06** cases with observed ROC-AUC below `0.5`.

An AUC near `0.5` means the score is approximately non-discriminative. An AUC below `0.5` can mean the ordering is effectively reversed for that distribution.

That evidence strongly suggested that V1 had learned attack- or corpus-specific signatures rather than a stable universal signal of synthetic speech.

## Stage 9 — generator independence was judged unproven

A deployable detector cannot assume that tomorrow's attacker uses one of the synthesis systems present in the training set.

If the model primarily learns fingerprints of known TTS / voice-conversion systems, the detector can fail as soon as the generator changes.

V1 therefore failed the key question:

> Can the model detect synthetic speech produced by generators it has never seen before?

The answer was not proven strongly enough.

## Stage 10 — corpus independence was also judged unproven

The same problem exists on the genuine side.

M-AILABS genuine speech is not equivalent to all human speech.

Real VaaniRakshak inputs may contain:

- mobile microphones,
- telephone band-limiting,
- VoIP compression,
- packet loss,
- echo cancellation,
- reverberation,
- background noise,
- different accents,
- multilingual speech,
- unknown recording hardware.

So even a detector that performs well on one genuine corpus may fail on real callers.

## Stage 11 — we realized early stopping did not protect us from this problem

V1 used normal model-selection controls such as checkpoint selection and early stopping.

But early stopping only helps with certain forms of conventional optimization overfitting.

It does **not** prevent shortcut learning when both training and validation contain the same shortcut.

A model can stop at the statistically "best" epoch and still be solving the wrong problem.

## Stage 12 — the log-mel + compact CNN design was reinterpreted

The compact 2-D CNN was initially attractive because it was fast and able to learn spectral structure from log-mel representations.

The failure investigation showed the limitation of that success: the representation is also very capable of exposing dataset-specific spectral signatures, codec residue, silence behavior and generator artifacts.

Therefore the fact that the CNN learned easily was not evidence that it had learned the right signal.

A larger CNN would not automatically solve this. More capacity could simply make shortcut learning easier.

The conclusion became:

> **Fix the experiment before increasing the model.**

## Stage 13 — numerical instability appeared during external inference

External evaluation also exposed an engineering weakness around reduced-precision inference.

Mixed / float16 inference can be useful for speed, but under some inputs we encountered non-finite behavior during debugging.

The engineering response was to require:

- explicit `NaN` / `Inf` checks,
- reliable float32 fallback,
- numerical validation before accepting a prediction,
- no silent propagation of invalid scores.

This was not the primary scientific reason V1 was frozen, but it became a mandatory reliability requirement.

## Stage 14 — malformed audio showed that evaluation needed record-level fault isolation

A malformed or problematic WAV must not terminate an entire benchmark or live-analysis job.

The robust behavior is:

```text
bad recording
    -> detect decode / validation failure
    -> record the failure
    -> skip or quarantine that item
    -> continue the run
    -> include failure counts in the report
```

This became part of the engineering requirements for later versions.

## Stage 15 — dependency and environment problems exposed reproducibility gaps

During V1 development, missing or inconsistent packages such as `numpy`, `huggingface_hub`, `requests`, `filelock` and experiment-specific dependencies repeatedly showed that "it worked once" was not sufficient.

A reproducible ML system needs a defined environment, versioned requirements and verification commands.

## Stage 16 — GPU presence was not the same as GPU-enabled PyTorch

Another practical failure occurred when the machine had a working NVIDIA GPU/driver but the installed PyTorch runtime did not initially provide the expected CUDA execution path.

The lesson was:

```text
GPU exists
    !=
framework is CUDA-enabled
    !=
model is actually running on CUDA
```

Later workflows therefore verify CUDA from PyTorch itself instead of assuming hardware acceleration from the machine configuration.

## Stage 17 — V1 was frozen instead of endlessly patched

At this point the central problem was no longer "tune the threshold" or "add more epochs."

The core problem was experimental validity.

Continuing to optimize V1 on the same development distributions risked producing a higher score on the wrong objective.

The correct engineering decision was therefore to freeze V1 as a historical baseline and redesign the data/evaluation system before training V2.

---

# Root causes, one by one

### 1. Dataset/class confounding

Different corpora could reveal the class through corpus fingerprints rather than authenticity.

### 2. Shortcut learning

The CNN had enough capacity to exploit easy spectral and recording-domain cues.

### 3. Validation-domain similarity

A held-out validation set can still reproduce the same shortcuts as training.

### 4. Non-representative initial external sampling

The first external sample was attack-family biased, especially toward A01.

### 5. Cross-domain collapse after corrected sampling

Stratified testing showed that strong development performance did not transfer reliably.

### 6. Catastrophic false-positive behavior in a critical configuration

Genuine speech could be flagged as synthetic at an unacceptable rate.

### 7. Threshold changes could not repair representation failure

Calibration and representation were separate problems.

### 8. Attack-family instability

A05/A06 analysis showed that some spoof families were not merely harder; the learned ranking could become unreliable.

### 9. Generator independence was unproven

Known-generator fingerprints are not the same thing as universal synthetic-speech artifacts.

### 10. Corpus independence was unproven

One genuine corpus is not representative of real phone/VoIP speech.

### 11. Log-mel features exposed shortcuts as well as useful artifacts

The feature representation was informative, but not automatically causal or generator-independent.

### 12. Early stopping did not protect against shortcut learning

Both train and validation can reward the same wrong feature.

### 13. Model capacity was not the primary bottleneck

A larger network could overfit the shortcut more effectively.

### 14. Calibration was not established as real-world probability

A sigmoid score is not automatically a trustworthy probability of fraud or synthetic speech.

### 15. Numerical robustness needed improvement

Non-finite values and reduced-precision behavior require explicit handling.

### 16. Audio fault tolerance needed improvement

One corrupt file must never bring down an evaluation or call-analysis pipeline.

### 17. Environment reproducibility needed improvement

Dependencies, CUDA and runtime configuration must be versioned and testable.

### 18. Evaluation provenance needed improvement

Predictions, class distributions, attack labels, thresholds and source revisions must be preserved so every reported number can be traced to an artifact.

---

# The actual V1 failure chain

```text
MLAAD synthetic / M-AILABS genuine source structure
                |
                v
class becomes correlated with dataset identity
                |
                v
CNN can exploit easy corpus / generator fingerprints
                |
                v
internal validation rewards the same shortcuts
                |
                v
near-perfect-looking development metrics create false confidence
                |
                v
external evaluation begins
                |
                v
first sample discovered to be attack-family biased
                |
                v
sampling corrected across attack families
                |
                v
cross-domain performance collapses
                |
                v
genuine false-positive behavior investigated
                |
                v
threshold hypothesis tested
                |
                v
per-attack ROC/AUC shows deeper representation failure
                |
                v
generator and corpus independence remain unproven
                |
                v
engineering robustness issues catalogued
                |
                v
V1 frozen
                |
                v
V2 redesigned around data provenance + harder evaluation gates
```

**Final V1 verdict:** a useful baseline and valuable engineering artifact, but **not a production-ready voice-cloning detector**.

---

# What Version 1 still accomplished

V1 still produced substantial reusable engineering work:

- speech ingestion,
- waveform validation and preprocessing,
- fixed-window segmentation,
- log-mel feature extraction,
- CNN training,
- checkpoints and early stopping,
- GPU/runtime configuration,
- inference pipelines,
- thresholded classification,
- confusion matrices,
- precision / recall / F1,
- ROC-AUC and PR analysis,
- calibration diagnostics,
- saved prediction records,
- reproducible performance reporting,
- local frontend integration,
- external distribution-shift analysis,
- failure analysis by attack family.

Most importantly, V1 changed the central engineering question from:

> **How high is our model accuracy?**

into:

> **What evidence proves that the model is detecting synthetic speech rather than the dataset or generator that produced it?**

---

# What Version 2 changes

V2 begins with the dataset and evaluation protocol instead of the neural network.

The current engineering principles are:

1. **Identical preprocessing** for bona fide and spoof audio.
2. **Immutable raw data and source provenance.**
3. **Speaker-disjoint train/dev/test partitions.**
4. **Generator-disjoint evaluation.**
5. **Corpus/source identity recorded for every example.**
6. **No class-specific DSP.**
7. **External benchmarks excluded from initial training and threshold tuning.**
8. **Audit sample rate, codec, channels, duration, language, gender, speakers, generators and recording conditions before training.**
9. **Preserve per-recording predictions, attack labels, thresholds and source revisions.**
10. **Evaluate per class and per attack family, not accuracy alone.**
11. **Cross-corpus testing before claiming generalization.**
12. **Realistic channel / phone / VoIP stress testing.**
13. **Explicit finite-value and decode-failure handling.**
14. **No deployment claim until unseen-generator and realistic-channel tests pass.**

The current dataset tooling exists specifically because V1 showed that **data provenance is part of the model architecture** in an anti-spoofing system.

---

# Version 2 execution roadmap

Version 2 is deliberately **evaluation-first and data-first**. We will not begin by choosing a larger neural network. We will first define what evidence a model must survive before we trust it.

## Milestone 0 — freeze the V2 evaluation contract

**Status: NEXT. No V2 training before this gate passes.**

Before another gradient update, we will define and commit:

- candidate training corpora and their exact roles,
- bona fide and spoof provenance requirements,
- speaker identity rules,
- generator identity rules,
- source/corpus identity rules,
- immutable external holdouts,
- allowed development datasets,
- forbidden test-set uses,
- split construction rules,
- attack-family stratification rules,
- primary and secondary metrics,
- minimum reporting requirements,
- model acceptance / rejection gates.

The intended evidence hierarchy is:

```text
in-domain holdout
        -> speaker-disjoint holdout
        -> generator-disjoint holdout
        -> attack-family breakdown
        -> cross-corpus holdout
        -> external benchmark
        -> codec / channel stress tests
        -> realistic phone / VoIP tests
        -> multilingual tests
        -> modern unseen generators
```

The test sets must not be used to choose architecture, preprocessing, augmentation, threshold or hyperparameters.

### Milestone 0 deliverable

A written V2 evaluation contract that answers, before training:

> **What exact result would convince us that V2 has learned synthetic-speech evidence rather than another shortcut?**

If that question cannot be answered from the protocol, training does not start.

## Milestone 1 — provenance inventory and shortcut audit

We will construct a canonical sample inventory containing, where available:

```text
sample_id
dataset
label
speaker_id
generator
attack_family
language
codec
container
sample_rate
channels
duration
source_recording
source_revision
recording_condition
```

Before training, we will audit whether the target label can be predicted from nuisance variables such as dataset, codec, language, duration or recording condition.

A particularly important diagnostic will be a **dataset-identity leakage baseline**. If a tiny classifier can distinguish the genuine corpus from the synthetic corpus almost perfectly, that is direct evidence that the authenticity detector has an easy shortcut available.

We will therefore ask not only:

```text
Can features predict genuine vs synthetic?
```

but also:

```text
Can the same features predict dataset?
Can they predict codec?
Can they predict generator?
Can they predict language?
Can they predict speaker/source group?
```

These are diagnostic experiments, not product models.

## Milestone 2 — construct hard train/dev/test boundaries

The V2 split must prevent related examples from leaking across evaluation boundaries.

Required controls:

- speaker-disjoint partitions where identity is known,
- generator-disjoint evaluation,
- grouping of synthetic variants derived from the same source recording,
- duplicate / near-duplicate checks where practical,
- attack-family-aware sampling,
- no random-row fallback when grouping constraints fail,
- locked external corpora that are never used for tuning.

The most important V2 experiment will intentionally hold out entire synthesis systems:

```text
TRAIN
Generator A
Generator B
Generator C
Generator D
...

TEST ONLY
Generator X
Generator Y
Generator Z
```

A detector that survives this test provides much stronger evidence than one evaluated only on generators it saw during training.

## Milestone 3 — V2 Baseline-0: deliberately keep the first model simple

The first V2 detector should remain intentionally modest: a reproducible log-mel + compact CNN baseline close enough to V1 that we can isolate the effect of **better experimental design**.

This baseline answers:

> Does fixing provenance, splitting and evaluation improve generalization even before we introduce a more sophisticated architecture?

That gives us a meaningful comparison:

```text
V1-style CNN + weak experimental boundaries
                vs
similar-capacity CNN + rigorous V2 boundaries
```

If the V2 baseline still shows near-perfect development performance but collapses on generator-disjoint or cross-corpus tests, we stop and inspect the data again. We do **not** hide the failure by immediately scaling the model.

## Milestone 4 — representation / architecture upgrade

Only after Baseline-0 survives the data and evaluation gates will we evaluate stronger anti-spoofing architectures.

Candidates may include an **AASIST-style baseline**, raw-waveform approaches or pretrained speech representations, but architecture selection comes **after** the protocol is trustworthy.

Every candidate must be compared on the same locked splits and external holdouts.

A bigger architecture is accepted only if it improves generalization, not merely in-domain accuracy.

## Milestone 5 — channel and communication stress testing

VaaniRakshak will eventually receive call audio, not pristine benchmark WAVs.

V2 therefore needs controlled stress tests involving conditions such as:

- resampling,
- telephone bandwidth limitation,
- common audio codecs,
- VoIP-style compression,
- moderate background noise,
- clipping,
- reverberation,
- microphone variation,
- packet-loss-like degradation where simulation is defensible.

We will compare the same underlying speech before and after transformation so that we can measure prediction stability rather than simply mixing another uncontrolled corpus into the test.

## Milestone 6 — calibration and call-level decision layer

The anti-spoofing network will produce **evidence**, not the final cybersecurity verdict.

The intended architecture is:

```text
call
  -> short overlapping windows
  -> anti-spoofing scores
  -> quality / validity checks
  -> temporal aggregation
  -> uncertainty handling
  -> call-level risk state
  -> likely genuine / inconclusive / likely synthetic
  -> warning or step-up verification
```

We will explicitly separate:

```text
raw model score
!= calibrated probability
!= call-level risk
!= final security decision
```

Thresholds and calibration must be selected on development data only and then frozen before final testing.

## Milestone 7 — real-time engineering and product integration

Only after the detector survives generalization and channel tests do we optimize product behavior:

- streaming / window scheduling,
- inference latency,
- CPU/GPU execution path,
- bounded memory use,
- fault-tolerant audio ingestion,
- live score aggregation,
- API / backend integration,
- frontend call-state display,
- audit logging,
- final post-call report.

This ordering is intentional. We do not optimize a detector for real-time execution until we have evidence that it detects the right phenomenon.

---

# V2 acceptance philosophy

Version 2 will **not** be declared successful because it reaches a headline accuracy target.

A result such as `99%` on a familiar distribution is less valuable than a lower but stable result on genuinely unseen generators and corpora.

The V2 report must include, at minimum:

- confusion matrix,
- per-class precision / recall / F1,
- specificity,
- balanced accuracy,
- ROC-AUC,
- PR-AUC / average precision,
- false-positive rate on genuine speech,
- false-negative rate on spoof speech,
- EER where appropriate,
- calibration diagnostics,
- results by corpus,
- results by generator,
- results by attack family,
- results by language where applicable,
- channel-stress results,
- failed / invalid audio counts,
- exact dataset and checkpoint provenance.

The central success criterion is:

> **The model must retain useful discrimination when speaker, generator, corpus and channel conditions move away from the training distribution.**

---

# Immediate next action

We now proceed with **V2 Milestone 0 — Dataset & Evaluation Contract**.

The next work item is not model training. It is to define and commit:

```text
V2 datasets and roles
        -> provenance schema
        -> split constraints
        -> locked external holdouts
        -> leakage diagnostics
        -> required metrics
        -> acceptance gates
        -> only then training
```

Once Milestone 0 is frozen, we can build V2 Baseline-0 with confidence that the experiment itself is testing the problem we actually care about.

---

# Problem statement

Recent generative speech systems can clone or synthesize convincing voices from very little source audio. Attackers can use those voices to impersonate executives, officials, colleagues or family members during high-pressure calls.

Traditional caller ID and voice familiarity do not reliably solve this problem.

VaaniRakshak is intended to evolve toward a pipeline that:

1. accepts live or uploaded speech,
2. analyses short audio windows,
3. estimates bona fide vs synthetic evidence,
4. aggregates evidence over the call,
5. produces an interpretable risk verdict,
6. requests stronger verification when risk is high.

The long-term target includes multilingual and Indian-language conditions rather than treating them as an afterthought.

---

# Current architecture direction

```text
Call / uploaded audio
        |
        v
Audio validation + canonical preprocessing
        |
        v
Anti-spoofing detector
        |
        v
Window-level evidence
        |
        v
Temporal risk aggregation
        |
        v
Cybersecurity decision layer
        |
        v
Likely genuine / inconclusive / likely synthetic
        |
        v
Warning or step-up authentication
```

The final call-level verdict is intentionally separated from a single neural-network score.

---

# Dataset strategy

The V2 data pipeline is designed around explicit provenance and shortcut auditing.

| Dataset | Intended role |
| --- | --- |
| IndicVoices-R | Genuine Indian speech candidate |
| IndicSynth | Synthetic / converted Indian speech candidate |
| MLAAD English workflow | Synthetic-speech experimentation and feature-cache workflow |
| M-AILABS | Genuine originals corresponding to the MLAAD workflow |
| ASVspoof 2019 / 2021 | External anti-spoofing benchmarks; never equivalent to proof of real-world performance |
| SEA-Spoof workflow | Additional spoofing / frontend experimentation |

A dataset appearing in the repository does **not** automatically mean it is approved for V2 training. Source identity, schema, licensing/access, speaker structure, generator structure and class confounding must be verified first.

---

# Preprocessing principles

```text
load audio
  -> validate decode / duration / silence / finite values
  -> convert to mono
  -> resample to configured sample rate
  -> segment into fixed windows
  -> retain source + speaker + generator provenance
  -> write processed artifact / manifest record
```

Important rule:

> **Never apply a transformation only to one class unless the experiment explicitly tests that transformation.**

Otherwise the preprocessing step itself can become the label.

---

# Evaluation philosophy

A future detector will not be accepted because of one large accuracy value.

The intended sequence of evidence is:

```text
in-domain holdout
    -> speaker-disjoint holdout
    -> generator-disjoint holdout
    -> per-attack evaluation
    -> cross-corpus evaluation
    -> codec / channel stress tests
    -> realistic phone / VoIP conditions
    -> multilingual evaluation
    -> modern unseen generators
```

Only after those gates should VaaniRakshak make strong deployment claims.

---

# Repository documentation

- **[Version 1 Final Postmortem](docs/version1_postmortem.md)** — formal V1 postmortem.
- **[Performance Report](docs/performance_report.md)** — preserved evaluation reporting and provenance rules.
- **[Fast Baseline](docs/fast_baseline.md)** — baseline experiment notes and shortcut limitations.
- **[English Training](docs/english_training.md)** — English experiment workflow.
- **[MLAAD Streaming](docs/mlaad_streaming.md)** — MLAAD + M-AILABS streaming / feature-cache workflow.
- **[SEA-Spoof Training](docs/sea30_training.md)** — SEA-Spoof experiment workflow.
- **[SEA-Spoof Frontend](docs/seaspoof_frontend.md)** — frontend integration notes.
- **[Dataset Sources](docs/dataset_sources.md)** — dataset source and provenance observations.
- **[Milestone 2 Handoff](docs/milestone2_handoff.md)** — dataset inspection / bias-audit handoff.
- **[Milestone 3](docs/milestone3.md)** — source-verification roadmap.

Historical experiments are intentionally retained as part of the engineering record. They are not all current production paths and must not be compared without checking dataset, checkpoint, split, threshold and provenance.

---

# Setup

Python 3.11 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Experiment-specific requirement files are also present in the repository.

---

# Run the test suite

```powershell
python -m pytest -q
```

Milestone-specific verification commands are documented in the corresponding docs.

---

# Research and engineering rule

The most important rule produced by Version 1 is now part of the project philosophy:

> **A model that recognizes the dataset is not a model that recognizes the attack.**

VaaniRakshak-AI therefore treats data provenance, distribution shift, generator independence, evaluation design and engineering reliability as first-class parts of the security system.
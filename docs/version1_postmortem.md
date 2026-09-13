# VaaniRakshak-AI Version 1 — Final Postmortem

**Status:** Frozen / concluded

Version 1 was the first end-to-end anti-spoofing baseline built for VaaniRakshak-AI. It proved that the repository could ingest speech, transform audio into model-ready representations, train a neural detector, score held-out recordings, expose predictions through a local demo, and generate reproducible performance artifacts.

It was **not accepted as the final VaaniRakshak detector**.

The reason is not that V1 could not fit its training data or produce good benchmark metrics. The opposite happened: V1 produced strong-looking metrics, but the experimental design did not give enough evidence that the model had learned a general notion of speech authenticity rather than corpus- and generator-specific shortcuts.

That distinction ended Version 1.

---

## 1. What Version 1 attempted

The V1 detector used a compact neural anti-spoofing pipeline built around audio preprocessing and spectral features. The broad experimental flow was:

```text
speech recording
    -> load / decode
    -> mono + fixed sample-rate preprocessing
    -> fixed-duration segment
    -> log-mel style time-frequency representation
    -> compact CNN classifier
    -> synthetic / genuine score
    -> thresholded decision
```

The goal was deliberately practical: obtain a working binary anti-spoofing baseline first, then determine whether it generalized strongly enough to justify continued development.

---

## 2. What looked successful

A preserved ASVspoof 2019 LA evaluation record exists under:

`records/performance/asv2019_20260909T183107466708Z/metrics.json`

For that recorded experiment, using checkpoint:

`models/english_20260907T044807696536Z/best.pt`

at a threshold of `0.5`, the saved metrics were:

| Metric | Recorded result |
| --- | ---: |
| Recordings | 71,237 |
| Accuracy | 87.77% |
| Precision, synthetic class | 99.62% |
| Recall, synthetic class | 86.70% |
| F1, synthetic class | 92.71% |
| Specificity, genuine class | 97.09% |
| Balanced accuracy | 91.89% |
| ROC-AUC | 0.9789 |
| Average precision | 0.9974 |

The corresponding confusion matrix was:

```text
                predicted genuine   predicted synthetic
true genuine          7141                 214
true synthetic        8497               55385
```

These numbers demonstrate that the implementation could learn and separate the recorded benchmark distribution reasonably well.

They do **not** demonstrate that the detector would remain reliable on a real phone call, an unseen text-to-speech system, a new voice-cloning model, a different language, a different microphone, a different codec, or a future attack family.

The saved evaluation record itself carries this limitation.

---

## 3. Why Version 1 was frozen

### 3.1 Dataset identity was too strongly correlated with class identity

The most important failure was experimental, not architectural.

During V1, bona fide and synthetic speech came from different source collections in major parts of the workflow. When one corpus mostly represents `genuine` and another corpus mostly represents `synthetic`, the classifier is offered an easier problem than the one we intended to solve.

Instead of learning only:

```text
human speech vs synthetic speech
```

it can partially learn:

```text
corpus A vs corpus B
```

through differences in microphones, codecs, silence, loudness, bandwidth, preprocessing, speaker population, recording environment, file construction, duration, or other dataset fingerprints.

This is **dataset shortcut learning**.

A high validation score does not remove that risk when the validation split inherits the same corpus structure.

### 3.2 The original train/validation logic could reward the same shortcut

Random or insufficiently constrained splitting can place highly related examples into both training and validation distributions.

Even without exact duplicate files, the model may repeatedly see:

- the same corpus pipeline,
- the same speakers,
- the same generators,
- the same recording conditions,
- the same preprocessing characteristics.

The resulting validation set can therefore confirm that the model learned the development distribution while saying little about external generalization.

V1 taught us that future experiments must explicitly separate **speakers**, **generators**, and ideally **corpora** across evaluation boundaries.

### 3.3 Generator independence was not proven

VaaniRakshak must encounter synthesis systems that were not present during training.

A detector that learns fingerprints of particular TTS or voice-conversion systems can perform strongly until the attack generator changes. That is unacceptable for an adversarial security system because new generators appear continuously.

V1 did not provide sufficient generator-disjoint evidence to claim that it had learned generator-independent synthetic-speech artifacts.

### 3.4 Corpus independence was not proven

The same concern exists for genuine speech.

A real-world caller may use:

- a phone microphone,
- VoIP,
- narrow-band telephony,
- aggressive compression,
- background noise,
- echo cancellation,
- packet-loss concealment,
- a language or accent absent from training.

Good performance on one genuine-speech corpus does not prove robustness to those conditions.

### 3.5 Strong ASVspoof performance was still not enough

The preserved ASVspoof 2019 result is useful evidence that V1 was not a completely broken classifier. However, it is still one benchmark family.

The experiment record explicitly states that the result does not establish performance on modern Qwen3 voices, telephone calls, or other languages.

That limitation matters more than a single headline accuracy number.

VaaniRakshak is intended to operate in a changing, adversarial environment. The acceptance criterion therefore became **cross-domain generalization**, not simply benchmark accuracy.

### 3.6 Calibration was not established as real-world probability

The network output can be thresholded, but that does not make the score a trustworthy probability that a call is synthetic.

The saved evaluation record notes that synthetic scores were not established as calibrated probabilities and that no test-set threshold or temperature fitting was performed.

A cybersecurity UI must distinguish between:

- model score,
- calibrated probability,
- decision threshold,
- final security verdict.

V1 did not yet establish that chain robustly enough for production claims.

### 3.7 Class imbalance made headline metrics easy to over-read

The saved ASVspoof evaluation contains 63,882 synthetic recordings and 7,355 genuine recordings, so the synthetic class dominates the test population.

This is why V1 records more than raw accuracy. Balanced accuracy, per-class recall, confusion matrices, MCC, calibration error and class prevalence are necessary to interpret the detector correctly.

A security detector cannot be judged from one scalar metric.

### 3.8 A larger neural network would not solve the core problem

The V1 CNN already had enough capacity to learn the development distribution.

The primary failure was therefore not "the network is too small." Increasing model size before fixing the data design could make shortcut learning even easier.

The V1 conclusion was:

> Fix the experiment before increasing the model.

### 3.9 Preprocessing and inference robustness also needed hardening

The project exposed several engineering requirements that are easy to miss in a notebook-only experiment:

- malformed audio must not terminate an evaluation run,
- NaN/Inf values must be detected explicitly,
- preprocessing must be identical for both classes,
- inference precision must remain numerically stable,
- dependencies must be reproducible,
- CUDA availability must be verified at the framework level rather than assumed from the presence of a GPU,
- predictions and evaluation artifacts must be saved so metrics can be recomputed without rerunning the model.

These are not the primary reason V1 was rejected, but they became mandatory requirements for later versions.

---

## 4. The actual failure chain

Version 1 can be summarized as:

```text
class and dataset source became strongly correlated
                |
                v
model could exploit dataset-specific shortcuts
                |
                v
ordinary validation could reward those shortcuts
                |
                v
strong benchmark metrics looked more general than they were
                |
                v
generator/corpus independence remained unproven
                |
                v
real-world calibration and robustness remained unproven
                |
                v
V1 failed the VaaniRakshak verification standard
```

The important point is that **V1 did not fail because neural networks cannot detect synthetic speech. V1 failed because our evidence was not strong enough to claim that this particular neural network had learned a portable authenticity signal.**

---

## 5. Why we did not simply keep tuning V1

Once dataset shortcut learning became the dominant concern, continuing to tune thresholds, add epochs, enlarge the CNN, or optimize benchmark accuracy would have risked improving the wrong objective.

The correct response was to freeze the baseline and redesign the data and evaluation protocol.

V1 remains in the project as:

- a historical baseline,
- a reproducible engineering artifact,
- a source of evaluation tooling,
- evidence for why dataset auditing matters,
- a reference point for future models.

It is **not** presented as a production-ready voice-cloning detector.

---

## 6. What Version 2 must do differently

The V2 data/evaluation design is being built around the lessons from V1.

### Required controls

1. **Identical preprocessing for both classes.**
2. **Speaker-disjoint train/dev/test partitions.**
3. **Generator-disjoint evaluation.**
4. **Dataset provenance stored for every example.**
5. **Explicit audit of codec, duration, sample rate, channels, language, gender, speaker and generator distributions.**
6. **External benchmarks excluded from training and threshold tuning.**
7. **Cross-corpus evaluation before claiming generalization.**
8. **Per-class and balanced metrics, not accuracy alone.**
9. **Saved per-recording predictions and reproducible reports.**
10. **No production claim until unseen-generator and realistic-channel tests pass.**

The present dataset-inspection and source-probe milestones exist specifically because V1 showed that **data provenance is part of the model architecture** in an anti-spoofing system.

---

## 7. What Version 1 accomplished despite being frozen

V1 still moved VaaniRakshak forward substantially.

It produced practical experience with:

- speech ingestion,
- waveform preprocessing,
- spectral feature extraction,
- CNN training,
- checkpoints and early stopping,
- GPU/runtime configuration,
- inference pipelines,
- thresholded classification,
- confusion matrices,
- precision / recall / F1,
- ROC-AUC and PR metrics,
- calibration diagnostics,
- saved predictions,
- reproducible evaluation reports,
- local frontend integration,
- failure analysis under distribution shift.

Most importantly, it changed the central engineering question from:

> "How high is our model accuracy?"

into:

> "What evidence proves that the model is detecting synthetic speech rather than the dataset that produced it?"

That question defines the next version of VaaniRakshak-AI.

---

## Final verdict

**Version 1: concluded and frozen.**

It remains a useful baseline, but it does not satisfy the project's generalization standard for deployment.

The project now moves forward by redesigning dataset acquisition, provenance, sampling, split constraints and external evaluation **before** training the next detector.

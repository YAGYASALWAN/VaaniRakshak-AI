# VaaniRakshak V2 — SIH Product Architecture

## Product definition

VaaniRakshak V2 is an Android-first voice-authenticity security product, not only a binary classifier.

The intended user experience is:

```text
User installs VaaniRakshak
        ↓
Android detects an incoming/outgoing call event
        ↓
Available call audio is streamed to the VaaniRakshak AI service
        ↓
Speech gate → normalization → overlapping 4 s windows → anti-spoof detector
        ↓
Temporal evidence is accumulated during the call
        ↓
Live state: collecting / likely genuine / suspicious / likely synthetic
        ↓
Call ends
        ↓
Final call-security record is stored by the application backend
        ↓
User can inspect the call, evidence timeline and model result
        ↓
User can mark the prediction correct / incorrect / unsure
        ↓
Only consented, quality-reviewed samples can enter a future training-data pipeline
```

The product therefore has three major runtime components:

1. **Android client** — automatic call-event detection, user-facing warnings, call history and feedback.
2. **Application backend** — owned by the main product/backend repository; manages users, sessions, persistence, reporting, permissions and feedback workflow.
3. **VaaniRakshak AI service** — this repository; owns speech preprocessing, anti-spoof inference, calibration, temporal evidence and model provenance.

---

## SIH architecture

```text
┌───────────────────────────┐
│      Android client       │
│                           │
│ Call detection            │
│ Live warning              │
│ Call history              │
│ User feedback             │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│    Application backend    │
│  (team backend repository)│
│                           │
│ Authentication            │
│ Call/session lifecycle    │
│ WebSocket/API gateway     │
│ Database                  │
│ Reports                   │
│ Feedback records          │
└─────────────┬─────────────┘
              │ audio/session protocol
              ▼
┌───────────────────────────┐
│ VaaniRakshak AI service   │
│   (this repository)       │
│                           │
│ WebRTC VAD                │
│ 16 kHz normalization      │
│ 4 s / 2 s windowing       │
│ WavLM anti-spoof model    │
│ Calibration               │
│ Temporal evidence         │
│ Risk/verdict metadata     │
│ Model/checkpoint identity │
└─────────────┬─────────────┘
              │
              ▼
      Result / evidence
```

---

## Important Android boundary

Automatic call-event detection is a realistic Android capability through the platform call-screening/dialer APIs.

However, a normal third-party Android application must **not** assume it can transparently read unrestricted two-way raw audio from every ordinary cellular call. Therefore the SIH prototype and the production architecture must distinguish call-event integration from media access.

### SIH demonstration path

For SIH, the demo should prove the complete product workflow while using an audio path we actually control. Acceptable demonstration paths include:

- in-app / VoIP demonstration call where media frames are available;
- enterprise/telephony media stream integration;
- controlled recorded/live audio source routed through the same inference service;
- Android call event triggering a demo media-analysis session.

The judge-facing explanation should state clearly that production cellular deployment can use deeper dialer/system, carrier/OEM or supported telephony-media integration rather than claiming unrestricted Android call recording.

---

## Runtime AI pipeline

```text
Audio stream
    ↓
PCM buffering
    ↓
Speech gate (WebRTC VAD)
    ↓
Quality gate
    ↓
Mono 16 kHz normalization
    ↓
4 second windows, 2 second hop
    ↓
WavLM anti-spoof detector
    ↓
Per-window score + threshold + quality + latency
    ↓
Temporal evidence aggregator
    ↓
Call-level risk/verdict
```

The product never treats one isolated model score as proof that an entire call is genuine or fake.

Current product verdict vocabulary:

```text
Collecting evidence
Insufficient evidence
Likely genuine
Inconclusive
Suspicious
Likely synthetic
```

---

## Call record shown to the user

After the call, the user should be able to inspect:

- caller/session identifier;
- date and duration;
- usable speech duration;
- final verdict;
- risk score;
- detector score semantics;
- suspicious-segment count;
- evidence timeline with timestamps;
- audio-quality / VAD information;
- model/checkpoint version and SHA-256;
- inference latency / realtime status;
- system limitations;
- user feedback controls.

The canonical structured artifact remains the V2 JSON audit. A printable/PDF presentation can be derived from the same data.

Raw microphone audio is not included in the report by default.

---

## Feedback and model-improvement loop

User feedback is valuable but must never directly retrain the deployed model.

```text
Prediction
   ↓
User feedback
   ↓
Consent gate
   ↓
Privacy / retention policy
   ↓
Label verification / quality review
   ↓
Curated real-world corpus
   ↓
Offline model training
   ↓
Frozen evaluation
   ↓
Versioned model promotion
```

Reasons for the review layer:

- user labels can be mistaken;
- attackers could deliberately submit false labels;
- class imbalance can distort learning;
- audio collection requires explicit consent and retention rules;
- training data needs provenance and leakage controls;
- a new checkpoint must pass the full evaluation gate before deployment.

The feedback system is therefore a **data acquisition and review pipeline**, not online self-training.

---

## Repository ownership boundary

### This AI repository owns

- SEA-Spoof preparation and audits;
- WavLM model/training;
- calibration;
- checkpoint schema and fingerprinting;
- speech gating and audio preprocessing;
- per-window anti-spoof inference;
- temporal evidence / risk calculation;
- in-domain, unseen-generator, cross-dataset and robustness evaluation;
- AI-service reference server;
- AI audit metadata.

### Main/backend repository owns

- user accounts / authentication;
- application database;
- Android/Web clients;
- call/session records;
- WebSocket/API gateway;
- history/dashboard UX;
- feedback persistence;
- consent and data-retention workflow;
- production deployment orchestration.

The existing `v2_server.py` remains useful as the reference inference server and integration-test harness even when the production backend lives elsewhere.

---

## SIH MVP vs future production

### SIH MVP must demonstrate

1. automatic or explicit call/session start;
2. streaming/near-live audio analysis;
3. speech/quality gating;
4. real V2 trained detector, not mock output;
5. live evidence updates;
6. final verdict;
7. call history/report screen;
8. evidence timeline;
9. user feedback control;
10. model/version/checkpoint traceability;
11. measurable latency;
12. honest deployment-boundary explanation.

### Production evolution

Future production can add:

- carrier/OEM/enterprise telephony integration;
- supported VoIP/SIP/WebRTC media adapters;
- fleet/model rollout controls;
- signed reports;
- privacy-preserving sample upload;
- analyst review tooling;
- trusted-feedback / active-learning queues;
- continuously versioned model releases.

---

## SIH demo story

A strong judge demo should feel like a security product rather than a model notebook:

```text
1. User receives / starts a call.
2. VaaniRakshak activates automatically or the demo media stream begins.
3. Dashboard says "Collecting evidence".
4. Speech windows appear as the call progresses.
5. Suspicious regions become visible when the model crosses its operating threshold.
6. The call ends.
7. VaaniRakshak produces a final verdict and explainable evidence timeline.
8. User opens call history and inspects the record.
9. User marks the result Correct / Incorrect / Unsure.
10. Team shows that feedback enters a consented review queue, not direct retraining.
11. Team shows checkpoint fingerprint, latency, external-test results and limitations.
```

---

## What determines whether V2 is ready

The product architecture can be complete while the detector is still scientifically unproven.

The detector should not be promoted as SIH-ready until the real WavLM checkpoint has completed:

- leakage-clean SEA in-domain evaluation;
- strict unseen-generator evaluation;
- frozen ASVspoof 2021 LA evaluation;
- frozen ASVspoof 5 evaluation;
- duration/telephone/noise robustness;
- false-positive analysis;
- calibration assessment if probability semantics are shown;
- live latency verification.

No model-quality numbers should be invented before those experiments run.

# VaaniRakshak V2 — Backend Integration Contract

This document defines the boundary between the application/backend repository and the VaaniRakshak AI service.

The contract is intentionally small so the production backend can change independently from the anti-spoof model implementation.

## Ownership

The **application backend** owns users, sessions, persistence, authorization, dashboards and feedback records.

The **VaaniRakshak AI service** owns audio preprocessing, speech gating, detector inference, calibration, temporal evidence and detector provenance.

## Session lifecycle

### 1. Start analysis session

Application backend creates a call/session identifier and opens an AI analysis session.

Conceptual request:

```json
{
  "session_id": "call_123",
  "sample_rate": 48000,
  "audio_encoding": "pcm_s16le",
  "channels": 1
}
```

Conceptual acknowledgement:

```json
{
  "session_id": "call_123",
  "state": "ready",
  "detector": {
    "name": "VaaniRakshak-WavLM-V2",
    "mode": "trained",
    "threshold": 0.67,
    "calibrated_probability": false,
    "checkpoint_sha256": "..."
  },
  "speech_gate": "webrtc-vad-m2",
  "window_seconds": 4.0,
  "hop_seconds": 2.0
}
```

## 2. Stream audio

The preferred transport for live audio is WebSocket or another binary streaming channel.

Audio frames should carry or inherit:

- session identity;
- mono PCM16 audio;
- original sample rate;
- monotonically ordered stream position.

The AI service performs its own 16 kHz normalization and framing. The application backend should not attempt to reproduce detector preprocessing independently.

## 3. Window evidence event

When enough audio exists for a new analysis window, the AI service emits an evidence event.

Conceptual payload:

```json
{
  "type": "window_evidence",
  "session_id": "call_123",
  "index": 6,
  "start_seconds": 12.0,
  "end_seconds": 16.0,
  "analyzed": true,
  "synthetic_score": 0.81,
  "score_semantics": "uncalibrated_detector_score",
  "threshold": 0.67,
  "suspicious": true,
  "evidence_signal": 0.72,
  "speech_ratio": 0.91,
  "speech_gate": "webrtc-vad-m2",
  "quality": {
    "usable": true,
    "reason": null
  },
  "latency": {
    "preprocessing_ms": 9.4,
    "inference_ms": 143.8,
    "total_analysis_ms": 154.2
  }
}
```

If a window is rejected by speech/quality gating, the event should preserve the timestamp and rejection reason while omitting the detector score.

## 4. Live summary event

The AI service can emit a rolling summary after evidence changes.

Conceptual payload:

```json
{
  "type": "live_summary",
  "session_id": "call_123",
  "verdict": "Collecting evidence",
  "risk_label": "Collecting evidence",
  "risk_score": 58,
  "enough_evidence": false,
  "segments_analyzed": 1,
  "segments_skipped": 1,
  "suspicious_segments": 1,
  "usable_speech_seconds": 2.7,
  "realtime": {
    "mean_total_window_ms": 154.2,
    "hop_budget_ms": 2000,
    "within_budget": true
  }
}
```

The application backend/client should not relabel `Insufficient evidence` as `Likely genuine`.

## 5. End session

At call end, the backend signals finalization.

The AI service returns the final analysis summary and evidence metadata.

Conceptual result:

```json
{
  "session_id": "call_123",
  "verdict": "Suspicious",
  "risk_label": "Suspicious",
  "risk_score": 72,
  "enough_evidence": true,
  "duration_seconds": 46.3,
  "usable_speech_seconds": 33.8,
  "segments_analyzed": 16,
  "segments_skipped": 4,
  "suspicious_segments": 9,
  "suspicious_ratio": 0.5625,
  "detector": {
    "name": "VaaniRakshak-WavLM-V2",
    "mode": "trained",
    "threshold": 0.67,
    "calibrated_probability": false,
    "checkpoint_sha256": "..."
  },
  "speech_gate": "webrtc-vad-m2",
  "notice": "..."
}
```

The application backend persists this result together with the application-side call/session metadata.

## Feedback contract

Feedback is an application-level record, not an immediate model update.

Recommended conceptual payload:

```json
{
  "session_id": "call_123",
  "feedback": "prediction_wrong",
  "user_confidence": "high",
  "consent_for_research_use": false,
  "submitted_at_utc": "..."
}
```

Allowed feedback states should be simple and auditable:

```text
prediction_correct
prediction_wrong
unsure
```

If the user explicitly consents to future research/training use, the application backend may create a separate data-review candidate. That candidate must not automatically become training data.

## Training-data candidate flow

```text
Feedback record
   ↓
Explicit consent check
   ↓
Audio availability / retention check
   ↓
De-identification / privacy policy
   ↓
Human or trusted-label verification
   ↓
Provenance assignment
   ↓
Leakage / duplicate checks
   ↓
Curated offline dataset
```

## Error states

The AI service should fail visibly rather than silently substitute another detector when:

- configured checkpoint cannot be loaded;
- checkpoint schema is invalid;
- CUDA is requested but unavailable;
- detector emits an invalid score;
- audio framing/encoding is invalid;
- session state is inconsistent.

The application backend should preserve these as analysis failures, not convert them into genuine-call verdicts.

## Security and privacy boundary

- Raw audio should not be written to persistent storage by the AI service unless a deployment explicitly enables an approved retention flow.
- User feedback alone must never modify the deployed checkpoint.
- Authentication/authorization belongs to the application backend.
- AI results should carry detector/checkpoint identity so stored reports remain traceable across model versions.
- The JSON audit result is the canonical machine-readable analysis artifact.

## SIH integration sequence

For the SIH prototype, integration can proceed in this order:

```text
1. Backend can create analysis session
2. Backend can stream a controlled audio source
3. AI service emits window evidence
4. Backend forwards live evidence to Android/Web UI
5. Backend stores final result
6. UI shows call history/detail
7. UI records Correct / Wrong / Unsure feedback
8. Feedback appears in a review queue
```

This sequence allows the Android/UI/backend team and AI team to work independently while sharing a stable protocol.

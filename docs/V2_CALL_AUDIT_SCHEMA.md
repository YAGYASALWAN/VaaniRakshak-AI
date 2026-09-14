# VaaniRakshak V2 Call Audit Report

The live V2 browser can export a call-analysis record after a session ends.

Schema identifier:

```text
vaanirakshak-v2-call-audit-v1
```

The export is intentionally an **analysis audit**, not a recording archive.

## Privacy contract

The client-side export must state:

```json
{
  "privacy": {
    "raw_audio_included": false,
    "raw_audio_persisted_by_report_export": false
  }
}
```

The report exporter does not add microphone PCM, WAV/FLAC data, browser audio buffers, transcripts, or voice embeddings to the downloaded file.

This does not make a broader claim about operating-system/browser/network retention. It describes VaaniRakshak V2's report-export behavior.

## Top-level structure

```json
{
  "schema": "vaanirakshak-v2-call-audit-v1",
  "generated_at_utc": "ISO-8601 timestamp",
  "privacy": {},
  "summary": {},
  "windows": []
}
```

## `summary`

The summary is the final server-produced call assessment. Important fields include:

- `session_id`: server-generated analysis-session identifier;
- `analysis_mode`: mock, trained, or explicit legacy-experimental mode;
- `model`: detector identity;
- `detector`: trained-checkpoint metadata when a real/legacy checkpoint is connected;
- `detector.checkpoint_sha256`: SHA-256 fingerprint of the exact checkpoint file loaded for inference;
- `calibrated_probability`: whether detector scores have validated probability calibration;
- `speech_gate`: active VAD/speech-gate identity;
- `duration_seconds`: streamed call audio duration;
- `usable_speech_seconds`: estimated unique usable speech duration;
- `windows_seen`: total framed analysis windows;
- `segments_analyzed`: windows passed to the detector;
- `segments_skipped`: windows rejected by speech/quality gating;
- `suspicious_segments`: analyzed windows crossing the detector's own operating threshold;
- `threshold`: detector threshold selected outside the test/live session;
- `risk_score`: product-level heuristic score, not a model probability;
- `risk_label`: call-level risk category;
- `verdict`: product verdict or insufficient-evidence refusal;
- `enough_evidence`: whether minimum evidence requirements were met;
- `mean_preprocessing_ms`: mean VAD/quality/resampling time per framed window;
- `mean_inference_ms`: mean detector inference time for analyzed windows;
- `mean_total_window_ms`: mean total processing time per framed window;
- `regions`: highest-scoring analyzed regions;
- `notice`: detector/product limitation text.

The checkpoint digest makes two reports comparable at the model-artifact level even when local checkpoint paths differ. It is an identity fingerprint, not a digital signature and not proof that a report itself was not edited after export.

## `windows`

Every streamed evidence window received by the browser is retained as metadata in the report.

A window can include:

- temporal start/end;
- raw detector score when analyzed;
- threshold-relative evidence signal;
- detector threshold;
- analyzed/skipped state;
- suspicious state;
- quality metrics;
- speech ratio and speech-gate identity;
- skip reason;
- preprocessing latency;
- detector inference latency;
- total analysis latency.

Skipped windows remain in the audit. They are not silently removed, because knowing **why evidence was unavailable** is part of an explainable security decision.

## Interpretation rules

1. `synthetic_score` is not automatically a probability.
2. `risk_score` is not a probability of fraud or fakery.
3. A skipped window contributes no detector score.
4. An `Insufficient evidence` verdict must remain distinguishable from `Likely genuine`.
5. The detector threshold stored in the report is the model's configured operating point; a live session must not tune it.
6. The report must preserve the active detector mode and speech gate so a later reviewer knows what system produced the evidence.
7. When a checkpoint-backed detector is active, preserve its checkpoint SHA-256 fingerprint.

## Future evolution

A later schema version may add:

- cryptographic report digest/signature;
- application/build commit identifier;
- calibrated confidence intervals;
- PDF rendering derived from the same JSON audit;
- enterprise policy/action metadata.

Any incompatible change should increment the schema identifier rather than silently changing the meaning of `vaanirakshak-v2-call-audit-v1`.

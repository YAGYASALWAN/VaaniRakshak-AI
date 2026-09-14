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
- `detector.calibration`: temperature-calibration metadata when the loaded checkpoint is calibrated;
- `calibrated_probability`: whether detector scores have validated probability calibration metadata;
- `speech_gate`: active VAD/speech-gate identity;
- `duration_seconds`: streamed call audio duration;
- `usable_speech_seconds`: estimated unique usable speech duration;
- `windows_seen`: total framed analysis windows;
- `segments_analyzed`: windows passed to the detector;
- `segments_skipped`: windows rejected by speech/quality gating;
- `suspicious_segments`: analyzed windows crossing the detector's own operating threshold;
- `threshold`: detector threshold selected outside the test/live session;
- `evidence_semantics`: currently `temperature-normalized-logit-margin` for product-risk evidence;
- `median_evidence_signal` / `mean_evidence_signal`: temporal evidence statistics used by the heuristic risk engine;
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
- `synthetic_score`: detector score when analyzed;
- `threshold`: detector operating threshold;
- `evidence_signal`: threshold-relative security evidence;
- `evidence_semantics`: evidence transform identity;
- `evidence_temperature`: temperature used to recover the detector's pre-calibration logit margin for risk evidence;
- analyzed/skipped state;
- suspicious state;
- quality metrics;
- speech ratio and speech-gate identity;
- skip reason;
- preprocessing latency;
- detector inference latency;
- total analysis latency.

Skipped windows remain in the audit. They are not silently removed, because knowing **why evidence was unavailable** is part of an explainable security decision.

## Probability score vs risk evidence

A calibrated V2 checkpoint may expose `synthetic_score` as a development-temperature-calibrated synthetic-speech probability. Calibration changes the numerical probability and transforms the checkpoint threshold through the same monotonic map.

The product risk engine intentionally does **not** use calibrated probability distance directly. Its window evidence is based on the detector's logit margin from the operating threshold:

```text
visible_margin = logit(synthetic_score) - logit(threshold)
raw_margin = visible_margin * evidence_temperature
evidence_signal = sigmoid(raw_margin)
```

For an uncalibrated detector, `evidence_temperature = 1`. For a temperature-calibrated detector, it is the saved calibration temperature. This reconstructs the underlying pre-calibration logit margin.

Therefore a raw checkpoint and its correctly temperature-calibrated counterpart preserve:

- threshold-crossing decisions;
- window evidence signals;
- suspicious-window counts;
- evidence consistency statistics;
- the current call-level heuristic risk score and verdict.

Calibration changes probability semantics; it must not silently change security policy.

## Interpretation rules

1. `synthetic_score` is not automatically a probability. Check `calibrated_probability` and detector calibration metadata.
2. Even when `synthetic_score` is calibrated, it is a synthetic-speech probability under the calibration procedure/distribution, not probability of fraud.
3. `risk_score` is not a probability of fraud or fakery.
4. `evidence_signal` is not a calibrated probability; it is threshold-relative product evidence reconstructed in detector-logit space.
5. A skipped window contributes no detector score or detector evidence.
6. An `Insufficient evidence` verdict must remain distinguishable from `Likely genuine`.
7. The detector threshold stored in the report is the model's configured operating point; a live session must not tune it.
8. The report must preserve the active detector mode and speech gate so a later reviewer knows what system produced the evidence.
9. When a checkpoint-backed detector is active, preserve its checkpoint SHA-256 fingerprint.
10. When calibration is active, preserve its temperature/provenance so evidence and score semantics can be reconstructed.

## Future evolution

A later schema version may add:

- cryptographic report digest/signature;
- application/build commit identifier;
- calibrated confidence intervals;
- PDF rendering derived from the same JSON audit;
- enterprise policy/action metadata.

Any incompatible change should increment the schema identifier rather than silently changing the meaning of `vaanirakshak-v2-call-audit-v1`.

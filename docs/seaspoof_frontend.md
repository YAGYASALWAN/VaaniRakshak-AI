# SEA-Spoof frontend integration

This change adds a restrained local frontend and an explicit model adapter interface. The user's newer SEA-Spoof training code was not available in the accessible GitHub branches when this was built. It does **not** replace that code, start ASVspoof 5 downloads, claim SEA-Spoof checkpoint compatibility, or claim a measured accuracy improvement.

## Start the interface

From the project root, using your working model environment:

```powershell
python scripts/serve_demo.py
```

Open http://127.0.0.1:8765. No frontend package installation is needed. Without a model adapter, it correctly shows that no model is connected. The interface provides file selection/drag-and-drop, playback, loading/error states and score display. It contains no emojis, simulated predictions or fabricated accuracy figures. It accepts WAV/FLAC uploads up to 20 MiB. Audio remains in memory.

## Connect the existing trained model

An adapter is a project module with a factory accepting `checkpoint` (Path or None) and `device` (`cpu` or `cuda`). It returns an object exposing:

- `status()`: return `{ "ready": bool, "model": "model name or null", "evaluation": null }`. Only return evaluation data when it belongs to that exact checkpoint and uses held-out data. When present, use `evaluation.test_metrics` with numeric `f1`, `roc_auc`, and `n` fields.
- `analyze(audio_bytes)`: decode/validate the input, apply **the exact training preprocessing**, and return `synthetic_score`, `threshold`, `duration_seconds`, `window_start_seconds`, `window_seconds`, `elapsed_seconds`, `model`, and `notice`.

`synthetic_score` must increase toward **spoof**, regardless of the model's internal class order. A two-logit model must use its documented spoof index; a single-logit model's sigmoid interpretation must match its training labels. Never infer the label order from a filename. Restore the original architecture and feature extractor when loading the checkpoint.

Run with the actual adapter module and checkpoint path:

```powershell
python scripts/serve_demo.py --adapter your_project.seaspoof_adapter:create_detector --checkpoint path/to/best.pt
```

Those module/path names are placeholders until the existing training implementation is supplied. The built-in `--checkpoint` loader supports only the earlier repository's `english-cnn-v1` checkpoint schema. It is not an automatic loader for an arbitrary SEA-Spoof checkpoint. Older model folders are never silently selected.

The server validates numeric output and interval bounds. It displays scores within 0.1 of the provided decision threshold as inconclusive. This is an interface rule, not calibrated uncertainty. Scores are not probabilities of authenticity. If a checkpoint is missing, it returns an error instead of a prediction. The server is loopback-only and rejects cross-origin requests and oversized/non-binary uploads. It is intended for a local academic demonstration, not public hosting.

## SEA-Spoof data checks

The [current source card](https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof) defines `train`, `validation`, and `evaluation`; labels are `bonafide` and `spoof`. Use `row_id` as the unique key, not `utterance_id`, which is not guaranteed globally unique. The release includes English (`en`); the complete multilingual release is larger than the user's 50 GB budget. Filtering rows while streaming does not itself enforce a download-byte cap.

Keep the existing authorized dataset access in the local environment; do not commit or paste tokens. No gated audio is fetched by these tools. The metadata auditor reads an existing CSV/JSONL export:

```powershell
python scripts/audit_seaspoof.py path/to/metadata.jsonl --language en
```

It checks label and split values, both classes in each partition, duplicate row IDs, and cross-partition speakers/audio hashes when supplied. Required fields: `row_id`, `language`, `split`, `label`. Optional fields: `utterance_id`, `speaker_id`, `audio_sha256`, `text`, `is_text_exact`.

The published row schema does not provide a speaker-ID column. The auditor reports speaker separation as **unverified** when IDs are absent; it never invents speaker identities. Repeated original utterance IDs and exact transcripts are review flags. Audit findings must be resolved against source provenance rather than blindly reassigning records. Preserve validation for model selection and evaluation for the final test; do not tune on evaluation results.

## Remaining integration work

The SEA-Spoof training script, config, checkpoint structure, preprocessing, split construction and current failures must be inspected before the adapter can be completed. Their absence is an access limitation, not evidence that they are broken. Upload the current training `.py` and config, or push them to an accessible branch. No raw dataset or token is needed for this review.

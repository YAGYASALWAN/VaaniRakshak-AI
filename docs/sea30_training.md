# English SEA-Spoof: 30 GB ceiling

This is a self-contained training workflow for the current VaaniRakshak CNN, connected to the existing local frontend. It does not depend on unpublished training scripts from the laptop.

## Start

In the project PowerShell terminal:

```powershell
git fetch origin
git switch codex/seaspoof-frontend
git pull --ff-only origin codex/seaspoof-frontend
$vrPython = "$env:LOCALAPPDATA\VaaniRakshak\gpu210-clean\Scripts\python.exe"
& $vrPython -m pip install -r requirements-sea30.txt
& $vrPython scripts/train_seaspoof30.py --epochs 10
```

Use your current working GPU Python instead of `$vrPython` if its location differs. Existing torch/torchaudio installations are preserved. If Hugging Face authentication is missing, sign in to your already-approved account locally:

```powershell
& $vrPython -c "from huggingface_hub import login; login()"
```

Never paste a token into a chat or commit it. The workflow uses Hugging Face's cached token or `HF_TOKEN`; it does not request additional author approval. Stop any earlier dataset-preparation command before starting this one. The 30 GB ledger applies to this workflow, not other programs or previously downloaded datasets.

## What 30 GB means

- **English only reaches training, validation and testing.** Filtering uses the verified `language == 'en'` field and labels `bonafide=0`, `spoof=1`.
- **30,000,000,000 bytes is a hard cumulative source-payload limit**, including metadata reads and reserved failed requests. HTTP/TLS overhead and dependency installations are outside that count. It is not a promise of exactly 30 GB of unique English audio.
- SEA-Spoof mixes languages inside Parquet files. The code reads language/label columns first, selects row groups with English examples, then reads those groups using HTTP byte ranges. Mixed groups can transfer unused non-English bytes; those rows never enter the model.
- The plan uses at most 26.5 GB of estimated preparation reads, with room for metadata and retries. Train gets 80% of that allocation; validation and evaluation get 10% each. Selection favors English yield and class coverage. It stops if it cannot obtain at least 100 examples of each class per partition.
- The number of usable English recordings is printed after the metadata scan and again after duplicate removal. If the English subset fits below the cap, the workflow does not add other languages or duplicate data to fill it.
- No full dataset or full source archives are stored. Compact float16 features and metadata persist under `data/sea30_en/`. At most the release's 130,776 English recordings would use approximately 6.7 GB for these features; allow **10 GB free disk space** for features and working overhead, separately from old caches.

The source file inventory and revision `132f5dca9b6efe39cf1d3b54a858f167f5a421fc` are pinned in `configs/seaspoof_source.json`. Range reads use HTTPS and verify returned byte offsets and lengths. They cannot verify a whole-file SHA-256 without fetching the whole file; listed source hashes are provenance, not a claim of full-file verification. Prepared feature files have SHA-256 checksums and are checked before reuse. Source: [SEA-Spoof](https://huggingface.co/datasets/Jack-ppkdczgx/SEA-Spoof).

## Training and evaluation

The pipeline prepares data, audits exact duplicate original audio and four-second windows, trains the existing small CNN, selects its best epoch using validation loss, then evaluates the selected official evaluation rows. Duplicates are removed from training before validation/evaluation; the original split roles are never reassigned. Training uses class-balanced replacement sampling from the selected English pool. Ten epochs is the default, with early stopping after five non-improving validation losses.

This is an English subset experiment, not a full official benchmark score. Known `speaker_or_voice`, `source_model`, and `source_dataset` fields are retained in the audit. A voice identifier is not necessarily an independent human speaker, and speaker/generator independence is **not established**. More data does not guarantee improved accuracy. Results on modern voices and live microphone recordings still require separate testing.

Preparation finishes before model training; it can take hours depending on network speed and metadata layout. Checkpoints become available after the first training epoch. This workflow has been tested with local Parquet/audio fixtures and CPU training; a full gated-source/GPU run must execute in the user's approved local environment.

Outputs:

- `data/sea30_en/plan.json`: selected English row groups and estimates.
- `data/sea30_en/transfer.json`: cumulative reserved bytes; **do not delete it to restart the budget**.
- `data/sea30_en/audit.json`: actual counts, duplicate removals and limitations.
- `models/sea30_<timestamp>/best.pt`, `last.pt`, `history.json`, `evaluation.json`.

Rerun the same command after interrupted preparation. Completed metadata scans and feature groups are reused; an interrupted group may be reread, within the same cumulative ceiling. Use a saved run folder to resume model training:

```powershell
& $vrPython scripts/train_seaspoof30.py --epochs 10 --resume models/sea30_REPLACE_WITH_TIMESTAMP
```

The last completed epoch is restored, including optimizer and RNG state. Keep the original batch size; a completed test-evaluated run cannot resume further training. Start with `--batch-size 16` if a new batch-size-32 run exceeds GPU memory. Only one preparation/training process can hold this cache's lock.

## Frontend

In a second terminal, set `$vrPython` again and run:

```powershell
& $vrPython scripts/serve_demo.py --adapter vaanirakshak.sea_training:create_detector
```

Open http://127.0.0.1:8765. This adapter selects only `sea30_*` checkpoints created by this workflow. Click Refresh status after the first epoch. CPU inference is the default so the frontend can coexist with GPU training. To fix a specific model for a presentation, add `--checkpoint models/sea30_REPLACE_WITH_TIMESTAMP/best.pt`.

The frontend shows actual results and held-out metrics when available, with no invented predictions. It supports WAV/FLAC and analyzes the same centre four-second window used for training. Its synthetic score is not a calibrated probability.

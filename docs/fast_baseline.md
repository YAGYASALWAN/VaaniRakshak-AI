# Experimental local GPU baseline

The authenticated 2026-09-06 17:11 UTC probe inspected all six language previews
in both repositories without errors. The user has now chosen a faster Hindi and
Punjabi baseline on an RTX 4050 Laptop GPU (6 GiB VRAM), 16 GiB system RAM,
Python 3.14.2, PyTorch/TorchAudio 2.10.0+cu128. Their GPU tensor check passed.

## Run from the project root in the active virtual environment

```powershell
python -m pip install -r requirements-baseline.txt
python scripts/run_baseline.py all
```

Paste a valid Hugging Face read token only at the hidden prompt. It needs the
existing IndicVoices-R dataset permission. Do not use the token exposed earlier
in chat; revoke that token. Tokens are neither command arguments nor saved files.

The command downloads original Parquet shards at the exact revisions observed
in the successful probe, extracts and validates audio, balances the usable
sample, then trains and evaluates. It does not use signed/transcoded viewer
previews, load_dataset, ASVspoof, or entire language downloads.

## Concrete bounds and resuming

- Four randomly selected original train shards per corpus/language: 16 total.
- No shard over 600 MiB; total planned original download at most 8 GiB.
- Metadata requests have a separate 16 MiB cumulative bound.
- Plan saved before audio transfer, with repository paths, revisions and sizes.
- Allow about 12 GiB free disk. Original Parquet files remain as provenance.
- Completed downloads are verified and reused after interruption. An incomplete
  shard restarts; its partial file is removed. An invocation never automatically
  retries or increases the payload limit. Repeated interrupted runs can transfer
  more bytes cumulatively, since the 8 GiB transfer cap is per invocation.
- At most 3,120 candidate four-second windows are written, at most 2,000 selected.
- Preparation can produce fewer than 2,000 usable recordings; actual counts are
  saved. It stops if any language/class lacks 30 train or 10 dev/test recordings,
  or at least two known speakers in a cell. It never substitutes a random clip
  split or fabricates metrics to make training proceed.

Rerun `all` to reuse verified shards or an already prepared manifest. Once data
is prepared, `python scripts/run_baseline.py train` starts a new training run.
Training runs are separate timestamped directories. Training does not resume
optimizer state; an interrupted run retains its best completed checkpoint for
prediction, and a new train command starts from scratch without redownloading.

## Sampling and evaluation meaning

This is a fast pipeline baseline, not a full Milestone 3 research audit.
Deterministic 60/20/20 speaker hash partitions are assigned before selection.
Synthetic source and target speakers use one corpus-wide namespace; pairs that
cross partitions are discarded. Shared reference recordings crossing partitions
are removed. Duplicate original bytes and identical processed WAVs are removed;
speaker/reference/hash separation is checked again before training.

Within each split, class/language counts are equal, capped at 300/100/100 per
cell for train/dev/test. This is a shard sample, not representative corpus-wide
sampling. Generator diversity and per-generator counts may be limited by shards.
The original datasets' train splits are repartitioned for this pilot; these are
not their official benchmark test splits. Speaker IDs across corpora have not
been reconciled, and near-duplicate audio is not yet fingerprinted.

Both classes get identical centre-crop/end-padding to four seconds, mono 16 kHz
PCM16, using the existing mono and sinc-resampling functions. Original duration,
rate, channels, source row/shard, and byte checksums are recorded. Durations come
from actual audio headers, never reference recordings. Clips outside 1–30 seconds
or with silent/non-finite windows are rejected. Only one window per recording.

The CNN uses 64-bin log-mel spectrograms, three convolution blocks and per-clip
normalization. Default batch size is 16, with CUDA mixed precision, 15 maximum
epochs and early stopping after four epochs without better dev loss. The best
epoch is selected only by dev loss. Threshold is fixed at 0.5. Test data is used
after checkpoint selection, with no test-based threshold tuning. Results include
F1, ROC-AUC and confusion matrices overall and by language.

IndicVoices-R and IndicSynth differ in corpus provenance. IndicSynth references
IndicSUPERB; IndicVoices-R is restored/enhanced speech. Class and source corpus
are confounded here. High scores may reflect source/processing signatures. There
is no unseen-generator holdout or external benchmark in this shortcut, so no
claim of production, telephone, financial-fraud or general deepfake readiness.
The synthetic score is uncalibrated. Publisher attribution and selected source
terms remain applicable (IndicSynth's card labels it CC BY-NC 4.0).

## Outputs and prediction

- `data/baseline/download_plan.json`: pinned source files and transfer size.
- `data/baseline/preparation_report.json`: actual counts and rejection reasons.
- `data/baseline/manifest.json`: baseline-specific provenance and split records;
  separate from the strict Milestone 2 manifest format.
- `models/baseline_<timestamp>/best.pt`: model checkpoint.
- `models/baseline_<timestamp>/evaluation.json`: evaluation to upload to chat.
- `models/baseline_<timestamp>/history.json`: training/validation curves as data.

```powershell
python scripts/run_baseline.py predict --checkpoint models/baseline_TIMESTAMP/best.pt --audio C:/path/to/clip.wav
```

Replace both paths with actual files. Prediction accepts a 1–30-second recording
and examines its centre four seconds, using the same preprocessing as training.

## Verification

```powershell
python -m unittest discover -s tests -p "test_baseline*.py"
```

The stdlib suite exercises split leakage, metric ties, download limits,
redirect token handling and cache corruption. Optional runtime tests exercise
actual audio, Parquet, model gradients and checkpoint prediction when those
dependencies are installed. Live authenticated acquisition and GPU training
must run on the user's laptop; they were not executed in the agent environment.

Sources: [IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r),
[IndicSynth](https://huggingface.co/datasets/vdivyasharma/IndicSynth),
[PyArrow ParquetFile](https://arrow.apache.org/docs/python/generated/pyarrow.parquet.ParquetFile.html),
[PyTorch mixed precision](https://docs.pytorch.org/docs/2.10/amp.html).

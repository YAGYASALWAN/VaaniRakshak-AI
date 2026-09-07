# Full English training

This experiment uses the public [Bisher/ASVspoof_2019_LA mirror](https://huggingface.co/datasets/Bisher/ASVspoof_2019_LA), pinned to revision `aea92dd83a9c56e070c0b1e9f02e7c0d96216a4c`. Original dataset: [ASVspoof 2019](https://datashare.ed.ac.uk/handle/10283/3336). Preserve the original dataset attribution and terms when redistributing data. The mirror is third-party; it is not an official ASVspoof publication.

| Official split | Genuine | Synthetic | Total |
| --- | ---: | ---: | ---: |
| Train | 2,580 | 22,800 | 25,380 |
| Validation | 2,548 | 22,296 | 24,844 |
| Test | 7,355 | 63,882 | 71,237 |
| Total | 12,483 | 108,978 | 121,461 |

The source is English speech and contains about 7.54 GB of Parquet data. Python loads the public dataset with `streaming=True`; there is no manual archive download or token prompt. Streaming still transfers audio over the internet. The first pass converts each recording to compact log-mel features. Complete caches work offline and later epochs do not stream audio again. Partial resumes can reread earlier remote rows.

Allow at least 12 GB of additional free disk space for the approximately 6.23 GB feature cache, metadata, checkpoints and library overhead. Old Hindi/Punjabi files are not used or deleted. Training does not require the old `data/baseline/manifest.json`.

## Windows PowerShell

Run inside the project directory, using the clean CUDA environment already set up:

```powershell
git switch codex/milestone-3-source-probe
git pull --ff-only origin codex/milestone-3-source-probe
$vrPython = "$env:LOCALAPPDATA\VaaniRakshak\gpu210-clean\Scripts\python.exe"
& $vrPython -m pip install -r requirements-english.txt
& $vrPython scripts/train_english.py --epochs 20
```

The requirements file preserves the existing paired PyTorch/torchaudio 2.10.0+cu128 installation. CUDA availability is checked before data loading. Default batch size 32, mixed precision, four CPU threads and zero DataLoader subprocesses target the RTX 4050 6 GB laptop. Actual memory and runtime depend on the machine. If CUDA reports out-of-memory, start with `--batch-size 16`.

The command prepares all train and validation recordings, trains for up to 20 epochs with early stopping after five non-improving validation losses, then prepares and evaluates the full held-out test set. Each epoch draws 25,380 class-balanced samples with replacement from the full training pool; every recording is eligible, but not every recording is necessarily drawn in each epoch. Audio preparation uses one centre four-second window per recording and zero-pads short recordings. Original label mapping is 0=genuine, 1=synthetic.

Outputs appear under `models/english_<timestamp>/`: `best.pt`, `last.pt`, `history.json`, `run_config.json`, `evaluation.json` and `test_predictions.json`. Validation loss selects the best epoch; the classification threshold is fixed at 0.5. Test audio does not participate in training or checkpoint selection. Split checks reject repeated recording IDs, known speakers or exact original audio across partitions.

Feature batches persist every 64 recordings. Epoch checkpoints save optimizer, mixed-precision and random-generator state. After interruption, rerunning the command reuses features but starts a new model; to resume a particular model use its printed folder:

```powershell
& $vrPython scripts/train_english.py --epochs 20 --resume models/english_REPLACE_WITH_TIMESTAMP
```

Use the original batch size when resuming. An interrupted epoch restarts from the last completed epoch. A run with completed test evaluation cannot be resumed for further training.

To classify a mono 16 kHz WAV or FLAC after training:

```powershell
& $vrPython scripts/train_english.py --predict sample.wav --checkpoint models/english_REPLACE_WITH_TIMESTAMP/best.pt
```

This small CNN is an English ASVspoof 2019 experiment. Its score is not a calibrated probability; benchmark results do not demonstrate reliable detection of modern Qwen3 voices, telephone audio or other languages. Metrics include accuracy, precision, recall, F1, ROC-AUC and a confusion matrix, not an official ASVspoof t-DCF score.

Offline rule tests and optional CPU runtime tests:

```powershell
$env:PYTHONPATH = "src"
& $vrPython -m unittest discover -s tests -p "test_english*.py" -v
```

Runtime tests use generated audio, check gradients and cache interruption/reuse/integrity, and never access the benchmark. They require the installed torch, torchaudio, SoundFile and NumPy packages.

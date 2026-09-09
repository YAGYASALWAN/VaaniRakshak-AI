# VaaniRakshak-AI

## Current runnable experiments

The historical milestone notes below predate the CNN and local upload demo.
The repository now includes ASVspoof English, SEA-Spoof, and an English MLAAD
feature-cache workflow. Implementation is not evidence of a completed GPU run.

**New: [English MLAAD streaming and offline training](docs/mlaad_streaming.md).**
Fetch each recording into memory, commit compact log-mel features, then train
offline with epoch checkpoints. Matching genuine M-AILABS originals are required.
See the guide for source checks, recovery, disk space, and evaluation limits.


AI-powered cybersecurity prototype for detecting voice-cloning and synthetic-speech impersonation attacks.

This repository is being built incrementally for SIH 2026. **This milestone is dataset engineering only.** There is no anti-spoofing model, API, frontend, risk engine, or blockchain layer yet.

## Problem

Voice-cloning attacks can impersonate trusted people and manipulate victims into sensitive actions (fraudulent transfers, disclosure of credentials, social-engineering of helpdesks). A defender needs a pipeline that:

1. Accepts live or uploaded speech.
2. Analyses short audio windows.
3. Estimates whether the speech is bona fide human speech or AI-generated / voice-converted.
4. Turns those estimates into a time-varying cyber-risk score.
5. Warns the user or demands extra verification when risk is high.

Indian languages and mixed recording conditions are first-class requirements, not an afterthought.

## Current milestone

**Milestone 2: Dataset Inspection, Sampling and Bias Audit (implementation; full-suite verification pending).**

**NEURAL NETWORK TRAINING NOT STARTED.**

You can:

- Keep raw datasets immutable.
- Load and validate local audio.
- Convert any file to mono 16 kHz WAV windows (~4 s).
- Write a unified manifest (`bonafide` / `spoof`).
- Split later by **speaker** and by **generator**, not by random rows.

You cannot yet train a detector. That is intentional.

## Planned architecture

```
Audio
  → preprocessing (this milestone)
  → anti-spoofing model          [not implemented]
  → temporal risk engine         [not implemented]
  → cybersecurity decision       [not implemented]
  → warning / step-up auth       [not implemented]
```

Future milestones (not in this repo yet): AASIST-style baseline, speaker verification, FastAPI + WebSockets, privacy-preserving logs, tamper-evident audit records, multilingual Indian evaluation.

## Dataset strategy

| Dataset | Role | Canonical label | Training use |
| --- | --- | --- | --- |
| [IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r) | Genuine Indian speech | `bonafide` | Train / development |
| [IndicSynth](https://aikosh.indiaai.gov.in/home/datasets/details/indicsynth.html) | Synthetic / converted Indian speech | `spoof` | Train / development |
| [ASVspoof 2021 DF](https://huggingface.co/datasets/SpeechAntiSpoofingBenchmarks/ASVspoof2021_DF) | External benchmark | as published, mapped to `bonafide`/`spoof` later | **Never for initial training or tuning** |

ASVspoof 2021 DF is held out so that hyperparameter choices cannot overfit a well-known English deepfake benchmark. If the model only works on ASVspoof, it has not solved the Indian-speech problem. If it works on IndicVoices-R + IndicSynth **and** still generalises to ASVspoof, that is evidence of a speech-authenticity detector rather than a dataset detector.

**Do not download the corpora yet.** Directories under `data/raw/` are placeholders. Work with locally supplied WAV files until a sampling strategy exists.

## Engineering principles

- **Identical preprocessing** for bona fide and spoof audio. The classifier must learn *human vs synthetic*, not *IndicVoices vs IndicSynth*.
- **Immutable raw data.** Scripts only write under `data/processed/` and `data/manifests/`.
- **Speaker-disjoint evaluation.** A speaker in train never appears in dev or test.
- **Generator-disjoint evaluation.** Synthesis systems (XTTS, VITS, FreeVC, …) are stored in metadata so a later test set can use an unseen generator.
- **No class-specific DSP.** No extra denoising, pause stripping, or loudness matching applied only to one label. Breathing, pauses, and prosody may be forensic cues.
- **Reproducible splits** when a seed is provided.
- **Structured validation** (no silent `except: pass`).
- **Canonical labels** everywhere: `bonafide` and `spoof` (not `0/1`, `real/fake`, `genuine/generated`).

## Shortcut-learning risks (to inspect later)

Because bona fide and spoof audio currently come from *different collections*, a naive model can cheat using dataset fingerprints instead of synthesis artifacts:

| Risk | Why it fools a model |
| --- | --- |
| Sample rate | 48 kHz studio vs 16 kHz TTS can be inferred from residual spectrum even after resampling. |
| Codec / container | MP3 vs uncompressed WAV leaves different quantization noise. |
| Loudness | TTS often sits in a narrow LUFS range; field recordings do not. |
| Recording environment | Room reverb and mic noise may be unique to IndicVoices. |
| Clip duration | If spoof clips are always ~10 s and genuine clips are 2 s, duration becomes the label. |
| Language / gender imbalance | Model predicts majority language/gender rather than authenticity. |
| Speaker overlap | Same speaker in train and test inflates accuracy. |
| Generator overlap | Test generators seen in training overstates generalization. |
| File-format differences | Header/parser artifacts if we accidentally leak non-audio features. |
| Dataset-specific silence | One corpus pads with zeros; the other does not. |
| Dataset-specific preprocessing | If we denoise only spoof audio, the denoiser *is* the classifier. |

The manifest stores `language`, `gender`, `speaker_id`, `generator`, `codec`, `duration`, `sample_rate`, and `dataset` so these factors can be tabulated **before** training. Statistical tests come in a later milestone.

## Preprocessing pipeline

```
load audio          (soundfile / libsndfile decodes PCM → float32 array)
  → validate file   (corrupt, empty, NaN/Inf, silence, duration, channels, sample rate)
  → convert to mono (mean of channels; same rule for both classes)
  → resample 16 kHz (windowed-sinc anti-aliasing via torchaudio)
  → amplitude clip  (clip to [-1, 1] on save only; no peak-normalization)
  → segment ~4 s    (configurable window/hop; default hop 2 s → 50% overlap)
  → metadata        (unified schema, nullable fields allowed)
  → write WAV       (under data/processed/<split>/)
  → manifest row    (CSV under data/manifests/)
```

### What the libraries are doing

- **NumPy**: the waveform is a 1-D array of amplitudes, typically in `[-1, 1]`.
- **soundfile**: reads/writes audio containers. It does not “understand speech”; it unpacks samples.
- **torchaudio.functional.resample**: conceptually, low-pass filter then interpolate onto a new time grid so duration stays the same while sample rate changes.
- **PyYAML**: loads `configs/data_config.yaml` so 4.0 s / 16 kHz / pad-vs-skip are not magic numbers in code.

### Short-clip policy

Default: **`pad`** (zeros at the **end**).

If a clip is shorter than `window_seconds`, skip would drop it. If one dataset has shorter utterances, skip would delete that class more often and create a duration shortcut. Pad keeps the clip, marks the window, and preserves onsets.

Set `segmentation.short_clip_policy: skip` in YAML if you explicitly want to drop short audio.

## Repository layout

```
VaaniRakshak-AI/
├── README.md
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── configs/data_config.yaml          # all DSP / path / split defaults
├── data/
│   ├── raw/{indicvoices,indicsynth,asvspoof,local}/
│   ├── processed/{train,dev,test}/
│   └── manifests/
├── src/vaanirakshak/
│   ├── config.py                     # YAML → typed dataclasses
│   ├── exceptions.py
│   └── data/
│       ├── audio_io.py               # load / save
│       ├── validation.py             # structured checks
│       ├── preprocessing.py          # mono, resample, segment
│       ├── metadata.py               # manifest schema
│       ├── splits.py                 # speaker- & generator-disjoint
│       └── pipeline.py               # ordered end-to-end path
├── scripts/inspect_dataset.py
├── scripts/preprocess_dataset.py
├── tests/test_preprocessing.py
└── notebooks/
```

`src/vaanirakshak/config.py` and `data/pipeline.py` were added on purpose: configuration must be typed and loadable from tests, and the CLI must not own the pipeline logic.

`data/raw/local/` is a small extra folder for smoke-test files you create yourself. It is not a fourth research corpus.

## Setup

Python 3.11 or newer.

```powershell
cd c:\Users\qwert\OneDrive\Desktop\SIH
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Equivalent: `pip install -r requirements.txt` then `pip install -e .`.

## Run tests

Tests synthesize waveforms. They do **not** download IndicVoices, IndicSynth, or ASVspoof.

```powershell
pytest -q
```

## Process one example WAV

Create a 5-second stereo file (this is not a real dataset sample):

```powershell
python -c "import numpy as np, soundfile as sf; from pathlib import Path; sr=22050; t=np.arange(int(5*sr))/sr; x=np.stack([0.2*np.sin(2*np.pi*220*t), 0.2*np.sin(2*np.pi*330*t)], axis=1).astype(np.float32); p=Path('data/raw/local/example.wav'); p.parent.mkdir(parents=True, exist_ok=True); sf.write(p, x, sr); print(p)"
```

Inspect (no writes):

```powershell
python scripts/inspect_dataset.py --input data/raw/local/example.wav
```

Preprocess:

```powershell
python scripts/preprocess_dataset.py --input data/raw/local/example.wav --dataset local --label bonafide --split train --speaker-id spk_demo --language en
```

**Expected output**

- Original `data/raw/local/example.wav` unchanged.
- One or more WAVs in `data/processed/train/` named like `local__example__0.000_4.000.wav`.
- Each processed file: **mono**, **16000 Hz**, **exactly 4.0 s** (64000 samples) with the default config.
- Rows appended to `data/manifests/manifest.csv` with `label=bonafide`, `dataset=local`, window start/end times.

A 5 s clip with a 4 s window and 2 s hop yields two windows: `0–4 s` (full) and `2–6 s` (last 1 s zero-padded), unless you set `short_clip_policy: skip` (then only `0–4 s`).

## Assumptions

- Processed windows are 16-bit PCM WAV. Raw files may be any libsndfile-readable format.
- Internal layout after load is `(samples, channels)` from soundfile.
- Missing `speaker_id` is **not** treated as one global speaker; splits fall back to `source_file` so overlapping windows of the same file stay in one split.
- `--split train` on a single smoke-test file is **not** a speaker-disjoint dataset split. Use `speaker_disjoint_split` on a full manifest later.
- Peak normalization is off. Clipping on write is logged.
- ASVspoof may be ingested later into the same schema but `use_for_training: false` in config.

## Dataset Inspection, Sampling and Bias Audit

**NEURAL NETWORK TRAINING NOT STARTED.** No dataset downloads occur in these tools.
The default sampling config uses tiny synthetic metadata fixtures, clearly named
`demo_generator_a` and `demo_generator_b`. Results from those fixtures describe
the software's behavior, never the actual corpora.

### Architecture and manifest compatibility

Milestone 1's audio loader, transforms, pipeline, processed-record dataclass,
split helpers, and original 12 tests are unchanged. The public data package now
loads audio exports lazily, so metadata work does not import torch or soundfile.

Adapters translate source-specific fields into the existing 16 manifest columns,
plus channels, file format, age group, transcript, source language/speaker values,
source speaker, source/target references, original split, role, metadata locator,
RMS and silence observations. This pre-acquisition DataFrame permits null duration,
sample rate, processed path and segment times. It is not a processed
`ManifestRecord`, whose stricter audio invariants remain intact. Existing split
helpers return the core columns; keep the inspection table and join by
`(dataset, sample_id)` to retain extra provenance after using those helpers.

An adapter is a translator, not a downloader or audio transform. IndicSynth's
target speaker becomes `speaker_id`; source speaker and reference recordings
stay separate. Reference audio is never treated as the generated audio path.
Speaker IDs are namespaced by dataset by default. Supply a verified shared
`namespace` only if two exports genuinely use the same identity domain.
Do not claim disjointness across source/target roles merely because target IDs
are disjoint: future split design must consider connected source speakers and
reference recordings too. Running speaker and generator split functions in
sequence can break the earlier constraint; joint constraints need explicit
validation before any future training experiment.

### Why these checks exist

A naive merge confounds dataset with class: genuine audio comes from one source
and spoof audio from another. A detector can get 99% accuracy on an equally
confounded test set by recognizing recording noise, codecs, padding or language
instead of synthesis. That result would not demonstrate cybersecurity utility.
The audit always flags complete dataset/class confounding, even when measured
distributions match.

Identical 16 kHz mono processing standardizes model input. It cannot undo lossy
codec artifacts, bandwidth limits, prior enhancement, microphones or room noise.
Keep original measurements for the acquisition audit and re-audit processed
audio later. Do not normalize only one class.

Hours balance exposure better than counts when clip lengths differ: 1,000
ten-second clips contain five times the time of 1,000 two-second clips. Sampling
therefore uses original clip seconds. It rejects processed windows so overlapping
audio is not counted repeatedly. Report both time and counts.

Language is restricted to the observed class intersection. Speaker diversity
reduces domination by a few voices. Generator names are retained verbatim,
including previously unseen names, so later generator-disjoint experiments can
test generalization to attacks produced by systems absent from training.
Preserving metadata supports that experiment; it does not run it.

ASVspoof is marked external and excluded from the primary audit/planner. It must
not guide training, tuning, threshold optimization, or model selection. Inspect
it separately when useful. Full datasets are intentionally not downloaded:
first determine a justified subset from metadata and verify access/identities.

### Offline examples

Install the project in your existing virtual environment:

```bash
python -m pip install -e ".[dev,analysis]"
python scripts/inspect_dataset.py --dataset indicvoices --metadata tests/fixtures/indicvoices.json --output data/reports/indicvoices-demo
python scripts/inspect_dataset.py --dataset indicsynth --metadata tests/fixtures/indicsynth.json --output data/reports/indicsynth-demo
python scripts/audit_datasets.py --config configs/sampling_config.yaml --plots
python scripts/plan_sample.py --config configs/sampling_config.yaml
python -m pytest -q
```

Without installing the project, set `PYTHONPATH=src` before commands (on Windows
CMD use `set PYTHONPATH=src`). Metadata-only commands require numpy, pandas and
PyYAML; plots additionally require matplotlib. A dependency-light test command is:

```bash
python -m unittest discover -s tests -p test_dataset_analysis.py -v
```

That command tests Milestone 2 only. It is not a substitute for the full pytest
suite, which also needs the existing torch, torchaudio and soundfile dependencies.

For real metadata, configure local CSV/JSON/JSONL exports in `inputs`.
Multiple exports per dataset are supported by using a list of input specs.
Paths in YAML resolve relative to the config file, not the working directory.
CSV IDs retain leading zeros. Numeric durations must be seconds. Files are capped
at 50 MiB and 100,000 rows; the in-memory greedy planner is for small development
exports, not million-row corpus optimization.

Each input spec can provide `language`, `namespace`, `field_map`, and
`locator_prefix`. A field map is canonical field -> actual source key, including
dotted nested keys such as `audio.path`. Known aliases are conveniences, not a
claim that every release uses them. Numeric ASVspoof labels require a verified
`label_map` in its input spec (for example, only after confirming the mirror's
class mapping). Unknown labels fail explicitly.

If sample ID/path is absent, use a stable locator prefix containing dataset
revision, config/language, split and export identity. Row position is appended,
so changing row order requires new provenance. Generated IDs are deterministic
hashes of this locator, not invented source metadata. Actual audio location
must still be resolved before acquisition. Source sample rate is not inferred
from a dataset card or generator name; missing values remain null.

The inspection command's `--audio-root` opts into local audio measurements using
Milestone 1's loader and silence validator. It restricts access to that root,
records decode errors, and never fetches URLs. RMS is an amplitude proxy, not
perceptual LUFS; its denominator is measured clips only.

### Sampling semantics and feasibility

- Empty `languages` uses all observed common languages. The six-language starter
  list is a suggestion until metadata confirms coverage.
- `target_total_hours`, when non-null, overrides class targets and divides equally.
  Otherwise specify `target_hours_per_class: {bonafide: ..., spoof: ...}`.
- `target_hours_per_language` means hours **per class** for each named language.
  Remaining class budget is divided across unspecified selected languages.
- `max_speakers` is global; minimum speakers is per language and class; the clip
  cap is global per namespaced speaker.
- Missing duration, speaker, language, or spoof generator prevents eligibility;
  excluded counts are reported. No labels, durations, or generators are fabricated.
- Greedy selection prioritizes speaker coverage, language budget progress,
  speaker clip counts, and optional generator time diversity, with seeded identity
  hashes breaking ties. Input order does not affect results with stable IDs.
- Whole clips never exceed time budgets. The maximum generator share applies to
  actual selected spoof time; excess is removed and shortfalls disclosed.
- Reserved `holdout_generators` are excluded using exact observed values.
- Relative class imbalance and language underfill use `balance_tolerance`.
  Unsatisfied coverage/balance yields `infeasible` and CLI exit code 2.
  The heuristic may underfill even where a different combination exists.
  Constraints are never silently relaxed. Candidate plans do not authorize downloads.

### Transparent audit thresholds

All thresholds are configurable through `bias_thresholds`. A value at or above
the high threshold is HIGH, at or above medium is MEDIUM, otherwise LOW.

| Check | Metric | MEDIUM | HIGH |
| --- | --- | ---: | ---: |
| Sample rate, channels, codec, format, language, gender | Total variation distance | 0.20 | 0.50 |
| Duration | Larger mean / smaller mean | 1.25 | 2.00 |
| Speaker dominance | Largest observed single-speaker share | 0.10 | 0.25 |
| Generator dominance | Largest observed single-generator spoof share | 0.60 | 0.80 |
| Loudness proxy | Absolute class mean RMS difference | 0.03 | 0.10 |
| Missing metadata | Missing fraction in either relevant class | 0.20 | 0.80 |

Categorical distance is half the sum of absolute probability differences: zero
means identical observed distributions; one means disjoint support. Both file
and audio-duration weights are evaluated; dominance also checks both. Missing
values are excluded from distributions and reported separately, including fully
unknown fields. One-class-only languages and complete dataset/class confounding
are HIGH. No opaque combined score or statistical significance claim is made.
Low risk on measured fields does not certify authenticity learning or fairness.

### Generated reports

Ignored under `data/reports/`: `dataset_summary.json`, `class_balance.csv`,
`language_balance.csv`, `speaker_balance.csv`, `generator_balance.csv`,
`language_generator.csv`, `bias_report.json`, `sampling_plan.json`,
`candidate_plan.csv`, and optional `plots/`. Inspection also emits
`normalized_metadata.csv`. Reports contain supplied-row scope and observed
duration coverage. All-unknown duration yields null hours; partially known hours
are a subtotal, not a full-corpus estimate.

See [schema observations](docs/dataset_sources.md) for publisher evidence and
manual checks, and [Milestone 2 handoff](docs/milestone2_handoff.md) for examples,
the final file tree, and verification status. Stop here; do not start Milestone 3.

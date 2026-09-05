# VoiceShield

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

**Dataset-engineering foundation.**

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
voiceshield/
├── README.md
├── .gitignore
├── requirements.txt
├── pyproject.toml
├── configs/data_config.yaml          # all DSP / path / split defaults
├── data/
│   ├── raw/{indicvoices,indicsynth,asvspoof,local}/
│   ├── processed/{train,dev,test}/
│   └── manifests/
├── src/voiceshield/
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

`src/voiceshield/config.py` and `data/pipeline.py` were added on purpose: configuration must be typed and loadable from tests, and the CLI must not own the pipeline logic.

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

## Logical next milestone (do not start it in this pass)

**Corpus sampling + bias audit (still no neural net):** draw a stratified subset of IndicVoices-R and IndicSynth, fill speaker/language/generator fields, run speaker-disjoint splits, and compute tables for duration, loudness, sample-rate, language, and gender by class. Only after those distributions are understood should an AASIST baseline be trained — still holding ASVspoof 2021 DF out of training and tuning.

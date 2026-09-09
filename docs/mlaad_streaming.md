# English MLAAD: stream once, cache features, train offline

This workflow is a **new CNN experiment**, not fine-tuning the ASVspoof checkpoint.
It preserves the previous model, caches, and SEA-Spoof transfer ledger.

## Start on Windows

From the project root, use the existing working CUDA environment:

```powershell
$vrPython = "$env:LOCALAPPDATA\VaaniRakshak\gpu210-clean\Scripts\python.exe"
& $vrPython -m pip install -r requirements-mlaad.txt
& $vrPython -u scripts/train_mlaad.py all --epochs 25 --batch-size 16
```

The command runs metadata inspection, genuine feature preparation, synthetic
feature preparation, split verification, CNN training, and final test evaluation.
It uses the saved Hugging Face login. Do not put a token in source code or chat.

If audio has already been downloaded with the previous command, reuse it:

```powershell
& $vrPython -u scripts/train_mlaad.py all --epochs 25 --batch-size 16 --audio-root data/raw/mlaad_english
```

The root must contain `fake/en/`. Local WAVs are verified against pinned file
hashes before use; they are not deleted. Missing WAVs are fetched into memory.

## Source and scope

- MLAAD revision: `30c3dec763fa4803f11c2df1557d3b0828395366`, from the user's
  authenticated inventory: 143,000 English WAVs, 143 generator folders, ~45.49 GB.
  `--revision` can select a different revision only with a separate cache.
- Metadata must describe **every** English WAV exactly once. Language, original
  file, generator, transcript and source hash must be available; failures stop
  rather than dropping an undisclosed subset.
- Genuine speech comes from the corresponding English M-AILABS originals.
  Publisher archive links are used (UK and US); no substitute source is silently
  mixed in. HTTP ETag/Last-Modified/length are saved and compared on later reads.
  These are version consistency checks, not cryptographic publisher signatures.
- Publisher HEAD probes on 2026-09-09 returned HTTP 200 for both archives:
  UK 3,720,763,355 bytes; US 8,016,794,960 bytes. Availability can change.
- All English MLAAD files are prepared; training/validation/test use separate
  groups. "All" does not mean test recordings enter training, nor does it mean
  the entire waveform of every recording is used.
- The same centre-four-second, mono, 16 kHz frontend is used for both classes
  and is compatible with the current demo. Genuine and synthetic samples are
  resampled alike. Features are float16, shape 64 x 401. No special denoising or
  silence trimming is applied to only one class.

## Storage and transfer

The command never writes newly fetched WAVs or source archives to disk and does
not use Hugging Face's audio snapshot cache. At most one bounded recording is
held in memory for transformation. Metadata and feature blobs persist in
`data/mlaad_en/`, ignored by Git. Each feature + provenance row is committed in
one SQLite transaction (`synchronous=FULL`). Database checksums detect corrupt
feature blobs before training. Existing completed features are reused.

143,000 synthetic features contain about **7.34 GB** of numerical data; SQLite,
metadata and genuine features add storage. The program checks estimated free
space before acquisition and stops if free space becomes low. The 45.49 GB of
synthetic source bytes still crosses the network once in a successful run.
Genuine archive reads add up to ~11.74 GB per full scan. The user's later
all-English request superseded the former 30 GB SEA-Spoof ceiling; that old
ledger is not reset or reused.

Synthetic downloads retry transient failures with bounded backoff. The current
in-flight recording can be reread after interruption, while committed recordings
are not requested again. Genuine `.tgz` archives are sequential: interruption
may require rereading compressed bytes from the beginning, but previously saved
features are skipped. This is **not** byte-level archive resumption. Once all
required originals from an archive are saved, that archive is never requested
again. The complete feature cache supports every epoch without network access.

## Commands and recovery

```powershell
# Metadata only; no audio:
& $vrPython -u scripts/train_mlaad.py plan
# Preparation only:
& $vrPython -u scripts/train_mlaad.py prepare
# Inspect progress (stop the active job first; the cache has one writer):
& $vrPython scripts/train_mlaad.py status
# Once preparation is complete, train without network/authentication:
& $vrPython -u scripts/train_mlaad.py train --epochs 25 --batch-size 16
# Resume an interrupted training run from its last completed epoch:
& $vrPython -u scripts/train_mlaad.py train --epochs 25 --batch-size 16 --resume models/mlaad_ACTUAL_TIMESTAMP
```

After interrupted preparation, rerun the original `all` or `prepare` command.
After interrupted training, use `train --resume` with the printed run directory
and the same batch size. Do not delete the cache, metadata, or source stamps.
An already test-evaluated run cannot resume additional training. `--epochs` is
the total target, not additional epochs. The existing training loop saves
model, optimizer, scaler and random states each completed epoch, selects best
validation loss, and stops after five non-improving epochs. Threshold is fixed
at 0.5. Test data is never used for model selection. Class-balanced replacement
sampling makes all training records eligible; it does not guarantee each one is
sampled in every epoch.

## Evaluation limitations and gates

The split groups connected original files, shared normalized transcripts and
identical decoded windows. All synthetic variants of an original remain with
the original. Identical windows of the same class are deduplicated and counted;
conflicting class labels block training. Each split must retain >=100 examples
per class. Missing originals, missing metadata, invalid audio, corruption and
infeasible groups block training; no random-row or one-class fallback exists.

This is **not speaker-disjoint or generator-disjoint**. `reference_speaker` is
retained but is not treated as a verified shared identity. Generator counts and
full-clip durations are reported; feature exposure remains four seconds per
recording. Codec, recording-condition and source shortcuts can remain. Strong
same-source scores do not demonstrate live-microphone or unseen-generator
performance. Existing ASVspoof 2019 test metrics are historical, not a fresh
validation set for this experiment. ASVspoof 2021 DF remains external-only.

## Demo

The checkpoint schema is compatible with the existing demo:

```powershell
& $vrPython scripts/serve_demo.py --checkpoint models/mlaad_ACTUAL_TIMESTAMP/best.pt
```

Use the exact printed folder; no automatic replacement of the working ASVspoof
model. Open http://127.0.0.1:8765. Scores are uncalibrated; the display's
inconclusive band is a heuristic, not a statistical confidence interval.

## Verification

```powershell
$env:PYTHONPATH = "src"
& $vrPython -m unittest discover -s tests -p "test_mlaad*.py" -v
```

Tests cover committed-record recovery, transaction rollback, cache corruption,
source revision changes, metadata contracts, integrity failures, source grouping,
conflicting labels, class coverage, and in-memory genuine archive processing.
Runtime tests additionally exercise audio features, a CPU training run and demo
compatibility when torch/torchaudio/soundfile are installed. The gated MLAAD
metadata and full 143,000-file GPU run must be verified on the user's machine;
no full acquisition or performance improvement is claimed from fixture tests.

Sources:
- https://huggingface.co/datasets/mueller91/MLAAD
- https://github.com/i-celeste-aurora/m-ailabs-dataset
- https://ics.tau-ceti.space/data/Training/stt_tts/en_UK.tgz
- https://ics.tau-ceti.space/data/Training/stt_tts/en_US.tgz

MLAAD's stated license is CC BY-NC 4.0 / non-commercial academic research.
Retain dataset attribution with reports and models. M-AILABS publisher terms
are recorded in its linked repository.

# Milestone 2 handoff

Implementation is available for review. **Milestone 2 is not yet declared complete:**
the required full pytest suite could not run in this environment. **NEURAL NETWORK
TRAINING NOT STARTED.** No corpus audio was downloaded.

## Verification

- 38 new metadata regression tests pass using Python unittest. The same tests are
  discoverable by pytest. Includes duration balance with unequal file lengths,
  row-order-independent determinism, all source adapters, leakage warnings,
  constraints, provenance, and existing speaker/generator split functions.
- Original 12 pytest tests are preserved without edits.
- Full command attempted: `python -m pytest -q`. It failed before collection:
  `No module named pytest`. The runtime also lacks torch, torchaudio and soundfile.
  Package installation was attempted, but shell network approval was cancelled.
  No network restriction was bypassed.
- All Python source, scripts and tests compile.
- Both metadata inspection modes, audit with plots, and sampling demo ran
  successfully. Repeated JSON report generation is byte-reproducible.
- Remaining gate: run `python -m pytest -q` in the existing project environment
  with all dependencies. There should be 50 tests including the original 12;
  this is an expected count, not a claimed full-suite result.

## Example outputs (synthetic fixtures only)

Combined summary excerpt:

```json
{
  "scope": "supplied metadata rows only",
  "total_rows": 32,
  "total_hours": 0.035555555555555556,
  "duration_complete": true,
  "unique_speakers": 16
}
```

Audit excerpt:

```json
{
  "code": "sample_rate",
  "severity": "HIGH",
  "value": 1.0,
  "medium_threshold": 0.2,
  "high_threshold": 0.5,
  "evidence": {"clips": 1.0, "duration": 1.0},
  "message": "sample_rate distributions may reveal class (total variation distance)."
}
```

The demo deliberately sets genuine source rate to 48 kHz and spoof to 24 kHz.
Codec and RMS are unknown, so the audit reports missing coverage, not equality.
Dataset/class confounding is also flagged.

Sampling excerpt:

```json
{
  "status": "feasible",
  "input_rows": 32,
  "eligible_rows": 32,
  "class_duration_relative_gap": 0.0,
  "problems": [],
  "acquisition_authorized": false
}
```

| Selected class | Clips | Seconds | Hours | Unique speakers |
| --- | ---: | ---: | ---: | ---: |
| bonafide | 6 | 24 | 0.006666667 | 6 |
| spoof | 6 | 24 | 0.006666667 | 6 |

Each class includes 12 seconds of Hindi and 12 seconds of Punjabi. This tests
planner behavior; it does not establish real-dataset coverage or sample quality.

## Architecture and file inventory

Core processed manifest schema and all Milestone 1 DSP remain unchanged.
New pre-acquisition adapters create manifest-compatible tables with nullable
audio fields and extra inspection/provenance columns. The only changed library
file from Milestone 1 is the data package initializer (lazy public imports).

Every added file:

- `configs/sampling_config.yaml`
- `docs/dataset_sources.md`
- `docs/milestone2_handoff.md`
- `scripts/audit_datasets.py`
- `scripts/plan_sample.py`
- `src/vaanirakshak/data/adapters/__init__.py`
- `src/vaanirakshak/data/adapters/indicvoices.py`
- `src/vaanirakshak/data/adapters/indicsynth.py`
- `src/vaanirakshak/data/adapters/asvspoof.py`
- `src/vaanirakshak/data/inspection.py`
- `src/vaanirakshak/data/audit.py`
- `src/vaanirakshak/data/sampling.py`
- `src/vaanirakshak/data/sampling_config.py`
- `src/vaanirakshak/data/metadata_io.py`
- `src/vaanirakshak/data/reporting.py`
- `tests/test_dataset_analysis.py`
- `tests/fixtures/indicvoices.json`
- `tests/fixtures/indicsynth.json`

Modified files: `README.md`, `.gitignore`, `pyproject.toml`,
`scripts/inspect_dataset.py`, `src/vaanirakshak/data/__init__.py`.
Matplotlib is optional through `.[analysis]`. Large reports remain ignored.

Final source tree (generated outputs summarized):

```text
VaaniRakshak-AI/
  .gitignore
  README.md
  pyproject.toml
  requirements.txt
  configs/
    data_config.yaml
    sampling_config.yaml
  docs/
    dataset_sources.md
    milestone2_handoff.md
  data/
    manifests/.gitkeep
    reports/                 # ignored: JSON, CSV and plots
  notebooks/.gitkeep
  scripts/
    inspect_dataset.py
    preprocess_dataset.py
    audit_datasets.py
    plan_sample.py
  src/vaanirakshak/
    __init__.py
    config.py
    exceptions.py
    data/
      __init__.py
      audio_io.py
      metadata.py
      pipeline.py
      preprocessing.py
      splits.py
      validation.py
      adapters/
        __init__.py
        indicvoices.py
        indicsynth.py
        asvspoof.py
      inspection.py
      audit.py
      sampling.py
      sampling_config.py
      metadata_io.py
      reporting.py
  tests/
    test_preprocessing.py
    test_dataset_analysis.py
    fixtures/
      indicvoices.json
      indicsynth.json
```

## Assumptions and unresolved checks

Durations are source seconds; planning uses original clips, never overlapping
windows. Generator share is measured over known selected spoof time. Missing
speaker/duration/spoof-generator data is excluded and counted. Identity namespaces
need manual verification. Stable row locators require stable versioned exports.
The planner is greedy, conservative, and intended for small local metadata exports;
it does not guarantee finding every mathematically feasible combination.

Real metadata exports, row-level duration coverage, stable generated-audio paths,
all language/generator combinations and joint source/target speaker groupings
remain unverified. The AIKosh page was unavailable; IndicSynth mapping is based
on the publisher's Hugging Face preview. The implementation does not automatically
acquire metadata remotely, because column projection/streaming could still transfer
audio payloads. Small local export ingestion is the supported entry point.

See [dataset_sources.md](dataset_sources.md) for the complete pre-acquisition
verification list and evidence links. Review restoration artifacts, codec history,
license/attribution, speaker namespaces, reference reuse and acquisition size before
approving real audio downloads. ASVspoof remains external. No Milestone 3 work.

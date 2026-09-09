# Performance records

Run from the project root using the working Python environment:

```powershell
$vrPython = "$env:LOCALAPPDATA\VaaniRakshak\gpu210-clean\Scripts\python.exe"
& $vrPython -m pip install -r requirements-evaluation.txt
& $vrPython scripts/evaluate_performance.py --run-dir models/english_20260907T044807696536Z
```

The script reads `evaluation.json`, optionally `test_predictions.json` and
`history.json`. It evaluates saved predictions rather than rerunning inference,
training, or changing the checkpoint. No GPU, audio download, torch or Hugging
Face login is required. Outputs default to a new timestamped folder under
`data/reports/performance/` (ignored by Git). Open the printed `report.html`.

Outputs: JSON and CSV metrics, per-class CSV, count/row-normalized confusion
matrix heatmaps, accuracy/precision/recall/F1/specificity/balanced-accuracy bars.
Per-recording predictions also enable ROC-AUC, average precision, ROC and PR
curves and CSV points, score histograms, Brier score, log loss, and a ten-bin
reliability diagram with expected calibration error. Epoch history enables
training/validation loss and validation F1/AUC curves. MCC, macro/weighted F1,
false-positive/negative rates, negative predictive value and class counts are
also recorded. Undefined metrics are null, not fabricated zeros.

The requested "temperature graph" is interpreted here as a confusion-matrix
heatmap. No model temperature is fitted; a reliability diagram assesses the
existing score calibration. Calibration metrics on this class mixture do not
establish calibration at deployment prevalence.

Rows of the confusion matrix are true labels, columns are predicted labels.
Genuine=0, synthetic=1 (positive class). Classification uses the saved threshold
(default 0.5); the demo's inconclusive band is not applied. Scores are not
established calibrated probabilities. The script rejects duplicate IDs, invalid
scores, missing classes, and metrics/predictions inconsistent with saved counts.
No test-set threshold search or tuning is performed.

To record an aggregate report without predictions:

```powershell
& $vrPython scripts/evaluate_performance.py --evaluation records/performance/asv2019_reported_evaluation.json
```

The committed historical record is transcribed from the user's screenshot:
71,237 test examples, best epoch 13, confusion matrix [[7141,214],[8497,55385]].
It is not a new checkpoint test. Its AUC is reported, not recomputed. Prediction
and training curves are intentionally absent because their inputs were not
provided. `records/performance/asv2019_historical_report/` contains the derived
HTML/PNG/CSV/JSON record. Inspect those HTML files locally after pulling the repo.

A custom `--output records/performance/UNIQUE_RUN_NAME` lets you explicitly
prepare an aggregate report for a later Git commit. Existing output directories
are never overwritten. Input SHA-256 digests and source provenance are recorded;
raw prediction IDs/audio are not copied into reports. Do not confuse test scores
from different datasets as a direct comparison or a guarantee of call security.

Verification: `python -m unittest discover -s tests -p test_performance_report.py -v`

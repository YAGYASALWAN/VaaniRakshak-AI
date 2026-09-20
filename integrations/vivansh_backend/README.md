# VaaniRakshak V2 -> Vivansh Backend Integration

This bundle connects the portable V2 WavLM checkpoint produced by this branch to
the session/risk/report backend in \`Vivansh-07/VaaniRakshak-AI\` without
rewriting that backend.

## What this milestone changes

It adds a V2 detector adapter that satisfies the teammate backend's existing
\`VoiceDetector\` contract:

\`\`\`
recording window (4 s, mono 16 kHz WAV)
        |
        v
WavLMV2Detector
        |
        v
WindowVerdict(synthetic_score, confidence, timestamp)
        |
        v
MeasuredQualityDetector
        |
        v
existing AnalysisWorker -> risk engine -> SSE -> final report
\`\`\`

The WavLM model is shared across sessions and protects inference with a lock, so
the ~379 MB checkpoint is not loaded once per call. Each session still receives
its own quality wrapper.

## Files

- \`backend/app/voice_detector/wavlm_v2.py\` - drop-in V2 WavLM detector.
- \`run_v2_prototype.py\` - additive launcher; does not replace the teammate's
  existing \`run_prototype.py\`.
- \`install_into_teammate_repo.ps1\` - copies both files into a local clone.

## Install into a local clone

From the YAGYASALWAN repository root:

\`\`\`powershell
.\\integrations\\vivansh_backend\\install_into_teammate_repo.ps1 \`
  -TargetRepo C:\\path\\to\\Vivansh-VaaniRakshak-AI
\`\`\`

Then activate the teammate repository environment and ensure the V2 inference
dependency exists:

\`\`\`powershell
python -m pip install transformers
\`\`\`

Use the same CUDA-enabled PyTorch build that already ran the V2 checkpoint.

## First integration test

From the teammate repository root:

\`\`\`powershell
python run_v2_prototype.py \`
  --checkpoint C:\\VaaniRakshakData\\models\\v2_wavlm_mlaad_tiny\\best.pt \`
  --device cuda \`
  --stt none \`
  --open
\`\`\`

Upload a WAV/FLAC/MP3 recording through the existing prototype UI. The
\`RecordingAnalyzer\` already converts it to mono 16 kHz and emits the same 4 s /
2 s windows used by the backend.

The detector should report:

- architecture: \`wavlm-attention-v1\`
- a real V2 checkpoint, not a scripted detector
- higher \`synthetic_score\` = more likely synthetic
- quality measured independently from the model

## Important limitation in this milestone

This first bridge makes **recording-upload analysis real with V2**. The existing
\`POST /sessions/{id}/audio\` live endpoint currently submits timestamps without
audio bytes, so arbitrary live-call ingestion is the next patch. Do not claim
that Android call audio is already wired into the backend until that endpoint is
changed to carry actual audio.

## Model evidence

The first frozen MLAAD-tiny test run for this checkpoint reported approximately:

- ROC-AUC: 0.967
- PR-AUC: 0.975
- spoof precision: 0.987
- spoof recall: 0.775

These are held-out MLAAD-tiny test metrics for the baseline checkpoint, not a
claim of real-world fraud-detection accuracy.

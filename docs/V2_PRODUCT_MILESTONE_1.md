# VaaniRakshak V2 — Product Milestone 1

## Goal

Build the complete product path before training the V2 anti-spoofing model:

**microphone -> streaming transport -> audio windows -> detector interface -> temporal aggregation -> risk score -> final call verdict**

This milestone intentionally uses a deterministic mock detector. Its scores are integration-test data and must never be reported as real synthetic-speech detection results.

## Why V2 starts here

V1 behaved primarily like an audio-file classifier: upload a recording, select one 4-second analysis window, run one model prediction, and display the result. V2 treats the call as a session and accumulates evidence across multiple windows.

## Architecture

1. Browser requests microphone access.
2. An `AudioWorklet` converts microphone samples to mono PCM16 chunks.
3. The browser streams PCM over a WebSocket.
4. `StreamingSession` buffers audio into 4-second windows.
5. A detector implementing the `Detector` protocol returns a score for each window.
6. `aggregate_call()` combines all window scores into a separate product-level risk signal.
7. The UI receives live segment events and updates the evidence timeline.
8. When the call ends, the backend analyzes a useful partial tail and returns the final report.

## Files

- `src/vaanirakshak/v2_engine.py` — streaming session, detector protocol, mock detector and risk aggregation.
- `src/vaanirakshak/v2_server.py` — FastAPI + WebSocket local backend.
- `frontend/v2/index.html` — V2 dashboard.
- `frontend/v2/app.js` — microphone, WebSocket and live UI logic.
- `frontend/v2/pcm-worklet.js` — raw PCM16 browser capture.
- `frontend/v2/style.css` — product UI styling.
- `tests/test_v2_engine.py` — core streaming/aggregation tests.
- `requirements-v2.txt` — V2 web dependencies.

## Run locally

From the repository root on the `v2-product-skeleton` branch:

```powershell
git pull
git checkout v2-product-skeleton
python -m pip install -e .
python -m pip install -r requirements-v2.txt
python -m vaanirakshak.v2_server
```

Then open:

```text
http://127.0.0.1:8766
```

Allow microphone access, speak for at least 8–12 seconds so multiple windows are produced, and end the call.

## Test the product engine

```powershell
python -m pytest tests/test_v2_engine.py -q
```

## Milestone 1 acceptance criteria

- Browser can request microphone input.
- Audio is streamed continuously rather than uploaded after recording.
- The backend maintains a call/session state.
- At least one result is emitted for each complete 4-second audio window.
- Live risk and segment counts update without reloading the page.
- Ending the call produces a final aggregated verdict.
- The mock mode is visibly disclosed in the UI.
- V1 remains untouched on `main`.

## What Milestone 1 does NOT claim

- It does not detect AI speech accurately yet.
- Its mock score is not a probability.
- Its risk score is not evidence of fraud.
- It does not yet resample audio to the future model's canonical sample rate.
- It does not yet use VAD, overlapping windows, a calibrated threshold, or a trained V2 model.

## Next milestone

Milestone 2 should replace the placeholder audio path with a production inference pipeline:

1. streaming resampling to 16 kHz mono,
2. VAD / speech-presence filtering,
3. overlapping inference windows,
4. a real detector adapter,
5. latency instrumentation,
6. explicit handling of insufficient/low-quality speech,
7. persistence-free suspicious-region extraction for the final report.

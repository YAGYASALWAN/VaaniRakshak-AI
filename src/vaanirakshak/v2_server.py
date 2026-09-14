"""VaaniRakshak V2 local streaming server.

Default run:
    python -m vaanirakshak.v2_server

Connect a trained checkpoint:
    VAANIRAKSHAK_V2_CHECKPOINT=/path/to/best.pt python -m vaanirakshak.v2_server

Legacy V1 checkpoints are refused unless VAANIRAKSHAK_ALLOW_LEGACY=1 is set.
The live product defaults to WebRTC VAD; set VAANIRAKSHAK_V2_VAD=energy only
for fallback/debugging.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from vaanirakshak.v2_audio import MODEL_SAMPLE_RATE, build_speech_gate
from vaanirakshak.v2_detectors import CheckpointDetector
from vaanirakshak.v2_engine import HOP_SECONDS, WINDOW_SECONDS, MockDetector, StreamingSession


ROOT = Path(__file__).resolve().parents[2]
V2_FRONTEND = ROOT / "frontend" / "v2"
REALTIME_BUDGET_MS = HOP_SECONDS * 1000.0

app = FastAPI(title="VaaniRakshak V2", version="2.4.0-alpha")
app.mount("/assets", StaticFiles(directory=V2_FRONTEND), name="assets")


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _load_configured_detector():
    checkpoint = os.environ.get("VAANIRAKSHAK_V2_CHECKPOINT")
    if not checkpoint:
        return MockDetector(), None

    device = os.environ.get("VAANIRAKSHAK_V2_DEVICE", "cpu").strip().lower()
    allow_legacy = _truthy(os.environ.get("VAANIRAKSHAK_ALLOW_LEGACY"))
    try:
        return CheckpointDetector(checkpoint, device=device, allow_legacy=allow_legacy), None
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        return None, str(exc)


def _load_configured_speech_gate():
    mode = os.environ.get("VAANIRAKSHAK_V2_VAD", "webrtc").strip().lower()
    try:
        aggressiveness = int(os.environ.get("VAANIRAKSHAK_V2_VAD_AGGRESSIVENESS", "2"))
        return build_speech_gate(mode, aggressiveness=aggressiveness), None
    except (TypeError, ValueError, RuntimeError) as exc:
        return None, str(exc)


DETECTOR, DETECTOR_ERROR = _load_configured_detector()
SPEECH_GATE, SPEECH_GATE_ERROR = _load_configured_speech_gate()


def _configuration_error() -> str | None:
    errors = [value for value in (DETECTOR_ERROR, SPEECH_GATE_ERROR) if value]
    return "; ".join(errors) if errors else None


def _origin_allowed(websocket: WebSocket) -> bool:
    origin = websocket.headers.get("origin")
    if origin is None:
        return True
    return origin.startswith("http://127.0.0.1:") or origin.startswith("http://localhost:")


def _detector_metadata() -> dict | None:
    if DETECTOR is None:
        return None
    info = getattr(DETECTOR, "info", None)
    return info.as_dict() if info is not None else None


def _with_realtime_budget(summary: dict) -> dict:
    """Annotate observed mean processing against the two-second sliding-window hop.

    This is deliberately an observed scheduling signal, not a claim that the system
    can sustain arbitrary concurrency or worst-case latency.
    """
    result = dict(summary)
    result["realtime_budget_ms"] = REALTIME_BUDGET_MS
    mean_total = result.get("mean_total_window_ms")
    if isinstance(mean_total, (int, float)):
        margin = REALTIME_BUDGET_MS - float(mean_total)
        result["realtime_margin_ms"] = round(margin, 3)
        result["observed_mean_within_hop_budget"] = margin >= 0.0
    else:
        result["realtime_margin_ms"] = None
        result["observed_mean_within_hop_budget"] = None
    return result


def _detector_status() -> dict:
    error = _configuration_error()
    if DETECTOR is None or SPEECH_GATE is None:
        return {
            "ready": False,
            "analysis_mode": "configuration-error",
            "model": getattr(DETECTOR, "name", None),
            "error": error,
            "streaming": True,
            "window_seconds": WINDOW_SECONDS,
            "hop_seconds": HOP_SECONDS,
            "model_sample_rate": MODEL_SAMPLE_RATE,
            "quality_gate": getattr(SPEECH_GATE, "name", None),
            "window_execution": "worker-thread-when-analysis-ready",
            "realtime_budget_ms": REALTIME_BUDGET_MS,
        }

    status = {
        "ready": True,
        "analysis_mode": DETECTOR.mode,
        "model": DETECTOR.name,
        "threshold": DETECTOR.threshold,
        "calibrated_probability": bool(DETECTOR.calibrated_probability),
        "streaming": True,
        "window_seconds": WINDOW_SECONDS,
        "hop_seconds": HOP_SECONDS,
        "model_sample_rate": MODEL_SAMPLE_RATE,
        "quality_gate": SPEECH_GATE.name,
        "window_execution": "worker-thread-when-analysis-ready",
        "realtime_budget_ms": REALTIME_BUDGET_MS,
        "notice": DETECTOR.notice + f" Speech gate: {SPEECH_GATE.name}.",
    }
    detector_metadata = _detector_metadata()
    if detector_metadata is not None:
        status["detector"] = detector_metadata
    return status


@app.get("/")
def index() -> FileResponse:
    return FileResponse(V2_FRONTEND / "index.html")


@app.get("/api/v2/status")
def status() -> JSONResponse:
    payload = _detector_status()
    return JSONResponse(payload, status_code=200 if payload["ready"] else 503)


async def _ingest_without_blocking_event_loop(session: StreamingSession, payload: bytes):
    """Keep cheap packet buffering inline; offload only packets that trigger window analysis."""
    incoming_samples = len(payload) // 2
    analysis_ready = len(session.buffer) + incoming_samples >= session.window_samples
    if analysis_ready:
        return await asyncio.to_thread(session.ingest_pcm16le, payload)
    return session.ingest_pcm16le(payload)


@app.websocket("/ws/v2/analyze")
async def analyze_stream(websocket: WebSocket) -> None:
    if not _origin_allowed(websocket):
        await websocket.close(code=1008, reason="Open VaaniRakshak from localhost")
        return
    if DETECTOR is None or SPEECH_GATE is None:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": _configuration_error() or "V2 configuration unavailable"})
        await websocket.close(code=1011, reason="V2 configuration error")
        return

    await websocket.accept()
    session_id = str(uuid.uuid4())
    session: StreamingSession | None = None

    connected = {
        "type": "connected",
        "session_id": session_id,
        "analysis_mode": DETECTOR.mode,
        "model": DETECTOR.name,
        "threshold": DETECTOR.threshold,
        "calibrated_probability": bool(DETECTOR.calibrated_probability),
        "speech_gate": SPEECH_GATE.name,
        "window_execution": "worker-thread-when-analysis-ready",
        "realtime_budget_ms": REALTIME_BUDGET_MS,
    }
    detector_metadata = _detector_metadata()
    if detector_metadata is not None:
        connected["detector"] = detector_metadata
    await websocket.send_json(connected)

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return

            if message.get("text") is not None:
                try:
                    command = json.loads(message["text"])
                except json.JSONDecodeError:
                    await websocket.send_json({"type": "error", "message": "Invalid JSON command"})
                    continue

                kind = command.get("type")
                if kind == "start":
                    if session is not None:
                        await websocket.send_json({"type": "error", "message": "Session already started"})
                        continue
                    try:
                        sample_rate = int(command["sample_rate"])
                        session = StreamingSession(
                            detector=DETECTOR,
                            sample_rate=sample_rate,
                            speech_gate=SPEECH_GATE,
                        )
                    except (KeyError, TypeError, ValueError) as exc:
                        await websocket.send_json({"type": "error", "message": str(exc)})
                        continue
                    started = {
                        "type": "started",
                        "session_id": session_id,
                        "sample_rate": sample_rate,
                        "model_sample_rate": MODEL_SAMPLE_RATE,
                        "window_seconds": WINDOW_SECONDS,
                        "hop_seconds": HOP_SECONDS,
                        "threshold": DETECTOR.threshold,
                        "analysis_mode": DETECTOR.mode,
                        "model": DETECTOR.name,
                        "speech_gate": SPEECH_GATE.name,
                        "window_execution": "worker-thread-when-analysis-ready",
                        "realtime_budget_ms": REALTIME_BUDGET_MS,
                        "notice": session.live_summary()["notice"],
                    }
                    if detector_metadata is not None:
                        started["detector"] = detector_metadata
                    await websocket.send_json(started)
                    continue

                if kind == "stop":
                    if session is None:
                        await websocket.send_json({"type": "error", "message": "Session has not started"})
                        continue
                    final = _with_realtime_budget(await asyncio.to_thread(session.finalize))
                    if detector_metadata is not None:
                        final["detector"] = detector_metadata
                    await websocket.send_json({"type": "final", "session_id": session_id, **final})
                    await websocket.close(code=1000)
                    return

                if kind == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                await websocket.send_json({"type": "error", "message": "Unknown command"})
                continue

            payload = message.get("bytes")
            if payload is not None:
                if session is None:
                    await websocket.send_json({"type": "error", "message": "Send start before audio"})
                    continue
                try:
                    emitted = await _ingest_without_blocking_event_loop(session, payload)
                    for window in emitted:
                        await websocket.send_json(
                            {
                                "type": "segment",
                                "segment": window.as_dict(),
                                "summary": _with_realtime_budget(session.live_summary()),
                            }
                        )
                except (ValueError, RuntimeError) as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})

    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("vaanirakshak.v2_server:app", host="127.0.0.1", port=8766, reload=False)

"""VaaniRakshak V2 local streaming server.

Run with:
    python -m vaanirakshak.v2_server

The current milestone keeps a deterministic mock detector while the surrounding
streaming, audio-quality and aggregation systems are made production-shaped.
"""
from __future__ import annotations

import json
from pathlib import Path
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from vaanirakshak.v2_audio import MODEL_SAMPLE_RATE
from vaanirakshak.v2_engine import HOP_SECONDS, WINDOW_SECONDS, MockDetector, StreamingSession


ROOT = Path(__file__).resolve().parents[2]
V2_FRONTEND = ROOT / "frontend" / "v2"

app = FastAPI(title="VaaniRakshak V2", version="2.1.0-alpha")
app.mount("/assets", StaticFiles(directory=V2_FRONTEND), name="assets")


def _origin_allowed(websocket: WebSocket) -> bool:
    origin = websocket.headers.get("origin")
    if origin is None:
        return True
    return origin.startswith("http://127.0.0.1:") or origin.startswith("http://localhost:")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(V2_FRONTEND / "index.html")


@app.get("/api/v2/status")
def status() -> JSONResponse:
    detector = MockDetector()
    return JSONResponse(
        {
            "ready": True,
            "analysis_mode": detector.mode,
            "model": detector.name,
            "streaming": True,
            "window_seconds": WINDOW_SECONDS,
            "hop_seconds": HOP_SECONDS,
            "model_sample_rate": MODEL_SAMPLE_RATE,
            "quality_gate": "energy-v1",
            "notice": (
                "Product-skeleton mode. Audio quality gating, overlapping windows and 16 kHz normalization "
                "are active, but live scores still come from a deterministic mock detector and must not be "
                "interpreted as authenticity findings."
            ),
        }
    )


@app.websocket("/ws/v2/analyze")
async def analyze_stream(websocket: WebSocket) -> None:
    if not _origin_allowed(websocket):
        await websocket.close(code=1008, reason="Open VaaniRakshak from localhost")
        return

    await websocket.accept()
    session_id = str(uuid.uuid4())
    detector = MockDetector()
    session: StreamingSession | None = None

    await websocket.send_json(
        {
            "type": "connected",
            "session_id": session_id,
            "analysis_mode": detector.mode,
            "model": detector.name,
        }
    )

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
                        session = StreamingSession(detector=detector, sample_rate=sample_rate)
                    except (KeyError, TypeError, ValueError) as exc:
                        await websocket.send_json({"type": "error", "message": str(exc)})
                        continue
                    await websocket.send_json(
                        {
                            "type": "started",
                            "session_id": session_id,
                            "sample_rate": sample_rate,
                            "model_sample_rate": MODEL_SAMPLE_RATE,
                            "window_seconds": WINDOW_SECONDS,
                            "hop_seconds": HOP_SECONDS,
                            "notice": session.live_summary()["notice"],
                        }
                    )
                    continue

                if kind == "stop":
                    if session is None:
                        await websocket.send_json({"type": "error", "message": "Session has not started"})
                        continue
                    final = session.finalize()
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
                    emitted = session.ingest_pcm16le(payload)
                    for window in emitted:
                        await websocket.send_json(
                            {
                                "type": "segment",
                                "segment": window.as_dict(),
                                "summary": session.live_summary(),
                            }
                        )
                except (ValueError, RuntimeError) as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})

    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("vaanirakshak.v2_server:app", host="127.0.0.1", port=8766, reload=False)

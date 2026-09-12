"""Loopback-only demo server; uploaded audio stays in memory."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
MAX_BYTES = 20 * 1024**2


class Detector:
    def __init__(self, checkpoint=None, device="cpu"):
        self.requested = checkpoint
        self.device = device
        self.loaded = None

    def checkpoint(self):
        if self.requested:
            path = Path(self.requested)
            return path if path.is_file() else None
        return None  # Never silently substitute an older model for a SEA-Spoof model.

    def status(self):
        path = self.checkpoint()
        report = None
        if path and (path.parent / "evaluation.json").is_file():
            report = json.loads((path.parent / "evaluation.json").read_text())
        return dict(ready=path is not None, model=path.parent.name if path else None,
                    evaluation=report, max_bytes=MAX_BYTES)

    def analyze(self, raw):
        path = self.checkpoint()
        if path is None:
            raise FileNotFoundError("No model is connected. Start the server with the trained model adapter.")
        import io
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
        from vaanirakshak.english_training import CACHE_VERSION, Frontend, EnglishCNN, score_features

        signature = (str(path.resolve()), path.stat().st_mtime_ns)
        if signature != self.loaded:
            state = torch.load(path, map_location="cpu", weights_only=True)
            if state.get("schema") != "english-cnn-v1" or state.get("frontend") != CACHE_VERSION:
                raise ValueError("The selected checkpoint is not supported by this demo")
            torch.set_num_threads(4)
            model = EnglishCNN().to(self.device).eval()
            model.load_state_dict(state["model"])
            self.model = model
            self.frontend = Frontend().to(self.device).eval()
            self.threshold = state["threshold"]
            self.notice = state["notice"]
            self.loaded = signature
        started = time.monotonic()
        with sf.SoundFile(io.BytesIO(raw)) as stream:
            if stream.format not in ("WAV", "WAVEX", "FLAC"):
                raise ValueError("Upload a WAV or FLAC recording")
            rate, frames = stream.samplerate, len(stream)
            if not 8000 <= rate <= 96000 or stream.channels not in (1, 2) or not rate <= frames <= rate * 120:
                raise ValueError("Use a 1–120 second mono or stereo recording at 8–96 kHz")
            start = max(0, (frames - rate * 4) // 2)
            stream.seek(start)
            wave = stream.read(rate * 4, dtype="float32", always_2d=True).mean(axis=1)
        if not np.isfinite(wave).all() or np.sqrt(np.mean(wave ** 2)) < 1e-5:
            raise ValueError("The recording is silent or too quiet to analyze")
        wave = torch.from_numpy(wave)
        if rate != 16000:
            wave = torchaudio.functional.resample(wave, rate, 16000)
        wave = torch.nn.functional.pad(wave[:64000], (0, max(0, 64000 - len(wave))))
        # Shared with predict() and the frozen evaluator, so the number shown here is
        # the same number those report for the same recording.
        (score,), (usable,) = score_features(self.frontend, self.model, wave.unsqueeze(0).to(self.device))
        if not usable or not np.isfinite(score):
            raise ValueError("Model returned an invalid score")
        uncertain = abs(score - self.threshold) < .1
        verdict = "Inconclusive" if uncertain else ("Likely synthetic" if score >= self.threshold else "Likely genuine")
        return dict(verdict=verdict, synthetic_score=score, threshold=self.threshold,
                    calibrated_probability=False, model=path.parent.name, notice=self.notice,
                    duration_seconds=frames / rate, window_start_seconds=start / rate,
                    window_seconds=min(4, frames / rate), elapsed_seconds=time.monotonic() - started,
                    interpretation="Scores close to the decision threshold are shown as inconclusive. This is a display rule, not calibrated uncertainty.")


def handler_for(detector, assets, port):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value, mime="application/json"):
            body = json.dumps(value, allow_nan=False).encode() if mime == "application/json" else value
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; media-src 'self' blob:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def allowed(self):
            hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
            origin = self.headers.get("Origin")
            return self.headers.get("Host") in hosts and (origin is None or origin in {"http://" + h for h in hosts})

        def do_GET(self):
            if not self.allowed():
                return self.reply(403, {"error": "Open the app using its localhost address"})
            if self.path == "/api/status":
                try:
                    return self.reply(200, detector.status())
                except Exception:
                    return self.reply(503, {"error": "Model status is temporarily unavailable"})
            files = {"/": ("index.html", "text/html; charset=utf-8"),
                     "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                     "/style.css": ("style.css", "text/css; charset=utf-8")}
            if self.path not in files:
                return self.reply(404, {"error": "Not found"})
            name, mime = files[self.path]
            return self.reply(200, (assets / name).read_bytes(), mime)

        def do_POST(self):
            if not self.allowed():
                return self.reply(403, {"error": "Cross-origin requests are not allowed"})
            if self.path != "/api/analyze":
                return self.reply(404, {"error": "Not found"})
            if self.headers.get("Content-Type", "").split(";")[0] != "application/octet-stream":
                return self.reply(415, {"error": "Expected a binary audio upload"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = 0
            if not 0 < size <= MAX_BYTES or self.headers.get("Transfer-Encoding"):
                return self.reply(413, {"error": "Choose a file smaller than 20 MiB"})
            try:
                self.connection.settimeout(30)
                raw = self.rfile.read(size)
                if len(raw) != size:
                    raise ValueError("Upload was interrupted")
                result = validate_result(detector.analyze(raw))
                return self.reply(200, result)
            except FileNotFoundError as exc:
                return self.reply(503, {"error": str(exc)})
            except (ValueError, RuntimeError) as exc:
                return self.reply(422, {"error": str(exc)})
            except Exception:
                import traceback
                traceback.print_exc()
                return self.reply(500, {"error": "Analysis failed. Check the terminal and model environment."})

        def log_message(self, fmt, *args):
            print(fmt % args, flush=True)

    return Handler


def validate_result(result):
    """Keep label meaning explicit and prevent invalid adapter output reaching the UI."""
    result = dict(result)
    for field in ("synthetic_score", "threshold", "duration_seconds", "window_start_seconds", "window_seconds", "elapsed_seconds"):
        value = result.get(field)
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"Model adapter returned an invalid {field}")
    score, threshold = result["synthetic_score"], result["threshold"]
    if not 0 <= score <= 1 or not 0 < threshold < 1:
        raise ValueError("Model adapter score or threshold is out of range")
    if not 0 < result["duration_seconds"] <= 120 or result["window_start_seconds"] < 0 or result["window_seconds"] <= 0 or result["elapsed_seconds"] < 0:
        raise ValueError("Invalid analyzed audio interval")
    if result["window_start_seconds"] + result["window_seconds"] > result["duration_seconds"] + .01:
        raise ValueError("Analyzed interval exceeds recording duration")
    if not isinstance(result.get("model"), str) or not result["model"] or not isinstance(result.get("notice"), str):
        raise ValueError("Model adapter must identify its model and limitations")
    result["verdict"] = "Inconclusive" if abs(score - threshold) < .1 else ("Likely synthetic" if score >= threshold else "Likely genuine")
    result["calibrated_probability"] = False
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--adapter", help="Project module:factory implementing status() and analyze(audio_bytes)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a port between 1024 and 65535")
    if args.adapter:
        import importlib
        module, separator, name = args.adapter.partition(":")
        if not separator or not module or not name:
            parser.error("Use --adapter module:factory")
        factory = getattr(importlib.import_module(module), name)
        detector = factory(checkpoint=args.checkpoint, device=args.device)
        if not callable(getattr(detector, "status", None)) or not callable(getattr(detector, "analyze", None)):
            parser.error("Adapter must implement status() and analyze(audio_bytes)")
    else:
        detector = Detector(args.checkpoint, args.device)
    server = HTTPServer(("127.0.0.1", args.port), handler_for(detector, ROOT / "frontend", args.port))
    server.timeout = 30
    print(f"VaaniRakshak: http://127.0.0.1:{args.port} (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

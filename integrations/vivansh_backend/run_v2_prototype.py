"""Run Vivansh's backend with the VaaniRakshak V2 WavLM checkpoint.

Copy this file to the teammate repository root as run_v2_prototype.py and copy
wavlm_v2.py to backend/app/voice_detector/wavlm_v2.py.

Example:
  python run_v2_prototype.py ^
    --checkpoint C:\VaaniRakshakData\models\v2_wavlm_mlaad_tiny\best.pt ^
    --device cuda --stt none --open
"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))

from app.analysis.recording import RecordingAnalyzer  # noqa: E402
from app.analysis.scenario import ScenarioRunner  # noqa: E402
from app.session.manager import SessionManager  # noqa: E402
from app.stt.engines import build_transcriber  # noqa: E402
from app.transport.http_sse import build_server  # noqa: E402
from app.voice_detector.measured import MeasuredQualityDetector  # noqa: E402
from app.voice_detector.wavlm_v2 import WavLMV2Detector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8770)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--stt",
        choices=("auto", "whisper", "vosk", "none"),
        default="auto",
    )
    parser.add_argument("--whisper-model", default="base")
    parser.add_argument("--language")
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()

    if not 1024 <= args.port <= 65535:
        parser.error("choose a port between 1024 and 65535")
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("the SIH prototype must remain loopback-only")

    shared_v2 = WavLMV2Detector(args.checkpoint, device=args.device)

    # The expensive WavLM model is shared across calls and serializes PyTorch
    # inference internally. Each session gets its own quality wrapper/state.
    def detector_factory():
        return MeasuredQualityDetector(shared_v2)

    manager = SessionManager(detector_factory=detector_factory, buffer_capacity=16)
    transcriber = build_transcriber(
        prefer=args.stt,
        model_size=args.whisper_model,
        language=args.language,
    )
    analyzer = RecordingAnalyzer(manager=manager, transcriber=transcriber)
    scenarios = ScenarioRunner(manager=manager, analyzer=analyzer)

    def detector_status():
        return MeasuredQualityDetector(shared_v2).status()

    detector = detector_status()
    print("\n  VaaniRakshak V2 + teammate backend")
    print(f"    checkpoint    : {args.checkpoint}")
    print(f"    architecture  : {detector.get('architecture')}")
    print(f"    device        : {args.device}")
    print(f"    model loaded  : {'yes' if detector.get('loaded') else 'lazy / first window'}")
    print("    window        : 4 s")
    print("    hop           : 2 s")
    print("    transport     : HTTP + SSE")
    print("    warning       : raw detector scores are not probabilities unless calibrated")

    server = build_server(
        manager,
        host=args.host,
        port=args.port,
        analyzer=analyzer,
        scenarios=scenarios,
        detector_status=detector_status,
    )
    url = f"http://{args.host}:{args.port}"
    print(f"\n    open          : {url}")
    print("    stop          : Ctrl-C\n")

    if args.open:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopping")
    finally:
        manager.end_all()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

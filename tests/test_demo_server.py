from http.client import HTTPConnection
from http.server import HTTPServer
from pathlib import Path
import threading
import unittest

from vaanirakshak.demo_server import Detector, handler_for, validate_result

ROOT = Path(__file__).resolve().parents[1]


class DemoServerTests(unittest.TestCase):
    def setUp(self):
        self.server = HTTPServer(("127.0.0.1", 0), handler_for(Detector(Path("missing-checkpoint.pt")), ROOT / "frontend", 0))
        port = self.server.server_port
        self.server.RequestHandlerClass = handler_for(Detector(Path("missing-checkpoint.pt")), ROOT / "frontend", port)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.client = HTTPConnection("127.0.0.1", port, timeout=3)

    def tearDown(self):
        self.client.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()

    def test_page_and_truthful_empty_status(self):
        self.client.request("GET", "/")
        response = self.client.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn(b"Analyze recording", response.read())
        self.client.request("GET", "/api/status")
        response = self.client.getresponse()
        self.assertEqual(response.status, 200)
        self.assertIn(b'"ready": false', response.read())

    def test_invalid_upload_and_cross_origin(self):
        self.client.request("POST", "/api/analyze", body=b"x", headers={"Content-Type": "text/plain"})
        response = self.client.getresponse()
        self.assertEqual(response.status, 415)
        response.read()
        self.client.request("GET", "/api/status", headers={"Origin": "https://unrelated.example"})
        response = self.client.getresponse()
        self.assertEqual(response.status, 403)
        response.read()

    def test_static_path_traversal_blocked(self):
        self.client.request("GET", "/../configs/asvspoof5_subset.json")
        response = self.client.getresponse()
        self.assertEqual(response.status, 404)
        response.read()

    def test_missing_model_never_returns_a_prediction(self):
        self.client.request("POST", "/api/analyze", body=b"audio fixture", headers={"Content-Type": "application/octet-stream"})
        response = self.client.getresponse()
        self.assertEqual(response.status, 503)
        self.assertNotIn(b"synthetic_score", response.read())

    def test_adapter_score_contract(self):
        sample = dict(synthetic_score=.8, threshold=.5, duration_seconds=5,
                      window_start_seconds=.5, window_seconds=4, elapsed_seconds=.1,
                      model="unit-test-only", notice="Fixture")
        self.assertEqual(validate_result(sample)["verdict"], "Likely synthetic")
        self.assertEqual(validate_result(sample | {"synthetic_score": .52})["verdict"], "Inconclusive")
        self.assertEqual(validate_result(sample | {"synthetic_score": .1})["verdict"], "Likely genuine")
        for changes in ({"synthetic_score": float("nan")}, {"synthetic_score": 2}, {"window_seconds": 50}):
            with self.assertRaises(ValueError):
                validate_result(sample | changes)


if __name__ == "__main__":
    unittest.main()

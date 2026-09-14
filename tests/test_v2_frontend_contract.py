import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "frontend" / "v2"


class V2FrontendReportContractTests(unittest.TestCase):
    def test_final_report_controls_and_detail_surface_exist(self):
        html = (FRONTEND / "index.html").read_text(encoding="utf-8")
        self.assertIn('id="final-card"', html)
        self.assertIn('id="final-details"', html)
        self.assertIn('id="report-generated"', html)
        self.assertIn('id="download-report"', html)
        self.assertIn('id="print-report"', html)
        self.assertIn("Print / Save PDF", html)

    def test_client_keeps_json_audit_canonical_and_audio_out_of_export(self):
        javascript = (FRONTEND / "app.js").read_text(encoding="utf-8")
        self.assertIn("vaanirakshak-v2-call-audit-v1", javascript)
        self.assertIn("raw_audio_included: false", javascript)
        self.assertIn("raw_audio_persisted_by_report_export: false", javascript)
        self.assertIn("window.print()", javascript)
        self.assertIn("print-report", javascript)
        self.assertIn("Checkpoint SHA-256", javascript)

    def test_print_styles_hide_live_controls_and_render_final_report(self):
        css = (FRONTEND / "style.css").read_text(encoding="utf-8")
        self.assertIn("@media print", css)
        self.assertIn(".topbar, .hero, .grid, .report-actions", css)
        self.assertIn(".final-card", css)
        self.assertIn("print-color-adjust", css)


if __name__ == "__main__":
    unittest.main()

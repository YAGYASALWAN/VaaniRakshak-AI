import importlib.util
from pathlib import Path
import tempfile
import unittest
import json

spec=importlib.util.spec_from_file_location('report',Path(__file__).resolve().parents[1]/'scripts/evaluate_performance.py')
report=importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)

class PerformanceTests(unittest.TestCase):
    def test_known_matrix_and_undefined_precision(self):
        r=report.from_confusion([[90,10],[20,80]])
        self.assertEqual(r['accuracy'],.85)
        self.assertEqual(r['recall'],.8)
        self.assertEqual(r['specificity'],.9)
        self.assertIsNone(report.from_confusion([[10,0],[10,0]])['precision'])

    def test_tied_scores_have_half_auc(self):
        rows=[{'id':str(i),'label':i%2,'synthetic_score':.5} for i in range(8)]
        r,_=report.record({'test_metrics':{'threshold':.5}},rows)
        self.assertEqual(r['metrics']['roc_auc'],.5)
        self.assertEqual(r['metrics']['confusion_matrix'],[[0,4],[0,4]])

    def test_duplicate_or_invalid_predictions_rejected(self):
        with self.assertRaises(ValueError):
            report.validate_predictions([{'id':'x','label':0,'synthetic_score':.2}]*2)
        with self.assertRaises(ValueError):
            report.validate_predictions([{'id':'x','label':0,'synthetic_score':float('nan')}])

    def test_inconsistent_record_rejected(self):
        with self.assertRaisesRegex(ValueError,'accuracy'):
            report.record({'confusion_matrix':[[5,0],[0,5]],'accuracy':.5})
        with self.assertRaises(ValueError):
            report.from_confusion([[1.5,2],[3,4]])

    def test_aggregate_does_not_invent_curves(self):
        r,c=report.record({'confusion_matrix':[[5,1],[2,4]],'roc_auc':.8})
        self.assertIsNone(c)
        self.assertNotIn('brier_score',r['metrics'])
        # EER needs per-recording scores; a counts-only record must not claim one.
        self.assertNotIn('eer',r['metrics'])

    def test_threshold_independent_metrics_recorded_from_predictions(self):
        rows=[{'id':str(i),'label':0,'synthetic_score':.1+i/100} for i in range(20)]
        rows+=[{'id':str(100+i),'label':1,'synthetic_score':.6+i/100} for i in range(20)]
        r,_=report.record({'test_metrics':{'threshold':.5}},rows)
        self.assertEqual(r['metrics']['eer'],0.)
        self.assertEqual(r['metrics']['min_dcf'],0.)
        self.assertIn('p_spoof',r['metrics']['dcf_parameters'])
        json.dumps(r,allow_nan=False)

    def test_full_report_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d)
            ev=d/'evaluation.json'; pred=d/'test_predictions.json'; hist=d/'history.json'
            ev.write_text(json.dumps({'test_metrics':{'confusion_matrix':[[1,1],[1,1]],'threshold':.5}}))
            pred.write_text(json.dumps([{'id':str(i),'label':y,'synthetic_score':p} for i,(y,p) in enumerate([(0,.1),(0,.7),(1,.3),(1,.9)])]))
            hist.write_text(json.dumps([{'epoch':1,'train_loss':.8,'dev_loss':.7,'dev_metrics':{'f1':.5,'roc_auc':.6}}]))
            r=report.generate(ev,d/'out',pred,hist)
            self.assertEqual(len(r['charts']),5)
            self.assertTrue((d/'out/report.html').exists())
            self.assertTrue((d/'out/calibration_bins.csv').exists())
            with self.assertRaises(FileExistsError):
                report.generate(ev,d/'out',pred,hist)

if __name__=='__main__': unittest.main()

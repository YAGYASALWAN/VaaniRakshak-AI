"""Offline regression tests; run with pytest or Python's built-in unittest runner."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import pandas as pd

from vaanirakshak.data.adapters import adapt, normalize_language
from vaanirakshak.data.audit import audit
from vaanirakshak.data.inspection import common_languages, summarize, balance_table
from vaanirakshak.data.metadata_io import read_rows, load_inputs
from vaanirakshak.data.reporting import write_reports
from vaanirakshak.data.sampling import plan_sample
from vaanirakshak.data.sampling_config import SamplingConfig, BiasThresholds, load_settings
from vaanirakshak.data.splits import speaker_disjoint_split, generator_disjoint_split

ROOT = Path(__file__).resolve().parents[1]


def fixture() -> pd.DataFrame:
    return pd.concat([
        adapt(read_rows(ROOT / f"tests/fixtures/{dataset}.json"), dataset)
        for dataset in ("indicvoices", "indicsynth")
    ], ignore_index=True)


def config() -> SamplingConfig:
    return SamplingConfig(target_total_hours=48 / 3600, languages=("Hindi", "Punjabi"),
                          max_clips_per_speaker=2, min_speakers_per_language=2)


def high_codes(frame: pd.DataFrame) -> set[str]:
    return {f["code"] for f in audit(frame)["findings"] if f["severity"] == "HIGH"}


class DatasetAnalysisTests(unittest.TestCase):
    def test_indicvoices_actual_fields(self) -> None:
        f = adapt([{"id": "x", "lang": "Hindi", "speaker_id": "001",
                    "age_group": "20-30", "duration": 9.5}], "indicvoices")
        self.assertEqual(f.iloc[0].language, "hi")
        self.assertEqual(f.iloc[0].speaker_id, "indicvoices::001")
        self.assertEqual(f.iloc[0].label, "bonafide")
        self.assertIsNone(f.iloc[0].sample_rate)
        self.assertEqual(f.iloc[0].age_group, "20-30")

    def test_indicsynth_actual_fields(self) -> None:
        f = adapt([{"id": "a", "Target Speaker ID": 123, "Generative Model": "new_model_v7",
                    "Source Speaker_ID": 456.0, "Gender": "Female",
                    "Source Reference Audio": "source.wav", "Target Reference Audio": "target.wav"}],
                  "indicsynth", language="Punjabi")
        row = f.iloc[0]
        self.assertEqual(row.generator, "new_model_v7")
        self.assertEqual(row.speaker_id, "indicsynth::123")
        self.assertEqual(row.source_speaker_id, "indicsynth::456")
        self.assertEqual(row.source_reference, "source.wav")
        self.assertIsNone(row.source_file)
        self.assertEqual(row.language, "pa")

    def test_bonafide_clears_attack(self) -> None:
        f = adapt([{"id": "a", "generator": "incorrect", "attack_type": "tts"}], "indicvoices")
        self.assertIsNone(f.iloc[0].generator)
        self.assertIsNone(f.iloc[0].attack_type)

    def test_generator_unknown_stays_null(self) -> None:
        f = adapt([{"id": "a"}], "indicsynth")
        self.assertIsNone(f.iloc[0].generator)

    def test_custom_field_map(self) -> None:
        f = adapt([{"id": "a", "custom_model": "v3"}], "indicsynth",
                  field_map={"generator": "custom_model"})
        self.assertEqual(f.iloc[0].generator, "v3")

    def test_language_normalization(self) -> None:
        for raw, expected in [("hin", "hi"), ("Tamil", "ta"), ("tel", "te"),
                              ("Bengali", "bn"), ("Marathi", "mr"), ("pan", "pa")]:
            self.assertEqual(normalize_language(raw), expected)

    def test_common_languages(self) -> None:
        f = fixture()
        f.loc[(f.label == "spoof") & (f.language == "pa"), "language"] = "ta"
        self.assertEqual(common_languages(f[f.label == "bonafide"], f[f.label == "spoof"]),
                         {"common": ["hi"], "only_indicvoices": ["pa"], "only_indicsynth": ["ta"]})

    def test_total_duration(self) -> None:
        self.assertEqual(summarize(fixture())["duration_seconds"]["total"], 128)

    def test_class_hours(self) -> None:
        t = balance_table(fixture(), ["label"])
        self.assertTrue((t.clips == 16).all())
        self.assertTrue((t.hours == 64 / 3600).all())

    def test_missing_duration_not_zero_total(self) -> None:
        f = fixture()
        f["duration"] = None
        summary = summarize(f)
        self.assertIsNone(summary["total_hours"])
        self.assertFalse(summary["duration_complete"])

    def test_deterministic_input_order_independent(self) -> None:
        a = plan_sample(fixture(), config()).selected
        b = plan_sample(fixture().sample(frac=1, random_state=9), config()).selected
        pd.testing.assert_frame_equal(a, b)

    def test_seed_changes_choice(self) -> None:
        a = plan_sample(fixture(), config()).selected.sample_id.tolist()
        b = plan_sample(fixture(), replace(config(), seed=901)).selected.sample_id.tolist()
        self.assertNotEqual(a, b)

    def test_speaker_cap(self) -> None:
        selected = plan_sample(fixture(), replace(config(), max_clips_per_speaker=1)).selected
        self.assertLessEqual(selected.speaker_id.value_counts().max(), 1)

    def test_max_speakers(self) -> None:
        selected = plan_sample(fixture(), replace(config(), max_speakers=4)).selected
        self.assertLessEqual(selected.speaker_id.nunique(), 4)

    def test_stratified_language_duration(self) -> None:
        result = plan_sample(fixture(), config())
        self.assertEqual(result.report["status"], "feasible")
        self.assertEqual(result.selected.groupby(["label", "language"]).duration.sum().tolist(),
                         [12, 12, 12, 12])

    def test_unequal_file_lengths_balance_by_time(self) -> None:
        f = fixture()
        f.loc[f.label == "bonafide", "duration"] = 8
        c = replace(config(), target_total_hours=64 / 3600)
        result = plan_sample(f, c)
        self.assertEqual(result.report["status"], "feasible")
        self.assertEqual(result.selected.groupby("label").duration.sum().tolist(), [32, 32])
        self.assertNotEqual(*result.selected.groupby("label").size().tolist())

    def test_sample_rate_leakage(self) -> None:
        self.assertIn("sample_rate", high_codes(fixture()))

    def test_duration_leakage(self) -> None:
        f = fixture()
        f.loc[f.label == "bonafide", "duration"] = 12
        self.assertIn("duration", high_codes(f))

    def test_one_class_language(self) -> None:
        f = fixture()
        f.loc[f.label == "spoof", "language"] = "ta"
        self.assertIn("one_class_language", high_codes(f))

    def test_generator_dominance(self) -> None:
        f = fixture()
        f.loc[f.label == "spoof", "generator"] = "dominant"
        self.assertIn("generator_dominance_spoof", high_codes(f))

    def test_actual_generator_share(self) -> None:
        selected = plan_sample(fixture(), config()).selected
        amounts = selected[selected.label == "spoof"].groupby("generator").duration.sum()
        self.assertLessEqual(amounts.max() / amounts.sum(), config().max_generator_share)

    def test_impossible_generator_cap_reports_failure(self) -> None:
        f = fixture()
        f.loc[f.label == "spoof", "generator"] = "single"
        result = plan_sample(f, config())
        self.assertEqual(result.report["status"], "infeasible")
        self.assertFalse((result.selected.label == "spoof").any())

    def test_missing_metadata_excluded(self) -> None:
        f = fixture()
        f.loc[0, "duration"] = None
        f.loc[1, "speaker_id"] = None
        result = plan_sample(f, config())
        self.assertEqual(result.report["excluded"]["missing_or_zero_duration"], 1)
        self.assertEqual(result.report["excluded"]["missing_speaker"], 1)

    def test_exclusive_language_rejected(self) -> None:
        with self.assertRaises(ValueError):
            plan_sample(fixture(), replace(config(), languages=("Tamil",)))

    def test_holdout_generator_excluded(self) -> None:
        result = plan_sample(fixture(), replace(config(), holdout_generators=("demo_generator_a",),
                                               max_generator_share=1))
        self.assertNotIn("demo_generator_a", result.selected.generator.dropna().tolist())

    def test_asvspoof_notes_and_label_map(self) -> None:
        f = adapt([{"path": "DF_1.flac", "label": 1, "notes": json.dumps({
            "utterance_id": "DF_1", "speaker_id": "LA_1", "codec": "mp3",
            "attack_id": "A14", "vocoder": "family"})}],
            "asvspoof", label_map={"1": "spoof"})
        self.assertEqual(f.iloc[0].codec, "mp3")
        self.assertEqual(f.iloc[0].generator, "family")
        self.assertEqual(f.iloc[0].attack_type, "A14")
        self.assertEqual(f.iloc[0].split, "external")

    def test_external_never_selected(self) -> None:
        f = fixture()
        external = f.copy()
        external["dataset"] = "asvspoof"
        external["sample_id"] = "external_" + external.sample_id
        combined = pd.concat([f, external], ignore_index=True)
        result = plan_sample(combined, config())
        self.assertEqual(result.report["excluded"]["external_or_nonprimary"], len(f))
        self.assertNotIn("asvspoof", result.selected.dataset.tolist())

    def test_speaker_disjoint_function_still_works(self) -> None:
        split = speaker_disjoint_split(fixture(), seed=9)
        self.assertTrue((split.groupby("speaker_id").split.nunique() == 1).all())

    def test_generator_disjoint_preserved(self) -> None:
        split = generator_disjoint_split(fixture(), train_generators=["demo_generator_a"],
                                         test_generators=["demo_generator_b"])
        self.assertEqual(set(split.loc[split.generator == "demo_generator_b", "split"]), {"test"})

    def test_duplicate_ids_rejected(self) -> None:
        with self.assertRaises(ValueError):
            adapt([{"id": "x"}, {"id": "x"}], "indicvoices")

    def test_processed_windows_rejected(self) -> None:
        f = fixture()
        f.loc[0, "processed_file"] = "window.wav"
        with self.assertRaises(ValueError):
            plan_sample(f, config())

    def test_duplicate_source_rejected(self) -> None:
        f = fixture()
        f.loc[1, "source_file"] = f.loc[0, "source_file"]
        with self.assertRaises(ValueError):
            plan_sample(f, config())

    def test_bad_measurements_rejected(self) -> None:
        for duration in (-1, "inf", "bad"):
            with self.assertRaises(ValueError):
                adapt([{"id": "x", "duration": duration}], "indicvoices")

    def test_identity_requires_locator(self) -> None:
        with self.assertRaises(ValueError):
            adapt([{"lang": "Hindi"}], "indicvoices")
        f = adapt([{"lang": "Hindi"}], "indicvoices", locator_prefix="revision/Hindi/train")
        self.assertEqual(f.iloc[0].metadata_locator, "revision/Hindi/train#row=0")

    def test_empty_and_single_class_audit(self) -> None:
        f = fixture()
        self.assertEqual(summarize(f.iloc[:0])["total_rows"], 0)
        self.assertEqual(audit(f[f.label == "bonafide"])["status"], "insufficient_data")

    def test_threshold_validation(self) -> None:
        with self.assertRaises(ValueError):
            BiasThresholds(categorical_medium=0.8, categorical_high=0.2)
        with self.assertRaises(ValueError):
            SamplingConfig(max_clips_per_speaker=0)

    def test_config_demo_and_reports_reproducible(self) -> None:
        cfg, thresholds, raw = load_settings(ROOT / "configs/sampling_config.yaml")
        f = load_inputs(raw["inputs"], ROOT / "configs")
        self.assertEqual(plan_sample(f, cfg).report["status"], "feasible")
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            write_reports(f, dest, bias=audit(f, thresholds))
            first = (dest / "dataset_summary.json").read_bytes()
            write_reports(f, dest, bias=audit(f, thresholds))
            self.assertEqual(first, (dest / "dataset_summary.json").read_bytes())
            self.assertTrue((dest / "generator_balance.csv").is_file())

    def test_csv_preserves_leading_zero_speaker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "small.csv"
            path.write_text("id,speaker_id\nx,001\n")
            f = adapt(read_rows(path), "indicvoices")
            self.assertEqual(f.iloc[0].speaker_id_raw, "001")


if __name__ == "__main__":
    unittest.main()

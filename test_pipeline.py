"""Regression checks for leakage, stored metrics, chronological prediction and fuzzy decisions."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import unittest

import numpy as np
import pandas as pd

from air_data import FEATURES, MIN_STEPS, TARGET, fit_transform, make_windows, prepare, read_data
from fuzzy_system import fuzzy_detail
from inference import DEFAULT, MODEL_KEYS, forecast, load_run, sample_input, verify_bundle
from train import metrics
from evaluation import comparison, error_groups


class ForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runinfo = load_run()
        cls.example = pd.read_csv(DEFAULT / "contoh_sensor.csv")
        cls.results = pd.read_csv(DEFAULT / "test_predictions.csv")

    def test_checkpoint_reproduces_test_prediction(self):
        prediction = forecast(self.example).iloc[0]
        target = pd.Timestamp(self.example.timestamp.iloc[-1]) + pd.Timedelta(hours=1)
        self.assertEqual(prediction.timestamp_target, target)
        actual = self.results.set_index("timestamp_target").loc[str(target), "hybrid"]
        self.assertAlmostEqual(prediction.CO_mg_m3, actual, places=5)
        self.assertEqual(forecast(BytesIO(self.example.to_csv(index=False).encode())).iloc[0].prioritas,
                         prediction.prioritas)

    def test_four_independent_results(self):
        self.assertEqual(len(self.results), self.runinfo["data"]["splits"]["test"]["n"])
        for key, spec in self.runinfo["models"].items():
            recalculated = metrics(self.results.aktual.to_numpy(), self.results[key].to_numpy())
            for m in ["mae", "rmse", "r2", "bias"]:
                self.assertAlmostEqual(spec["test"][m], recalculated[m], places=6)

    def test_all_checkpoints_reproduce_the_same_historical_target(self):
        for key in MODEL_KEYS:
            with self.subTest(model=key):
                sample = sample_input(DEFAULT, key)
                row = forecast(sample, DEFAULT, key).iloc[-1]
                actual = self.results.set_index("timestamp_target").loc[str(row.timestamp_target), key]
                self.assertAlmostEqual(row.CO_mg_m3, actual, places=5)
                self.assertGreater(row.kelengkapan_pct, 0)
                self.assertLessEqual(row.referensi_bawah, row.CO_mg_m3)
                self.assertGreaterEqual(row.referensi_atas, row.CO_mg_m3)
                self.assertGreaterEqual(row.referensi_bawah, 0)

    def test_validation_error_reference_and_grouped_residuals(self):
        val = pd.read_csv(DEFAULT / "validation_predictions.csv")
        for key in MODEL_KEYS:
            spec = self.runinfo["models"][key]
            actual = val.aktual.to_numpy()
            pred = val[key].to_numpy()
            self.assertAlmostEqual(metrics(actual, pred)["rmse"], spec["val"]["rmse"], places=6)
            self.assertAlmostEqual(np.quantile(abs(actual - pred), .9), spec["val_abs_error_p90"], places=6)
            counts = error_groups(self.results, key)["n"].sum()
            self.assertEqual(counts, len(self.results))
        self.assertEqual(len(comparison(self.runinfo)), 4)

    def test_bundle_integrity(self):
        self.assertIn("validation_predictions.csv", verify_bundle())
        with TemporaryDirectory() as tmp:
            dest = Path(tmp) / "copy"
            shutil.copytree(DEFAULT, dest)
            with (dest / "correlation.csv").open("a") as out:
                out.write("\n")
            with self.assertRaisesRegex(ValueError, "Artefak correlation.csv berubah"):
                verify_bundle(dest)

    def test_validation_convergence(self):
        for group in ["ga", "pso_full", "pso_ga"]:
            for log in self.runinfo[group]:
                self.assertTrue(np.all(np.diff(log["history"]) <= 1e-12))
                self.assertGreater(len(log["candidates"]), 1)

    def test_input_and_recency_checks(self):
        for bad in [self.example.iloc[1:], self.example.drop(columns=[self.example.columns[-1]]),
                    pd.concat([self.example, self.example.tail(1)]), self.example.drop(2)]:
            with self.assertRaises(ValueError):
                forecast(bad)
        bad = self.example.copy()
        bad.iloc[-MIN_STEPS:, 1] = -200
        self.assertEqual(forecast(bad).iloc[-1].prioritas, "Data kurang")

    def test_quality_flags_and_out_of_range(self):
        sample = sample_input()
        partial = sample.copy()
        partial.iloc[0, 1] = -200
        row = forecast(partial).iloc[-1]
        self.assertTrue(np.isfinite(row.CO_mg_m3))
        self.assertEqual(row.jumlah_imputasi, 1)
        self.assertEqual(row.kualitas_input, "Ada imputasi")
        ood = sample.copy()
        ood.iloc[0, 1] = self.runinfo["train_ranges"][ood.columns[1]]["max"] + 100
        row = forecast(ood).iloc[-1]
        self.assertEqual(row.kualitas_input, "Periksa rentang")
        self.assertEqual(row.di_luar_rentang_latih, 1)
        self.assertTrue(np.isfinite(row.CO_mg_m3))

    def test_timezone_and_non_finite_input(self):
        sample = sample_input()
        bad = sample.copy()
        bad["timestamp"] += "+08:00"
        with self.assertRaises(ValueError):
            forecast(bad)
        bad = sample.copy()
        bad.iloc[0, 1] = 1e100
        with self.assertRaises(ValueError):
            forecast(bad)

    def test_true_hour_ahead_and_train_only_transform(self):
        n = 160
        frame = pd.DataFrame(np.repeat(np.arange(n)[:, None], len(FEATURES) + 1, axis=1).astype(float),
                             columns=[*FEATURES, TARGET], index=pd.date_range("2020-01-01", periods=n, freq="h"))
        frame.loc[frame.index[120], TARGET] = np.nan
        prep, x, _, splits, (cut, _) = prepare(frame)
        changed = frame.copy()
        changed.iloc[cut:] = 9000
        self.assertEqual(prep, fit_transform(changed, cut))
        self.assertNotIn(120, splits["val"])
        i = int(splits["test"][0])
        np.testing.assert_equal(make_windows(x, 6)[i-6], x[i-6:i])
        for earlier, later in [("train", "val"), ("val", "test")]:
            self.assertLess(max(splits[earlier]), min(splits[later]))

    def test_fuzzy_boundaries_and_monotonicity(self):
        grid = np.linspace(0, 15, 1501)
        scores = np.array([fuzzy_detail(x)[0] for x in grid])
        self.assertTrue(np.all(np.diff(scores) >= -1e-10))
        self.assertTrue(((scores >= 0) & (scores <= 100)).all())
        self.assertAlmostEqual(fuzzy_detail(3)[0], 37.5, places=5)
        for invalid in [-1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                fuzzy_detail(invalid)


if __name__ == "__main__":
    unittest.main()

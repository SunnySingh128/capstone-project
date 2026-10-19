"""
Unit tests for model training, evaluation, versioning, and prediction.
Uses isolated temporary directories to ensure test isolation.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from model import (
    build_time_series_features,
    evaluate_models_on_holdout,
    load_model,
    predict,
    save_model_artifact,
    train_models,
)


class TestModel(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.test_dir, "models")
        self.log_dir = os.path.join(self.test_dir, "logs")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Generate a synthetic monthly series of 36 months
        dates = pd.date_range(start="2024-01-01", periods=36, freq="MS")
        # Steady upward trend + noise
        revs = [100.0 + i * 15.0 + (i % 3) * 5.0 for i in range(36)]
        self.india_df = pd.DataFrame({"date": dates, "country": "India", "revenue": revs})

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_feature_engineering(self):
        feat_df = build_time_series_features(self.india_df)
        self.assertTrue(len(feat_df) > 0)
        expected_cols = ["time_idx", "month", "year", "quarter", "lag_1", "lag_2", "lag_3", "lag_6", "rolling_mean_3"]
        for col in expected_cols:
            self.assertIn(col, feat_df.columns)

    def test_evaluate_models_on_holdout(self):
        feat_df = build_time_series_features(self.india_df)
        best_name, final_model, metrics, train_df, test_df = evaluate_models_on_holdout(
            feat_df, test_size=0.2, random_state=42
        )

        self.assertIn(best_name, ["Linear Regression", "Random Forest", "Gradient Boosting"])
        self.assertIn("holdout_rmse", metrics)
        self.assertIn("holdout_mae", metrics)
        self.assertIn("baseline_rmse", metrics)
        self.assertIn("baseline_mae", metrics)
        self.assertIn("all_ml_metrics", metrics)
        self.assertIn("naive_baseline", metrics)

        # Verify holdout split was chronological
        self.assertGreater(test_df["date"].min(), train_df["date"].max())

    def test_train_save_load_and_predict(self):
        all_df = self.india_df.copy()
        all_df["country"] = "all"
        combined_df = pd.concat([self.india_df, all_df], ignore_index=True)

        # Train with custom isolated model and log directories
        results = train_models(
            df=combined_df,
            model_dir=self.model_dir,
            log_dir=self.log_dir,
            test=True,
        )

        self.assertIn("India", results)
        self.assertIn("all", results)

        # Test load_model dictionary access as specified in requirements
        india_model = load_model("India", model_dir=self.model_dir)
        self.assertIn("version", india_model)
        self.assertIn("model", india_model)
        self.assertIn("metrics", india_model)
        self.assertEqual(india_model["country"], "India")

        all_model = load_model("all", model_dir=self.model_dir)
        self.assertTrue(all_model["version"].startswith("all_"))

        # Test future prediction (recursive multi-step)
        pred_res = predict(
            country="India",
            date="2027-06-01",
            model_dir=self.model_dir,
            log_dir=self.log_dir,
            test=True,
        )
        self.assertEqual(pred_res["status"], "success")
        self.assertEqual(pred_res["country"], "India")
        self.assertEqual(pred_res["date"], "2027-06-01")
        self.assertGreater(pred_res["prediction"], 0.0)
        self.assertIn("model_version", pred_res)


if __name__ == "__main__":
    unittest.main()

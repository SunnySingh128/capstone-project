"""
Unit tests for audit and CSV logging module.
Uses isolated temporary directories to ensure test isolation.
"""

import os
import shutil
import tempfile
import unittest

from logger import get_logs, log_prediction, log_training


class TestLogger(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_log_training_and_retrieval(self):
        log_training(
            country="USA",
            model_name="Random Forest",
            holdout_rmse=123.45,
            holdout_mae=89.12,
            baseline_rmse=200.0,
            baseline_mae=150.0,
            version="USA_20261019_000000",
            num_samples=48,
            log_dir=self.test_dir,
        )

        entries = get_logs(kind="training", log_dir=self.test_dir)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["country"], "USA")
        self.assertEqual(entries[0]["model_name"], "Random Forest")
        self.assertEqual(float(entries[0]["holdout_rmse"]), 123.45)
        self.assertEqual(entries[0]["log_type"], "training")

    def test_log_prediction_and_retrieval(self):
        log_prediction(
            country="India",
            target_date="2028-01-01",
            prediction=5432.10,
            model_version="India_20261019_000000",
            model_name="Linear Regression",
            log_dir=self.test_dir,
        )

        entries = get_logs(kind="predictions", log_dir=self.test_dir)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["country"], "India")
        self.assertEqual(entries[0]["target_date"], "2028-01-01")
        self.assertEqual(float(entries[0]["prediction"]), 5432.10)
        self.assertEqual(entries[0]["log_type"], "prediction")

    def test_combined_logs_and_limit(self):
        for i in range(5):
            log_prediction(
                country="India",
                target_date=f"2028-0{i+1}-01",
                prediction=100.0 * i,
                model_version="v1",
                log_dir=self.test_dir,
            )

        log_training(
            country="India",
            model_name="Random Forest",
            holdout_rmse=10.0,
            holdout_mae=8.0,
            baseline_rmse=15.0,
            baseline_mae=12.0,
            version="v1",
            log_dir=self.test_dir,
        )

        # Retrieve all logs with limit 3
        all_logs = get_logs(kind="all", limit=3, log_dir=self.test_dir)
        self.assertEqual(len(all_logs), 3)


if __name__ == "__main__":
    unittest.main()

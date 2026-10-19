"""
Unit tests for performance and drift monitoring module.
Uses isolated temporary directories to ensure test isolation.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from model import train_models
from monitoring import (
    check_mean_shift_drift,
    check_model_performance,
    monitor_pipeline,
)


class TestMonitoring(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model_dir = os.path.join(self.test_dir, "models")
        self.log_dir = os.path.join(self.test_dir, "logs")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        dates = pd.date_range(start="2024-01-01", periods=24, freq="MS")
        rng = np.random.RandomState(42)
        revs = [100.0 + i * 10.0 + float(rng.normal(0, 20)) for i in range(24)]
        df = pd.DataFrame({"date": dates, "country": "all", "revenue": revs})
        train_models(df, model_dir=self.model_dir, log_dir=self.log_dir, test=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_drift_normal_case(self):
        ref = [100.0, 102.0, 98.0, 101.0, 99.0]
        curr = [101.0, 100.0, 99.5]
        res = check_mean_shift_drift(ref, curr, country="all", threshold=2.0)

        self.assertFalse(res["drift_detected"])
        self.assertEqual(res["status"], "normal")
        self.assertLess(res["standardized_shift"], 2.0)

    def test_drift_detected_case(self):
        # Example from prompt: reference_values=[100,110,105,98], current_values=[250,260,255]
        ref = [100.0, 110.0, 105.0, 98.0]
        curr = [250.0, 260.0, 255.0]
        res = check_mean_shift_drift(ref, curr, country="all", threshold=2.0)

        self.assertTrue(res["drift_detected"])
        self.assertEqual(res["status"], "drift_detected")
        self.assertGreater(res["standardized_shift"], 2.0)

    def test_model_performance_check(self):
        # Threshold higher than RMSE -> healthy
        healthy_res = check_model_performance(
            country="all",
            threshold=10000.0,
            model_dir=self.model_dir,
            test=True,
        )
        self.assertFalse(healthy_res["performance_degraded"])
        self.assertEqual(healthy_res["status"], "healthy")

        # Threshold lower than RMSE -> alert
        alert_res = check_model_performance(
            country="all",
            threshold=1.0,
            model_dir=self.model_dir,
            test=True,
        )
        self.assertTrue(alert_res["performance_degraded"])
        self.assertEqual(alert_res["status"], "alert")

    def test_monitor_pipeline_dispatch(self):
        # Drift payload
        drift_payload = {
            "country": "all",
            "reference_values": [10.0, 12.0, 11.0],
            "current_values": [50.0, 52.0, 55.0],
            "threshold": 2.0,
        }
        res_drift = monitor_pipeline(drift_payload, model_dir=self.model_dir, test=True)
        self.assertIn("drift_check", res_drift)

        # Performance payload
        perf_payload = {
            "country": "all",
            "threshold": 5000.0,
        }
        res_perf = monitor_pipeline(perf_payload, model_dir=self.model_dir, test=True)
        self.assertIn("performance_check", res_perf)


if __name__ == "__main__":
    unittest.main()

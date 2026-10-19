"""
Integration and unit tests for Flask API endpoints.
Uses Flask test client and isolated test paths.
"""

import json
import os
import shutil
import tempfile
import unittest

import pandas as pd

from app import app
from model import train_models


class TestAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()
        self.test_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.test_dir, "timeseries.csv")

        # Create sample data for test training
        dates = pd.date_range(start="2024-01-01", periods=24, freq="MS")
        revs = [200.0 + i * 20.0 for i in range(24)]
        df_all = pd.DataFrame({"date": dates, "country": "all", "revenue": revs})
        df_ind = pd.DataFrame({"date": dates, "country": "India", "revenue": [r * 0.4 for r in revs]})
        combined = pd.concat([df_all, df_ind], ignore_index=True)
        combined.to_csv(self.data_path, index=False)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["author"], "Sunny Singh")

    def test_train_endpoint(self):
        resp = self.client.post(
            "/train",
            json={"data_path": self.data_path, "test": True, "test_size": 0.2},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["mode"], "test")
        self.assertIn("all", data["models_trained"])

    def test_predict_endpoint_get_and_post(self):
        # Ensure test model is trained
        df = pd.read_csv(self.data_path)
        train_models(df, test=True)

        # GET /predict
        resp_get = self.client.get("/predict?country=all&date=2028-01-01&test=true")
        self.assertEqual(resp_get.status_code, 200)
        data_get = resp_get.get_json()
        self.assertEqual(data_get["country"], "all")
        self.assertEqual(data_get["date"], "2028-01-01")
        self.assertGreater(data_get["prediction"], 0.0)

        # POST /predict
        resp_post = self.client.post(
            "/predict",
            json={"country": "India", "date": "2028-01-01", "test": True},
        )
        self.assertEqual(resp_post.status_code, 200)
        data_post = resp_post.get_json()
        self.assertEqual(data_post["country"], "India")
        self.assertIn("model_version", data_post)

    def test_predict_validation_errors(self):
        # Missing country
        resp = self.client.get("/predict?date=2028-01-01")
        self.assertEqual(resp.status_code, 400)

        # Missing date
        resp = self.client.get("/predict?country=all")
        self.assertEqual(resp.status_code, 400)

    def test_logs_endpoint(self):
        resp = self.client.get("/logs?kind=all&limit=50&test=true")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIsInstance(data["logs"], list)

    def test_monitor_endpoint(self):
        df = pd.read_csv(self.data_path)
        train_models(df, test=True)

        # Performance check
        resp_perf = self.client.post(
            "/monitor",
            json={"country": "all", "threshold": 5000, "test": True},
        )
        self.assertEqual(resp_perf.status_code, 200)
        data_perf = resp_perf.get_json()
        self.assertEqual(data_perf["status"], "success")
        self.assertIn("performance_check", data_perf["monitoring"])

        # Drift check
        resp_drift = self.client.post(
            "/monitor",
            json={
                "country": "all",
                "reference_values": [100, 110, 105, 98],
                "current_values": [250, 260, 255],
                "test": True,
            },
        )
        self.assertEqual(resp_drift.status_code, 200)
        data_drift = resp_drift.get_json()
        self.assertEqual(data_drift["status"], "success")
        self.assertIn("drift_check", data_drift["monitoring"])


if __name__ == "__main__":
    unittest.main()

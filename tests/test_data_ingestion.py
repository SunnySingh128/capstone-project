"""
Unit tests for data ingestion pipeline.
Uses isolated temporary directories without touching production paths.
"""

import json
import os
import shutil
import tempfile
import unittest

import pandas as pd

from data_ingestion import (
    aggregate_revenue,
    ingest_data,
    load_raw_data,
    validate_and_clean_data,
)


class TestDataIngestion(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.test_dir, "raw")
        self.processed_dir = os.path.join(self.test_dir, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_csv_ingestion_and_aggregation(self):
        csv_file = os.path.join(self.raw_dir, "invoices.csv")
        sample_data = pd.DataFrame([
            {"date": "2024-01-05", "country": "India", "price": 100.0},
            {"date": "2024-01-15", "country": "India", "price": 50.0},
            {"date": "2024-01-20", "country": "USA", "price": 200.0},
            {"date": "2024-03-10", "country": "India", "price": 300.0},  # note gap in Feb 2024
        ])
        sample_data.to_csv(csv_file, index=False)

        result_df = ingest_data(
            input_dir=self.raw_dir,
            output_dir=self.processed_dir,
            frequency="monthly",
            output_file="test_series.csv",
        )

        # Check saved file exists
        saved_file = os.path.join(self.processed_dir, "test_series.csv")
        self.assertTrue(os.path.exists(saved_file))

        # Check series contains India, USA, and 'all'
        countries = set(result_df["country"].unique())
        self.assertIn("India", countries)
        self.assertIn("USA", countries)
        self.assertIn("all", countries)

        # Check missing periods inside observed range are filled with 0.0
        # Global range is 2024-01 to 2024-03 (3 months)
        india_series = result_df[result_df["country"] == "India"].set_index("date")
        self.assertEqual(len(india_series), 3)
        self.assertAlmostEqual(india_series.loc["2024-01-01", "revenue"], 150.0)
        self.assertAlmostEqual(india_series.loc["2024-02-01", "revenue"], 0.0)  # filled zero
        self.assertAlmostEqual(india_series.loc["2024-03-01", "revenue"], 300.0)

        # Check 'all' aggregate series
        all_series = result_df[result_df["country"] == "all"].set_index("date")
        self.assertEqual(len(all_series), 3)
        self.assertAlmostEqual(all_series.loc["2024-01-01", "revenue"], 350.0)
        self.assertAlmostEqual(all_series.loc["2024-02-01", "revenue"], 0.0)
        self.assertAlmostEqual(all_series.loc["2024-03-01", "revenue"], 300.0)

    def test_json_and_jsonl_ingestion(self):
        # Create JSON file
        json_file = os.path.join(self.raw_dir, "invoices_1.json")
        data_json = [
            {"date": "2024-05-01", "country": "UK", "price": 120.0},
        ]
        with open(json_file, "w") as f:
            json.dump(data_json, f)

        # Create JSONL file
        jsonl_file = os.path.join(self.raw_dir, "invoices_2.jsonl")
        with open(jsonl_file, "w") as f:
            f.write(json.dumps({"date": "2024-05-15", "country": "Germany", "price": 250.0}) + "\n")
            f.write(json.dumps({"date": "2024-06-01", "country": "UK", "price": 180.0}) + "\n")

        raw_df = load_raw_data(self.raw_dir)
        self.assertEqual(len(raw_df), 3)

        cleaned_df, _, _, _ = validate_and_clean_data(raw_df)
        self.assertEqual(len(cleaned_df), 3)

    def test_column_mapping_and_validation(self):
        csv_file = os.path.join(self.raw_dir, "custom_cols.csv")
        custom_df = pd.DataFrame([
            {"invoice_date": "2024-02-01", "billing_country": "Japan", "amount": 99.5},
            {"invoice_date": "invalid-date", "billing_country": "Japan", "amount": 50.0},  # invalid date
            {"invoice_date": "2024-02-10", "billing_country": "", "amount": 40.0},  # invalid country
        ])
        custom_df.to_csv(csv_file, index=False)

        raw_df = load_raw_data(csv_file)
        cleaned_df, date_col, country_col, rev_col = validate_and_clean_data(
            raw_df,
            date_col="invoice_date",
            country_col="billing_country",
            revenue_col="amount",
        )

        self.assertEqual(date_col, "invoice_date")
        self.assertEqual(country_col, "billing_country")
        self.assertEqual(rev_col, "amount")
        # Only 1 row was valid
        self.assertEqual(len(cleaned_df), 1)
        self.assertEqual(cleaned_df.iloc[0]["clean_country"], "Japan")
        self.assertAlmostEqual(cleaned_df.iloc[0]["clean_revenue"], 99.5)


if __name__ == "__main__":
    unittest.main()

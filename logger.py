"""
Audit and CSV Logging Module for Subscription Revenue Forecasting Service.

Maintains structured CSV audit logs for training runs and prediction calls.
Supports isolation between production and test environments via directories.
"""

import csv
from datetime import datetime, timezone
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PRODUCTION_LOG_DIR = "logs/production"
TEST_LOG_DIR = "logs/test"

TRAINING_LOG_FILE = "training.csv"
PREDICTION_LOG_FILE = "predictions.csv"

TRAINING_HEADERS = [
    "timestamp",
    "country",
    "model_name",
    "holdout_rmse",
    "holdout_mae",
    "baseline_rmse",
    "baseline_mae",
    "version",
    "num_samples",
]

PREDICTION_HEADERS = [
    "timestamp",
    "country",
    "target_date",
    "prediction",
    "model_version",
    "model_name",
]


def _get_log_dir(test: bool = False, custom_dir: Optional[str] = None) -> str:
    """Resolve directory based on test flag or explicit override."""
    if custom_dir:
        return custom_dir
    return TEST_LOG_DIR if test else PRODUCTION_LOG_DIR


def _append_csv_row(file_path: str, headers: List[str], row_data: Dict[str, Any]):
    """Safely append a single row to a CSV file, writing header if new."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path) and os.path.getsize(file_path) > 0

    with open(file_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)


def log_training(
    country: str,
    model_name: str,
    holdout_rmse: float,
    holdout_mae: float,
    baseline_rmse: float,
    baseline_mae: float,
    version: str,
    num_samples: int = 0,
    test: bool = False,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Append a training event to training.csv."""
    target_dir = _get_log_dir(test=test, custom_dir=log_dir)
    file_path = os.path.join(target_dir, TRAINING_LOG_FILE)
    timestamp = datetime.now(timezone.utc).isoformat()

    row = {
        "timestamp": timestamp,
        "country": country,
        "model_name": model_name,
        "holdout_rmse": round(float(holdout_rmse), 4),
        "holdout_mae": round(float(holdout_mae), 4),
        "baseline_rmse": round(float(baseline_rmse), 4),
        "baseline_mae": round(float(baseline_mae), 4),
        "version": version,
        "num_samples": int(num_samples),
    }

    _append_csv_row(file_path, TRAINING_HEADERS, row)
    return row


def log_prediction(
    country: str,
    target_date: str,
    prediction: float,
    model_version: str,
    model_name: str = "",
    test: bool = False,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Append a prediction event to predictions.csv."""
    target_dir = _get_log_dir(test=test, custom_dir=log_dir)
    file_path = os.path.join(target_dir, PREDICTION_LOG_FILE)
    timestamp = datetime.now(timezone.utc).isoformat()

    row = {
        "timestamp": timestamp,
        "country": country,
        "target_date": target_date,
        "prediction": round(float(prediction), 2),
        "model_version": model_version,
        "model_name": model_name,
    }

    _append_csv_row(file_path, PREDICTION_HEADERS, row)
    return row


def get_logs(
    kind: str = "all",
    limit: int = 100,
    test: bool = False,
    log_dir: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Retrieve logged events.
    kind: 'all', 'predictions', or 'training'
    limit: maximum number of records to return (most recent first)
    """
    target_dir = _get_log_dir(test=test, custom_dir=log_dir)
    entries: List[Dict[str, Any]] = []

    if kind in ["all", "training"]:
        train_path = os.path.join(target_dir, TRAINING_LOG_FILE)
        if os.path.exists(train_path):
            with open(train_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_copy = dict(row)
                    row_copy["log_type"] = "training"
                    entries.append(row_copy)

    if kind in ["all", "predictions"]:
        pred_path = os.path.join(target_dir, PREDICTION_LOG_FILE)
        if os.path.exists(pred_path):
            with open(pred_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_copy = dict(row)
                    row_copy["log_type"] = "prediction"
                    entries.append(row_copy)

    # Sort descending by timestamp (newest first)
    entries.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return entries[:limit]

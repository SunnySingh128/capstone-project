"""
Flask API Service for Subscription Revenue Forecasting.

Endpoints:
- GET /health
- POST /train
- GET/POST /predict
- GET /logs
- POST /monitor

Author: Sunny Singh
"""

import logging
import os
from typing import Any, Dict

from flask import Flask, jsonify, request
import pandas as pd

import logger as audit_logger
import model as forecasting_model
import monitoring

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "service": "subscription-revenue-forecast",
        "author": "Sunny Singh",
    }), 200


@app.route("/train", methods=["POST"])
def train():
    """
    Train time-series forecasting models for all countries.
    Accepts optional JSON fields:
    - data_path: path to processed CSV (default: 'data/processed/revenue_timeseries.csv')
    - test: bool (if True, routes artifacts to models/test and logs to logs/test)
    - test_size: float (fraction of series for chronological holdout, default: 0.2)
    - random_state: int (default: 42)
    """
    body: Dict[str, Any] = request.get_json(silent=True) or {}

    data_path = body.get("data_path", "data/processed/revenue_timeseries.csv")
    is_test = bool(body.get("test", False))
    test_size = float(body.get("test_size", 0.2))
    random_state = int(body.get("random_state", 42))

    if not os.path.exists(data_path):
        return jsonify({
            "status": "error",
            "message": f"Data file not found at '{data_path}'. Please run data_ingestion.py first.",
        }), 404

    try:
        df = pd.read_csv(data_path)
        training_results = forecasting_model.train_models(
            df=df,
            test=is_test,
            test_size=test_size,
            random_state=random_state,
        )

        return jsonify({
            "status": "success",
            "mode": "test" if is_test else "production",
            "models_trained": list(training_results.keys()),
            "details": training_results,
        }), 200

    except Exception as e:
        logger.exception("Error during model training")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/predict", methods=["GET", "POST"])
def predict():
    """
    Forecast revenue for a country at a given target date.
    Accepts GET query parameters or POST JSON payload:
    - country: str (e.g. 'India', 'USA', 'all')
    - date: str (e.g. '2028-01-01')
    - test: bool (optional, routes to test models/logs)
    """
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        country = payload.get("country")
        target_date = payload.get("date")
        is_test = bool(payload.get("test", False))
    else:
        country = request.args.get("country")
        target_date = request.args.get("date")
        is_test = request.args.get("test", "false").lower() in ["true", "1", "yes"]

    if not country:
        return jsonify({"status": "error", "message": "Missing required parameter 'country'."}), 400

    if not target_date:
        return jsonify({"status": "error", "message": "Missing required parameter 'date'."}), 400

    try:
        result = forecasting_model.predict(
            country=country,
            date=target_date,
            test=is_test,
        )
        return jsonify(result), 200

    except FileNotFoundError as fnf_err:
        return jsonify({"status": "error", "message": str(fnf_err)}), 404
    except KeyError as key_err:
        return jsonify({"status": "error", "message": str(key_err)}), 404
    except ValueError as val_err:
        return jsonify({"status": "error", "message": str(val_err)}), 400
    except Exception as exc:
        logger.exception("Unexpected error in /predict")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/logs", methods=["GET"])
def logs():
    """
    Retrieve audit logs for predictions and training.
    Query parameters:
    - kind: 'all' (default), 'predictions', or 'training'
    - limit: int (default: 100)
    - test: bool (default: false)
    """
    kind = request.args.get("kind", "all")
    limit_str = request.args.get("limit", "100")
    is_test = request.args.get("test", "false").lower() in ["true", "1", "yes"]

    try:
        limit = max(1, int(limit_str))
    except ValueError:
        limit = 100

    try:
        entries = audit_logger.get_logs(kind=kind, limit=limit, test=is_test)
        return jsonify({
            "status": "success",
            "kind": kind,
            "count": len(entries),
            "logs": entries,
        }), 200
    except Exception as exc:
        logger.exception("Error reading logs")
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/monitor", methods=["POST"])
def monitor():
    """
    Run performance monitoring or standardized mean-shift drift checks.
    Accepts JSON body:
    - country: str (default: 'all')
    - threshold: float (for performance RMSE check or drift Z-score threshold)
    - reference_values: list of floats (optional, for drift check)
    - current_values: list of floats (optional, for drift check)
    - test: bool (optional)
    """
    body = request.get_json(silent=True) or {}
    is_test = bool(body.get("test", False))

    try:
        result = monitoring.monitor_pipeline(payload=body, test=is_test)
        return jsonify({
            "status": "success",
            "monitoring": result,
        }), 200
    except FileNotFoundError as fnf:
        return jsonify({"status": "error", "message": str(fnf)}), 404
    except ValueError as val_err:
        return jsonify({"status": "error", "message": str(val_err)}), 400
    except Exception as exc:
        logger.exception("Error in /monitor")
        return jsonify({"status": "error", "message": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)

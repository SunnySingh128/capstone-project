"""
Model Training, Evaluation, Versioning, and Prediction Module.

Evaluates Naive Last-Observation Baseline, Random Forest, Gradient Boosting,
and Linear Regression on chronological holdout sets using RMSE and MAE.
Selects best ML model (RMSE tie-broken by MAE), refits on full history,
saves timestamped artifacts with latest.json pointer, and supports recursive forecasting.
"""

from datetime import datetime, timezone
import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

import logger as audit_logger

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PRODUCTION_MODEL_DIR = "models/production"
TEST_MODEL_DIR = "models/test"
LATEST_POINTER_FILE = "latest.json"


def _get_model_dir(test: bool = False, custom_dir: Optional[str] = None) -> str:
    """Resolve model artifact directory."""
    if custom_dir:
        return custom_dir
    return TEST_MODEL_DIR if test else PRODUCTION_MODEL_DIR


def build_time_series_features(df_series: pd.DataFrame) -> pd.DataFrame:
    """
    Build lag, rolling, and calendar features for a single country time series.
    Input df_series must have columns: 'date' (datetime) and 'revenue' (float).
    """
    df = df_series.sort_values("date").copy().reset_index(drop=True)
    n = len(df)

    # Trend index
    df["time_idx"] = np.arange(n)

    # Calendar features
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter

    # Lag features
    df["lag_1"] = df["revenue"].shift(1)
    df["lag_2"] = df["revenue"].shift(2)
    df["lag_3"] = df["revenue"].shift(3)
    df["lag_6"] = df["revenue"].shift(6)

    # Rolling statistics
    df["rolling_mean_3"] = df["revenue"].shift(1).rolling(window=3, min_periods=1).mean()

    # Backfill earliest rows with available lags so we don't drop too much data
    df["lag_6"] = df["lag_6"].fillna(df["lag_3"]).fillna(df["lag_2"]).fillna(df["lag_1"])
    df["lag_3"] = df["lag_3"].fillna(df["lag_2"]).fillna(df["lag_1"])
    df["lag_2"] = df["lag_2"].fillna(df["lag_1"])
    df["rolling_mean_3"] = df["rolling_mean_3"].fillna(df["lag_1"])

    # Drop row 0 since lag_1 is NaN
    valid_df = df.dropna(subset=["lag_1"]).reset_index(drop=True)
    return valid_df


FEATURE_COLUMNS = ["time_idx", "month", "year", "quarter", "lag_1", "lag_2", "lag_3", "lag_6", "rolling_mean_3"]


def evaluate_models_on_holdout(
    features_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[str, Any, Dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """
    Chronologically split features into train/test holdout, evaluate candidates,
    and select the best ML model using holdout RMSE then MAE.
    """
    n_samples = len(features_df)
    holdout_count = max(1, int(math.ceil(n_samples * test_size)))

    train_df = features_df.iloc[:-holdout_count].copy()
    test_df = features_df.iloc[-holdout_count:].copy()

    if len(train_df) == 0:
        # Edge case for extremely small datasets: train on all except 1
        train_df = features_df.iloc[:-1].copy()
        test_df = features_df.iloc[-1:].copy()

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df["revenue"]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df["revenue"]

    # 1. Naive Baseline: predicts lag_1
    baseline_pred = test_df["lag_1"].values
    baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
    baseline_mae = float(mean_absolute_error(y_test, baseline_pred))

    # 2. Machine Learning Candidates
    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=random_state),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
    }

    ml_metrics = {}
    best_name = None
    best_model = None
    best_rmse = float("inf")
    best_mae = float("inf")

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        ml_metrics[name] = {"rmse": round(rmse, 4), "mae": round(mae, 4)}

        # Selection rule: lowest RMSE, MAE tie-breaker
        if (rmse < best_rmse - 1e-6) or (abs(rmse - best_rmse) <= 1e-6 and mae < best_mae):
            best_rmse = rmse
            best_mae = mae
            best_name = name
            best_model = model

    # Refit the winning ML model on ALL historical data (train + holdout test)
    X_full = features_df[FEATURE_COLUMNS]
    y_full = features_df["revenue"]
    final_model = candidates[best_name]
    final_model.fit(X_full, y_full)

    summary_metrics = {
        "selected_ml_model": best_name,
        "holdout_rmse": round(best_rmse, 4),
        "holdout_mae": round(best_mae, 4),
        "baseline_rmse": round(baseline_rmse, 4),
        "baseline_mae": round(baseline_mae, 4),
        "all_ml_metrics": ml_metrics,
        "naive_baseline": {"rmse": round(baseline_rmse, 4), "mae": round(baseline_mae, 4)},
        "holdout_size": len(test_df),
        "total_training_samples": len(features_df),
    }

    return best_name, final_model, summary_metrics, train_df, test_df


def save_model_artifact(
    country: str,
    model: Any,
    model_name: str,
    metrics: Dict[str, Any],
    raw_series_df: pd.DataFrame,
    test: bool = False,
    model_dir: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Save timestamped model artifact and update latest.json pointer.
    """
    target_dir = _get_model_dir(test=test, custom_dir=model_dir)
    os.makedirs(target_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    version = f"{country}_{timestamp}"
    artifact_filename = f"model_{country}_{timestamp}.joblib"
    artifact_path = os.path.join(target_dir, artifact_filename)

    # Store series history for recursive forecasting
    history_dates = pd.to_datetime(raw_series_df["date"]).dt.strftime("%Y-%m-%d").tolist()
    history_revenues = raw_series_df["revenue"].astype(float).tolist()

    artifact = {
        "country": country,
        "version": version,
        "model_name": model_name,
        "model": model,
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
        "history": {
            "dates": history_dates,
            "revenues": history_revenues,
        },
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }

    joblib.dump(artifact, artifact_path)

    # Update latest pointer file in model directory
    latest_file = os.path.join(target_dir, LATEST_POINTER_FILE)
    latest_data = {}
    if os.path.exists(latest_file):
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                latest_data = json.load(f)
        except Exception:
            latest_data = {}

    latest_data[country] = {
        "version": version,
        "artifact_path": artifact_path,
        "model_name": model_name,
        "holdout_rmse": metrics.get("holdout_rmse"),
        "holdout_mae": metrics.get("holdout_mae"),
        "saved_at": artifact["saved_at"],
    }

    with open(latest_file, "w", encoding="utf-8") as f:
        json.dump(latest_data, f, indent=2)

    logger.info("Saved model artifact for %s (version=%s) to %s", country, version, artifact_path)
    return version, artifact_path


def load_model(
    country: str,
    test: bool = False,
    model_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Load latest model artifact for a country.
    Usage: load_model('all')['version']
    """
    target_dir = _get_model_dir(test=test, custom_dir=model_dir)
    latest_file = os.path.join(target_dir, LATEST_POINTER_FILE)

    if not os.path.exists(latest_file):
        raise FileNotFoundError(
            f"No trained models found. '{latest_file}' does not exist. Train models first."
        )

    with open(latest_file, "r", encoding="utf-8") as f:
        latest_data = json.load(f)

    if country not in latest_data:
        available = list(latest_data.keys())
        raise KeyError(f"No latest model for country '{country}'. Available: {available}")

    artifact_path = latest_data[country]["artifact_path"]
    if not os.path.exists(artifact_path):
        # Check relative to model directory if moved
        alt_path = os.path.join(target_dir, os.path.basename(artifact_path))
        if os.path.exists(alt_path):
            artifact_path = alt_path
        else:
            raise FileNotFoundError(f"Model artifact file not found: {artifact_path}")

    artifact = joblib.load(artifact_path)
    return artifact


def train_models(
    df: pd.DataFrame,
    test: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    model_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Train, evaluate, and save models for each country in df and for 'all'.
    """
    if "country" not in df.columns or "date" not in df.columns or "revenue" not in df.columns:
        raise ValueError("DataFrame must contain 'date', 'country', and 'revenue' columns.")

    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])
    countries = sorted(data["country"].unique())
    results: Dict[str, Any] = {}

    for country in countries:
        c_df = data[data["country"] == country].sort_values("date").reset_index(drop=True)
        if len(c_df) < 3:
            logger.warning("Skipping %s: insufficient data (%d rows)", country, len(c_df))
            continue

        feat_df = build_time_series_features(c_df)
        best_name, model, metrics, _, _ = evaluate_models_on_holdout(
            feat_df, test_size=test_size, random_state=random_state
        )

        version, artifact_path = save_model_artifact(
            country=country,
            model=model,
            model_name=best_name,
            metrics=metrics,
            raw_series_df=c_df,
            test=test,
            model_dir=model_dir,
        )

        # Log training event to CSV audit log
        audit_logger.log_training(
            country=country,
            model_name=best_name,
            holdout_rmse=metrics["holdout_rmse"],
            holdout_mae=metrics["holdout_mae"],
            baseline_rmse=metrics["baseline_rmse"],
            baseline_mae=metrics["baseline_mae"],
            version=version,
            num_samples=len(feat_df),
            test=test,
            log_dir=log_dir,
        )

        results[country] = {
            "version": version,
            "model_name": best_name,
            "metrics": metrics,
            "artifact_path": artifact_path,
        }

    return results


def predict(
    country: str,
    date: Union[str, pd.Timestamp, datetime],
    test: bool = False,
    model_dir: Optional[str] = None,
    log_dir: Optional[str] = None,
    record_log: bool = True,
) -> Dict[str, Any]:
    """
    Forecast revenue for a country at a given future date.
    Performs recursive multi-step forecasting from the last historical date.
    """
    target_dt = pd.to_datetime(date)
    artifact = load_model(country=country, test=test, model_dir=model_dir)

    model = artifact["model"]
    model_name = artifact["model_name"]
    version = artifact["version"]
    history = artifact["history"]

    hist_dates = [pd.to_datetime(d) for d in history["dates"]]
    hist_revenues = [float(r) for r in history["revenues"]]

    if not hist_dates:
        raise ValueError(f"Model history for country '{country}' is empty.")

    last_date = hist_dates[-1]

    # Detect step frequency (monthly vs daily)
    if len(hist_dates) > 1:
        median_delta_days = np.median([(hist_dates[i] - hist_dates[i - 1]).days for i in range(1, len(hist_dates))])
        is_monthly = median_delta_days >= 25
    else:
        is_monthly = True

    # If target date is equal to or before the last date, return or predict directly
    if target_dt <= last_date:
        # If in history, return historical value
        for d, r in zip(hist_dates, hist_revenues):
            if d.strftime("%Y-%m-%d") == target_dt.strftime("%Y-%m-%d"):
                prediction_val = float(r)
                break
        else:
            prediction_val = float(hist_revenues[-1])
    else:
        # Step forward recursively from last historical point
        curr_dates = list(hist_dates)
        curr_revs = list(hist_revenues)
        curr_step = last_date

        prediction_val = None
        while curr_step < target_dt:
            if is_monthly:
                # Next month start
                next_step = (curr_step + pd.DateOffset(months=1)).replace(day=1)
            else:
                next_step = curr_step + pd.DateOffset(days=1)

            curr_step = next_step
            n = len(curr_revs)
            time_idx = n
            month = curr_step.month
            year = curr_step.year
            quarter = curr_step.quarter

            lag_1 = curr_revs[-1]
            lag_2 = curr_revs[-2] if n >= 2 else lag_1
            lag_3 = curr_revs[-3] if n >= 3 else lag_2
            lag_6 = curr_revs[-6] if n >= 6 else lag_3
            rolling_mean_3 = float(np.mean(curr_revs[-3:]))

            features = pd.DataFrame([{
                "time_idx": time_idx,
                "month": month,
                "year": year,
                "quarter": quarter,
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "lag_6": lag_6,
                "rolling_mean_3": rolling_mean_3,
            }])[FEATURE_COLUMNS]

            step_pred = float(model.predict(features)[0])
            # Ensure non-negative revenue forecast
            step_pred = max(0.0, step_pred)
            curr_dates.append(curr_step)
            curr_revs.append(step_pred)

            if curr_step >= target_dt:
                prediction_val = step_pred
                break

        if prediction_val is None:
            prediction_val = curr_revs[-1]

    formatted_target = target_dt.strftime("%Y-%m-%d")
    prediction_val = round(float(prediction_val), 2)

    # Log prediction to CSV
    if record_log:
        audit_logger.log_prediction(
            country=country,
            target_date=formatted_target,
            prediction=prediction_val,
            model_version=version,
            model_name=model_name,
            test=test,
            log_dir=log_dir,
        )

    return {
        "status": "success",
        "country": country,
        "date": formatted_target,
        "prediction": prediction_val,
        "model_version": version,
        "model_name": model_name,
    }

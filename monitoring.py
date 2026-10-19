"""
Monitoring Module for Subscription Revenue Forecasting Service.

Provides:
1. Performance monitoring using the model's holdout metrics against a threshold.
2. Standardized mean-shift drift check between reference and current distribution values.
"""

import logging
from typing import Any, Dict, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


def check_mean_shift_drift(
    reference_values: List[float],
    current_values: List[float],
    country: str = "all",
    threshold: float = 2.0,
) -> Dict[str, Any]:
    """
    Standardized mean-shift drift check:
    Z_shift = |mean(current) - mean(reference)| / (std(reference))
    """
    if not reference_values or len(reference_values) == 0:
        raise ValueError("reference_values must contain at least one value.")
    if not current_values or len(current_values) == 0:
        raise ValueError("current_values must contain at least one value.")

    ref_arr = np.array(reference_values, dtype=float)
    curr_arr = np.array(current_values, dtype=float)

    ref_mean = float(np.mean(ref_arr))
    ref_std = float(np.std(ref_arr, ddof=1)) if len(ref_arr) > 1 else float(np.std(ref_arr))
    curr_mean = float(np.mean(curr_arr))

    # Guard against zero variance
    effective_std = ref_std if ref_std > 1e-8 else 1e-8
    standardized_shift = float(abs(curr_mean - ref_mean) / effective_std)
    drift_detected = standardized_shift > threshold

    return {
        "check_type": "data_drift",
        "country": country,
        "drift_detected": bool(drift_detected),
        "standardized_shift": round(standardized_shift, 4),
        "reference_mean": round(ref_mean, 4),
        "reference_std": round(ref_std, 4),
        "current_mean": round(curr_mean, 4),
        "threshold": float(threshold),
        "status": "drift_detected" if drift_detected else "normal",
        "message": (
            f"Mean shift of {standardized_shift:.2f} standard deviations "
            f"{'exceeds' if drift_detected else 'is within'} threshold of {threshold:.2f}."
        ),
    }


def check_model_performance(
    country: str,
    threshold: float,
    test: bool = False,
    model_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Performance monitoring using the trained model's holdout RMSE vs a specified threshold.
    """
    # Import model lazily to avoid circular dependencies
    from model import load_model

    model_artifact = load_model(country=country, test=test, model_dir=model_dir)
    metrics = model_artifact.get("metrics", {})
    holdout_rmse = metrics.get("holdout_rmse", 0.0)
    holdout_mae = metrics.get("holdout_mae", 0.0)
    version = model_artifact.get("version", "unknown")
    model_name = model_artifact.get("model_name", "unknown")

    performance_degraded = holdout_rmse > threshold

    return {
        "check_type": "model_performance",
        "country": country,
        "model_version": version,
        "model_name": model_name,
        "holdout_rmse": round(float(holdout_rmse), 4),
        "holdout_mae": round(float(holdout_mae), 4),
        "threshold": float(threshold),
        "performance_degraded": bool(performance_degraded),
        "status": "alert" if performance_degraded else "healthy",
        "message": (
            f"Holdout RMSE ({holdout_rmse:.2f}) "
            f"{'exceeds' if performance_degraded else 'meets'} acceptable threshold ({threshold:.2f})."
        ),
    }


def monitor_pipeline(
    payload: Dict[str, Any],
    test: bool = False,
    model_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Main monitoring entrypoint accepting JSON payload from /monitor endpoint.
    Payload can specify:
    - country (default: 'all')
    - threshold (float)
    - reference_values (list of floats)
    - current_values (list of floats)
    """
    country = payload.get("country", "all")
    ref_vals = payload.get("reference_values")
    curr_vals = payload.get("current_values")
    threshold = payload.get("threshold")

    results: Dict[str, Any] = {"country": country}

    # If drift check values are provided
    if ref_vals is not None and curr_vals is not None:
        drift_threshold = float(threshold) if threshold is not None else 2.0
        drift_res = check_mean_shift_drift(
            reference_values=ref_vals,
            current_values=curr_vals,
            country=country,
            threshold=drift_threshold,
        )
        results["drift_check"] = drift_res
        results["status"] = drift_res["status"]

    # If performance check threshold is provided without explicit sample arrays
    elif threshold is not None:
        perf_res = check_model_performance(
            country=country,
            threshold=float(threshold),
            test=test,
            model_dir=model_dir,
        )
        results["performance_check"] = perf_res
        results["status"] = perf_res["status"]

    else:
        raise ValueError(
            "Invalid monitor payload. Provide 'threshold' for performance monitoring, "
            "or 'reference_values' and 'current_values' for drift monitoring."
        )

    return results

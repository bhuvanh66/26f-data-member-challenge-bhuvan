"""
Evaluating the models.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score


def dollar_metrics(y_true_log, y_pred_log) -> dict:
    """Metrics in dollars, from predictions made on the log scale."""
    y_true_log = np.asarray(y_true_log, dtype=float)
    y_pred_log = np.asarray(y_pred_log, dtype=float)
    actual, pred = np.exp(y_true_log), np.exp(y_pred_log)
    err = pred - actual
    ratio = pred / actual
    return {
        "n": int(len(actual)),
        "mdape_pct": float(np.median(np.abs(ratio - 1)) * 100), # Median Absolute Percentage Error
        "mae_usd": float(np.mean(np.abs(err))),
        "medae_usd": float(np.median(np.abs(err))),
        "rmse_usd": float(np.sqrt(np.mean(err ** 2))),
        "within_25pct": float(np.mean(np.abs(ratio - 1) <= 0.25) * 100),
        "within_50pct": float(np.mean(np.abs(ratio - 1) <= 0.50) * 100),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
        "r2_usd": float(r2_score(actual, pred)),
        "mae_log": float(np.mean(np.abs(y_true_log - y_pred_log))),
        # >1 means the model reads high overall; a systematic tilt shows up here and
        # nowhere else, because MdAPE is an absolute error and hides direction
        "median_ratio": float(np.median(ratio)),
    }



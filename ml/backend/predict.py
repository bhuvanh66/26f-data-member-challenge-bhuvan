"""
Prediction logic for the saved model. Used by `ml/backend/api.py`; there is no CLI here —
the frontend is the interface.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from ml.pipeline.clean import INPUT_COLS

ARTIFACT = "ml/model.joblib"


def load_artifact(path: str = ARTIFACT) -> dict:
    return joblib.load(path)


def predict_one(artifact: dict, answers: dict) -> dict:
    """One row of raw survey answers -> predicted salary + prediction interval."""
    X = pd.DataFrame([{c: answers.get(c) for c in INPUT_COLS}])
    pred_log = float(artifact["pipeline"].predict(X)[0])
    q = artifact["log_residual_quantiles"]
    return {
        "predicted_usd": float(np.exp(pred_log)),
        "interval_80_usd": (float(np.exp(pred_log + q[0.10])), float(np.exp(pred_log + q[0.90]))),
        "interval_50_usd": (float(np.exp(pred_log + q[0.25])), float(np.exp(pred_log + q[0.75]))),
    }

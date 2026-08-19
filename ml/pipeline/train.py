"""
Train, validate, and save the salary model.

  .venv/bin/python -m ml.pipeline.train

Steps:

  1. Clean the raw file (row-level: target gate)
  2. Split into a training 80% and a held-out test 20%, stratified by country so a
     thin country doesn't land entirely on one side.
  3. Fit hist_gbdt and the two baselines on the training 80%, then evaluate all three
     once against the held-out 20%. 
  4. Refit hist_gbdt on 100% of the rows and save that as the artifact, because the
     deployed model should use every row available. Its expected error is the
     step-3 number.
"""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from ml.pipeline.clean import (
    COUNTRY_RENAME, INPUT_COLS, RAW_CSV, TEXT_INPUTS, country_strata, load_raw, training_frame,
)
from ml.evaluation.evaluate import dollar_metrics
from ml.evaluation.figures import predicted_vs_actual_figure
from ml.pipeline.models import RANDOM_STATE, candidates

OUT = Path("ml/outputs")
FIGDIR = OUT / "figures"
ARTIFACT = Path("ml/model.joblib")
TEST_SIZE = 0.20
MODEL_NAME = "hist_gbdt"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- 1. clean ----
    data = training_frame(RAW_CSV)
    X, y, y_log = data["X"], data["y"], data["y_log"]
    print(f"{RAW_CSV} -> {data['n_raw']:,} raw rows -> {len(X):,} usable rows")

    # ---- 2. train/test split ----
    strata = country_strata(X)
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                              stratify=strata)
    Xtr, Xte = X.iloc[tr].reset_index(drop=True), X.iloc[te].reset_index(drop=True)
    ytr, yte = y_log.to_numpy()[tr], y_log.to_numpy()[te]

    # ---- 3. hist_gbdt vs. the baselines ----
    scores = {}
    for name, est in candidates().items():
        fitted = clone(est).fit(Xtr, ytr)
        p = fitted.predict(Xte)
        scores[name] = dollar_metrics(yte, p)
        if name == MODEL_NAME:
            best, pte = fitted, p
        print(f"  {name:15s} MdAPE {scores[name]['mdape_pct']:5.1f}%  "
              f"MAE ${scores[name]['mae_usd']:,.0f}")

    fig_path = predicted_vs_actual_figure(yte, pte, FIGDIR)

    # ---- 4. refit on everything + save ----
    final = clone(best).fit(X, y_log)
    resid = yte - pte     # held-out, so the interval is not measured on training fits
    q = {p: float(np.quantile(resid, p)) for p in (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)}
    country_clean = X["Country"].replace(COUNTRY_RENAME)
    artifact = {
        "pipeline": final,
        "model_name": MODEL_NAME,
        "input_cols": INPUT_COLS,
        # every level seen in the raw file, most common first: predict.py offers these as
        # the menu, so a typo becomes an unseen category instead of a silent wrong answer
        "choices": {c: load_raw(RAW_CSV)[c].dropna().str.replace("’", "'", regex=False)
                    .str.replace(r"\s+", " ", regex=True).str.strip()
                    .value_counts().index.tolist()
                    for c in TEXT_INPUTS},
        "log_residual_quantiles": q,
        "country_reference": pd.DataFrame({
            "country": country_clean, "salary": y}).groupby("country")["salary"]
            .agg(["median", "size"]).to_dict("index"),
        "meta": {"trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "n_rows_fitted": int(len(X)), "raw_csv": str(RAW_CSV),
                 "sklearn": sklearn.__version__, "pandas": pd.__version__,
                 "python": platform.python_version(), "target": "log(annual_salary_usd)"},
    }
    joblib.dump(artifact, ARTIFACT, compress=3)

    (OUT / "metrics.json").write_text(json.dumps({
        "model": MODEL_NAME,
        "test_scores": scores,
    }, indent=2, default=float))

    print(f"\nsaved {ARTIFACT}, {OUT/'metrics.json'}, {fig_path} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

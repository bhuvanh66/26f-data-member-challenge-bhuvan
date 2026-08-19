"""
HTTP wrapper around the saved model, for the frontend in `frontend/`.

  .venv/bin/uvicorn ml.backend.api:app --reload --port 8000

Three endpoints:
  GET  /meta      the control ranges and dropdown choices the frontend needs to render
  POST /predict    raw survey answers -> predicted salary + interval (ml/backend/predict.py logic)
  GET  /context    where that prediction sits against real respondents in the same
                    country, from the cleaned survey rows directly
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml.pipeline.clean import AGE_MID, COUNTRY_RENAME, ED_ORDINAL, ORG_MID
from ml.backend.predict import load_artifact, predict_one

app = FastAPI(title="salary-predictor")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

ARTIFACT = load_artifact()

# The cleaned survey rows, for the real salary distribution behind the "Data" tab —
# this is the same file ml/pipeline/train.py trained on, not something recomputed on the fly.
SURVEY = pd.read_csv("data/survey_clean.csv")[["country_clean", "annual_salary_usd"]]

# Ordered slider options for the fields that have a natural order. Country and DevType
# are dropdowns instead — they have no order sliders could sensibly express.
AGE_OPTIONS = list(AGE_MID.keys())
ED_OPTIONS = list(ED_ORDINAL.keys())
ORG_SIZE_OPTIONS = list(ORG_MID.keys())
REMOTE_OPTIONS = [
    "In-person",
    "Hybrid (some remote, leans heavy to in-person)",
    "Hybrid (some in-person, leans heavy to flexibility)",
    "Remote",
]
IC_OR_PM_OPTIONS = ["Individual contributor", "People manager"]


class PredictRequest(BaseModel):
    years_code: float | None = None
    work_exp: float | None = None
    age: str | None = None
    ed_level: str | None = None
    org_size: str | None = None
    remote_work: str | None = None
    ic_or_pm: str | None = None
    country: str | None = None
    dev_type: str | None = None


@app.get("/meta")
def meta() -> dict:
    return {
        "age_options": AGE_OPTIONS,
        "ed_level_options": ED_OPTIONS,
        "org_size_options": ORG_SIZE_OPTIONS,
        "remote_options": REMOTE_OPTIONS,
        "ic_or_pm_options": IC_OR_PM_OPTIONS,
        "country_options": ARTIFACT["choices"]["Country"],
        "dev_type_options": ARTIFACT["choices"]["DevType"],
        "years_code_max": 50,
        "work_exp_max": 50,
    }


@app.post("/predict")
def predict(req: PredictRequest) -> dict:
    answers = {
        "YearsCode": req.years_code,
        "WorkExp": req.work_exp,
        "Age": req.age,
        "EdLevel": req.ed_level,
        "OrgSize": req.org_size,
        "RemoteWork": req.remote_work,
        "ICorPM": req.ic_or_pm,
        "Country": req.country,
        "DevType": req.dev_type,
        # every OrgSize/ICorPM answer implies employment; the model only reads
        # employment for respondents the survey never asked these of
        "Employment": "Employed" if req.org_size else None,
    }
    result = predict_one(ARTIFACT, answers)
    return {
        "predicted_usd": round(result["predicted_usd"]),
        "interval_50_usd": [round(v) for v in result["interval_50_usd"]],
        "interval_80_usd": [round(v) for v in result["interval_80_usd"]],
        "model_name": ARTIFACT["model_name"],
    }


@app.get("/context")
def context(country: str, predicted_usd: float) -> dict:
    country = COUNTRY_RENAME.get(country, country)

    rows = SURVEY.loc[SURVEY["country_clean"] == country, "annual_salary_usd"]
    country_block = None
    if len(rows) >= 5:
        salaries = np.sort(rows.to_numpy())
        percentile = float(np.searchsorted(salaries, predicted_usd) / len(salaries) * 100)
        # log-spaced bins: salary is right-skewed, so equal-width bins would put
        # almost every respondent in the first one and tell you nothing
        edges = np.geomspace(salaries.min(), salaries.max(), num=9)
        counts, _ = np.histogram(salaries, bins=edges)
        country_block = {
            "name": country, "n": int(len(salaries)),
            "median_usd": round(float(np.median(salaries))),
            "percentile": round(percentile, 1),
            "histogram": {"edges": [round(e) for e in edges], "counts": counts.tolist()},
        }

    return {"country": country_block}

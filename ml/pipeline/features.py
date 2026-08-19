"""
Feature engineering: raw survey columns -> model matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.pipeline.clean import (
    AGE_MID, COUNTRY_RENAME, COUNTRY_TO_REGION, DEVTYPE_TO_FAMILY, ED_ORDINAL,
    EMP_GROUP, HARD_CURRENCIES, KEPT_COUNTRIES, KEPT_DEVTYPES, MAX_YEARS,
    MIN_CODE_START_AGE, MIN_WORK_START_AGE, ORG_MID, REMOTE_ORD, ensure_schema,
    normalise_text,
)


CATEGORICAL_FEATURES = [
    "country_grouped",     # 44 levels: countries with n>=30, rest as "Other: <region>"
    "region",              # 11 levels: the coarse geography fallback
    "employment_group",
    "ed_level_clean",
    "org_size_cat",
    "ic_or_pm",
    "remote_cat",          # all five raw levels, including both Hybrid phrasings
    "industry_clean",
    "dev_type_grouped",
    "role_family",
]



def _age_and_experience(X: pd.DataFrame) -> dict:
    age_mid = X["Age"].map(AGE_MID)
    cols = {"age_mid": age_mid}
    # Two failure modes repaired, not deleted: the literal 100 (one respondent
    # claims 100 years of both, in the 35-44 age bucket), and values implying someone
    # started coding at 4 or working at 12.
    for col, min_start in [("YearsCode", MIN_CODE_START_AGE), ("WorkExp", MIN_WORK_START_AGE)]:
        v = pd.to_numeric(X[col], errors="coerce")
        impossible = (v > MAX_YEARS) | ((age_mid - v) < min_start)  # NaN compares False
        cols[col] = v.where(~impossible)
    return cols


def _education(X: pd.DataFrame) -> dict:
    return {"ed_ordinal": X["EdLevel"].map(ED_ORDINAL),          # 1-7, ordered
            "ed_level_clean": X["EdLevel"].fillna("Unknown")}


def _employment_and_org(X: pd.DataFrame) -> dict:
    """
    Everything gated by the survey's skip-logic block.

    OrgSize/ICorPM/RemoteWork/Industry are blank together for anyone the survey never
    asked (freelancers, retirees, the not-employed) — that's `struct` below. Kept as an
    explicit "Not applicable" level rather than imputed, because a modal office size on
    a retiree would invent data.
    """
    emp = X["Employment"]
    struct = X[["OrgSize", "ICorPM", "RemoteWork"]].isna().all(axis=1)

    org_size_log = np.log10(X["OrgSize"].map(ORG_MID))
    return {
        "has_employer": emp.eq("Employed").astype(float),
        "job_block_missing": struct.astype(float),
        "org_size_log": org_size_log,
        "is_manager": X["ICorPM"].map({"People manager": 1.0, "Individual contributor": 0.0}),
        "remote_ordinal": X["RemoteWork"].map(REMOTE_ORD).astype(float),
        "employment_group": emp.map(EMP_GROUP).fillna("Unknown"),
        "org_size_cat": np.where(struct, "Not applicable", X["OrgSize"].fillna("Unknown")),
        "ic_or_pm": np.where(struct, "Not applicable", X["ICorPM"].fillna("Unknown")),
        "remote_cat": np.where(struct, "Not applicable", X["RemoteWork"].fillna("Unknown")),
        "industry_clean": np.where(struct & X["Industry"].isna(), "Not applicable",
                                   X["Industry"].fillna("Unknown")),
    }


def _geography(X: pd.DataFrame) -> dict:
    country = X["Country"].replace(COUNTRY_RENAME)
    region = country.map(COUNTRY_TO_REGION).fillna("Unknown")

    country_grouped = np.where(country.isin(KEPT_COUNTRIES), country, "Other: " + region)
    return {
        "country_grouped": country_grouped,
        "region": region,
        # Currency is not redundant with country: in Ukraine, being paid in USD/EUR
        # rather than the local currency roughly doubles the median.
        "paid_in_hard_currency": X["currency_code"].isin(HARD_CURRENCIES).astype(float),
    }


def _role(X: pd.DataFrame) -> dict:
    dev_clean = X["DevType"].fillna("Unknown")
    dev_grouped = np.where(dev_clean.isin(KEPT_DEVTYPES), dev_clean, "Other role")
    fallback = np.where(dev_clean.str.startswith("Developer"), "Engineering", "Other")
    role_family = dev_clean.map(DEVTYPE_TO_FAMILY).fillna(pd.Series(fallback, index=X.index))
    return {"dev_type_grouped": dev_grouped, "role_family": role_family}


def build_features(X: pd.DataFrame) -> pd.DataFrame:
    """Raw survey columns -> model matrix. Same output for one row or a million."""
    Xn = normalise_text(ensure_schema(X))
    cols = {
        **_age_and_experience(Xn),
        **_education(Xn),
        **_employment_and_org(Xn),
        **_geography(Xn),
        **_role(Xn),
    }
    f = pd.DataFrame(cols, index=Xn.index)

    # non-null so the encoder never sees a NaN level; everything else is numeric
    for c in CATEGORICAL_FEATURES:
        f[c] = f[c].astype(object).fillna("Unknown").astype(str)
    for c in f.columns:
        if c not in CATEGORICAL_FEATURES:
            f[c] = pd.to_numeric(f[c], errors="coerce").astype(float)
    return f

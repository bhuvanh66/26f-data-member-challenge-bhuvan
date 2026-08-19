"""
Clean row-level data
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_CSV = Path("data/survey.csv")

# ---- tunable
SALARY_FLOOR = 1_000
SALARY_CEIL = 1_000_000
MAX_YEARS = 60
MIN_CODE_START_AGE = 5
MIN_WORK_START_AGE = 14
RARE_COUNTRY_MIN = 30    # documents the threshold KEPT_COUNTRIES below was computed with
RARE_DEVTYPE_MIN = 50    # documents the threshold KEPT_DEVTYPES below was computed with


KEPT_COUNTRIES = [
    "Australia", "Austria", "Belgium", "Brazil", "Canada", "Czech Republic", "Denmark",
    "Finland", "France", "Germany", "Greece", "Hungary", "India", "Ireland", "Israel",
    "Italy", "Mexico", "Netherlands", "New Zealand", "Norway", "Poland", "Portugal",
    "Romania", "Russia", "South Africa", "Spain", "Sweden", "Switzerland", "Turkey",
    "Ukraine", "United Kingdom", "United States",
]
KEPT_DEVTYPES = [
    "AI/ML engineer", "Architect, software or solutions", "Data engineer", "Data scientist",
    "DevOps engineer or professional", "Developer, back-end",
    "Developer, desktop or enterprise applications",
    "Developer, embedded applications or devices", "Developer, front-end",
    "Developer, full-stack", "Developer, mobile", "Engineering manager", "Other",
    "Senior executive (C-suite, VP, etc.)",
]

TARGET = "annual_salary_usd"
ID_COL = "ResponseId"


TEXT_INPUTS = [
    "Age", "EdLevel", "Employment", "DevType", "OrgSize", "ICorPM", "RemoteWork",
    "Industry", "Country", "Currency",
]
NUM_INPUTS = ["WorkExp", "YearsCode"]
INPUT_COLS = TEXT_INPUTS + NUM_INPUTS


# ============================================================ lookup tables
AGE_MID = {
    "18-24 years old": 21.0, "25-34 years old": 29.5, "35-44 years old": 39.5,
    "45-54 years old": 49.5, "55-64 years old": 59.5, "65 years or older": 70.0,
}

ED_ORDINAL = {
    "Primary/elementary school": 1,
    "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 2,
    "Some college/university study without earning a degree": 3,
    "Associate degree (A.A., A.S., etc.)": 4,
    "Bachelor's degree (B.A., B.S., B.Eng., etc.)": 5,
    "Master's degree (M.A., M.S., M.Eng., MBA, etc.)": 6,
    "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 7,
}

EMP_GROUP = {
    "Employed": "Employed",
    "Independent contractor, freelancer, or self-employed": "Self-employed",
    "Student": "Student",
    "Not employed": "Not employed",
    "Retired": "Retired",
    "I prefer not to say": "Undisclosed",
}

ORG_MID = {
    "Just me - I am a freelancer, sole proprietor, etc.": 1.0,
    "Less than 20 employees": 10.0,
    "20 to 99 employees": 60.0,
    "100 to 499 employees": 300.0,
    "500 to 999 employees": 750.0,
    "1,000 to 4,999 employees": 3000.0,
    "5,000 to 9,999 employees": 7500.0,
    "10,000 or more employees": 15000.0,
}


REMOTE_ORD = {
    "In-person": 0,
    "Hybrid (some remote, leans heavy to in-person)": 1,
    "Hybrid (some in-person, leans heavy to flexibility)": 2,
    "Remote": 3,
    "Your choice (very flexible, you can come in when you want or just as needed)": 3,
}

COUNTRY_RENAME = {
    "United States of America": "United States",
    "United Kingdom of Great Britain and Northern Ireland": "United Kingdom",
    "Russian Federation": "Russia",
    "Iran, Islamic Republic of...": "Iran",
    "Venezuela, Bolivarian Republic of...": "Venezuela",
    "Republic of Moldova": "Moldova",
    "Republic of North Macedonia": "North Macedonia",
    "United Republic of Tanzania": "Tanzania",
    "Viet Nam": "Vietnam",
    "Hong Kong (S.A.R.)": "Hong Kong",
    "Republic of Korea": "South Korea",
    "Syrian Arab Republic": "Syria",
}

REGIONS = {
    "North America": ["United States", "Canada"],
    "Latin America": ["Brazil", "Mexico", "Argentina", "Colombia", "Chile", "Venezuela",
                      "Paraguay", "Bolivia", "Peru", "Uruguay", "Honduras", "Dominican Republic",
                      "Costa Rica", "Guatemala", "El Salvador", "Cuba", "Jamaica", "Nicaragua",
                      "Trinidad and Tobago"],
    "Western/Northern Europe": ["Germany", "United Kingdom", "France", "Netherlands", "Sweden",
                                "Switzerland", "Austria", "Denmark", "Belgium", "Ireland",
                                "Finland", "Norway", "Iceland", "Luxembourg", "Isle of Man"],
    "Southern Europe": ["Italy", "Spain", "Portugal", "Greece", "Cyprus", "Malta"],
    "Eastern Europe/Central Asia": ["Poland", "Ukraine", "Czech Republic", "Hungary", "Romania",
                                    "Russia", "Serbia", "Bulgaria", "Croatia", "Slovakia",
                                    "Slovenia", "Estonia", "Lithuania", "Latvia", "Belarus",
                                    "Bosnia and Herzegovina", "Moldova", "North Macedonia",
                                    "Albania", "Montenegro", "Georgia", "Armenia", "Azerbaijan",
                                    "Kazakhstan", "Kyrgyzstan", "Uzbekistan"],
    "Middle East/North Africa": ["Israel", "Turkey", "Iran", "United Arab Emirates", "Lebanon",
                                 "Morocco", "Algeria", "Tunisia", "Egypt", "Saudi Arabia",
                                 "Qatar", "Kuwait", "Jordan", "Iraq", "Syria", "Yemen", "Sudan"],
    "Sub-Saharan Africa": ["South Africa", "Nigeria", "Kenya", "Cameroon", "Namibia", "Zimbabwe",
                           "Rwanda", "Mauritius", "Ethiopia", "Ghana", "Madagascar", "Senegal",
                           "Lesotho", "Malawi", "Mozambique", "Togo", "Tanzania", "Angola",
                           "Mauritania", "Cote d'Ivoire", "Côte d'Ivoire"],
    "South Asia": ["India", "Bangladesh", "Pakistan", "Sri Lanka", "Nepal", "Maldives"],
    "East/Southeast Asia": ["China", "Japan", "Taiwan", "South Korea", "North Korea", "Hong Kong",
                            "Singapore", "Vietnam", "Indonesia", "Philippines", "Malaysia",
                            "Thailand", "Cambodia", "Myanmar", "Mongolia"],
    "Oceania": ["Australia", "New Zealand"],
}
COUNTRY_TO_REGION = {c: r for r, cs in REGIONS.items() for c in cs}

ROLE_FAMILY = {
    "Data/AI": ["AI/ML engineer", "Data engineer", "Data scientist", "Applied scientist",
                "Data or business analyst", "Developer, AI apps or physical AI",
                "Database administrator or engineer", "Academic researcher"],
    "Management": ["Engineering manager", "Senior executive (C-suite, VP, etc.)",
                   "Product manager", "Project manager", "Founder, technology or otherwise"],
    "Infrastructure": ["DevOps engineer or professional", "Cloud infrastructure engineer",
                       "System administrator", "Cybersecurity or InfoSec professional"],
}
DEVTYPE_TO_FAMILY = {d: f for f, ds in ROLE_FAMILY.items() for d in ds}

HARD_CURRENCIES = ["USD", "EUR", "GBP", "CHF"]


# ============================================================ load + hygiene
def load_raw(path: str | Path = RAW_CSV) -> pd.DataFrame:
    """
    Read the raw survey file.

    'NA' is the only missing-value token in this file (verified in 01_profile_raw.py).
    keep_default_na=False stops pandas applying its wider default list, which would be
    guesswork on a file whose sentinel we already know — and it stops a country or a
    language literally named "None"/"NaN" from silently becoming a null.
    """
    df = pd.read_csv(path, na_values=["NA"], keep_default_na=False, dtype=str)
    for c in [ID_COL, "WorkExp", "YearsCode", TARGET]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ensure_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee the 14 input columns exist, in a known order.

    A prediction request may only supply Country and YearsCode. Rather than special-case
    that, missing fields are added as NaN and flow through the same missing-value
    handling as a respondent who skipped the question.
    """
    out = df.copy()
    for c in INPUT_COLS:
        if c not in out.columns:
            out[c] = np.nan
    for c in NUM_INPUTS:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def normalise_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Text hygiene, then derive currency_code / currency_name.

    Three real problems in this file, all of which break lookup keys silently:
      - a literal TAB inside Currency ("AUD\\tAustralian dollar") in 1,133 of 5,000 rows,
        where every other row uses a space
      - curly apostrophes (Master's with U+2019) in 3,794 rows
      - "Other (please specify):" style labels, which are the same category as "Other"
    """
    out = df.copy()
    for c in TEXT_INPUTS:
        s = out[c]
        if s.isna().all():
            out[c] = s.astype(object)
            continue
        s = s.astype(object).where(s.notna(), None)
        s = pd.Series([x if x is None else str(x) for x in s], index=out.index, dtype=object)
        s = s.str.replace("’", "'", regex=False)              # curly -> ASCII apostrophe
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()     # tabs/newlines/doubles -> one space
        out[c] = s.replace({"": None})

    # "AUD Australian dollar" -> ("AUD", "Australian dollar"). Only possible after the
    # tab above has become a space, otherwise 1,133 rows split differently to the rest.
    out["currency_code"] = out["Currency"].str.split(" ", n=1).str[0]
    out["currency_name"] = out["Currency"].str.split(" ", n=1).str[1]

    for c in ["EdLevel", "DevType", "Industry"]:
        out[c] = out[c].str.replace(r"^Other.*$", "Other", regex=True)
    return out


# ============================================================ row-level filtering
def training_frame(path: str | Path = RAW_CSV) -> dict:
    """
    Raw CSV -> (X, y) ready to split, plus a record of everything removed.

    Rows are removed for one reason only: a target that is missing, below $1,000, or
    above $1,000,000.

    Returned dict keys:
      X        4,783 x 14 raw predictors (the model's input schema)
      y        annual_salary_usd in dollars
      y_log    log(annual_salary_usd) — what the model actually fits
      dropped  every removed row with the reason
      log      the actions, for reporting
    """
    raw = load_raw(path)
    n_start = len(raw)
    # ensure_schema keeps ResponseId and the target alongside the 14 inputs; they are
    # needed for the target gate, then dropped from X.
    df = normalise_text(ensure_schema(raw))
    actions: list[dict] = []
    dropped: list[pd.DataFrame] = []

    assert df[ID_COL].is_unique, "ResponseId is not unique"

    sal = df[TARGET]
    drop_any = pd.Series(False, index=df.index)
    for mask, reason in [
        (sal.isna(), "target missing (annual_salary_usd = 'NA')"),
        (sal < SALARY_FLOOR, f"target below plausibility floor (< ${SALARY_FLOOR:,})"),
        (sal > SALARY_CEIL, f"target above plausibility ceiling (> ${SALARY_CEIL:,})"),
    ]:
        mask = mask & ~drop_any
        if mask.any():
            d = df[mask].copy()
            d["drop_reason"] = reason
            dropped.append(d)
            actions.append({"action": f"dropped rows: {reason}", "n": int(mask.sum())})
        drop_any |= mask

    keep = ~drop_any
    df = df[keep].copy()
    actions.append({"action": "rows retained", "n": len(df)})

    return {
        "X": df[INPUT_COLS].copy(),
        "y": df[TARGET].astype(float),
        "y_log": np.log(df[TARGET].astype(float)),
        "ids": df[ID_COL],
        "dropped": pd.concat(dropped, ignore_index=True) if dropped else pd.DataFrame(),
        "log": actions,
        "n_raw": n_start,
    }


def country_strata(X: pd.DataFrame, min_count: int = 25) -> pd.Series:
    """
    Labels for the stratified train/test split.

    Rows are stratified on country where the
    country has enough respondents to land on both sides, and on region otherwise.
    """
    country = X["Country"].replace(COUNTRY_RENAME)
    region = country.map(COUNTRY_TO_REGION).fillna("Unknown")
    counts = country.map(country.value_counts())
    strata = pd.Series(np.where(counts >= min_count, country, "region: " + region),
                       index=X.index, dtype=object)
    # At least one country has to be "rare" for the stratification to work. If the rare
    # bucket is too small, it is merged into the most common non-rare bucket.
    thin = strata.map(strata.value_counts()) < 2
    strata = strata.where(~thin, "rare")
    counts = strata.value_counts()
    if counts.get("rare", 0) < 2:
        strata = strata.replace({"rare": counts.drop(labels=["rare"], errors="ignore").idxmax()})
    return strata

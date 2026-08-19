"""
The models this pipeline fits, each as a single end-to-end pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from ml.pipeline.clean import COUNTRY_RENAME, COUNTRY_TO_REGION
from ml.pipeline.features import CATEGORICAL_FEATURES, build_features

RANDOM_STATE = 42


HGB_PARAMS = dict(
    categorical_features="from_dtype", learning_rate=0.06, max_leaf_nodes=31,
    min_samples_leaf=20, l2_regularization=1.0, max_iter=400, early_stopping=True,
    n_iter_no_change=25, validation_fraction=0.15, random_state=RANDOM_STATE,
)


# ============================================================ baselines
class CountryMedianRegressor(BaseEstimator, RegressorMixin):
    """
    Predict the median log-salary of the respondent's country.

    This is the baseline the model has to beat, not a strawman. It sits directly on the
    raw columns — it needs only Country — and falls back through region to the global
    median for a country hich is what a random split will
    hand it for the thin countries.
    """

    def fit(self, X: pd.DataFrame, y):
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        country = X["Country"].replace(COUNTRY_RENAME)
        region = country.map(COUNTRY_TO_REGION).fillna("Unknown")
        self.country_median_ = y.groupby(country).median().to_dict()
        self.region_median_ = y.groupby(region).median().to_dict()
        self.global_median_ = float(y.median())
        return self

    def predict(self, X: pd.DataFrame):
        country = X["Country"].replace(COUNTRY_RENAME)
        region = country.map(COUNTRY_TO_REGION).fillna("Unknown")
        out = country.map(self.country_median_)
        out = out.fillna(region.map(self.region_median_)).fillna(self.global_median_)
        return out.to_numpy(dtype=float)


class CategoryCaster(BaseEstimator, TransformerMixin):
    """
    Cast the categorical columns to pandas `category` with categories fixed at fit time.
    """

    def fit(self, X: pd.DataFrame, y=None):
        self.categories_ = {c: pd.Index(sorted(X[c].dropna().unique()))
                            for c in CATEGORICAL_FEATURES if c in X.columns}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for c, cats in self.categories_.items():
            out[c] = pd.Categorical(out[c], categories=cats)
        return out


# ============================================================ candidates
def candidates() -> dict[str, Pipeline | BaseEstimator]:
    """name -> unfitted estimator, all taking the 12 raw columns and predicting log salary."""
    return {
        "global_median": DummyRegressor(strategy="median"),
        "country_median": CountryMedianRegressor(),
        "hist_gbdt": Pipeline([
            ("features", FunctionTransformer(build_features)),
            ("cast", CategoryCaster()),
            ("model", HistGradientBoostingRegressor(**HGB_PARAMS)),
        ]),
    }

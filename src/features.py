"""
features.py — Feature engineering and preprocessing pipeline
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


NUMERIC_FEATURES = [
    "age",
    "los_days",
    "baseline_creatinine",
    "baseline_hemoglobin",
    "egfr",
    "mean_map",
    "min_map",
    "pct_hypotensive",
    "mean_hr",
    "n_icu_stays",
    "icu_los_days",
]

CATEGORICAL_FEATURES = [
    "gender",
    "admission_type",
    "insurance",
]

TARGET = "aki_label"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features on top of the raw cohort."""
    df = df.copy()

    # Flag: missing hemodynamics (patient not in ICU)
    df["has_icu_data"] = (df["mean_map"] != -1).astype(int)

    # Replace -1 sentinel with NaN for proper imputation
    sentinel_cols = ["mean_map", "min_map", "pct_hypotensive", "mean_hr"]
    for col in sentinel_cols:
        df[col] = df[col].replace(-1, np.nan)

    # CKD flag from eGFR
    df["ckd_flag"] = (df["egfr"] < 60).astype(int)

    # Severe AKI flag (stage 2+)
    df["severe_aki"] = (df["aki_stage"] >= 2).astype(int)

    return df


def build_preprocessor() -> ColumnTransformer:
    """Sklearn ColumnTransformer: impute + scale numeric, OHE categorical."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipe,      NUMERIC_FEATURES),
        ("cat", categorical_pipe,  CATEGORICAL_FEATURES),
    ])


def get_X_y(df: pd.DataFrame):
    """Return feature matrix X and target vector y."""
    df = engineer_features(df)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ["has_icu_data", "ckd_flag"]
    X = df[feature_cols].copy()
    y = df[TARGET].copy()
    return X, y

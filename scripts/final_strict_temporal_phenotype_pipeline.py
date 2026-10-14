from __future__ import annotations


import numpy as np


import pandas as pd


STRESS_FEATURE_SIGNS = {
    "lactate_mean": 1.0,
    "lactate_last": 1.0,
    "glucose_mean": 1.0,
    "ph_mean": -1.0,
    "be_mean": -1.0,
    "hco3_act_mean": -1.0,
}


def laboratory_stress_score(imputed_features: pd.DataFrame) -> pd.Series:
    """Prespecified laboratory-only axis used to give GMM components stable names.

    Higher values represent greater metabolic stress: higher lactate/glucose and
    lower pH/base excess/bicarbonate. Outcomes and clinical covariates are not
    used in this mapping.
    """
    terms = []
    for feature, sign in STRESS_FEATURE_SIGNS.items():
        values = imputed_features[feature].astype(float)
        scale = values.std(ddof=0)
        if not np.isfinite(scale) or scale <= 1e-12:
            continue
        terms.append(sign * (values - values.mean()) / scale)
    if not terms:
        raise ValueError("No prespecified stress-score features were available.")
    return pd.concat(terms, axis=1).mean(axis=1)

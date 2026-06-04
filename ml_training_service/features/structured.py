"""
Structured feature engineering: derive additional tabular features from raw columns.
"""
import numpy as np
import pandas as pd

TABULAR_FEATURE_COLS = [
    "category_id",
    "goal",
    "descr_len",
    "title_len",
    "primary_phone_checks__line_type",
    "identity_check_score",
    "primary_email_address_checks__is_disposable",
    "primary_email_address_checks__email_domain_creation_days",
    "log_goal",
    "email_trust_score",
    "text_ratio",
    "identity_norm",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered tabular features. Input df must be preprocessed."""
    df = df.copy()

    df["log_goal"] = np.log1p(df["goal"].clip(lower=0))

    domain_days = df["primary_email_address_checks__email_domain_creation_days"].clip(lower=0)
    is_disposable = df["primary_email_address_checks__is_disposable"].fillna(0)
    df["email_trust_score"] = domain_days / (is_disposable + 1.0)

    title_len = df["title_len"].clip(lower=1)
    df["text_ratio"] = df["descr_len"] / title_len

    df["identity_norm"] = df["identity_check_score"] / 100.0

    return df


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the model-ready tabular feature columns."""
    available = [c for c in TABULAR_FEATURE_COLS if c in df.columns]
    return df[available].copy()

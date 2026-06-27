"""
data_prep.py
Loads the raw Telco Customer Churn dataset, cleans it, engineers a few
features, and saves a processed version ready for modeling.

Usage:
    python src/data_prep.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path("data/raw/Telco-Customer-Churn.csv")
PROCESSED_PATH = Path("data/processed/churn_processed.csv")


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Load the raw CSV."""
    df = pd.read_csv(path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning: fix dtypes, handle missing values."""
    df = df.copy()

    # TotalCharges is read as object because a handful of rows (tenure == 0,
    # i.e. brand-new customers) have a blank string instead of a number.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Target to binary int (only present for labeled/training data)
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Drop the ID column, it carries no signal
    df = df.drop(columns=["customerID"])

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few derived features that help tell a richer story."""
    df = df.copy()

    # Average monthly spend over the customer's lifetime
    df["avg_monthly_spend"] = np.where(
        df["tenure"] > 0, df["TotalCharges"] / df["tenure"], df["MonthlyCharges"]
    )

    # Tenure buckets — easier to read in EDA than raw months
    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=[-1, 6, 12, 24, 48, 72],
        labels=["0-6mo", "6-12mo", "1-2yr", "2-4yr", "4-6yr"],
    )

    # Count of add-on services each customer has (signal for "stickiness")
    service_cols = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_addon_services"] = (df[service_cols] == "Yes").sum(axis=1)

    return df


def encode(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns, leave numeric ones as-is."""
    df = df.copy()
    categorical_cols = df.select_dtypes(include=["object", "str", "category"]).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df


def main():
    print(f"Loading raw data from {RAW_PATH} ...")
    df = load_raw()
    print(f"Raw shape: {df.shape}")

    df = clean(df)
    df = engineer_features(df)

    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Save a pre-encoding version too — useful for EDA notebooks
    df.to_csv(PROCESSED_PATH.with_name("churn_clean.csv"), index=False)

    df_encoded = encode(df)
    df_encoded.to_csv(PROCESSED_PATH, index=False)

    print(f"Saved cleaned data to {PROCESSED_PATH.with_name('churn_clean.csv')}")
    print(f"Saved encoded data to {PROCESSED_PATH}")
    print(f"Final encoded shape: {df_encoded.shape}")


if __name__ == "__main__":
    main()

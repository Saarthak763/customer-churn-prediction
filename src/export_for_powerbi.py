"""
export_for_powerbi.py
Builds a single, human-readable CSV combining the original (readable)
customer fields with the trained model's churn probability and a risk
segment label. Designed to be dropped straight into Power BI — no
one-hot encoded columns, no scaling, just clean categorical + numeric
fields a BI tool can slice and dice.

Usage:
    python src/export_for_powerbi.py
"""

import glob
import joblib
import pandas as pd
from pathlib import Path

from data_prep import load_raw, clean, engineer_features, encode

RAW_PATH = Path("data/raw/Telco-Customer-Churn.csv")
OUT_PATH = Path("data/processed/churn_powerbi.csv")
MODEL_DIR = Path("models")


def risk_segment(prob: float) -> str:
    if prob >= 0.5:
        return "High"
    elif prob >= 0.25:
        return "Medium"
    return "Low"


def main():
    # Readable version (no one-hot encoding) for display in Power BI
    raw = load_raw(RAW_PATH)
    readable = clean(raw)
    readable = engineer_features(readable)

    # Encoded version (matches what the model was trained on) for scoring
    encoded = encode(readable)

    feature_cols = joblib.load(MODEL_DIR / "feature_columns.pkl")
    for col in feature_cols:
        if col not in encoded.columns:
            encoded[col] = 0
    X = encoded[feature_cols]

    model_path = glob.glob(str(MODEL_DIR / "best_model_*.pkl"))[0]
    model = joblib.load(model_path)

    if "LogisticRegression" in str(type(model)):
        scaler = joblib.load(MODEL_DIR / "scaler.pkl")
        X_for_predict = scaler.transform(X)
    else:
        X_for_predict = X

    probs = model.predict_proba(X_for_predict)[:, 1]

    # Build the final export: readable fields + predictions
    export = readable.copy()
    export["churn_probability"] = probs.round(4)
    export["risk_segment"] = [risk_segment(p) for p in probs]
    export["actual_churn"] = export["Churn"].map({1: "Yes", 0: "No"})
    export = export.drop(columns=["Churn"])

    # Re-attach customerID for readability (was dropped in clean())
    export.insert(0, "customerID", raw["customerID"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    export.to_csv(OUT_PATH, index=False)
    print(f"Saved Power BI export to {OUT_PATH} ({export.shape[0]} rows, {export.shape[1]} cols)")
    print(f"\nRisk segment breakdown:\n{export['risk_segment'].value_counts()}")


if __name__ == "__main__":
    main()

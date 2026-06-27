"""
predict.py
Loads the saved model and scores new customer records.

Usage (as a script):
    python src/predict.py

Usage (as a module, e.g. from a Streamlit app):
    from src.predict import load_artifacts, predict_batch
    model, scaler, feature_cols = load_artifacts()
"""

import glob
import joblib
import pandas as pd
from pathlib import Path

from data_prep import clean, engineer_features, encode

MODEL_DIR = Path("models")


def load_artifacts():
    model_path = glob.glob(str(MODEL_DIR / "best_model_*.pkl"))[0]
    model = joblib.load(model_path)
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_columns.pkl")
    return model, scaler, feature_cols


def preprocess_new(raw_df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Run a raw (uncleaned) dataframe through the same pipeline as training,
    then align columns to match what the model expects."""
    df = clean(raw_df) if "customerID" in raw_df.columns else raw_df
    df = engineer_features(df)
    df_encoded = encode(df)

    # Add any missing one-hot columns as 0, drop any extras, preserve order
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[feature_cols]
    return df_encoded


def predict_batch(raw_df: pd.DataFrame) -> pd.Series:
    model, scaler, feature_cols = load_artifacts()
    X = preprocess_new(raw_df, feature_cols)

    # Logistic regression needs scaled input; tree models don't, but scaling
    # doesn't hurt them, so we only scale if the loaded model is logreg.
    if "LogisticRegression" in str(type(model)):
        X = scaler.transform(X)

    probs = model.predict_proba(X)[:, 1]
    return pd.Series(probs, name="churn_probability")


if __name__ == "__main__":
    # Quick smoke test using a slice of the raw data
    sample = pd.read_csv("data/raw/Telco-Customer-Churn.csv").sample(5, random_state=1)
    probs = predict_batch(sample.drop(columns=["Churn"]))
    print(pd.concat([sample[["customerID"]].reset_index(drop=True), probs], axis=1))

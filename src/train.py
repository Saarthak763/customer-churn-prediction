"""
train.py
Trains and compares three models on the processed churn data:
Logistic Regression (baseline), Random Forest, and XGBoost.
Saves the best model (by ROC-AUC) to models/.

Usage:
    python src/train.py
"""

import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

from evaluate import metrics_table, plot_roc_curves

PROCESSED_PATH = Path("data/processed/churn_processed.csv")
MODEL_DIR = Path("models")
RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(PROCESSED_PATH)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    return X, y


def split_and_scale(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler


def train_logreg(X_train_scaled, y_train):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train_scaled, y_train)
    return model


def train_random_forest(X_train, y_train):
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [4, 6, 8, 10, None],
        "min_samples_split": [2, 5, 10],
        "class_weight": ["balanced"],
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=param_dist,
        n_iter=15,
        scoring="roc_auc",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"Best RF params: {search.best_params_}")
    return search.best_estimator_


def train_gradient_boosting(X_train, y_train):
    """HistGradientBoostingClassifier — scikit-learn's native histogram-based
    gradient boosting. Comparable to XGBoost/LightGBM in performance, ships
    with sklearn so there's no extra dependency to install."""
    # Class weighting: HGB doesn't take scale_pos_weight directly, so we
    # pass sample_weight at fit time instead.
    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    sample_weight = y_train.map({1: neg / pos, 0: 1.0}).values

    param_dist = {
        "max_iter": [100, 200, 300],
        "max_depth": [3, 4, 5, 6, None],
        "learning_rate": [0.01, 0.05, 0.1],
        "l2_regularization": [0.0, 0.1, 1.0],
        "max_leaf_nodes": [15, 31, 63],
    }
    base_model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        base_model,
        param_distributions=param_dist,
        n_iter=15,
        scoring="roc_auc",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train, sample_weight=sample_weight)
    print(f"Best HGB params: {search.best_params_}")
    return search.best_estimator_


def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"\n--- {name} ---")
    print(f"ROC-AUC: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    return auc


def main():
    X, y = load_data()
    (
        X_train, X_test,
        X_train_scaled, X_test_scaled,
        y_train, y_test, scaler,
    ) = split_and_scale(X, y)

    results = {}

    logreg = train_logreg(X_train_scaled, y_train)
    results["logreg"] = (logreg, evaluate("Logistic Regression", logreg, X_test_scaled, y_test))

    rf = train_random_forest(X_train, y_train)
    results["rf"] = (rf, evaluate("Random Forest", rf, X_test, y_test))

    xgb_model = train_gradient_boosting(X_train, y_train)
    results["hgb"] = (xgb_model, evaluate("HistGradientBoosting", xgb_model, X_test, y_test))

    best_name = max(results, key=lambda k: results[k][1])
    best_model, best_auc = results[best_name]
    print(f"\nBest model: {best_name} (ROC-AUC = {best_auc:.4f})")

    # Build a combined metrics table + ROC plot using the right X for each model
    model_inputs = {
        "Logistic Regression": (logreg, X_test_scaled),
        "Random Forest": (rf, X_test),
        "HistGradientBoosting": (xgb_model, X_test),
    }
    table = metrics_table(model_inputs, y_test)
    print("\nMetrics comparison:")
    print(table.to_string(index=False))
    table.to_csv("reports/metrics_comparison.csv", index=False)
    plot_roc_curves(model_inputs, y_test)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(best_model, MODEL_DIR / f"best_model_{best_name}.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")  # needed if best model is logreg
    joblib.dump(list(X.columns), MODEL_DIR / "feature_columns.pkl")
    print(f"Saved best model to {MODEL_DIR / f'best_model_{best_name}.pkl'}")


if __name__ == "__main__":
    main()

"""
evaluate.py
Standalone evaluation utilities — reusable from notebooks or train.py.
Produces a metrics table and saves an ROC curve figure.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_score,
    recall_score, f1_score, accuracy_score,
)

FIGURES_DIR = Path("reports/figures")


def metrics_table(models: dict, y_test) -> pd.DataFrame:
    """
    models: dict of {name: (fitted_model, X_test_for_this_model)}
    Returns a tidy DataFrame comparing Accuracy / Precision / Recall / F1 / ROC-AUC.
    """
    rows = []
    for name, (model, X) in models.items():
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]
        rows.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        })
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)


def plot_roc_curves(models: dict, y_test, save_path: Path = None):
    """
    models: dict of {name: (fitted_model, X_test_for_this_model)}
    Plots ROC curves for all models on one chart.
    """
    plt.figure(figsize=(7, 6))
    for name, (model, X) in models.items():
        y_proba = model.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Model Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()

    save_path = save_path or (FIGURES_DIR / "roc_curves.png")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"Saved ROC curve plot to {save_path}")
    plt.close()

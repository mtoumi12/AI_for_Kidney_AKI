"""
models.py — Model definitions, training, evaluation, and calibration
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score, brier_score_loss, classification_report,
    RocCurveDisplay, ConfusionMatrixDisplay
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in all environments


def get_models():
    """Return dict of base estimators (without preprocessor)."""
    return {
        "Logistic Regression": LogisticRegression(
            C=0.1, max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42
        ),
    }


def build_pipeline(preprocessor, estimator) -> Pipeline:
    return Pipeline([("prep", preprocessor), ("clf", estimator)])


def evaluate(model, X_test, y_test, name: str) -> dict:
    """Compute AUROC, Brier score, and print classification report."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auroc  = roc_auc_score(y_test, y_prob)
    brier  = brier_score_loss(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"  AUROC : {auroc:.3f}")
    print(f"  Brier : {brier:.3f}  (0=perfect, 0.25=baseline)")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No AKI", "AKI"]))

    return {"name": name, "auroc": auroc, "brier": brier,
            "y_prob": y_prob, "y_pred": y_pred}


def calibrate(pipeline, X_train, y_train) -> CalibratedClassifierCV:
    """
    Isotonic calibration — best for large datasets (Niculescu-Mizil & Caruana, ICML 2005).
    Uses cross-validated calibration to avoid data leakage.
    """
    cal = CalibratedClassifierCV(pipeline, method="isotonic", cv=5)
    cal.fit(X_train, y_train)
    return cal


def plot_calibration(results: list, y_test, save_path: str = None):
    """Reliability diagram for all models."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for r in results:
        frac_pos, mean_pred = calibration_curve(y_test, r["y_prob"], n_bins=8)
        ax.plot(mean_pred, frac_pos, marker="o", label=f"{r['name']} (Brier={r['brier']:.3f})")

    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives (observed AKI rate)")
    ax.set_title("Calibration Curves — MIMIC-IV AKI Prediction")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()


def plot_roc(results: list, y_test, save_path: str = None):
    """ROC curves for all models."""
    from sklearn.metrics import roc_curve, auc
    fig, ax = plt.subplots(figsize=(7, 6))

    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
        ax.plot(fpr, tpr, lw=2, label=f"{r['name']} (AUC={r['auroc']:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — MIMIC-IV AKI Prediction")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")
    plt.close()

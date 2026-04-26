"""
explain.py — SHAP-based model interpretability
"""

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_feature_names(preprocessor) -> list:
    """Recover human-readable feature names after ColumnTransformer."""
    num_names = preprocessor.transformers_[0][2]
    ohe       = preprocessor.transformers_[1][1].named_steps["ohe"]
    cat_names = list(ohe.get_feature_names_out(
        preprocessor.transformers_[1][2]
    ))
    return num_names + cat_names


def shap_summary(pipeline, X_train, X_test, save_path: str = None):
    """
    Global SHAP summary plot.
    Uses TreeExplainer for tree-based models, LinearExplainer for LR.
    """
    prep       = pipeline.named_steps["prep"]
    clf        = pipeline.named_steps["clf"]
    feature_names = get_feature_names(prep)

    X_train_t = prep.transform(X_train)
    X_test_t  = prep.transform(X_test)

    try:
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_t)
        # For binary classifiers, shap_values may be a list [neg, pos]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    except Exception:
        explainer   = shap.LinearExplainer(clf, X_train_t)
        shap_values = explainer.shap_values(X_test_t)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_test_t,
        feature_names=feature_names,
        show=False, max_display=15
    )
    plt.title("SHAP Feature Importance — AKI Prediction", pad=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()

    # Return mean |SHAP| as a tidy DataFrame
    importance = pd.DataFrame({
        "feature":    feature_names,
        "mean_shap":  np.abs(shap_values).mean(axis=0)
    }).sort_values("mean_shap", ascending=False).reset_index(drop=True)
    return importance


def shap_waterfall_single(pipeline, X_test, patient_idx: int,
                          save_path: str = None):
    """Local explanation for a single patient (waterfall plot)."""
    prep          = pipeline.named_steps["prep"]
    clf           = pipeline.named_steps["clf"]
    feature_names = get_feature_names(prep)
    X_test_t      = prep.transform(X_test)

    try:
        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_test_t)
        base_value  = explainer.expected_value
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
            base_value  = base_value[1]
    except Exception:
        print("Waterfall plot requires a tree-based model — skipping.")
        return

    explanation = shap.Explanation(
        values       = shap_values[patient_idx],
        base_values  = base_value,
        data         = X_test_t[patient_idx],
        feature_names= feature_names
    )
    plt.figure()
    shap.plots.waterfall(explanation, show=False, max_display=12)
    plt.title(f"Patient explanation (index {patient_idx})", pad=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.close()

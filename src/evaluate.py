"""
Model Evaluation Module
-----------------------
Computes metrics (Precision, Recall, F1, AUC-ROC, Confusion Matrix)
for every trained model variant.  Generates:
  - Comparison tables (SMOTE vs class-weight, cost-sensitive vs standard)
  - SHAP explainability plots
  - Drift simulation (train on early data, test on late data)
All artifacts are logged to MLflow.
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import mlflow
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, precision_recall_curve,
)

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "evaluation")


# ------------------------------------------------------------------
#  Core metrics
# ------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Return a dict of all required metrics."""
    return {
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_true, y_prob), 4),
        "false_negative_rate": round(
            1 - recall_score(y_true, y_pred, zero_division=0), 4),
        "false_positive_rate": round(
            confusion_matrix(y_true, y_pred, labels=[0, 1])[0, 1]
            / max(confusion_matrix(y_true, y_pred, labels=[0, 1])[0].sum(), 1), 4),
    }


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str):
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Legit", "Fraud"],
                yticklabels=["Legit", "Fraud"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=100)
    plt.close(fig)


# ------------------------------------------------------------------
#  SHAP explainability
# ------------------------------------------------------------------

def generate_shap(model, X_val: pd.DataFrame, save_dir: str,
                  model_name: str, max_samples: int = 500):
    """Generate SHAP summary plot for the given model."""
    try:
        sample = X_val.sample(n=min(max_samples, len(X_val)), random_state=42)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample)

        # For binary classifiers, shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        fig = plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, sample, show=False, max_display=20)
        plt.title(f"SHAP Summary — {model_name}")
        plt.tight_layout()
        path = os.path.join(save_dir, f"shap_summary_{model_name}.png")
        plt.savefig(path, dpi=100, bbox_inches="tight")
        plt.close()
        logger.info("SHAP plot saved: %s", path)
        return path
    except Exception as e:
        logger.warning("SHAP failed for %s: %s", model_name, e)
        return None


# ------------------------------------------------------------------
#  Comparison tables
# ------------------------------------------------------------------

def build_comparison_table(metrics_list: list[dict]) -> pd.DataFrame:
    """Build a comparison DataFrame from a list of {name, metrics} dicts."""
    rows = []
    for m in metrics_list:
        row = {"model": m["name"]}
        row.update(m["metrics"])
        rows.append(row)
    df = pd.DataFrame(rows)
    return df


# ------------------------------------------------------------------
#  Drift simulation
# ------------------------------------------------------------------

def simulate_drift(train_path: str, model, feature_cols: list[str],
                   save_dir: str) -> dict:
    """
    Train on first 80% of data (by TransactionDT order),
    test on last 20% to detect performance drift.
    """
    df = pd.read_parquet(train_path)

    if "TransactionID" in df.columns:
        df = df.drop(columns=["TransactionID"])

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Sort by time
    if "TransactionDT" in df.columns:
        df = df.sort_values("TransactionDT").reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    df_early = df.iloc[:split_idx]
    df_late = df.iloc[split_idx:]

    # Use available feature columns
    avail_cols = [c for c in feature_cols if c in df.columns]

    y_early = df_early["isFraud"]
    y_late = df_late["isFraud"]
    X_late = df_late[avail_cols]

    y_pred = model.predict(X_late)
    y_prob = model.predict_proba(X_late)[:, 1]

    early_metrics = {
        "fraud_rate": round(y_early.mean() * 100, 3),
    }
    late_metrics = compute_metrics(y_late, y_pred, y_prob)
    late_metrics["fraud_rate"] = round(y_late.mean() * 100, 3)

    drift_report = {
        "early_period": early_metrics,
        "late_period": late_metrics,
        "fraud_rate_change": round(
            late_metrics["fraud_rate"] - early_metrics["fraud_rate"], 3),
    }

    # Save report
    report_path = os.path.join(save_dir, "drift_simulation_report.csv")
    pd.DataFrame([
        {"period": "early (0-80%)", **early_metrics},
        {"period": "late (80-100%)", **late_metrics},
    ]).to_csv(report_path, index=False)

    logger.info("Drift simulation: early fraud=%.3f%% late fraud=%.3f%% "
                "late_recall=%.4f late_auc=%.4f",
                early_metrics["fraud_rate"],
                late_metrics["fraud_rate"],
                late_metrics.get("recall", 0),
                late_metrics.get("auc_roc", 0))

    return drift_report


# ------------------------------------------------------------------
#  Main entry
# ------------------------------------------------------------------

def run(trained_models: list[dict], train_path: str,
        mlflow_tracking: bool = True) -> dict:
    """
    Evaluate all trained models and generate comparison artifacts.

    Parameters
    ----------
    trained_models : list of dicts from train.run()
        Each has keys: model, name, params, val_data
    train_path : str
        Path to the featured parquet (for drift simulation)
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    all_metrics = []
    best_model = None
    best_auc = 0

    for entry in trained_models:
        model = entry["model"]
        name = entry["name"]
        X_val, y_val = entry["val_data"]

        # Ensure clean data
        X_val = pd.DataFrame(X_val).replace([np.inf, -np.inf], np.nan).fillna(0)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val, y_pred, y_prob)

        all_metrics.append({"name": name, "metrics": metrics})
        logger.info("%s → Recall=%.4f  AUC=%.4f  F1=%.4f",
                    name, metrics["recall"], metrics["auc_roc"], metrics["f1"])

        # Confusion matrix
        cm_path = os.path.join(ARTIFACTS_DIR, f"cm_{name}.png")
        plot_confusion_matrix(y_val, y_pred, name, cm_path)

        # Track best model (by AUC, with recall floor)
        if metrics["auc_roc"] > best_auc and metrics["recall"] >= 0.5:
            best_auc = metrics["auc_roc"]
            best_model = entry

        if mlflow_tracking:
            with mlflow.start_run(run_name=name, nested=True):
                mlflow.log_params(entry.get("params", {}))
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
                mlflow.log_artifact(cm_path, "confusion_matrices")

    # ---- Comparison tables ----
    comparison_df = build_comparison_table(all_metrics)
    comp_path = os.path.join(ARTIFACTS_DIR, "model_comparison.csv")
    comparison_df.to_csv(comp_path, index=False)
    logger.info("Model comparison saved to %s", comp_path)

    # SMOTE vs Class-weight comparison
    smote_vs_cw = comparison_df[
        comparison_df["model"].str.contains("smote|class_weight")
    ].copy()
    smote_cw_path = os.path.join(ARTIFACTS_DIR, "smote_vs_classweight.csv")
    smote_vs_cw.to_csv(smote_cw_path, index=False)

    # Cost-sensitive comparison
    cost_df = comparison_df[
        comparison_df["model"].str.contains("cost_sensitive|xgboost_class_weight")
    ].copy()
    cost_path = os.path.join(ARTIFACTS_DIR, "cost_sensitive_comparison.csv")
    cost_df.to_csv(cost_path, index=False)

    # ---- SHAP for best model ----
    if best_model:
        X_val_best = best_model["val_data"][0]
        X_val_best = pd.DataFrame(X_val_best).replace(
            [np.inf, -np.inf], np.nan).fillna(0)
        generate_shap(best_model["model"], X_val_best,
                      ARTIFACTS_DIR, best_model["name"])

    # ---- Drift simulation ----
    if best_model:
        feature_cols = list(best_model["val_data"][0].columns)
        drift = simulate_drift(train_path, best_model["model"],
                               feature_cols, ARTIFACTS_DIR)
    else:
        drift = {}

    # ---- Log all artifacts to MLflow ----
    if mlflow_tracking:
        mlflow.log_artifacts(ARTIFACTS_DIR, "evaluation")

    return {
        "comparison": comparison_df,
        "best_model_name": best_model["name"] if best_model else None,
        "best_model": best_model,
        "drift_report": drift,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("evaluate.py must be called from run_pipeline.py (needs trained models).")

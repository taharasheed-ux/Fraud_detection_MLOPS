"""
Conditional Deployment Module
-----------------------------
Deploys the best model ONLY if it meets the performance thresholds:
  - Recall > 0.80
  - AUC-ROC > 0.85

If thresholds are met, the model is:
  1. Registered in the MLflow Model Registry
  2. Exported locally for API consumption
"""

import os
import logging
import joblib
import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)

DEPLOY_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "deployed_model")

RECALL_THRESHOLD = 0.80
AUC_THRESHOLD = 0.85


def run(eval_results: dict, mlflow_tracking: bool = True) -> dict:
    """
    Conditionally deploy the best model.

    Parameters
    ----------
    eval_results : dict from evaluate.run()
        Must contain 'best_model' and 'comparison' DataFrame.

    Returns
    -------
    dict with 'deployed' (bool), 'model_name', 'reason'.
    """
    best = eval_results.get("best_model")

    if best is None:
        logger.warning("No best model found — deployment REJECTED")
        return {"deployed": False, "model_name": None,
                "reason": "No model available"}

    # Look up metrics from comparison table
    comparison = eval_results["comparison"]
    row = comparison[comparison["model"] == best["name"]]
    if row.empty:
        return {"deployed": False, "model_name": best["name"],
                "reason": "Metrics not found in comparison table"}

    recall = float(row["recall"].iloc[0])
    auc = float(row["auc_roc"].iloc[0])

    logger.info("Deployment gate — %s: recall=%.4f (threshold=%.2f) "
                "auc=%.4f (threshold=%.2f)",
                best["name"], recall, RECALL_THRESHOLD, auc, AUC_THRESHOLD)

    if recall > RECALL_THRESHOLD and auc > AUC_THRESHOLD:
        # ---- Deploy ----
        os.makedirs(DEPLOY_DIR, exist_ok=True)

        # Save locally
        model_path = os.path.join(DEPLOY_DIR, "model.joblib")
        joblib.dump(best["model"], model_path)

        # Save feature list
        feature_cols = list(best["val_data"][0].columns)
        joblib.dump(feature_cols,
                    os.path.join(DEPLOY_DIR, "feature_columns.joblib"))

        # Save metadata
        meta = {
            "model_name": best["name"],
            "recall": recall,
            "auc_roc": auc,
            "params": best.get("params", {}),
        }
        joblib.dump(meta, os.path.join(DEPLOY_DIR, "model_meta.joblib"))

        if mlflow_tracking:
            # Register in MLflow Model Registry
            try:
                mlflow.sklearn.log_model(
                    best["model"],
                    artifact_path="deployed_model",
                    registered_model_name="fraud-detection-champion",
                )
                logger.info("Model registered in MLflow as "
                            "'fraud-detection-champion'")
            except Exception as e:
                logger.warning("MLflow model registration failed: %s", e)

            mlflow.log_metric("deployed_recall", recall)
            mlflow.log_metric("deployed_auc", auc)
            mlflow.set_tag("deployment_status", "DEPLOYED")

        logger.info("✅ Model DEPLOYED: %s (recall=%.4f, auc=%.4f)",
                    best["name"], recall, auc)
        return {"deployed": True, "model_name": best["name"],
                "reason": "Passed deployment gate"}

    else:
        reason_parts = []
        if recall <= RECALL_THRESHOLD:
            reason_parts.append(f"recall={recall:.4f} ≤ {RECALL_THRESHOLD}")
        if auc <= AUC_THRESHOLD:
            reason_parts.append(f"auc={auc:.4f} ≤ {AUC_THRESHOLD}")
        reason = "Failed: " + ", ".join(reason_parts)

        if mlflow_tracking:
            mlflow.set_tag("deployment_status", "REJECTED")
            mlflow.log_param("rejection_reason", reason)

        logger.warning("❌ Model REJECTED: %s — %s", best["name"], reason)
        return {"deployed": False, "model_name": best["name"],
                "reason": reason}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("deploy.py must be called from run_pipeline.py (needs eval results).")

"""
Model Training Module
---------------------
Trains XGBoost, LightGBM, and Random Forest (hybrid with feature selection)
models.  For each model, trains variants with:
  - SMOTE oversampling
  - Class weighting
  - Cost-sensitive learning (XGBoost only)

All runs are logged to MLflow.  Retry logic wraps each training call.
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from tenacity import retry, stop_after_attempt, wait_fixed

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "models")
RANDOM_STATE = 42


# ------------------------------------------------------------------
#  Data helpers
# ------------------------------------------------------------------

def prepare_data(train_path: str, val_size: float = 0.2):
    """Load featured data, split into X/y train/val."""
    df = pd.read_parquet(train_path)

    # Drop ID column
    if "TransactionID" in df.columns:
        df = df.drop(columns=["TransactionID"])

    y = df["isFraud"]
    X = df.drop(columns=["isFraud"])

    # Replace inf values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # Ensure all float32 to save memory
    for col in X.select_dtypes(include=["float64"]).columns:
        X[col] = X[col].astype("float32")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d samples | Val: %d samples | Fraud rate: %.3f%%",
                len(X_train), len(X_val), y_train.mean() * 100)
    return X_train, X_val, y_train, y_val


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series,
                max_samples: int = 200_000):
    """
    Apply SMOTE oversampling.  If the dataset is too large, subsample
    the majority class first to keep memory usage manageable.
    """
    if len(X_train) > max_samples:
        logger.info("Subsampling majority class to %d before SMOTE", max_samples)
        df_tmp = X_train.copy()
        df_tmp["__target"] = y_train.values
        fraud = df_tmp[df_tmp["__target"] == 1]
        legit = df_tmp[df_tmp["__target"] == 0].sample(
            n=max_samples - len(fraud), random_state=RANDOM_STATE
        )
        df_tmp = pd.concat([fraud, legit])
        y_sub = df_tmp["__target"]
        X_sub = df_tmp.drop(columns=["__target"])
    else:
        X_sub, y_sub = X_train, y_train

    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_sub, y_sub)
    logger.info("SMOTE: %d → %d samples", len(X_sub), len(X_res))
    return X_res, y_res


# ------------------------------------------------------------------
#  Feature selection for hybrid model
# ------------------------------------------------------------------

def select_features(model, X: pd.DataFrame, top_k: int = 50):
    """Select top-K features by importance from a fitted tree model."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_k]
    selected = [X.columns[i] for i in indices]
    logger.info("Selected top-%d features for hybrid model", top_k)
    return selected


# ------------------------------------------------------------------
#  Model training (with retry)
# ------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def train_xgboost(X_train, y_train, X_val, y_val,
                  variant: str = "class_weight",
                  cost_sensitive: bool = False) -> dict:
    """Train XGBoost with class weighting or SMOTE-preprocessed data."""
    fraud_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "eval_metric": "aucpr",
        "use_label_encoder": False,
        "n_jobs": -1,
        "tree_method": "hist",
    }

    if variant == "class_weight" or cost_sensitive:
        params["scale_pos_weight"] = 10 if cost_sensitive else fraud_ratio

    run_name = f"xgboost_{variant}"
    if cost_sensitive:
        run_name += "_cost_sensitive"

    model = XGBClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)], verbose=False)

    logger.info("Trained %s", run_name)
    return {"model": model, "name": run_name, "params": params}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def train_lightgbm(X_train, y_train, X_val, y_val,
                   variant: str = "class_weight") -> dict:
    """Train LightGBM with class weighting or SMOTE-preprocessed data."""
    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        "verbose": -1,
    }

    if variant == "class_weight":
        params["is_unbalance"] = True

    run_name = f"lightgbm_{variant}"
    model = LGBMClassifier(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)])

    logger.info("Trained %s", run_name)
    return {"model": model, "name": run_name, "params": params}


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def train_random_forest(X_train, y_train, X_val, y_val,
                        variant: str = "class_weight",
                        feature_subset: list[str] | None = None) -> dict:
    """Train Random Forest (hybrid) with optional feature selection."""
    params = {
        "n_estimators": 200,
        "max_depth": 12,
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }

    if variant == "class_weight":
        params["class_weight"] = "balanced"

    if feature_subset:
        X_train = X_train[feature_subset]
        X_val = X_val[feature_subset]

    run_name = f"rf_hybrid_{variant}"
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    logger.info("Trained %s", run_name)
    return {"model": model, "name": run_name, "params": params,
            "feature_subset": feature_subset}


# ------------------------------------------------------------------
#  Main entry
# ------------------------------------------------------------------

def run(train_path: str, mlflow_tracking: bool = True) -> list[dict]:
    """
    Train all model variants and log to MLflow.

    Returns list of dicts: [{model, name, params, val_data}, ...]
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    X_train, X_val, y_train, y_val = prepare_data(train_path)

    # Pre-compute SMOTE-resampled data once
    logger.info("Applying SMOTE oversampling...")
    X_smote, y_smote = apply_smote(X_train, y_train)

    results = []
    feature_cols = list(X_train.columns)

    # ================================================================
    # 1. XGBoost — class weight
    # ================================================================
    res = train_xgboost(X_train, y_train, X_val, y_val, variant="class_weight")
    res["val_data"] = (X_val, y_val)
    results.append(res)

    # 2. XGBoost — SMOTE
    res = train_xgboost(X_smote, y_smote, X_val, y_val, variant="smote")
    res["val_data"] = (X_val, y_val)
    results.append(res)

    # 3. XGBoost — cost-sensitive (scale_pos_weight=10)
    res = train_xgboost(X_train, y_train, X_val, y_val,
                        variant="class_weight", cost_sensitive=True)
    res["val_data"] = (X_val, y_val)
    results.append(res)

    # ================================================================
    # 4. LightGBM — class weight
    # ================================================================
    res = train_lightgbm(X_train, y_train, X_val, y_val, variant="class_weight")
    res["val_data"] = (X_val, y_val)
    results.append(res)

    # 5. LightGBM — SMOTE
    res = train_lightgbm(X_smote, y_smote, X_val, y_val, variant="smote")
    res["val_data"] = (X_val, y_val)
    results.append(res)

    # ================================================================
    # 6. Random Forest Hybrid — class weight (full features first)
    # ================================================================
    res_rf_full = train_random_forest(X_train, y_train, X_val, y_val,
                                      variant="class_weight")

    # Feature selection from the full RF
    selected = select_features(res_rf_full["model"], X_train, top_k=50)
    joblib.dump(selected, os.path.join(ARTIFACTS_DIR, "selected_features.joblib"))

    # 7. RF Hybrid — class weight + feature selection
    res = train_random_forest(X_train, y_train, X_val, y_val,
                              variant="class_weight", feature_subset=selected)
    res["val_data"] = (X_val[selected], y_val)
    results.append(res)

    # 8. RF Hybrid — SMOTE + feature selection
    res = train_random_forest(X_smote, y_smote, X_val, y_val,
                              variant="smote", feature_subset=selected)
    res["val_data"] = (X_val[selected], y_val)
    results.append(res)

    # ---- Save all models locally ----
    for r in results:
        model_path = os.path.join(ARTIFACTS_DIR, f"{r['name']}.joblib")
        joblib.dump(r["model"], model_path)
        logger.info("Saved model: %s", model_path)

    # ---- Save feature list ----
    joblib.dump(feature_cols, os.path.join(ARTIFACTS_DIR, "feature_columns.joblib"))

    logger.info("Training complete: %d model variants", len(results))
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    mlflow.set_experiment("fraud-detection-experiments")
    with mlflow.start_run(run_name="training"):
        run(train_path=os.path.join(base, "train_featured.parquet"))
    print("Training complete.")

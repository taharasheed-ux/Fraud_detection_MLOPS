"""
Feature Engineering Module
--------------------------
Creates time-based features, user-level aggregations, and encodes
categorical variables (frequency encoding + target encoding with
cross-validation, one-hot for low-cardinality columns).
Saves encoders for inference reuse.
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "features")

# Columns for target encoding (high cardinality)
HIGH_CARD_COLS = ["ProductCD", "P_emaildomain", "R_emaildomain",
                  "card4", "card6", "M1", "M2", "M3", "M4", "M5", "M6",
                  "M7", "M8", "M9"]


# ------------------------------------------------------------------
#  Time features
# ------------------------------------------------------------------

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour-of-day and day-of-week from TransactionDT (seconds)."""
    if "TransactionDT" not in df.columns:
        return df
    # TransactionDT is delta-seconds from some reference; interpret modularly
    df["hour"] = np.floor(df["TransactionDT"] / 3600 % 24).astype("int8")
    df["dayofweek"] = np.floor(df["TransactionDT"] / 86400 % 7).astype("int8")
    logger.info("Added time features: hour, dayofweek")
    return df


# ------------------------------------------------------------------
#  Aggregations
# ------------------------------------------------------------------

def add_user_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """Card-level (user-proxy) aggregations on TransactionAmt."""
    if "card1" not in df.columns or "TransactionAmt" not in df.columns:
        return df

    agg = df.groupby("card1")["TransactionAmt"].agg(["count", "mean", "std"])
    agg.columns = ["card1_tx_count", "card1_tx_mean", "card1_tx_std"]
    agg["card1_tx_std"] = agg["card1_tx_std"].fillna(0)
    df = df.merge(agg, on="card1", how="left")
    logger.info("Added user-level aggregations (card1): count, mean, std")
    return df


# ------------------------------------------------------------------
#  Encoding
# ------------------------------------------------------------------

def frequency_encode(df: pd.DataFrame, cols: list[str],
                     freq_maps: dict | None = None, fit: bool = True):
    """Replace categories with their frequency counts."""
    if fit:
        freq_maps = {}
    for col in cols:
        if col not in df.columns:
            continue
        if fit:
            freq_maps[col] = df[col].value_counts(normalize=True).to_dict()
        df[f"{col}_freq"] = df[col].map(freq_maps[col]).fillna(0).astype("float32")
    return df, freq_maps


def target_encode(df: pd.DataFrame, cols: list[str], target: str = "isFraud",
                  n_splits: int = 5, te_maps: dict | None = None,
                  fit: bool = True):
    """
    Target encoding with K-Fold cross-validation to prevent leakage.
    At inference time, uses the global map built during fit.
    """
    if fit:
        te_maps = {}
        global_mean = df[target].mean()
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for col in cols:
            if col not in df.columns:
                continue
            df[f"{col}_target"] = np.nan

            # Compute global map for later inference
            te_maps[col] = df.groupby(col)[target].mean().to_dict()
            te_maps[f"{col}__global_mean"] = global_mean

            # CV-based filling for train set
            for train_idx, val_idx in kf.split(df):
                means = df.iloc[train_idx].groupby(col)[target].mean()
                df.loc[df.index[val_idx], f"{col}_target"] = (
                    df.iloc[val_idx][col].map(means)
                )
            df[f"{col}_target"] = df[f"{col}_target"].fillna(global_mean).astype("float32")
    else:
        for col in cols:
            if col not in df.columns:
                continue
            global_mean = te_maps.get(f"{col}__global_mean", 0)
            df[f"{col}_target"] = (
                df[col].map(te_maps.get(col, {})).fillna(global_mean).astype("float32")
            )
    return df, te_maps


def drop_raw_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Drop original object columns after encoding."""
    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        df = df.drop(columns=obj_cols)
        logger.info("Dropped %d raw categorical columns", len(obj_cols))
    return df


# ------------------------------------------------------------------
#  Main entry
# ------------------------------------------------------------------

def run(train_path: str, test_path: str | None = None,
        mlflow_tracking: bool = True) -> dict:
    """
    Execute the full feature engineering step.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    processed_dir = os.path.dirname(train_path)

    df_train = pd.read_parquet(train_path)
    df_test = pd.read_parquet(test_path) if test_path else None

    # ---- Time features ----
    df_train = add_time_features(df_train)
    if df_test is not None:
        df_test = add_time_features(df_test)

    # ---- User aggregations  ----
    df_train = add_user_aggregations(df_train)
    if df_test is not None:
        df_test = add_user_aggregations(df_test)

    # ---- Frequency encoding ----
    encode_cols = [c for c in HIGH_CARD_COLS if c in df_train.columns]
    df_train, freq_maps = frequency_encode(df_train, encode_cols, fit=True)
    if df_test is not None:
        df_test, _ = frequency_encode(df_test, encode_cols,
                                      freq_maps=freq_maps, fit=False)

    # ---- Target encoding (train only has isFraud) ----
    te_cols = [c for c in HIGH_CARD_COLS if c in df_train.columns]
    df_train, te_maps = target_encode(df_train, te_cols, fit=True)
    if df_test is not None:
        df_test, _ = target_encode(df_test, te_cols, te_maps=te_maps, fit=False)

    # ---- Drop raw categoricals ----
    df_train = drop_raw_categoricals(df_train)
    if df_test is not None:
        df_test = drop_raw_categoricals(df_test)

    # ---- Save encoders ----
    joblib.dump(freq_maps, os.path.join(ARTIFACTS_DIR, "freq_maps.joblib"))
    joblib.dump(te_maps, os.path.join(ARTIFACTS_DIR, "te_maps.joblib"))

    # ---- Persist ----
    train_out = os.path.join(processed_dir, "train_featured.parquet")
    df_train.to_parquet(train_out, index=False)
    paths = {"train_path": train_out}

    if df_test is not None:
        test_out = os.path.join(processed_dir, "test_featured.parquet")
        df_test.to_parquet(test_out, index=False)
        paths["test_path"] = test_out

    logger.info("Feature engineering complete → %d features", df_train.shape[1])

    if mlflow_tracking:
        mlflow.log_param("num_features_final", df_train.shape[1])
        mlflow.log_artifacts(ARTIFACTS_DIR, "features")

    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    mlflow.set_experiment("fraud-detection-experiments")
    with mlflow.start_run(run_name="feature_engineering"):
        run(
            train_path=os.path.join(base, "train_preprocessed.parquet"),
            test_path=os.path.join(base, "test_preprocessed.parquet"),
        )
    print("Feature engineering complete.")

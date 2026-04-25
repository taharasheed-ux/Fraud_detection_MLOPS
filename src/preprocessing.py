"""
Data Preprocessing Module
-------------------------
Handles missing-value imputation, scaling, anomaly cleaning, and
missing-indicator creation.  Saves fitted transformers for inference reuse.

Memory-optimized: processes train and test sequentially, drops
high-missing columns, and uses float32 throughout.
"""

import os
import gc
import logging
import joblib
import numpy as np
import pandas as pd
import mlflow
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts", "preprocessing")

# Drop columns with > this % missing values to reduce dimensionality
MISSING_DROP_THRESHOLD = 90.0


# ------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------

def _identify_column_types(df: pd.DataFrame):
    """Split columns into numeric and categorical lists (excluding target/id)."""
    exclude = {"TransactionID", "isFraud"}
    numeric = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    categorical = [c for c in df.select_dtypes(include=["object", "category"]).columns if c not in exclude]
    return numeric, categorical


def drop_high_missing(df: pd.DataFrame, threshold: float = MISSING_DROP_THRESHOLD) -> tuple:
    """Drop columns with > threshold % missing. Returns (df, dropped_cols)."""
    pcts = df.isnull().mean() * 100
    drop_cols = pcts[pcts > threshold].index.tolist()
    # Don't drop target or id
    drop_cols = [c for c in drop_cols if c not in ("TransactionID", "isFraud")]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        logger.info("Dropped %d columns with >%.0f%% missing values", len(drop_cols), threshold)
    return df, drop_cols


# ------------------------------------------------------------------
#  Missing-value handling
# ------------------------------------------------------------------

def add_missing_indicators(df: pd.DataFrame, cols: list[str],
                           threshold: float = 5.0) -> pd.DataFrame:
    """Create binary flags for columns with >= threshold % missing."""
    new_cols = {}
    for col in cols:
        if col not in df.columns:
            continue
        pct = df[col].isnull().mean() * 100
        if pct >= threshold:
            new_cols[f"{col}_missing"] = df[col].isnull().astype("int8")
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def impute_numeric(df: pd.DataFrame, cols: list[str],
                   fit: bool = True, imputer: dict | None = None):
    """Impute numeric columns with median (robust to outliers)."""
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return df, imputer
    if fit:
        imputer = df[valid_cols].median().to_dict()
        for col in valid_cols:
            val = imputer.get(col, 0)
            df[col] = df[col].fillna(val)
    else:
        for col in valid_cols:
            if col in imputer:
                df[col] = df[col].fillna(imputer[col])
    return df, imputer


def impute_categorical(df: pd.DataFrame, cols: list[str],
                       fit: bool = True, imputer: dict | None = None):
    """Impute categorical columns with most frequent value."""
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return df, imputer
    if fit:
        imputer = {}
        for col in valid_cols:
            mode_val = df[col].mode(dropna=True)
            val = mode_val.iloc[0] if not mode_val.empty else "Unknown"
            imputer[col] = val
            df[col] = df[col].fillna(val)
    else:
        for col in valid_cols:
            if col in imputer:
                df[col] = df[col].fillna(imputer[col])
    return df, imputer


# ------------------------------------------------------------------
#  Scaling
# ------------------------------------------------------------------

def scale_numeric(df: pd.DataFrame, cols: list[str],
                  fit: bool = True, scaler: dict | None = None):
    """StandardScaler on numeric features."""
    valid_cols = [c for c in cols if c in df.columns]
    if not valid_cols:
        return df, scaler
    if fit:
        scaler = {}
        for col in valid_cols:
            m = float(df[col].mean())
            s = float(df[col].std())
            if s == 0 or pd.isna(s):
                s = 1.0
            scaler[col] = {"mean": m, "scale": s}
            df[col] = ((df[col] - m) / s).astype("float32")
    else:
        for col in valid_cols:
            if col in scaler:
                m = scaler[col]["mean"]
                s = scaler[col]["scale"]
                df[col] = ((df[col] - m) / s).astype("float32")
    return df, scaler


# ------------------------------------------------------------------
#  Anomaly cleaning
# ------------------------------------------------------------------

def clip_outliers(df: pd.DataFrame, cols: list[str],
                  lower_q: float = 0.001, upper_q: float = 0.999) -> pd.DataFrame:
    """Clip extreme outliers to quantile boundaries."""
    for col in cols:
        if col not in df.columns:
            continue
        lo = df[col].quantile(lower_q)
        hi = df[col].quantile(upper_q)
        df[col] = df[col].clip(lo, hi)
    return df


# ------------------------------------------------------------------
#  Main entry — sequential processing to save memory
# ------------------------------------------------------------------

def run(train_path: str, test_path: str | None = None,
        mlflow_tracking: bool = True) -> dict:
    """
    Run the full preprocessing step.
    Processes train first, saves artifacts, frees memory,
    then processes test.
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    processed_dir = os.path.dirname(train_path)

    # ==========================
    # PASS 1: Process TRAIN
    # ==========================
    logger.info("Loading train data...")
    df_train = pd.read_parquet(train_path)

    # Drop high-missing columns first for memory savings
    df_train, dropped_cols = drop_high_missing(df_train, MISSING_DROP_THRESHOLD)
    joblib.dump(dropped_cols, os.path.join(ARTIFACTS_DIR, "dropped_columns.joblib"))

    numeric_cols, cat_cols = _identify_column_types(df_train)
    logger.info("Numeric features: %d | Categorical features: %d",
                len(numeric_cols), len(cat_cols))

    # Missing indicators
    df_train = add_missing_indicators(df_train, numeric_cols + cat_cols)

    # Impute
    df_train, num_imputer = impute_numeric(df_train, numeric_cols, fit=True)
    df_train, cat_imputer = impute_categorical(df_train, cat_cols, fit=True)

    # Clip outliers
    clip_cols = [c for c in numeric_cols if c != "TransactionDT"]
    df_train = clip_outliers(df_train, clip_cols)

    # Scale
    df_train, scaler = scale_numeric(df_train, numeric_cols, fit=True)

    # Save transformers
    joblib.dump(num_imputer, os.path.join(ARTIFACTS_DIR, "num_imputer.joblib"))
    joblib.dump(cat_imputer, os.path.join(ARTIFACTS_DIR, "cat_imputer.joblib"))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, "scaler.joblib"))
    joblib.dump(numeric_cols, os.path.join(ARTIFACTS_DIR, "numeric_cols.joblib"))
    joblib.dump(cat_cols, os.path.join(ARTIFACTS_DIR, "cat_cols.joblib"))
    logger.info("Saved preprocessing artifacts to %s", ARTIFACTS_DIR)

    # Save processed train
    train_out = os.path.join(processed_dir, "train_preprocessed.parquet")
    df_train.to_parquet(train_out, index=False)
    train_rows = df_train.shape[0]
    train_cols = df_train.shape[1]
    logger.info("Saved train → %s (%d rows, %d cols)", train_out, train_rows, train_cols)

    # Free memory
    del df_train
    gc.collect()

    paths = {"train_path": train_out}

    # ==========================
    # PASS 2: Process TEST
    # ==========================
    if test_path is not None:
        logger.info("Loading test data...")
        df_test = pd.read_parquet(test_path)

        # Drop same columns
        cols_to_drop = [c for c in dropped_cols if c in df_test.columns]
        if cols_to_drop:
            df_test = df_test.drop(columns=cols_to_drop)

        test_num = [c for c in numeric_cols if c in df_test.columns]
        test_cat = [c for c in cat_cols if c in df_test.columns]

        df_test = add_missing_indicators(df_test, test_num + test_cat)
        df_test, _ = impute_numeric(df_test, test_num, fit=False, imputer=num_imputer)
        df_test, _ = impute_categorical(df_test, test_cat, fit=False, imputer=cat_imputer)

        test_clip = [c for c in clip_cols if c in df_test.columns]
        df_test = clip_outliers(df_test, test_clip)
        df_test, _ = scale_numeric(df_test, test_num, fit=False, scaler=scaler)

        test_out = os.path.join(processed_dir, "test_preprocessed.parquet")
        df_test.to_parquet(test_out, index=False)
        paths["test_path"] = test_out
        logger.info("Saved test → %s (%d rows, %d cols)", test_out, *df_test.shape)

        del df_test
        gc.collect()

    logger.info("Preprocessing complete → %s", paths)

    if mlflow_tracking:
        mlflow.log_param("num_numeric_features", len(numeric_cols))
        mlflow.log_param("num_categorical_features", len(cat_cols))
        mlflow.log_param("dropped_high_missing_cols", len(dropped_cols))
        mlflow.log_metric("train_rows_after_preproc", train_rows)
        mlflow.log_artifacts(ARTIFACTS_DIR, "preprocessing")

    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    mlflow.set_experiment("fraud-detection-experiments")
    with mlflow.start_run(run_name="preprocessing"):
        run(
            train_path=os.path.join(base, "train_merged.parquet"),
            test_path=os.path.join(base, "test_merged.parquet"),
        )
    print("Preprocessing complete.")

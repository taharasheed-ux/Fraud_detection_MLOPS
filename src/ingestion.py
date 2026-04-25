"""
Data Ingestion Module
---------------------
Loads the IEEE-CIS Fraud Detection dataset (transaction + identity tables),
merges them, down-casts dtypes to save memory, and persists the merged
DataFrames as Parquet files under data/processed/.
"""

import os
import gc
import logging
import pandas as pd
import mlflow

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")


def _downcast(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce memory footprint by down-casting numeric columns."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        # keep TransactionID as int64 for merge key safety
        if col != "TransactionID":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def load_and_merge(split: str = "train") -> pd.DataFrame:
    """
    Load transaction and identity CSVs for the given split,
    merge on TransactionID, and return a single DataFrame.
    """
    tx_path = os.path.join(DATA_DIR, f"{split}_transaction.csv")
    id_path = os.path.join(DATA_DIR, f"{split}_identity.csv")

    if not os.path.exists(tx_path):
        raise FileNotFoundError(f"Transaction file not found: {tx_path}")

    logger.info("Loading %s transaction data from %s", split, tx_path)
    df_tx = pd.read_csv(tx_path)
    logger.info("  → %d rows, %d cols", *df_tx.shape)

    if os.path.exists(id_path):
        logger.info("Loading %s identity data from %s", split, id_path)
        df_id = pd.read_csv(id_path)
        logger.info("  → %d rows, %d cols", *df_id.shape)
        df = df_tx.merge(df_id, on="TransactionID", how="left")
        logger.info("  → Merged shape: %d rows, %d cols", *df.shape)
    else:
        logger.warning("Identity file not found, using transaction data only.")
        df = df_tx

    df = _downcast(df)
    logger.info("  → Memory after downcast: %.1f MB", df.memory_usage(deep=True).sum() / 1e6)
    return df


def run(mlflow_tracking: bool = True) -> dict:
    """
    Execute the full ingestion step.

    Returns
    -------
    dict with keys 'train_path' and 'test_path' pointing to saved parquet files.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    paths = {}

    for split in ("train", "test"):
        df = load_and_merge(split)

        out_path = os.path.join(PROCESSED_DIR, f"{split}_merged.parquet")
        df.to_parquet(out_path, index=False)
        logger.info("Saved %s → %s (%.1f MB)", split, out_path,
                     os.path.getsize(out_path) / 1e6)
        paths[f"{split}_path"] = out_path

        if mlflow_tracking:
            mlflow.log_param(f"{split}_rows", df.shape[0])
            mlflow.log_param(f"{split}_cols", df.shape[1])
            mlflow.log_metric(f"{split}_memory_mb",
                              round(df.memory_usage(deep=True).sum() / 1e6, 1))

        # Free memory before loading next split
        del df
        gc.collect()

    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    mlflow.set_experiment("fraud-detection-experiments")
    with mlflow.start_run(run_name="ingestion"):
        result = run()
    print("Ingestion complete:", result)

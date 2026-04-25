"""
Data Validation Module
----------------------
Validates the merged dataset against an expected schema, checks data types,
and reports missing-value statistics.  Logs validation results to MLflow.
"""

import logging
import mlflow
import pandas as pd

logger = logging.getLogger(__name__)

# ---------- Expected schema (key columns) ----------
REQUIRED_COLUMNS = [
    "TransactionID", "isFraud", "TransactionDT", "TransactionAmt",
    "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
    "addr1", "addr2", "P_emaildomain", "R_emaildomain",
]

NUMERIC_COLUMNS = [
    "TransactionDT", "TransactionAmt", "card1", "card2", "card3", "card5",
    "addr1", "addr2",
]

CATEGORICAL_COLUMNS = [
    "ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain",
]


def validate_schema(df: pd.DataFrame, split: str = "train") -> dict:
    """Check that required columns exist."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    # isFraud only exists in train
    if split == "test" and "isFraud" in missing_cols:
        missing_cols.remove("isFraud")

    result = {
        "schema_valid": len(missing_cols) == 0,
        "missing_columns": missing_cols,
        "total_columns": df.shape[1],
    }
    if missing_cols:
        logger.warning("Schema validation FAILED — missing: %s", missing_cols)
    else:
        logger.info("Schema validation PASSED (%d columns present)", df.shape[1])
    return result


def validate_dtypes(df: pd.DataFrame) -> dict:
    """Verify numeric and categorical columns have correct dtypes."""
    issues = []
    for col in NUMERIC_COLUMNS:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"{col}: expected numeric, got {df[col].dtype}")

    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and pd.api.types.is_float_dtype(df[col]):
            issues.append(f"{col}: expected categorical/object, got {df[col].dtype}")

    result = {"dtype_valid": len(issues) == 0, "dtype_issues": issues}
    if issues:
        logger.warning("Dtype issues: %s", issues)
    else:
        logger.info("Dtype validation PASSED")
    return result


def validate_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a DataFrame summarising missing values per column.
    Columns: column, missing_count, missing_pct
    """
    total = len(df)
    missing = df.isnull().sum()
    missing_df = (
        pd.DataFrame({"column": missing.index,
                       "missing_count": missing.values,
                       "missing_pct": (missing.values / total * 100).round(2)})
        .query("missing_count > 0")
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )
    logger.info("Columns with missing values: %d / %d", len(missing_df), df.shape[1])
    return missing_df


def run(train_path: str, test_path: str | None = None,
        mlflow_tracking: bool = True) -> dict:
    """
    Run the full validation step on train (and optionally test) data.

    Returns
    -------
    dict with keys 'train' and optionally 'test', each containing
    schema, dtype, and missing-value validation results.
    """
    results = {}

    for split, path in [("train", train_path), ("test", test_path)]:
        if path is None:
            continue

        logger.info("— Validating %s data —", split)
        df = pd.read_parquet(path)
        schema = validate_schema(df, split)
        dtypes = validate_dtypes(df)
        missing_df = validate_missing(df)

        results[split] = {
            "schema": schema,
            "dtypes": dtypes,
            "missing_summary": missing_df,
        }

        if mlflow_tracking:
            mlflow.log_param(f"{split}_schema_valid", schema["schema_valid"])
            mlflow.log_param(f"{split}_dtype_valid", dtypes["dtype_valid"])
            mlflow.log_metric(f"{split}_cols_with_missing", len(missing_df))
            mlflow.log_metric(f"{split}_max_missing_pct",
                              float(missing_df["missing_pct"].max()) if len(missing_df) else 0.0)

            # Save missing-value report as artifact
            artifact_path = f"/tmp/{split}_missing_report.csv"
            missing_df.to_csv(artifact_path, index=False)
            mlflow.log_artifact(artifact_path, "validation")

    return results


if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    base = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    mlflow.set_experiment("fraud-detection-experiments")
    with mlflow.start_run(run_name="validation"):
        run(
            train_path=os.path.join(base, "train_merged.parquet"),
            test_path=os.path.join(base, "test_merged.parquet"),
        )
    print("Validation complete.")

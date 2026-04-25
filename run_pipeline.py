"""
Pipeline Orchestrator
---------------------
Runs the full ML pipeline end-to-end:
  ingestion → validation → preprocessing → feature_engineering →
  training → evaluation → deployment

Each step has retry logic (3 attempts).  All results are tracked
in a single MLflow experiment.
"""

import os
import sys
import logging
import argparse
import mlflow
from tenacity import retry, stop_after_attempt, wait_fixed

# ---- Setup path ----
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from src import ingestion, validation, preprocessing  # noqa: E402
from src import feature_engineering, train, evaluate, deploy  # noqa: E402

# ---- Logging ----
LOG_FILE = os.path.join(ROOT_DIR, "pipeline.log")


def setup_logging():
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(LOG_FILE, mode="w"),
        ],
    )


logger = logging.getLogger("pipeline")


# ------------------------------------------------------------------
#  Pipeline step wrappers with retry
# ------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def step_ingestion():
    logger.info("=" * 60)
    logger.info("STEP 1 / 7 — DATA INGESTION")
    logger.info("=" * 60)
    return ingestion.run(mlflow_tracking=True)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def step_validation(paths):
    logger.info("=" * 60)
    logger.info("STEP 2 / 7 — DATA VALIDATION")
    logger.info("=" * 60)
    return validation.run(
        train_path=paths["train_path"],
        test_path=paths.get("test_path"),
        mlflow_tracking=True,
    )


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def step_preprocessing(paths):
    logger.info("=" * 60)
    logger.info("STEP 3 / 7 — DATA PREPROCESSING")
    logger.info("=" * 60)
    return preprocessing.run(
        train_path=paths["train_path"],
        test_path=paths.get("test_path"),
        mlflow_tracking=True,
    )


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def step_feature_engineering(paths):
    logger.info("=" * 60)
    logger.info("STEP 4 / 7 — FEATURE ENGINEERING")
    logger.info("=" * 60)
    return feature_engineering.run(
        train_path=paths["train_path"],
        test_path=paths.get("test_path"),
        mlflow_tracking=True,
    )


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def step_training(paths):
    logger.info("=" * 60)
    logger.info("STEP 5 / 7 — MODEL TRAINING")
    logger.info("=" * 60)
    return train.run(
        train_path=paths["train_path"],
        mlflow_tracking=True,
    )


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def step_evaluation(trained_models, train_path):
    logger.info("=" * 60)
    logger.info("STEP 6 / 7 — MODEL EVALUATION")
    logger.info("=" * 60)
    return evaluate.run(
        trained_models=trained_models,
        train_path=train_path,
        mlflow_tracking=True,
    )


@retry(stop=stop_after_attempt(3), wait=wait_fixed(5))
def step_deployment(eval_results):
    logger.info("=" * 60)
    logger.info("STEP 7 / 7 — CONDITIONAL DEPLOYMENT")
    logger.info("=" * 60)
    return deploy.run(eval_results=eval_results, mlflow_tracking=True)


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------

def main(experiment_name: str = "fraud-detection-experiments"):
    setup_logging()
    logger.info("Starting Fraud Detection Pipeline")
    logger.info("MLflow experiment: %s", experiment_name)

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="full-pipeline") as parent_run:
        logger.info("MLflow run ID: %s", parent_run.info.run_id)

        # 1. Ingestion
        ingestion_paths = step_ingestion()

        # 2. Validation
        step_validation(ingestion_paths)

        # 3. Preprocessing
        preproc_paths = step_preprocessing(ingestion_paths)

        # 4. Feature Engineering
        feature_paths = step_feature_engineering(preproc_paths)

        # 5. Training
        trained_models = step_training(feature_paths)

        # 6. Evaluation
        eval_results = step_evaluation(
            trained_models, feature_paths["train_path"])

        # 7. Deployment
        deploy_result = step_deployment(eval_results)

        # ---- Final summary ----
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)

        if eval_results.get("comparison") is not None:
            logger.info("\n%s", eval_results["comparison"].to_string(index=False))

        logger.info("Deployment: %s", deploy_result)

        mlflow.log_artifact(LOG_FILE, "logs")

    return deploy_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fraud Detection ML Pipeline")
    parser.add_argument("--experiment-name",
                        default="fraud-detection-experiments",
                        help="MLflow experiment name")
    args = parser.parse_args()
    main(experiment_name=args.experiment_name)

"""
scripts/train_sage_policy.py

Phase 4: Trains and evaluates the SAGE Random Forest Adaptive Policy
on the quality-aware, leakage-free Oracle dataset:
    experiments/oracle_results/oracle_dataset.csv

Requirements enforced:
- Strict validation of all 10 FEATURE_NAMES and valid oracle_k labels.
- Explicit check against NaN/Inf values and leakage columns.
- Query-grouped train/validation split (zero query_id overlap).
- Paper Random Forest hyperparameters: n_estimators=100, max_depth=10, random_state=42.
- Outputs Accuracy, MAE, Confusion Matrix, Classification Report, Feature Importances.
- Saves trained model to artifacts/models/random_forest/sage_policy.pkl
- Saves versioned metadata to artifacts/metadata/models/models_v2.json
"""

from datetime import datetime, timezone
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive.policy import SAGEPolicy
from src.config import (
    CANDIDATE_K_VALUES,
    EXPERIMENTS_METADATA_DIR,
    MODELS_METADATA_DIR,
    ORACLE_DATASET_PATH,
    ORACLE_RESULTS_DIR,
    RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_STATE,
    SAGE_MODEL_DIR,
)
from src.retrieval.features import FEATURE_NAMES
from src.utils.logger import setup_logger

logger = setup_logger("train_policy")

FORBIDDEN_LEAKAGE_COLUMNS = {
    "ground_truth_answer",
    "generated_answer",
    "answer_quality",
    "exact_match",
    "token_f1",
    "answer_presence",
    "retrieval_hit",
    "retrieval_latency_ms",
    "generation_latency_ms",
    "end_to_end_latency_ms",
    "slo_satisfied",
}


def validate_oracle_dataset(df: pd.DataFrame) -> None:
    """Validates schema, missing/infinite values, label bounds, and zero label leakage."""
    if len(df) == 0:
        raise ValueError("Oracle dataset is empty; cannot train Random Forest policy.")

    required_cols = {"query_id", *FEATURE_NAMES, "oracle_k"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise KeyError(f"Oracle dataset missing required columns: {sorted(missing_cols)}")

    leakage_found = FORBIDDEN_LEAKAGE_COLUMNS & set(df.columns)
    if leakage_found:
        raise ValueError(
            f"Data leakage detected! Prohibited outcome/ground-truth columns present in training data: "
            f"{sorted(leakage_found)}"
        )

    # Check missing/NaN or Inf values in features
    feat_df = df[FEATURE_NAMES]
    if feat_df.isna().any().any():
        bad = feat_df.columns[feat_df.isna().any()].tolist()
        raise ValueError(f"Missing (NaN) values found in feature columns: {bad}")

    if np.isinf(feat_df.to_numpy(dtype=np.float64)).any():
        raise ValueError("Infinite values found in feature matrix.")

    # Validate target labels
    if df["oracle_k"].isna().any():
        raise ValueError("Missing (NaN) values detected in oracle_k labels.")

    invalid_labels = set(df["oracle_k"].astype(int).unique()) - set(CANDIDATE_K_VALUES)
    if invalid_labels:
        raise ValueError(
            f"Invalid oracle_k labels found: {sorted(invalid_labels)}. "
            f"Expected subset of {CANDIDATE_K_VALUES}."
        )


def main() -> None:
    logger.info("=================================================================")
    logger.info("PHASE 4: TRAINING SAGE RANDOM FOREST POLICY (QUALITY-AWARE)")
    logger.info("=================================================================")

    csv_path = ORACLE_DATASET_PATH
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Oracle training dataset not found at: {csv_path}. "
            f"Run scripts/generate_oracle_dataset.py first."
        )

    df = pd.read_csv(csv_path)
    validate_oracle_dataset(df)
    df["oracle_k"] = df["oracle_k"].astype(int)
    logger.info(f"Validated {len(df)} leakage-free training samples from {csv_path}")

    # Display K class distribution
    label_counts = df["oracle_k"].value_counts().sort_index().to_dict()
    logger.info("Oracle K* Class Distribution:")
    for k_val, cnt in label_counts.items():
        pct = (cnt / len(df)) * 100.0
        logger.info(f"  K* = {k_val:2d}: {cnt:3d} queries ({pct:5.1f}%)")

    # Query-grouped train/validation split (by unique query_id)
    unique_qids = list(df["query_id"].astype(str).unique())
    if len(unique_qids) < 2:
        raise ValueError(f"Need at least 2 unique queries to split train/validation; found {len(unique_qids)}.")

    train_qids, val_qids = train_test_split(
        unique_qids,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_df = df[df["query_id"].isin(train_qids)].copy()
    val_df = df[df["query_id"].isin(val_qids)].copy()

    # Enforce strict zero-overlap guarantee
    overlap = set(train_df["query_id"]) & set(val_df["query_id"])
    assert len(overlap) == 0, f"Train/Validation query overlap detected: {overlap}"

    logger.info(
        f"Query-grouped split (0 overlap verified): "
        f"Train={len(train_df)} rows ({len(train_qids)} qids), "
        f"Validation={len(val_df)} rows ({len(val_qids)} qids)"
    )

    # Initialize policy with paper hyperparameters
    policy = SAGEPolicy(
        n_estimators=RANDOM_FOREST_N_ESTIMATORS,
        max_depth=RANDOM_FOREST_MAX_DEPTH,
        random_state=RANDOM_STATE,
    )

    # Train Random Forest
    policy.train(train_df, train_df["oracle_k"])

    # Evaluate on Train and Validation sets
    train_report = policy.evaluate(train_df, train_df["oracle_k"])
    val_report = policy.evaluate(val_df, val_df["oracle_k"])

    val_true = val_df["oracle_k"].to_numpy(dtype=np.int32)
    val_pred = policy.predict(val_df)
    present_labels = sorted(list(set(val_true.tolist()) | set(val_pred.tolist())))

    cm = confusion_matrix(val_true, val_pred, labels=present_labels)
    clf_text = classification_report(val_true, val_pred, labels=present_labels, zero_division=0)
    importances = policy.get_feature_importances()

    logger.info("-----------------------------------------------------------------")
    logger.info(
        f"TRAIN METRICS      : Accuracy = {train_report.accuracy:.4f} | "
        f"MAE = {train_report.mean_absolute_error:.4f}"
    )
    logger.info(
        f"VALIDATION METRICS : Accuracy = {val_report.accuracy:.4f} | "
        f"MAE = {val_report.mean_absolute_error:.4f}"
    )
    logger.info("-----------------------------------------------------------------")
    logger.info(f"Validation Confusion Matrix (labels={present_labels}):\n{cm}")
    logger.info(f"Validation Classification Report:\n{clf_text}")
    logger.info("-----------------------------------------------------------------")
    logger.info("FEATURE IMPORTANCES (Ranked):")
    for rank, (feat, score) in enumerate(importances.items(), start=1):
        logger.info(f"  {rank:2d}. {feat:<24}: {score * 100:6.2f}%")
    logger.info("-----------------------------------------------------------------")

    # Save trained Random Forest artifact
    saved_model_path = policy.save(SAGE_MODEL_DIR / "sage_policy.pkl")

    # Save evaluation summary under experiments/oracle_results/
    eval_summary_path = ORACLE_RESULTS_DIR / "policy_evaluation_results.json"
    eval_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_source": str(csv_path.relative_to(PROJECT_ROOT)),
        "feature_names": list(FEATURE_NAMES),
        "total_samples": int(len(df)),
        "train_samples": int(len(train_df)),
        "val_samples": int(len(val_df)),
        "query_overlap_count": len(overlap),
        "label_distribution": {str(int(k)): int(v) for k, v in label_counts.items()},
        "train_accuracy": float(train_report.accuracy),
        "train_mae": float(train_report.mean_absolute_error),
        "val_accuracy": float(val_report.accuracy),
        "val_mae": float(val_report.mean_absolute_error),
        "confusion_matrix_labels": [int(x) for x in present_labels],
        "confusion_matrix": cm.tolist(),
        "classification_report": val_report.detailed_report,
        "feature_importances": importances,
    }
    with open(eval_summary_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    # Save versioned model metadata under artifacts/metadata/models/models_v2.json
    MODELS_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    model_meta_path = MODELS_METADATA_DIR / "models_v2.json"

    oracle_meta_path = EXPERIMENTS_METADATA_DIR / "oracle_experiment_v2.json"
    model_meta = {
        "version": "v2",
        "model_type": "RandomForestClassifier",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "oracle_experiment_metadata": (
            str(oracle_meta_path.relative_to(PROJECT_ROOT)) if oracle_meta_path.exists() else None
        ),
        "hyperparameters": {
            "n_estimators": RANDOM_FOREST_N_ESTIMATORS,
            "max_depth": RANDOM_FOREST_MAX_DEPTH,
            "random_state": RANDOM_STATE,
            "criterion": "gini",
        },
        "feature_space": {
            "feature_count": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
        },
        "candidate_k_values": policy.candidate_k_values,
        "dataset": {
            "source": str(csv_path.relative_to(PROJECT_ROOT)),
            "total_samples": int(len(df)),
            "train_samples": int(len(train_df)),
            "val_samples": int(len(val_df)),
            "query_id_overlap": 0,
            "label_distribution": {str(int(k)): int(v) for k, v in label_counts.items()},
        },
        "evaluation": {
            "train_accuracy": float(train_report.accuracy),
            "train_mae": float(train_report.mean_absolute_error),
            "val_accuracy": float(val_report.accuracy),
            "val_mae": float(val_report.mean_absolute_error),
            "confusion_matrix_labels": [int(x) for x in present_labels],
            "confusion_matrix": cm.tolist(),
            "feature_importances": importances,
        },
        "artifact_path": str(saved_model_path.relative_to(PROJECT_ROOT)),
        "evaluation_results_path": str(eval_summary_path.relative_to(PROJECT_ROOT)),
    }
    with open(model_meta_path, "w", encoding="utf-8") as f:
        json.dump(model_meta, f, indent=2)

    logger.info(f"Saved evaluation results : {eval_summary_path}")
    logger.info(f"Saved model metadata     : {model_meta_path}")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()

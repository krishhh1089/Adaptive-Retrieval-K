"""
scripts/train_sage_policy.py

Trains and evaluates the SAGE Random Forest Adaptive Policy model
on the generated Oracle dataset (experiments/oracle_results/oracle_dataset.csv).

Saves the trained model artifact to:
artifacts/models/random_forest/sage_policy.pkl

Outputs evaluation metrics and logs feature importances.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive.policy import SAGEPolicy
from src.config import (
    ORACLE_RESULTS_DIR,
    RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_STATE,
    SAGE_MODEL_DIR,
)
from src.retrieval.features import FEATURE_NAMES
from src.utils.logger import setup_logger

logger = setup_logger("train_policy")


def main():
    logger.info("=================================================================")
    logger.info("STEP 8.2.3: TRAINING SAGE RANDOM FOREST POLICY")
    logger.info("=================================================================")

    csv_path = ORACLE_RESULTS_DIR / "oracle_dataset.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Oracle dataset not found at: {csv_path}. Please run build_oracle_dataset.py first.")

    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} samples from {csv_path}")

    # Inspect class distribution
    label_counts = df["oracle_k"].value_counts().sort_index().to_dict()
    logger.info(f"Oracle K* label distribution: {label_counts}")

    # Verify features exist
    for f in FEATURE_NAMES:
        if f not in df.columns:
            raise KeyError(f"Missing required feature column: {f}")

    # Train/Validation split (80/20)
    # Using non-stratified split because minority classes have < 2 members
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    logger.info(f"Dataset split: Train={len(train_df)} samples, Validation={len(val_df)} samples")

    # Initialize policy model with paper configuration
    policy = SAGEPolicy(
        n_estimators=RANDOM_FOREST_N_ESTIMATORS,
        max_depth=RANDOM_FOREST_MAX_DEPTH,
        random_state=RANDOM_STATE,
    )

    # Train model
    policy.train(train_df, train_df["oracle_k"])

    # Evaluate on Train Set
    train_report = policy.evaluate(train_df, train_df["oracle_k"])
    logger.info("-----------------------------------------------------------------")
    logger.info(f"TRAIN METRICS: Accuracy = {train_report.accuracy:.4f} | MAE = {train_report.mean_absolute_error:.4f}")

    # Evaluate on Validation Set
    val_report = policy.evaluate(val_df, val_df["oracle_k"])
    logger.info(f"VALIDATION METRICS: Accuracy = {val_report.accuracy:.4f} | MAE = {val_report.mean_absolute_error:.4f}")
    logger.info("-----------------------------------------------------------------")

    # Feature Importances
    importances = policy.get_feature_importances()
    logger.info("FEATURE IMPORTANCES (Ranked):")
    for rank, (feat, score) in enumerate(importances.items(), start=1):
        logger.info(f"  {rank:2d}. {feat:<24}: {score * 100:6.2f}%")
    logger.info("-----------------------------------------------------------------")

    # Save model artifact
    saved_path = policy.save()
    logger.info(f"Successfully saved SAGE Policy to: {saved_path}")
    logger.info("=================================================================")
    logger.info("SAGE POLICY TRAINING COMPLETE")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()


"""
src/adaptive/policy.py

Adaptive retrieval budget policy model for SAGE.
Reference: "SAGE: SLO-Aware Adaptive Retrieval for Production RAG Systems" (2026).

Implements the Random Forest policy:
    pi(q) -> K in {2, 3, 5, 7, 10, 15, 20, 25, 30}

The policy receives the 10 lightweight probe features (query characteristics,
probe retrieval signals, sparse-dense agreement, remaining latency budget)
and predicts the minimal optimal retrieval budget K*.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error

from src.config import (
    CANDIDATE_K_VALUES,
    RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_STATE,
    SAGE_MODEL_DIR,
)
from src.retrieval.features import FEATURE_NAMES, QueryRetrievalFeatures
from src.utils.logger import setup_logger

logger = setup_logger("sage_policy")


@dataclass
class PolicyEvaluationReport:
    """Evaluation summary metrics for SAGE policy."""
    accuracy: float
    mean_absolute_error: float
    feature_importances: Dict[str, float]
    detailed_report: Dict[str, Any]
    samples_evaluated: int


class SAGEPolicy:
    """
    Random Forest Adaptive Retrieval Policy for SAGE.

    Predicts the minimal sufficient retrieval budget K* given probe features.
    """

    def __init__(
        self,
        n_estimators: int = RANDOM_FOREST_N_ESTIMATORS,
        max_depth: Optional[int] = RANDOM_FOREST_MAX_DEPTH,
        random_state: int = RANDOM_STATE,
        candidate_k_values: Sequence[int] = CANDIDATE_K_VALUES,
        class_weight: Optional[str] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.candidate_k_values = sorted(list(candidate_k_values))
        self.feature_names = list(FEATURE_NAMES)
        self.class_weight = class_weight

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight=self.class_weight,
        )
        self.is_trained = False

    def _prepare_features(
        self,
        X: Union[np.ndarray, pd.DataFrame, List[List[float]], Sequence[QueryRetrievalFeatures]],
    ) -> np.ndarray:
        """
        Normalizes input feature representations into a 2D float32 numpy array.
        """
        if isinstance(X, pd.DataFrame):
            # Select required features in strict canonical order
            missing = [f for f in self.feature_names if f not in X.columns]
            if missing:
                raise ValueError(f"Input DataFrame missing required feature columns: {missing}")
            return X[self.feature_names].to_numpy(dtype=np.float32)

        if isinstance(X, (list, tuple)) and len(X) > 0 and isinstance(X[0], QueryRetrievalFeatures):
            return np.array([f.to_array() for f in X], dtype=np.float32)

        arr = np.asarray(X, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if arr.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature dimension mismatch: expected {len(self.feature_names)} features, got {arr.shape[1]}"
            )
        return arr

    def snap_to_candidate_budget(self, raw_value: float) -> int:
        """
        Snaps an arbitrary numeric budget prediction to the nearest valid candidate K.
        """
        return min(self.candidate_k_values, key=lambda k: abs(k - raw_value))

    def train(
        self,
        X: Union[np.ndarray, pd.DataFrame, Sequence[QueryRetrievalFeatures]],
        y: Union[np.ndarray, pd.Series, Sequence[int]],
    ) -> "SAGEPolicy":
        """
        Trains the Random Forest policy on extracted features and oracle target budgets.
        """
        X_mat = self._prepare_features(X)
        y_vec = np.asarray(y, dtype=np.int32)

        if len(X_mat) != len(y_vec):
            raise ValueError(f"Length mismatch: X has {len(X_mat)} rows, y has {len(y_vec)} labels.")

        logger.info(
            f"Training SAGE Random Forest Policy (n_estimators={self.n_estimators}, "
            f"max_depth={self.max_depth}, samples={len(X_mat)})..."
        )
        self.model.fit(X_mat, y_vec)
        self.is_trained = True
        logger.info("SAGE Random Forest Policy training complete.")
        return self

    def predict(
        self,
        X: Union[np.ndarray, pd.DataFrame, Sequence[QueryRetrievalFeatures]],
    ) -> np.ndarray:
        """
        Predicts adaptive retrieval budget K for each sample.
        Ensures output values strictly adhere to candidate_k_values.
        """
        if not self.is_trained:
            raise RuntimeError("SAGEPolicy must be trained before calling predict().")

        X_mat = self._prepare_features(X)
        raw_preds = self.model.predict(X_mat)

        # Snap each prediction to valid candidate budget
        snapped = np.array([self.snap_to_candidate_budget(float(p)) for p in raw_preds], dtype=np.int32)
        return snapped

    def predict_proba(
        self,
        X: Union[np.ndarray, pd.DataFrame, Sequence[QueryRetrievalFeatures]],
    ) -> np.ndarray:
        """
        Returns class probabilities for each sample.
        """
        if not self.is_trained:
            raise RuntimeError("SAGEPolicy must be trained before calling predict_proba().")

        X_mat = self._prepare_features(X)
        return self.model.predict_proba(X_mat)

    def predict_single(
        self,
        features: Union[QueryRetrievalFeatures, np.ndarray, Dict[str, float]],
    ) -> int:
        """
        Inference helper for a single query instance.
        """
        if isinstance(features, QueryRetrievalFeatures):
            feat_arr = features.to_array().reshape(1, -1)
        elif isinstance(features, dict):
            feat_arr = np.array([[features[k] for k in self.feature_names]], dtype=np.float32)
        else:
            feat_arr = np.asarray(features, dtype=np.float32)
            if feat_arr.ndim == 1:
                feat_arr = feat_arr.reshape(1, -1)

        pred = self.predict(feat_arr)
        return int(pred[0])

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Returns feature importances mapped to their feature names, sorted descending.
        """
        if not self.is_trained:
            raise RuntimeError("Policy must be trained before extracting feature importances.")

        importances = self.model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        return {
            self.feature_names[i]: float(importances[i])
            for i in sorted_indices
        }

    def evaluate(
        self,
        X: Union[np.ndarray, pd.DataFrame, Sequence[QueryRetrievalFeatures]],
        y: Union[np.ndarray, pd.Series, Sequence[int]],
    ) -> PolicyEvaluationReport:
        """
        Evaluates the policy against ground-truth labels.
        """
        if not self.is_trained:
            raise RuntimeError("Policy must be trained before evaluation.")

        X_mat = self._prepare_features(X)
        y_true = np.asarray(y, dtype=np.int32)
        y_pred = self.predict(X_mat)

        acc = float(accuracy_score(y_true, y_pred))
        mae = float(mean_absolute_error(y_true, y_pred))
        report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        importances = self.get_feature_importances()

        logger.info(f"Evaluation Results: Accuracy={acc:.4f}, MAE={mae:.4f}")
        return PolicyEvaluationReport(
            accuracy=acc,
            mean_absolute_error=mae,
            feature_importances=importances,
            detailed_report=report_dict,
            samples_evaluated=len(y_true),
        )

    def save(self, output_path: Optional[Union[str, Path]] = None) -> Path:
        """
        Serializes the trained policy to disk via joblib.
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained policy.")

        save_path = Path(output_path) if output_path else (SAGE_MODEL_DIR / "sage_policy.pkl")
        save_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "candidate_k_values": self.candidate_k_values,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "random_state": self.random_state,
        }
        joblib.dump(payload, save_path)
        logger.info(f"Saved SAGE policy artifact to: {save_path}")
        return save_path

    @classmethod
    def load(cls, model_path: Optional[Union[str, Path]] = None) -> "SAGEPolicy":
        """
        Loads a serialized policy from disk.
        """
        load_path = Path(model_path) if model_path else (SAGE_MODEL_DIR / "sage_policy.pkl")
        if not load_path.exists():
            raise FileNotFoundError(f"Policy artifact not found at {load_path}")

        payload = joblib.load(load_path)
        instance = cls(
            n_estimators=payload.get("n_estimators", RANDOM_FOREST_N_ESTIMATORS),
            max_depth=payload.get("max_depth", RANDOM_FOREST_MAX_DEPTH),
            random_state=payload.get("random_state", RANDOM_STATE),
            candidate_k_values=payload.get("candidate_k_values", CANDIDATE_K_VALUES),
        )
        instance.model = payload["model"]
        instance.feature_names = payload.get("feature_names", list(FEATURE_NAMES))
        instance.is_trained = True
        logger.info(f"Loaded SAGE policy artifact from: {load_path}")
        return instance


"""
tests/test_policy.py

Unit tests for SAGEPolicy (Random Forest adaptive retrieval budget model).
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.adaptive.policy import SAGEPolicy, PolicyEvaluationReport
from src.config import CANDIDATE_K_VALUES
from src.retrieval.features import FEATURE_NAMES, QueryRetrievalFeatures


@pytest.fixture
def dummy_features_and_labels():
    """Generates synthetic dataset matching the 10 SAGE features."""
    np.random.seed(42)
    n_samples = 30
    # 10 features
    X_mat = np.random.randn(n_samples, len(FEATURE_NAMES)).astype(np.float32)
    # Target K* drawn from candidate budgets
    y_labels = np.random.choice(CANDIDATE_K_VALUES, size=n_samples)
    return X_mat, y_labels


def test_policy_init():
    policy = SAGEPolicy(n_estimators=50, max_depth=5, random_state=123)
    assert policy.n_estimators == 50
    assert policy.max_depth == 5
    assert policy.random_state == 123
    assert policy.is_trained is False
    assert policy.candidate_k_values == sorted(CANDIDATE_K_VALUES)
    assert policy.feature_names == FEATURE_NAMES


def test_untrained_policy_guards():
    policy = SAGEPolicy()
    dummy_x = np.zeros((1, 10), dtype=np.float32)

    with pytest.raises(RuntimeError, match="must be trained before calling predict"):
        policy.predict(dummy_x)

    with pytest.raises(RuntimeError, match="must be trained before calling predict_proba"):
        policy.predict_proba(dummy_x)

    with pytest.raises(RuntimeError, match="must be trained before evaluation"):
        policy.evaluate(dummy_x, [2])

    with pytest.raises(RuntimeError, match="Cannot save an untrained policy"):
        policy.save("dummy.pkl")

    with pytest.raises(RuntimeError, match="must be trained before extracting feature importances"):
        policy.get_feature_importances()


def test_policy_train_and_predict(dummy_features_and_labels):
    X, y = dummy_features_and_labels
    policy = SAGEPolicy(n_estimators=20, max_depth=4, random_state=42)
    policy.train(X, y)

    assert policy.is_trained is True

    preds = policy.predict(X)
    assert len(preds) == len(y)
    # Every prediction must strictly be in CANDIDATE_K_VALUES
    for p in preds:
        assert p in CANDIDATE_K_VALUES


def test_budget_snapping():
    policy = SAGEPolicy()
    assert policy.snap_to_candidate_budget(1.8) == 2
    assert policy.snap_to_candidate_budget(3.1) == 3
    assert policy.snap_to_candidate_budget(4.4) == 5
    assert policy.snap_to_candidate_budget(6.8) == 7
    assert policy.snap_to_candidate_budget(12.0) == 10  # 10 vs 15 -> abs(10-12)=2, abs(15-12)=3
    assert policy.snap_to_candidate_budget(28.0) == 30


def test_policy_dataframe_input(dummy_features_and_labels):
    X, y = dummy_features_and_labels
    df = pd.DataFrame(X, columns=FEATURE_NAMES)

    policy = SAGEPolicy(n_estimators=10, max_depth=3, random_state=42)
    policy.train(df, y)

    preds = policy.predict(df)
    assert len(preds) == len(y)
    assert all(p in CANDIDATE_K_VALUES for p in preds)


def test_policy_feature_mismatch_raises(dummy_features_and_labels):
    X, y = dummy_features_and_labels
    policy = SAGEPolicy()

    # Wrong number of columns
    bad_x = X[:, :5]
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        policy.train(bad_x, y)

    # DataFrame with missing columns
    df_missing = pd.DataFrame(X[:, :8], columns=FEATURE_NAMES[:8])
    with pytest.raises(ValueError, match="missing required feature columns"):
        policy.train(df_missing, y)


def test_predict_single_methods(dummy_features_and_labels):
    X, y = dummy_features_and_labels
    policy = SAGEPolicy(n_estimators=10, max_depth=3, random_state=42)
    policy.train(X, y)

    # 1. Using QueryRetrievalFeatures dataclass
    q_feat = QueryRetrievalFeatures(
        query_char_len=45.0,
        query_word_count=8.0,
        query_type=1.0,
        query_complexity=0.0,
        dense_top1_score=0.72,
        dense_score_drop_1_2=0.08,
        bm25_top1_score=25.4,
        bm25_score_drop_1_2=3.2,
        jaccard_top2=0.33,
        time_remaining_ms=4500.0,
    )
    k_pred1 = policy.predict_single(q_feat)
    assert k_pred1 in CANDIDATE_K_VALUES

    # 2. Using dictionary
    feat_dict = q_feat.to_feature_dict()
    k_pred2 = policy.predict_single(feat_dict)
    assert k_pred2 in CANDIDATE_K_VALUES
    assert k_pred1 == k_pred2

    # 3. Using 1D numpy array
    k_pred3 = policy.predict_single(X[0])
    assert k_pred3 in CANDIDATE_K_VALUES


def test_predict_proba(dummy_features_and_labels):
    X, y = dummy_features_and_labels
    policy = SAGEPolicy(n_estimators=10, max_depth=3, random_state=42)
    policy.train(X, y)

    probs = policy.predict_proba(X)
    assert probs.shape[0] == len(X)
    assert probs.shape[1] == len(policy.model.classes_)
    # Probabilities per sample sum to 1.0
    for row in probs:
        assert np.isclose(np.sum(row), 1.0, atol=1e-4)


def test_feature_importances(dummy_features_and_labels):
    X, y = dummy_features_and_labels
    policy = SAGEPolicy(n_estimators=20, max_depth=4, random_state=42)
    policy.train(X, y)

    importances = policy.get_feature_importances()
    assert len(importances) == len(FEATURE_NAMES)
    assert set(importances.keys()) == set(FEATURE_NAMES)
    # Probabilities sum to ~1.0
    assert np.isclose(sum(importances.values()), 1.0, atol=1e-3)


def test_evaluate(dummy_features_and_labels):
    X, y = dummy_features_and_labels
    policy = SAGEPolicy(n_estimators=20, max_depth=4, random_state=42)
    policy.train(X, y)

    report = policy.evaluate(X, y)
    assert isinstance(report, PolicyEvaluationReport)
    assert 0.0 <= report.accuracy <= 1.0
    assert report.mean_absolute_error >= 0.0
    assert report.samples_evaluated == len(y)
    assert len(report.feature_importances) == len(FEATURE_NAMES)


def test_save_and_load(tmp_path, dummy_features_and_labels):
    X, y = dummy_features_and_labels
    policy = SAGEPolicy(n_estimators=15, max_depth=3, random_state=42)
    policy.train(X, y)

    orig_preds = policy.predict(X)
    save_file = tmp_path / "test_sage_policy.pkl"
    policy.save(save_file)

    assert save_file.exists()

    # Load in new instance
    loaded_policy = SAGEPolicy.load(save_file)
    assert loaded_policy.is_trained is True
    assert loaded_policy.n_estimators == 15
    assert loaded_policy.max_depth == 3

    loaded_preds = loaded_policy.predict(X)
    np.testing.assert_array_equal(orig_preds, loaded_preds)


def test_validate_oracle_dataset_leakage_and_bounds(dummy_features_and_labels):
    from scripts.train_sage_policy import validate_oracle_dataset

    X, y = dummy_features_and_labels
    valid_df = pd.DataFrame(X, columns=FEATURE_NAMES)
    valid_df["query_id"] = [f"q_{i}" for i in range(len(X))]
    valid_df["oracle_k"] = y

    # Valid dataframe passes cleanly
    validate_oracle_dataset(valid_df)

    # Leakage column triggers ValueError
    leak_df = valid_df.copy()
    leak_df["answer_quality"] = 0.5
    with pytest.raises(ValueError, match="Data leakage detected"):
        validate_oracle_dataset(leak_df)

    # Invalid label triggers ValueError
    bad_label_df = valid_df.copy()
    bad_label_df.loc[0, "oracle_k"] = 99
    with pytest.raises(ValueError, match="Invalid oracle_k labels"):
        validate_oracle_dataset(bad_label_df)

    # NaN feature triggers ValueError
    nan_df = valid_df.copy()
    nan_df.loc[0, FEATURE_NAMES[0]] = np.nan
    with pytest.raises(ValueError, match="Missing \\(NaN\\) values"):
        validate_oracle_dataset(nan_df)


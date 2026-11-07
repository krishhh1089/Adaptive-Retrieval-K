"""
tests/test_features.py

Comprehensive test suite for SAGE research-aligned feature extraction:
- Query-side features: char length, word count, query type, and deterministic complexity.
- Dense probe features: top1 score, score drop 1-2, empty / single edge cases.
- Sparse probe features: top1 score, score drop 1-2, empty / single edge cases.
- Sparse-dense agreement: Jaccard similarity across Top-2 probe passage IDs.
- Latency feature: Time remaining under target SLO budget.
- Model compatibility: Deterministic ordering, shape, and Random Forest input validation.
"""

import sys
from pathlib import Path
import math
import numpy as np
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SLO_P95_LATENCY_SECONDS
from src.retrieval.features import (
    FEATURE_NAMES,
    QUERY_TYPE_MAP,
    FeatureExtractor,
    QueryRetrievalFeatures,
    classify_query_type,
    compute_jaccard_similarity,
    estimate_query_complexity,
    sanitize_float,
)


# ==============================================================================
# 1. Query-Side Feature Tests
# ==============================================================================

def test_query_char_len():
    """Character length must include all spaces and punctuation."""
    q1 = "Who is the CEO of Apple?"
    features = FeatureExtractor.extract_features(q1)
    assert features.query_char_len == len(q1)
    assert features.query_char_len == 24

    q2 = " "
    features_empty = FeatureExtractor.extract_features(q2)
    assert features_empty.query_char_len == 1


def test_query_word_count():
    """Word count must use whitespace-based tokenization."""
    q1 = "When was Python released?"
    features = FeatureExtractor.extract_features(q1)
    assert features.query_word_count == 4

    # Multiple spaces and tabs
    q2 = "  What   is   machine    learning?  "
    features_spaces = FeatureExtractor.extract_features(q2)
    assert features_spaces.query_word_count == 4

    # Empty and whitespace-only
    features_empty = FeatureExtractor.extract_features("")
    assert features_empty.query_word_count == 0
    features_ws = FeatureExtractor.extract_features("   \t\n  ")
    assert features_ws.query_word_count == 0


@pytest.mark.parametrize(
    "query,expected_category",
    [
        ("Who directed Inception?", "who"),
        ("In Greek mythology who was the goddess of spring?", "who"),
        ("What is the capital of France?", "what"),
        ("When did World War II end?", "when"),
        ("Where is the Great Barrier Reef located?", "where"),
        ("Why does the moon have craters?", "why"),
        ("Which planet is closest to the Sun?", "which"),
        ("How does a diesel engine operate?", "how"),
        ("How many episodes are in season 8 of The Walking Dead?", "numeric"),
        ("How much does the Eiffel Tower weigh?", "numeric"),
        ("What year was the declaration signed?", "numeric"),
        ("In what year did Apollo 11 land?", "numeric"),
        ("Which year was Python first released?", "numeric"),
        ("1994 winter olympics location", "numeric"),
    ],
)
def test_query_types(query, expected_category):
    """Each canonical question type must be recognized and mapped correctly."""
    cat = classify_query_type(query)
    assert cat == expected_category
    features = FeatureExtractor.extract_features(query)
    assert features.query_type == QUERY_TYPE_MAP[expected_category]
    assert features.query_type_name == expected_category


def test_unknown_and_empty_query_types():
    """Queries without clear interrogatives must map to 'other' (8)."""
    assert classify_query_type("Tell me about photosynthesis.") == "other"
    assert classify_query_type("Random factual query about space.") == "other"
    assert classify_query_type("") == "other"
    assert classify_query_type("???") == "other"

    f_other = FeatureExtractor.extract_features("Just notes.")
    assert f_other.query_type == QUERY_TYPE_MAP["other"]
    assert f_other.query_type == 8


def test_query_complexity():
    """Estimator must distinguish Simple (0), Medium (1), and Complex (2)."""
    # Simple (0): single-hop, direct factoid
    q_simple = "Who is the president of France?"
    assert estimate_query_complexity(q_simple) == 0
    assert FeatureExtractor.extract_features(q_simple).query_complexity == 0

    # Medium (1): comparison or moderate length/clause
    q_medium = "What is the difference between RAM and ROM?"
    assert estimate_query_complexity(q_medium) == 1
    assert FeatureExtractor.extract_features(q_medium).query_complexity == 1

    # Complex (2): multi-hop reasoning + comparison or multiple subquestions
    q_complex = (
        "Why did the Roman Empire collapse because of internal economic conflict "
        "versus external barbarian invasions, and what were the consequences?"
    )
    assert estimate_query_complexity(q_complex) == 2
    assert FeatureExtractor.extract_features(q_complex).query_complexity == 2


# ==============================================================================
# 2. Dense Retrieval Feature Tests
# ==============================================================================

def test_dense_features_normal():
    """Dense Top-1 and drop 1-2 must match the scores."""
    scores = [0.85, 0.70, 0.50]
    features = FeatureExtractor.extract_features("Test query", dense_scores=scores)
    assert pytest.approx(features.dense_top1_score, 1e-4) == 0.85
    assert pytest.approx(features.dense_score_drop_1_2, 1e-4) == 0.15


def test_dense_features_single_result():
    """When only 1 dense result is returned, drop 1-2 must default to 0.0."""
    scores = [0.92]
    features = FeatureExtractor.extract_features("Test query", dense_scores=scores)
    assert pytest.approx(features.dense_top1_score, 1e-4) == 0.92
    assert features.dense_score_drop_1_2 == 0.0


def test_dense_features_empty():
    """Empty dense results must safely default to 0.0."""
    features = FeatureExtractor.extract_features("Test query", dense_scores=[])
    assert features.dense_top1_score == 0.0
    assert features.dense_score_drop_1_2 == 0.0


# ==============================================================================
# 3. Sparse (BM25) Retrieval Feature Tests
# ==============================================================================

def test_bm25_features_normal():
    """BM25 Top-1 and drop 1-2 must match the scores."""
    scores = [35.5, 20.0, 10.0]
    features = FeatureExtractor.extract_features("Test query", sparse_scores=scores)
    assert pytest.approx(features.bm25_top1_score, 1e-4) == 35.5
    assert pytest.approx(features.bm25_score_drop_1_2, 1e-4) == 15.5


def test_bm25_features_single_result():
    """When only 1 BM25 result is returned, drop 1-2 must default to 0.0."""
    scores = [42.1]
    features = FeatureExtractor.extract_features("Test query", sparse_scores=scores)
    assert pytest.approx(features.bm25_top1_score, 1e-4) == 42.1
    assert features.bm25_score_drop_1_2 == 0.0


def test_bm25_features_empty():
    """Empty BM25 results must safely default to 0.0."""
    features = FeatureExtractor.extract_features("Test query", sparse_scores=[])
    assert features.bm25_top1_score == 0.0
    assert features.bm25_score_drop_1_2 == 0.0


# ==============================================================================
# 4. Sparse-Dense Agreement Tests (k=2 Jaccard)
# ==============================================================================

def test_jaccard_top2_identical():
    """Identical top-2 sets must produce Jaccard = 1.0."""
    d_ids = ["doc_1_chunk_1", "doc_1_chunk_2"]
    s_ids = ["doc_1_chunk_1", "doc_1_chunk_2"]
    j = compute_jaccard_similarity(d_ids[:2], s_ids[:2])
    assert j == 1.0

    features = FeatureExtractor.extract_features(
        "query", dense_passage_ids=d_ids, sparse_passage_ids=s_ids
    )
    assert features.jaccard_top2 == 1.0


def test_jaccard_top2_disjoint():
    """Completely disjoint top-2 sets must produce Jaccard = 0.0."""
    d_ids = ["doc_1_chunk_1", "doc_1_chunk_2"]
    s_ids = ["doc_2_chunk_1", "doc_2_chunk_2"]
    features = FeatureExtractor.extract_features(
        "query", dense_passage_ids=d_ids, sparse_passage_ids=s_ids
    )
    assert features.jaccard_top2 == 0.0


def test_jaccard_top2_one_overlap():
    """One common passage out of 3 unique passages: 1 / 3 ≈ 0.3333."""
    d_ids = ["doc_1_chunk_1", "doc_1_chunk_2"]
    s_ids = ["doc_1_chunk_2", "doc_3_chunk_1"]
    features = FeatureExtractor.extract_features(
        "query", dense_passage_ids=d_ids, sparse_passage_ids=s_ids
    )
    assert pytest.approx(features.jaccard_top2, 1e-4) == (1.0 / 3.0)


def test_jaccard_top2_empty():
    """Empty retrieval result lists must return 0.0 without division by zero."""
    features = FeatureExtractor.extract_features("query", dense_passage_ids=[], sparse_passage_ids=[])
    assert features.jaccard_top2 == 0.0


def test_jaccard_top2_duplicates():
    """Duplicates within a single channel must be deduplicated by set conversion."""
    d_ids = ["doc_1", "doc_1"]  # set size 1
    s_ids = ["doc_1", "doc_2"]  # set size 2
    # Intersection = {'doc_1'} (1), Union = {'doc_1', 'doc_2'} (2) -> 1/2 = 0.5
    features = FeatureExtractor.extract_features(
        "query", dense_passage_ids=d_ids, sparse_passage_ids=s_ids
    )
    assert pytest.approx(features.jaccard_top2, 1e-4) == 0.5


# ==============================================================================
# 5. Latency Feature Tests
# ==============================================================================

def test_time_remaining_ms_normal():
    """Normal remaining budget under configured 5.0s SLO."""
    slo_budget = 5000.0  # 5 seconds
    elapsed = 1200.0     # 1.2 seconds elapsed
    features = FeatureExtractor.extract_features(
        "query", elapsed_ms=elapsed, slo_budget_ms=slo_budget
    )
    assert pytest.approx(features.time_remaining_ms, 1e-2) == 3800.0


def test_time_remaining_ms_nearly_exhausted():
    """Nearly exhausted latency budget."""
    slo_budget = 5000.0
    elapsed = 4950.0
    features = FeatureExtractor.extract_features(
        "query", elapsed_ms=elapsed, slo_budget_ms=slo_budget
    )
    assert pytest.approx(features.time_remaining_ms, 1e-2) == 50.0


def test_time_remaining_ms_exhausted():
    """Exhausted latency budget must floor at 0.0."""
    slo_budget = 5000.0
    elapsed = 6200.0
    features = FeatureExtractor.extract_features(
        "query", elapsed_ms=elapsed, slo_budget_ms=slo_budget
    )
    assert features.time_remaining_ms == 0.0


def test_time_remaining_default_slo():
    """Defaults to config.SLO_P95_LATENCY_SECONDS * 1000 if not specified."""
    features = FeatureExtractor.extract_features("query", elapsed_ms=1000.0)
    expected = (SLO_P95_LATENCY_SECONDS * 1000.0) - 1000.0
    assert pytest.approx(features.time_remaining_ms, 1e-2) == expected


# ==============================================================================
# 6. Model Input & Pipeline Integration Tests
# ==============================================================================

def test_feature_names_and_deterministic_order():
    """Verifies feature names, count, ordering, and array export."""
    assert len(FEATURE_NAMES) == 10
    features = FeatureExtractor.extract_features(
        query="When is the final episode?",
        dense_scores=[0.8, 0.6],
        dense_passage_ids=["p1", "p2"],
        sparse_scores=[30.0, 15.0],
        sparse_passage_ids=["p1", "p3"],
        elapsed_ms=500.0,
    )

    feat_dict = features.to_feature_dict()
    assert list(feat_dict.keys()) == FEATURE_NAMES

    feat_array = features.to_array()
    assert feat_array.shape == (10,)
    assert feat_array.dtype == np.float32

    # Verify no NaN or Inf
    assert not np.isnan(feat_array).any()
    assert not np.isinf(feat_array).any()


def test_random_forest_compatibility():
    """Validates that extracted vectors train and infer with RandomForest models."""
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

    # Generate small synthetic batch
    queries = [
        "Who is the author?",
        "When was it published?",
        "Where was it found?",
        "Why did it happen?",
        "How many chapters exist?",
    ]
    X_list = []
    for q in queries:
        f = FeatureExtractor.extract_features(
            query=q,
            dense_scores=[0.7, 0.5],
            dense_passage_ids=["p1", "p2"],
            sparse_scores=[25.0, 10.0],
            sparse_passage_ids=["p2", "p3"],
            elapsed_ms=200.0,
        )
        X_list.append(f.to_array())

    X = np.stack(X_list)
    assert X.shape == (5, 10)

    # Train Regressor
    y_reg = np.array([2.0, 5.0, 7.0, 10.0, 3.0])
    reg = RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
    reg.fit(X, y_reg)
    preds = reg.predict(X[:1])
    assert len(preds) == 1
    assert not np.isnan(preds[0])

    # Train Classifier
    y_clf = np.array([2, 5, 7, 10, 3])
    clf = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
    clf.fit(X, y_clf)
    preds_clf = clf.predict(X[:1])
    assert preds_clf[0] in [2, 3, 5, 7, 10]


def test_live_retrieval_smoke_test():
    """Smoke test running on disk index if available."""
    from src.data.chunker import DocumentChunker
    from src.retrieval.hybrid import HybridRetriever
    from src import config

    try:
        chunker = DocumentChunker()
        chunks = chunker.load_chunks()
        retriever = HybridRetriever(
            rrf_k=config.RRF_K_CONSTANT, candidate_pool=config.RETRIEVAL_CANDIDATE_POOL
        )
        retriever.initialize(chunks)
    except Exception as e:
        pytest.skip(f"Live retrieval index not available: {e}")

    query = "when is the last episode of season 8 of the walking dead"
    res = retriever.search(query, top_k=5)
    features = FeatureExtractor.extract_features(query, retrieval_result=res)

    assert features.query_char_len == len(query)
    assert features.dense_top1_score > 0.0
    assert features.bm25_top1_score > 0.0
    assert 0.0 <= features.jaccard_top2 <= 1.0
    assert features.to_array().shape == (10,)


if __name__ == "__main__":
    pytest.main(["-v", __file__])

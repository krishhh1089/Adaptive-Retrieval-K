"""
src/retrieval/features.py

Feature extraction module for adaptive retrieval budget allocation in SAGE.
Reference: "SAGE: SLO-Aware Adaptive Retrieval for Production RAG Systems" (2026).

This module extracts:
1. Query-side features: character length, word count, query category, and complexity.
2. Dense retrieval features: Top-1 similarity score and score drop (margin).
3. Sparse retrieval features: Top-1 BM25 score and score drop (margin).
4. Sparse-Dense agreement: Jaccard similarity across Top-2 probe passage IDs.
5. Latency feature: Remaining SLO latency budget in milliseconds.

Research Distinction Note:
- Features explicitly described in the SAGE paper:
  * Lightweight k=2 retrieval probe.
  * Dense and sparse top scores and score margins.
  * Sparse-dense rank agreement across probe candidates.
  * Remaining latency budget relative to target SLO.
- Engineered implementation choices in this project:
  * Exact heuristic scoring for query_complexity (0=Simple, 1=Medium, 2=Complex).
  * Specific question-type regex classifier mapping to categorical integer IDs.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import math
import re
import time
import numpy as np

from src.config import SLO_P95_LATENCY_SECONDS
from src.utils.logger import setup_logger

logger = setup_logger("feature_extractor")

# Ordered feature names expected by the SAGE policy Random Forest model
FEATURE_NAMES: List[str] = [
    "query_char_len",
    "query_word_count",
    "query_type",
    "query_complexity",
    "dense_top1_score",
    "dense_score_drop_1_2",
    "bm25_top1_score",
    "bm25_score_drop_1_2",
    "jaccard_top2",
    "time_remaining_ms",
]

# Categorical mapping for query_type feature (0 to 8)
QUERY_TYPE_MAP: Dict[str, int] = {
    "who": 0,
    "what": 1,
    "when": 2,
    "where": 3,
    "why": 4,
    "which": 5,
    "how": 6,
    "numeric": 7,
    "other": 8,
}

REVERSE_QUERY_TYPE_MAP: Dict[int, str] = {v: k for k, v in QUERY_TYPE_MAP.items()}

# Linguistic markers for deterministic question classification
NUMERIC_PATTERNS: Sequence[str] = (
    r"\bhow (many|much|old|long|far|tall|often|frequently)\b",
    r"\b(what|which|in what|in which) (year|date|century|decade|percentage|ratio|fraction|amount|number|quantity)\b",
    r"^\s*(what|which) year\b",
    r"^\s*\d{1,4}\b",
    r"\b(total|average|population|cost|price|temperature|speed|distance|height|weight)\b",
)

COMPARISON_KEYWORDS: Set[str] = {
    "versus",
    "vs",
    "vs.",
    "compare",
    "compared",
    "comparison",
    "difference",
    "differ",
    "better",
    "worse",
    "either",
    "neither",
}

REASONING_KEYWORDS: Set[str] = {
    "because",
    "cause",
    "caused",
    "causes",
    "effect",
    "effects",
    "impact",
    "impacts",
    "relationship",
    "relation",
    "influence",
    "consequence",
    "reason",
    "reasons",
    "both",
}


def classify_query_type(query: str) -> str:
    """
    Classifies a query into one of 9 canonical categories:
    ['who', 'what', 'when', 'where', 'why', 'which', 'how', 'numeric', 'other'].
    """
    clean_q = query.strip().lower()
    if not clean_q:
        return "other"

    # 1. Check numeric patterns first
    for pattern in NUMERIC_PATTERNS:
        if re.search(pattern, clean_q):
            return "numeric"

    # 2. Extract leading tokens (handling introductory prepositions)
    tokens = re.findall(r"\b\w+\b", clean_q)
    if not tokens:
        return "other"

    interrogatives = ["who", "what", "when", "where", "why", "which", "how"]

    # Check for direct leading question word
    for token in tokens:
        if token in interrogatives:
            return token

    return "other"


def estimate_query_complexity(query: str) -> int:
    """
    Lightweight, deterministic query complexity estimator.
    
    Returns:
        0: Simple  - Direct single-hop, single-entity factoid query.
        1: Medium  - Moderate length or compound clause.
        2: Complex - Multi-hop reasoning, explicit comparison, or multiple subquestions.
        
    Note:
        This deterministic heuristic is our project implementation choice for
        capturing reasoning complexity without expensive LLM classification calls.
    """
    clean_q = query.strip()
    if not clean_q:
        return 0

    words = clean_q.split()
    word_count = len(words)
    lower_q = clean_q.lower()
    tokens = set(re.findall(r"\b\w+\b", lower_q))

    score = 0

    # Signal 1: Comparison requirement
    if any(k in tokens for k in COMPARISON_KEYWORDS):
        score += 1

    # Signal 2: Multi-hop reasoning / causal dependencies
    if any(k in tokens for k in REASONING_KEYWORDS):
        score += 1

    # Signal 3: Multiple sub-questions (multiple '?' or conjunction of clauses)
    if clean_q.count("?") > 1 or ";" in clean_q:
        score += 1
    elif any(conj in lower_q for conj in (" and ", " or ")) and ("," in clean_q or len(tokens) > 10):
        score += 1

    # Signal 4: Query length penalty
    if word_count >= 15:
        score += 1

    if score == 0:
        return 0
    elif score == 1:
        return 1
    return 2


def compute_jaccard_similarity(set_a: Sequence[Any], set_b: Sequence[Any]) -> float:
    """
    Computes Jaccard overlap between two identifier lists: |A ∩ B| / |A ∪ B|.
    Handles duplicates, ordering, and empty lists safely.
    """
    sa = set(set_a)
    sb = set(set_b)
    union_size = len(sa | sb)
    if union_size == 0:
        return 0.0
    return float(len(sa & sb) / union_size)


def sanitize_float(val: Any, default: float = 0.0) -> float:
    """Safely converts input to a finite float, preventing NaN or Inf values."""
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


@dataclass
class QueryRetrievalFeatures:
    """
    Structured feature vector for SAGE adaptive retrieval policy prediction.
    Features strictly align with Section 2 of the research specification.
    """
    # A. Query-side features
    query_char_len: int
    query_word_count: int
    query_type: int
    query_complexity: int

    # B. Dense retrieval probe features (k=2)
    dense_top1_score: float
    dense_score_drop_1_2: float

    # C. Sparse retrieval probe features (BM25, k=2)
    bm25_top1_score: float
    bm25_score_drop_1_2: float

    # D. Sparse-Dense agreement (k=2 probe)
    jaccard_top2: float

    # E. Latency feature
    time_remaining_ms: float

    # Metadata helper (excluded from numeric array)
    query_type_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Returns all attributes as a dictionary including metadata."""
        return asdict(self)

    def to_feature_dict(self) -> Dict[str, float]:
        """Returns only the 10 numeric model features in canonical order."""
        return {
            name: float(getattr(self, name))
            for name in FEATURE_NAMES
        }

    def to_array(self) -> np.ndarray:
        """
        Converts the 10 numeric features to a 1D NumPy float32 array
        in exact canonical feature order.
        """
        vals = [float(getattr(self, name)) for name in FEATURE_NAMES]
        arr = np.array(vals, dtype=np.float32)
        # Ensure no NaN or Inf slips through
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr


class FeatureExtractor:
    """
    Extracts research-aligned SAGE features from query text and retrieval probes.
    """

    @classmethod
    def extract_features(
        cls,
        query: str,
        dense_scores: Optional[Sequence[float]] = None,
        dense_passage_ids: Optional[Sequence[Any]] = None,
        sparse_scores: Optional[Sequence[float]] = None,
        sparse_passage_ids: Optional[Sequence[Any]] = None,
        request_start_time: Optional[float] = None,
        elapsed_ms: Optional[float] = None,
        slo_budget_ms: Optional[float] = None,
        retrieval_result: Optional[Any] = None,
    ) -> QueryRetrievalFeatures:
        """
        Extracts the 10 SAGE features.

        Args:
            query: The user question string.
            dense_scores: Dense similarity scores (at least top-2 from probe).
            dense_passage_ids: Canonical passage IDs for top dense results.
            sparse_scores: BM25 scores (at least top-2 from probe).
            sparse_passage_ids: Canonical passage IDs for top BM25 results.
            request_start_time: Monotonic start timestamp from time.perf_counter().
            elapsed_ms: Explicit elapsed request latency in ms (overrides request_start_time).
            slo_budget_ms: Target SLO deadline in ms (defaults to config.SLO_P95_LATENCY_SECONDS * 1000).
            retrieval_result: Optional HybridRetrievalResult for backward compatibility.

        Returns:
            QueryRetrievalFeatures instance.
        """
        # 1. Query-side features
        query_char_len = len(query)
        # Whitespace-based tokenization
        query_word_count = len(query.strip().split()) if query.strip() else 0
        q_type_name = classify_query_type(query)
        query_type_id = QUERY_TYPE_MAP.get(q_type_name, QUERY_TYPE_MAP["other"])
        query_complexity = estimate_query_complexity(query)

        # 2. Extract probe scores & IDs (support fallback to retrieval_result)
        d_scores = list(dense_scores) if dense_scores is not None else []
        d_ids = list(dense_passage_ids) if dense_passage_ids is not None else []
        s_scores = list(sparse_scores) if sparse_scores is not None else []
        s_ids = list(sparse_passage_ids) if sparse_passage_ids is not None else []

        if retrieval_result is not None and (not d_scores or not s_scores):
            # Extract from HybridRetrievalResult passages
            passages = getattr(retrieval_result, "passages", [])
            if not d_scores:
                dense_sorted = sorted(
                    [p for p in passages if getattr(p, "dense_score", None) is not None],
                    key=lambda x: getattr(x, "dense_rank", 999),
                )
                d_scores = [p.dense_score for p in dense_sorted]
                d_ids = [p.chunk_id for p in dense_sorted]

            if not s_scores:
                sparse_sorted = sorted(
                    [p for p in passages if getattr(p, "sparse_score", None) is not None],
                    key=lambda x: getattr(x, "sparse_rank", 999),
                )
                s_scores = [p.sparse_score for p in sparse_sorted]
                s_ids = [p.chunk_id for p in sparse_sorted]

        # 3. Dense retrieval features
        if len(d_scores) >= 1:
            dense_top1 = sanitize_float(d_scores[0])
            dense_drop = sanitize_float(d_scores[0] - d_scores[1]) if len(d_scores) >= 2 else 0.0
        else:
            dense_top1 = 0.0
            dense_drop = 0.0

        # 4. Sparse retrieval features (BM25)
        if len(s_scores) >= 1:
            bm25_top1 = sanitize_float(s_scores[0])
            bm25_drop = sanitize_float(s_scores[0] - s_scores[1]) if len(s_scores) >= 2 else 0.0
        else:
            bm25_top1 = 0.0
            bm25_drop = 0.0

        # 5. Sparse-Dense agreement (k=2 probe)
        # Use top-2 canonical passage IDs
        top2_dense_ids = d_ids[:2]
        top2_sparse_ids = s_ids[:2]
        jaccard_top2 = compute_jaccard_similarity(top2_dense_ids, top2_sparse_ids)

        # 6. Latency feature
        target_slo_ms = slo_budget_ms if slo_budget_ms is not None else (SLO_P95_LATENCY_SECONDS * 1000.0)

        if elapsed_ms is not None:
            actual_elapsed = max(0.0, float(elapsed_ms))
        elif request_start_time is not None:
            actual_elapsed = max(0.0, (time.perf_counter() - request_start_time) * 1000.0)
        elif retrieval_result is not None and hasattr(retrieval_result, "total_retrieval_latency_ms"):
            actual_elapsed = max(0.0, float(retrieval_result.total_retrieval_latency_ms))
        else:
            actual_elapsed = 0.0

        time_remaining_ms = max(0.0, target_slo_ms - actual_elapsed)

        return QueryRetrievalFeatures(
            query_char_len=query_char_len,
            query_word_count=query_word_count,
            query_type=query_type_id,
            query_complexity=query_complexity,
            dense_top1_score=dense_top1,
            dense_score_drop_1_2=dense_drop,
            bm25_top1_score=bm25_top1,
            bm25_score_drop_1_2=bm25_drop,
            jaccard_top2=jaccard_top2,
            time_remaining_ms=time_remaining_ms,
            query_type_name=q_type_name,
        )

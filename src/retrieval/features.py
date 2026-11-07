"""
src/retrieval/features.py

Feature extraction module for adaptive retrieval budget allocation in SAGE.
Extracts query linguistic signals and retrieval distribution confidence metrics.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set
import re
import numpy as np

from src.retrieval.hybrid import HybridRetrievalResult
from src.utils.logger import setup_logger

logger = setup_logger("feature_extractor")

WH_WORDS: Set[str] = {"who", "what", "when", "where", "why", "which", "how"}


@dataclass
class QueryRetrievalFeatures:
    # Query-side features
    query_char_len: int
    query_word_count: int
    is_wh_question: int

    # Dense retrieval distribution features
    dense_top1_score: float
    dense_top3_mean: float
    dense_top5_mean: float
    dense_score_drop_1_2: float
    dense_score_std: float

    # Sparse BM25 retrieval distribution features
    bm25_top1_score: float
    bm25_top3_mean: float
    bm25_top5_mean: float
    bm25_score_drop_1_2: float

    # Multi-retriever agreement features (Jaccard overlap)
    jaccard_top5: float
    jaccard_top10: float

    # System telemetry signal
    probe_retrieval_latency_ms: float

    def to_dict(self) -> Dict[str, float]:
        """Converts feature dataclass to a dictionary."""
        return asdict(self)

    def to_array(self) -> np.ndarray:
        """Converts feature values to a 1D NumPy array for ML model input."""
        return np.array(list(self.to_dict().values()), dtype=np.float32)


class FeatureExtractor:
    """
    Extracts structured query and retrieval features from a HybridRetrievalResult.
    """

    @staticmethod
    def extract_features(
        query: str,
        retrieval_result: HybridRetrievalResult,
        dense_scores_pool: Optional[List[float]] = None,
        sparse_scores_pool: Optional[List[float]] = None,
        dense_indices_pool: Optional[List[int]] = None,
        sparse_indices_pool: Optional[List[int]] = None,
    ) -> QueryRetrievalFeatures:
        """
        Extracts feature vectors from query text and raw candidate retrieval pools.
        """
        clean_q = query.strip()
        words = re.findall(r"\b\w+\b", clean_q.lower())
        char_len = len(clean_q)
        word_count = len(words)
        is_wh = 1 if words and words[0] in WH_WORDS else 0

        # Dense signal extraction
        dense_scores = dense_scores_pool or [
            p.dense_score for p in retrieval_result.passages if p.dense_score is not None
        ]
        if dense_scores:
            d_top1 = float(dense_scores[0])
            d_top3_m = float(np.mean(dense_scores[:3])) if len(dense_scores) >= 3 else d_top1
            d_top5_m = float(np.mean(dense_scores[:5])) if len(dense_scores) >= 5 else d_top1
            d_drop_1_2 = float(dense_scores[0] - dense_scores[1]) if len(dense_scores) >= 2 else 0.0
            d_std = float(np.std(dense_scores))
        else:
            d_top1, d_top3_m, d_top5_m, d_drop_1_2, d_std = 0.0, 0.0, 0.0, 0.0, 0.0

        # BM25 signal extraction
        sparse_scores = sparse_scores_pool or [
            p.sparse_score for p in retrieval_result.passages if p.sparse_score is not None
        ]
        if sparse_scores:
            s_top1 = float(sparse_scores[0])
            s_top3_m = float(np.mean(sparse_scores[:3])) if len(sparse_scores) >= 3 else s_top1
            s_top5_m = float(np.mean(sparse_scores[:5])) if len(sparse_scores) >= 5 else s_top1
            s_drop_1_2 = float(sparse_scores[0] - sparse_scores[1]) if len(sparse_scores) >= 2 else 0.0
        else:
            s_top1, s_top3_m, s_top5_m, s_drop_1_2 = 0.0, 0.0, 0.0, 0.0

        # Channel agreement (Jaccard similarity between top sets)
        d_ids = dense_indices_pool or [
            p.chunk_index for p in sorted(
                [p for p in retrieval_result.passages if p.dense_rank is not None],
                key=lambda x: x.dense_rank or 999
            )
        ]
        s_ids = sparse_indices_pool or [
            p.chunk_index for p in sorted(
                [p for p in retrieval_result.passages if p.sparse_rank is not None],
                key=lambda x: x.sparse_rank or 999
            )
        ]

        def jaccard(set_a: Set[int], set_b: Set[int]) -> float:
            union = len(set_a | set_b)
            return len(set_a & set_b) / union if union > 0 else 0.0

        j_5 = jaccard(set(d_ids[:5]), set(s_ids[:5]))
        j_10 = jaccard(set(d_ids[:10]), set(s_ids[:10]))

        return QueryRetrievalFeatures(
            query_char_len=char_len,
            query_word_count=word_count,
            is_wh_question=is_wh,
            dense_top1_score=d_top1,
            dense_top3_mean=d_top3_m,
            dense_top5_mean=d_top5_m,
            dense_score_drop_1_2=d_drop_1_2,
            dense_score_std=d_std,
            bm25_top1_score=s_top1,
            bm25_top3_mean=s_top3_m,
            bm25_top5_mean=s_top5_m,
            bm25_score_drop_1_2=s_drop_1_2,
            jaccard_top5=j_5,
            jaccard_top10=j_10,
            probe_retrieval_latency_ms=float(retrieval_result.total_retrieval_latency_ms),
        )


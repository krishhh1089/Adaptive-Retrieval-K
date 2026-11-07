"""
src/adaptive/oracle.py

Oracle label generator for SAGE adaptive retrieval policy training.
Computes optimal minimal budget K* across candidate values:
K in [2, 3, 5, 7, 10, 15, 20, 25, 30] using the formalized decision rule.
"""

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple
import json
import numpy as np

from src.config import (
    CANDIDATE_K_VALUES,
    ORACLE_RESULTS_DIR,
    RETRIEVAL_CANDIDATE_POOL,
    SLO_P95_LATENCY_SECONDS,
)
from src.data.loader import QuestionRecord
from src.retrieval.features import FeatureExtractor, QueryRetrievalFeatures
from src.retrieval.hybrid import HybridRetriever, ScoredPassage
from src.utils.logger import setup_logger

logger = setup_logger("oracle_generator")


@dataclass
class OracleTrainingSample:
    qid: str
    question: str
    gold_answers: List[str]
    oracle_k: int
    hit_at_candidate_k: Dict[int, int]
    features: QueryRetrievalFeatures

    def to_flat_dict(self) -> Dict[str, object]:
        """Flattens sample into a tabular dictionary for ML training."""
        row: Dict[str, object] = {
            "qid": self.qid,
            "question": self.question,
            "oracle_k": self.oracle_k,
        }
        # Add the 10 numeric features
        row.update(self.features.to_feature_dict())
        # Add Hit@K indicator for each candidate budget
        for k, hit in self.hit_at_candidate_k.items():
            row[f"hit_k{k}"] = hit
        return row


class OracleLabeler:
    """
    Implements the SAGE Oracle Decision Rule to build training datasets.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        candidate_k_values: Sequence[int] = CANDIDATE_K_VALUES,
        slo_latency_seconds: float = SLO_P95_LATENCY_SECONDS,
    ):
        self.retriever = retriever
        self.candidate_k_values = sorted(list(candidate_k_values))
        self.slo_latency_ms = slo_latency_seconds * 1000.0

    @staticmethod
    def check_hit(passages: List[ScoredPassage], gold_answers: Sequence[str]) -> bool:
        """
        Returns True if any gold answer substring appears in the retrieved passage text.
        """
        if not gold_answers or not passages:
            return False

        clean_golds = [g.strip().lower() for g in gold_answers if g.strip()]
        if not clean_golds:
            return False

        for p in passages:
            text_lower = p.text.lower()
            if any(gold in text_lower for gold in clean_golds):
                return True
        return False

    def compute_oracle_k(
        self,
        candidate_passages: List[ScoredPassage],
        gold_answers: Sequence[str],
        total_latency_ms: float,
    ) -> Tuple[int, Dict[int, int]]:
        """
        Determines the minimal optimal budget K* using the SAGE Decision Rule:
        K* = min { K in candidate_k | Hit@K == 1 and Latency <= SLO }
        If no K achieves a hit, K* = min(candidate_k) (save tokens/latency).
        """
        hit_map: Dict[int, int] = {}
        min_k = self.candidate_k_values[0]
        optimal_k: Optional[int] = None

        # Check Hit@K for each candidate budget in ascending order
        for k in self.candidate_k_values:
            slice_k = candidate_passages[:k]
            is_hit = self.check_hit(slice_k, gold_answers)
            hit_map[k] = 1 if is_hit else 0

            # First K that achieves a hit within latency budget is optimal
            if is_hit and optimal_k is None:
                if total_latency_ms <= self.slo_latency_ms:
                    optimal_k = k

        # Fallback to minimum budget if no hit occurred
        if optimal_k is None:
            optimal_k = min_k

        return optimal_k, hit_map

    def process_query(self, record: QuestionRecord) -> OracleTrainingSample:
        """
        Runs candidate retrieval, extracts probe features, and determines Oracle K*.
        """
        # 1. Retrieve full candidate pool (depth=30)
        max_depth = max(self.candidate_k_values)
        retrieval_res = self.retriever.search(record.question, top_k=max_depth)

        # 2. Extract probe features (using top-2 slices from retrieval_res)
        features = FeatureExtractor.extract_features(
            query=record.question,
            retrieval_result=retrieval_res,
            slo_budget_ms=self.slo_latency_ms,
        )

        # 3. Compute Oracle K*
        oracle_k, hit_map = self.compute_oracle_k(
            candidate_passages=retrieval_res.passages,
            gold_answers=record.short_answers,
            total_latency_ms=retrieval_res.total_retrieval_latency_ms,
        )

        return OracleTrainingSample(
            qid=record.qid,
            question=record.question,
            gold_answers=record.short_answers,
            oracle_k=oracle_k,
            hit_at_candidate_k=hit_map,
            features=features,
        )

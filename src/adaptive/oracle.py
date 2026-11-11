"""
src/adaptive/oracle.py

Quality-Aware, SLO-Constrained Oracle Labeler for SAGE adaptive retrieval.
Selects the optimal retrieval budget K* per query using the formal objective:

    K* = argmax_K Q(q, K)   subject to   L(q, K) <= T_SLO (5.0 seconds)

Tie-breaker:
    Among feasible candidates with equivalent maximum answer quality,
    choose the smallest K.

Infeasibility / Missing Ground Truth:
    - If no candidate satisfies L(q, K) <= T_SLO, the query is marked as having
      no feasible candidate (selected_k=None) and excluded from supervised training.
    - Queries lacking short-answer annotations preserve missing-ground-truth
      status (answer_quality=None) and are never assigned zero quality.
"""

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence
import time

from src.config import (
    CANDIDATE_K_VALUES,
    RETRIEVAL_CANDIDATE_POOL,
    SLO_P95_LATENCY_SECONDS,
)
from src.data.loader import QuestionRecord
from src.evaluation.benchmark_metrics import (
    compute_answer_presence,
    compute_answer_quality,
    compute_exact_match,
    compute_retrieval_hit,
    compute_token_f1,
)
from src.generation.generator import GroqGenerator
from src.retrieval.features import FEATURE_NAMES, FeatureExtractor, QueryRetrievalFeatures
from src.retrieval.hybrid import HybridRetriever
from src.utils.logger import setup_logger

logger = setup_logger("oracle_generator")


@dataclass
class CandidateEvaluation:
    """Single (query, candidate_k) evaluation record for oracle_candidate_evaluations.csv."""
    query_id: str
    query: str
    candidate_k: int
    generated_answer: str
    ground_truth_answer: str
    answer_quality: Optional[float]
    exact_match: Optional[int]
    token_f1: Optional[float]
    answer_presence: Optional[int]
    retrieval_hit: Optional[int]
    retrieval_latency_ms: float
    generation_latency_ms: float
    end_to_end_latency_ms: float
    slo_satisfied: bool
    evaluation_status: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    error_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OracleSelectionResult:
    """Per-query Oracle K* selection summary for oracle_selected_labels.csv."""
    query_id: str
    query: str
    ground_truth_status: str
    selected_k: Optional[int]
    selected_answer_quality: Optional[float]
    selected_latency_ms: Optional[float]
    feasible_candidate_count: int
    has_feasible_candidate: bool
    selection_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OracleTrainingRow:
    """Leakage-free supervised training row for oracle_dataset.csv."""
    query_id: str
    features: QueryRetrievalFeatures
    oracle_k: int

    def to_dict(self) -> Dict[str, Any]:
        row: Dict[str, Any] = {"query_id": self.query_id}
        row.update(self.features.to_feature_dict())
        row["oracle_k"] = int(self.oracle_k)
        return row


class QualityAwareOracle:
    """
    Implements the quality-maximizing, latency-constrained SAGE Oracle.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        generator: Optional[GroqGenerator] = None,
        candidate_k_values: Sequence[int] = CANDIDATE_K_VALUES,
        slo_latency_seconds: float = SLO_P95_LATENCY_SECONDS,
        quality_tie_tol: float = 1e-6,
    ):
        self.retriever = retriever
        self.generator = generator
        self.candidate_k_values = sorted(list(candidate_k_values))
        self.slo_latency_ms = float(slo_latency_seconds) * 1000.0
        self.quality_tie_tol = float(quality_tie_tol)

    @staticmethod
    def is_query_eligible(record: QuestionRecord) -> bool:
        """
        Determines whether a question has valid short-answer ground truth
        eligible for answer-quality scoring.
        """
        clean_answers = [a.strip() for a in record.short_answers if a and a.strip()]
        return bool(record.has_answer and len(clean_answers) > 0)

    def extract_probe_features(self, record: QuestionRecord) -> QueryRetrievalFeatures:
        """
        Extracts the 10 inference-time SAGE probe features cleanly without ground-truth leakage.
        Starts request latency timer right before initial probe retrieval so
        time_remaining_ms reflects steady-state retrieval budget consumption.
        """
        if self.retriever is None:
            raise RuntimeError("HybridRetriever must be initialized before extracting probe features.")

        req_start = time.perf_counter()
        pool_depth = max(RETRIEVAL_CANDIDATE_POOL, 2)
        dense_candidates = self.retriever.dense_retriever.search(record.question, top_k=pool_depth)
        sparse_candidates = self.retriever.sparse_retriever.search(record.question, top_k=pool_depth)

        chunks = self.retriever.chunks
        dense_ids = [
            chunks[idx].chunk_id if chunks and idx < len(chunks) else f"chunk_{idx}"
            for idx, _ in dense_candidates
        ]
        sparse_ids = [
            chunks[idx].chunk_id if chunks and idx < len(chunks) else f"chunk_{idx}"
            for idx, _ in sparse_candidates
        ]
        dense_scores = [score for _, score in dense_candidates]
        sparse_scores = [score for _, score in sparse_candidates]

        return FeatureExtractor.extract_features(
            query=record.question,
            dense_scores=dense_scores,
            dense_passage_ids=dense_ids,
            sparse_scores=sparse_scores,
            sparse_passage_ids=sparse_ids,
            request_start_time=req_start,
            slo_budget_ms=self.slo_latency_ms,
        )

    def evaluate_candidate(self, record: QuestionRecord, candidate_k: int) -> CandidateEvaluation:
        """
        Executes hybrid retrieval at top_k=candidate_k, generates an answer via LLM,
        measures stage and end-to-end latencies, and scores answer quality against ground truth.
        """
        if not self.is_query_eligible(record):
            return CandidateEvaluation(
                query_id=record.qid,
                query=record.question,
                candidate_k=candidate_k,
                generated_answer="",
                ground_truth_answer="",
                answer_quality=None,
                exact_match=None,
                token_f1=None,
                answer_presence=None,
                retrieval_hit=None,
                retrieval_latency_ms=0.0,
                generation_latency_ms=0.0,
                end_to_end_latency_ms=0.0,
                slo_satisfied=False,
                evaluation_status="missing_ground_truth",
                error_message="Missing short-answer annotations; excluded from quality scoring.",
            )

        if self.retriever is None or self.generator is None:
            raise RuntimeError("Both retriever and generator must be initialized for candidate evaluation.")

        clean_golds = [a.strip() for a in record.short_answers if a and a.strip()]
        gt_joined = " | ".join(clean_golds)

        try:
            t_e2e_start = time.perf_counter()

            # 1. Hybrid retrieval for candidate budget K
            ret_res = self.retriever.search(record.question, top_k=candidate_k)

            # 2. LLM answer generation with configured model and greedy decoding
            gen_res = self.generator.generate(query=record.question, passages=ret_res.passages)

            wall_elapsed_ms = (time.perf_counter() - t_e2e_start) * 1000.0
            # Exclude external free-tier TPM throttling waits so L(q, K) reflects inference latency
            e2e_latency_ms = max(0.0, wall_elapsed_ms - getattr(gen_res, "rate_limit_wait_ms", 0.0))

            # 3. Compute answer quality using verified evaluation metrics
            ans_quality = compute_answer_quality(gen_res.generated_answer, clean_golds)
            em = compute_exact_match(gen_res.generated_answer, clean_golds)
            f1 = compute_token_f1(gen_res.generated_answer, clean_golds)
            presence = compute_answer_presence(gen_res.generated_answer, clean_golds)
            ret_hit = compute_retrieval_hit(ret_res.passages, clean_golds)

            slo_ok = bool(e2e_latency_ms <= self.slo_latency_ms)

            return CandidateEvaluation(
                query_id=record.qid,
                query=record.question,
                candidate_k=candidate_k,
                generated_answer=gen_res.generated_answer,
                ground_truth_answer=gt_joined,
                answer_quality=ans_quality,
                exact_match=em,
                token_f1=f1,
                answer_presence=presence,
                retrieval_hit=ret_hit,
                retrieval_latency_ms=float(ret_res.total_retrieval_latency_ms),
                generation_latency_ms=float(gen_res.latency_ms),
                end_to_end_latency_ms=float(e2e_latency_ms),
                slo_satisfied=slo_ok,
                evaluation_status="evaluated",
                prompt_tokens=gen_res.prompt_tokens,
                completion_tokens=gen_res.completion_tokens,
                total_tokens=gen_res.total_tokens,
                error_message="",
            )

        except Exception as exc:
            logger.error(f"Evaluation failed for qid={record.qid}, K={candidate_k}: {exc}")
            return CandidateEvaluation(
                query_id=record.qid,
                query=record.question,
                candidate_k=candidate_k,
                generated_answer="",
                ground_truth_answer=gt_joined,
                answer_quality=None,
                exact_match=None,
                token_f1=None,
                answer_presence=None,
                retrieval_hit=None,
                retrieval_latency_ms=0.0,
                generation_latency_ms=0.0,
                end_to_end_latency_ms=0.0,
                slo_satisfied=False,
                evaluation_status="error",
                error_message=str(exc),
            )

    def select_oracle_k(
        self,
        query_id: str,
        query: str,
        evaluations: Sequence[CandidateEvaluation],
        is_eligible: bool = True,
    ) -> OracleSelectionResult:
        """
        Applies the formal SAGE Oracle objective:
            K* = argmax Q(q, K)  subject to  L(q, K) <= T_SLO
        Tie-breaker: smallest K among feasible candidates tied for max Q.
        Does NOT silently pick infeasible candidates when none satisfy SLO.
        """
        if not is_eligible:
            return OracleSelectionResult(
                query_id=query_id,
                query=query,
                ground_truth_status="missing_ground_truth",
                selected_k=None,
                selected_answer_quality=None,
                selected_latency_ms=None,
                feasible_candidate_count=0,
                has_feasible_candidate=False,
                selection_reason="missing_ground_truth_excluded",
            )

        # Evaluate only properly evaluated candidates
        evaluated = [
            e for e in evaluations
            if e.evaluation_status == "evaluated" and e.answer_quality is not None
        ]
        if not evaluated:
            return OracleSelectionResult(
                query_id=query_id,
                query=query,
                ground_truth_status="eligible",
                selected_k=None,
                selected_answer_quality=None,
                selected_latency_ms=None,
                feasible_candidate_count=0,
                has_feasible_candidate=False,
                selection_reason="all_candidates_failed_evaluation",
            )

        # Rule 1: Filter strictly by SLO constraint L(q, K) <= SLO
        feasible = [
            e for e in evaluated
            if e.end_to_end_latency_ms <= self.slo_latency_ms and e.slo_satisfied
        ]

        # Rule 4 & 5: Reject infeasible runs without fallback
        if not feasible:
            return OracleSelectionResult(
                query_id=query_id,
                query=query,
                ground_truth_status="eligible",
                selected_k=None,
                selected_answer_quality=None,
                selected_latency_ms=None,
                feasible_candidate_count=0,
                has_feasible_candidate=False,
                selection_reason="no_feasible_candidate_within_slo",
            )

        # Rule 2: Maximize answer quality among feasible candidates
        max_quality = max(float(e.answer_quality) for e in feasible if e.answer_quality is not None)
        tied_best = [
            e for e in feasible
            if e.answer_quality is not None and abs(float(e.answer_quality) - max_quality) <= self.quality_tie_tol
        ]

        # Rule 3: Tie-breaker -> smallest K
        chosen = min(tied_best, key=lambda e: e.candidate_k)

        if max_quality <= 0.0:
            reason = "all_feasible_zero_quality_smallest_k"
        elif len(tied_best) > 1:
            reason = "max_quality_tie_smallest_k"
        else:
            reason = "max_quality_within_slo"

        return OracleSelectionResult(
            query_id=query_id,
            query=query,
            ground_truth_status="eligible",
            selected_k=int(chosen.candidate_k),
            selected_answer_quality=float(chosen.answer_quality),
            selected_latency_ms=float(chosen.end_to_end_latency_ms),
            feasible_candidate_count=len(feasible),
            has_feasible_candidate=True,
            selection_reason=reason,
        )

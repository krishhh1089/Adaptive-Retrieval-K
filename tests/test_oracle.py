"""
tests/test_oracle.py

Unit tests for the redesigned quality-aware, SLO-constrained SAGE Oracle.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from src.adaptive.oracle import (
    CandidateEvaluation,
    OracleSelectionResult,
    OracleTrainingRow,
    QualityAwareOracle,
)
from src.data.loader import QuestionRecord
from src.evaluation.benchmark_metrics import compute_answer_quality
from src.generation.generator import GenerationResult
from src.retrieval.features import FEATURE_NAMES, QueryRetrievalFeatures
from src.retrieval.hybrid import HybridRetrievalResult, ScoredPassage


def make_candidate_eval(
    qid: str,
    k: int,
    quality: float,
    latency_ms: float,
    slo_ms: float = 5000.0,
    status: str = "evaluated",
) -> CandidateEvaluation:
    return CandidateEvaluation(
        query_id=qid,
        query="sample question",
        candidate_k=k,
        generated_answer="sample answer",
        ground_truth_answer="sample answer",
        answer_quality=quality if status == "evaluated" else None,
        exact_match=1 if quality == 1.0 else 0,
        token_f1=quality if status == "evaluated" else None,
        answer_presence=1 if quality > 0 else 0,
        retrieval_hit=1,
        retrieval_latency_ms=300.0,
        generation_latency_ms=max(0.0, latency_ms - 300.0),
        end_to_end_latency_ms=latency_ms,
        slo_satisfied=latency_ms <= slo_ms,
        evaluation_status=status,
    )


def test_query_eligibility_and_missing_gt_preservation():
    eligible_q = QuestionRecord(qid="q_1", question="Who?", target_doc_id="d1", short_answers=["Apollo"], has_answer=True)
    missing_q = QuestionRecord(qid="q_2", question="Why?", target_doc_id="d2", short_answers=[], has_answer=False)
    blank_q = QuestionRecord(qid="q_3", question="What?", target_doc_id="d3", short_answers=["  "], has_answer=True)

    assert QualityAwareOracle.is_query_eligible(eligible_q) is True
    assert QualityAwareOracle.is_query_eligible(missing_q) is False
    assert QualityAwareOracle.is_query_eligible(blank_q) is False

    oracle = QualityAwareOracle(slo_latency_seconds=5.0)
    ev = oracle.evaluate_candidate(missing_q, candidate_k=2)
    assert ev.evaluation_status == "missing_ground_truth"
    assert ev.answer_quality is None
    assert ev.token_f1 is None

    sel = oracle.select_oracle_k(missing_q.qid, missing_q.question, [ev], is_eligible=False)
    assert sel.ground_truth_status == "missing_ground_truth"
    assert sel.has_feasible_candidate is False
    assert sel.selected_k is None
    assert sel.selection_reason == "missing_ground_truth_excluded"


def test_answer_quality_scoring_and_abstention():
    assert compute_answer_quality("Paris", []) is None
    assert compute_answer_quality("I cannot answer based on the provided context.", ["Paris"]) == 0.0
    assert compute_answer_quality("I cannot answer based on the provided context.", ["context"]) == 0.0
    assert compute_answer_quality("Paris", ["Paris"]) == 1.0
    partial = compute_answer_quality("The capital is Paris.", ["Paris"])
    assert 0.0 < partial < 1.0


def test_oracle_maximizes_quality_subject_to_slo():
    oracle = QualityAwareOracle(slo_latency_seconds=5.0)
    evals = [
        make_candidate_eval("q_1", 2, quality=0.25, latency_ms=1200.0),
        make_candidate_eval("q_1", 3, quality=0.50, latency_ms=1800.0),
        make_candidate_eval("q_1", 5, quality=0.90, latency_ms=3400.0),
        make_candidate_eval("q_1", 10, quality=1.00, latency_ms=5200.0),  # Infeasible (> 5000ms)
        make_candidate_eval("q_1", 30, quality=1.00, latency_ms=15000.0), # Infeasible (> 5000ms)
    ]

    res = oracle.select_oracle_k("q_1", "sample", evals, is_eligible=True)
    assert res.has_feasible_candidate is True
    assert res.selected_k == 5
    assert res.selected_answer_quality == 0.90
    assert res.selected_latency_ms == 3400.0
    assert res.feasible_candidate_count == 3
    assert res.selection_reason == "max_quality_within_slo"


def test_oracle_tie_breaker_prefers_smaller_k():
    oracle = QualityAwareOracle(slo_latency_seconds=5.0)
    evals = [
        make_candidate_eval("q_2", 2, quality=0.0, latency_ms=1000.0),
        make_candidate_eval("q_2", 3, quality=0.80, latency_ms=2100.0),
        make_candidate_eval("q_2", 5, quality=0.80, latency_ms=2900.0),
        make_candidate_eval("q_2", 7, quality=0.80, latency_ms=4100.0),
    ]

    res = oracle.select_oracle_k("q_2", "sample", evals, is_eligible=True)
    assert res.has_feasible_candidate is True
    assert res.selected_k == 3
    assert res.selection_reason == "max_quality_tie_smallest_k"


def test_oracle_rejects_infeasible_queries():
    oracle = QualityAwareOracle(slo_latency_seconds=5.0)
    evals = [
        make_candidate_eval("q_3", 2, quality=1.0, latency_ms=5100.0),
        make_candidate_eval("q_3", 5, quality=1.0, latency_ms=6500.0),
    ]

    res = oracle.select_oracle_k("q_3", "sample", evals, is_eligible=True)
    assert res.has_feasible_candidate is False
    assert res.selected_k is None
    assert res.selected_answer_quality is None
    assert res.feasible_candidate_count == 0
    assert res.selection_reason == "no_feasible_candidate_within_slo"


def test_oracle_training_row_leakage_free():
    features = QueryRetrievalFeatures(
        query_char_len=40.0,
        query_word_count=8.0,
        query_type=1.0,
        query_complexity=0.0,
        dense_top1_score=0.75,
        dense_score_drop_1_2=0.10,
        bm25_top1_score=22.0,
        bm25_score_drop_1_2=2.0,
        jaccard_top2=0.33,
        time_remaining_ms=4600.0,
    )
    row = OracleTrainingRow(query_id="q_1", features=features, oracle_k=5).to_dict()

    expected_keys = {"query_id", *FEATURE_NAMES, "oracle_k"}
    assert set(row.keys()) == expected_keys
    # Confirm leakage columns are strictly absent
    forbidden = {
        "ground_truth_answer",
        "generated_answer",
        "answer_quality",
        "end_to_end_latency_ms",
        "slo_satisfied",
    }
    assert not (set(row.keys()) & forbidden)


def test_evaluate_candidate_excludes_rate_limit_sleep():
    mock_retriever = MagicMock()
    mock_passage = ScoredPassage(0, "c0", "d0", "Title", "The capital of France is Paris.", 0.5)
    mock_retriever.search.return_value = HybridRetrievalResult(
        query="What is the capital of France?",
        passages=[mock_passage],
        dense_latency_ms=250.0,
        sparse_latency_ms=40.0,
        fusion_latency_ms=0.1,
        total_retrieval_latency_ms=290.1,
    )

    mock_generator = MagicMock()
    mock_generator.generate.return_value = GenerationResult(
        query="What is the capital of France?",
        context_used="context",
        num_passages=1,
        generated_answer="Paris",
        prompt_tokens=100,
        completion_tokens=10,
        total_tokens=110,
        latency_ms=900.0,
        model_id="openai/gpt-oss-120b",
        rate_limit_wait_ms=0.0,
    )

    oracle = QualityAwareOracle(retriever=mock_retriever, generator=mock_generator, slo_latency_seconds=5.0)
    record = QuestionRecord(qid="q_10", question="What is the capital of France?", target_doc_id="d0", short_answers=["Paris"], has_answer=True)

    ev = oracle.evaluate_candidate(record, candidate_k=2)
    assert ev.evaluation_status == "evaluated"
    assert ev.answer_quality == 1.0
    assert ev.exact_match == 1
    assert ev.slo_satisfied is True
    assert ev.end_to_end_latency_ms >= 0.0

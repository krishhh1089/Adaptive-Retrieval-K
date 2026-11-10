"""Retrieval and generation evaluation metrics and latency instrumentation module."""

from src.evaluation.latency_tracker import StageLatencies, StageLatencyTracker
from src.evaluation.benchmark_metrics import (
    compute_retrieval_hit,
    compute_first_hit_rank,
    compute_answer_presence,
    compute_exact_match,
    compute_token_f1,
    compute_faithfulness,
    calculate_distribution_stats,
    compute_relative_efficiency,
)
from src.evaluation.latency_analysis import LatencyAnalyzer

__all__ = [
    "StageLatencies",
    "StageLatencyTracker",
    "compute_retrieval_hit",
    "compute_first_hit_rank",
    "compute_answer_presence",
    "compute_exact_match",
    "compute_token_f1",
    "compute_faithfulness",
    "calculate_distribution_stats",
    "compute_relative_efficiency",
    "LatencyAnalyzer",
]

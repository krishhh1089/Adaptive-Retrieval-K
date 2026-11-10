"""
tests/test_evaluation.py

Unit tests for evaluation metrics, latency tracker, and latency analysis module.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.benchmark_metrics import (
    calculate_distribution_stats,
    compute_answer_presence,
    compute_exact_match,
    compute_faithfulness,
    compute_first_hit_rank,
    compute_relative_efficiency,
    compute_retrieval_hit,
    compute_token_f1,
    normalize_text,
)
from src.evaluation.latency_analysis import LatencyAnalyzer
from src.evaluation.latency_tracker import StageLatencies, StageLatencyTracker
from src.retrieval.hybrid import ScoredPassage


def test_normalize_text():
    raw = "The quick, BROWN FOX!   "
    expected = "quick brown fox"
    assert normalize_text(raw) == expected


def test_compute_retrieval_hit_and_rank():
    p1 = ScoredPassage(chunk_index=0, chunk_id="c0", doc_id="d0", title="T1", text="The capital of France is Paris.", rrf_score=0.1)
    p2 = ScoredPassage(chunk_index=1, chunk_id="c1", doc_id="d1", title="T2", text="Berlin is in Germany.", rrf_score=0.05)

    assert compute_retrieval_hit([p1, p2], ["Paris"]) == 1
    assert compute_retrieval_hit([p1, p2], ["Rome"]) == 0
    assert compute_first_hit_rank([p1, p2], ["Berlin"]) == 2
    assert compute_first_hit_rank([p1, p2], ["Rome"]) is None


def test_compute_answer_metrics():
    gold = ["Persephone", "Kore"]

    # Exact Match
    assert compute_exact_match("Persephone", gold) == 1
    assert compute_exact_match("persephone.", gold) == 1
    assert compute_exact_match("Demeter", gold) == 0

    # Answer presence
    assert compute_answer_presence("In Greek myth, the goddess was Persephone.", gold) == 1
    assert compute_answer_presence("Athena was the goddess of wisdom.", gold) == 0

    # Token F1
    f1_exact = compute_token_f1("Persephone", gold)
    assert np.isclose(f1_exact, 1.0)

    f1_partial = compute_token_f1("Goddess Persephone", gold)
    # gen tokens: [goddess, persephone], gold: [persephone] -> precision: 1/2, recall: 1/1 -> F1: 2*(0.5*1)/(0.5+1) = 2/3 = 0.6667
    assert np.isclose(f1_partial, 2.0 / 3.0, atol=1e-3)


def test_compute_faithfulness():
    p = ScoredPassage(chunk_index=0, chunk_id="c0", doc_id="d0", title="T", text="Apollo was the god of music and poetry.", rrf_score=0.1)

    # Grounded answer
    faith_grounded = compute_faithfulness("Apollo was the god of music.", [p])
    assert faith_grounded > 0.8

    # Refusal / can't answer
    faith_refusal = compute_faithfulness("I cannot answer based on the provided context.", [p])
    assert faith_refusal == 1.0


def test_distribution_stats():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    stats = calculate_distribution_stats(vals)
    assert stats["mean"] == 30.0
    assert stats["p50"] == 30.0
    assert stats["min"] == 10.0
    assert stats["max"] == 50.0
    assert stats["p95"] == 48.0


def test_relative_efficiency():
    # Latency: baseline=100ms, target=60ms -> 40% reduction
    eff = compute_relative_efficiency(baseline_val=100.0, target_val=60.0, higher_is_better=False)
    assert np.isclose(eff, 40.0)

    # Token savings: baseline=1000, target=400 -> 60% savings
    tok_eff = compute_relative_efficiency(baseline_val=1000.0, target_val=400.0, higher_is_better=False)
    assert np.isclose(tok_eff, 60.0)


def test_stage_latency_tracker():
    tracker = StageLatencyTracker()
    tracker.start_request()

    with tracker.time_stage("dense_retrieval_ms"):
        x = sum(i for i in range(1000))

    tracker.record_stage("sparse_retrieval_ms", 12.5)
    total_ms = tracker.stop_request()

    assert tracker.latencies.dense_retrieval_ms >= 0.0
    assert tracker.latencies.sparse_retrieval_ms == 12.5
    assert total_ms >= tracker.latencies.dense_retrieval_ms


def test_latency_analyzer_and_plots(tmp_path):
    # Create synthetic query DataFrame across 2 configs
    records = []
    for cfg in ["fixed_k2", "adaptive_sage"]:
        for q in range(4):
            records.append({
                "config_name": cfg,
                "selected_k": 2 if cfg == "fixed_k2" else 3,
                "dense_retrieval_ms": 3.0,
                "sparse_retrieval_ms": 40.0,
                "hybrid_fusion_ms": 1.0,
                "feature_extraction_ms": 0.0 if cfg == "fixed_k2" else 2.0,
                "policy_inference_ms": 0.0 if cfg == "fixed_k2" else 1.0,
                "context_preparation_ms": 0.5,
                "llm_generation_ms": 400.0,
                "total_end_to_end_ms": 444.5 if cfg == "fixed_k2" else 447.5,
                "prompt_tokens": 150,
                "completion_tokens": 20,
                "total_tokens": 170,
                "token_f1": 0.75,
                "retrieval_hit": 1,
            })
    df = pd.DataFrame(records)

    analyzer = LatencyAnalyzer(query_results_df=df, output_dir=tmp_path)
    summary_df = analyzer.compute_stage_summary()
    assert len(summary_df) > 0

    bottlenecks = analyzer.identify_bottlenecks()
    assert "fixed_k2" in bottlenecks
    assert "adaptive_sage" in bottlenecks
    assert bottlenecks["fixed_k2"]["primary_bottleneck"] == "LLM Generation (Groq)"

    plots = analyzer.generate_all_visualizations()
    assert len(plots) == 6
    for p in plots:
        assert p.exists()


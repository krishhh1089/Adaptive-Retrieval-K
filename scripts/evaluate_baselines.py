"""
scripts/evaluate_baselines.py

Phase 8.3 & Phase 8.4: Baseline Benchmarking and Comprehensive Latency Analysis.
Evaluates Fixed-K baselines (K in {2, 5, 10, 30}) against the trained SAGE Adaptive Policy.
Instruments stage-wise latencies, token consumption, and retrieval/answer quality.
Generates research metrics and comparative visualizations.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.adaptive.policy import SAGEPolicy
from src.data.chunker import DocumentChunker
from src.data.loader import NQDataLoader, QuestionRecord
from src.evaluation.benchmark_metrics import (
    calculate_distribution_stats,
    compute_answer_presence,
    compute_exact_match,
    compute_faithfulness,
    compute_first_hit_rank,
    compute_relative_efficiency,
    compute_retrieval_hit,
    compute_token_f1,
)
from src.evaluation.latency_analysis import LatencyAnalyzer
from src.evaluation.latency_tracker import StageLatencyTracker
from src.generation.generator import GroqGenerator
from src.retrieval.features import FeatureExtractor
from src.retrieval.hybrid import HybridRetriever, ScoredPassage
from src.utils.logger import setup_logger

logger = setup_logger("evaluate_baselines")

# Evaluation configurations
CONFIGS = [
    {"name": "fixed_k2", "type": "fixed", "k": 2},
    {"name": "fixed_k5", "type": "fixed", "k": 5},
    {"name": "fixed_k10", "type": "fixed", "k": 10},
    {"name": "fixed_k30", "type": "fixed", "k": 30},
    {"name": "adaptive_sage", "type": "adaptive", "k": None},
]


def run_single_pipeline(
    query: str,
    gold_answers: List[str],
    cfg: Dict[str, Any],
    retriever: HybridRetriever,
    generator: GroqGenerator,
    policy: Optional[SAGEPolicy],
) -> Dict[str, Any]:
    """
    Executes a single query through a specified retrieval-generation configuration
    with granular stage-wise monotonic latency instrumentation.
    """
    tracker = StageLatencyTracker()
    req_start = tracker.start_request()

    cfg_name = cfg["name"]
    is_adaptive = cfg["type"] == "adaptive"
    fixed_k = cfg["k"]

    dense_candidates = []
    sparse_candidates = []
    fused_passages: List[ScoredPassage] = []
    selected_k = fixed_k or 5
    gen_result = None
    error_msg = ""
    success = 1

    try:
        # Candidate pool depth
        pool_depth = max(config.RETRIEVAL_CANDIDATE_POOL, 30)

        # 1. Dense retrieval
        with tracker.time_stage("dense_retrieval_ms"):
            dense_candidates = retriever.dense_retriever.search(query, top_k=pool_depth)

        # 2. Sparse (BM25) retrieval
        with tracker.time_stage("sparse_retrieval_ms"):
            sparse_candidates = retriever.sparse_retriever.search(query, top_k=pool_depth)

        # 3. If Adaptive SAGE: Feature Extraction & Policy Inference
        if is_adaptive and policy is not None:
            # Map candidate chunk IDs and scores for probe
            chunks = retriever.chunks
            dense_ids = [chunks[idx].chunk_id if chunks and idx < len(chunks) else f"chunk_{idx}" for idx, _ in dense_candidates]
            sparse_ids = [chunks[idx].chunk_id if chunks and idx < len(chunks) else f"chunk_{idx}" for idx, _ in sparse_candidates]
            dense_scores = [score for _, score in dense_candidates]
            sparse_scores = [score for _, score in sparse_candidates]

            with tracker.time_stage("feature_extraction_ms"):
                probe_features = FeatureExtractor.extract_features(
                    query=query,
                    dense_scores=dense_scores,
                    dense_passage_ids=dense_ids,
                    sparse_scores=sparse_scores,
                    sparse_passage_ids=sparse_ids,
                    request_start_time=req_start,
                    slo_budget_ms=config.SLO_P95_LATENCY_SECONDS * 1000.0,
                )

            with tracker.time_stage("policy_inference_ms"):
                selected_k = policy.predict_single(probe_features)

        # 4. Hybrid Reciprocal Rank Fusion (RRF) up to selected_k
        with tracker.time_stage("hybrid_fusion_ms"):
            fused_passages = retriever.fuse_candidates(
                dense_results=dense_candidates,
                sparse_results=sparse_candidates,
                top_k=selected_k,
            )

        # 5. Context formatting & Prompt Preparation
        with tracker.time_stage("context_preparation_ms"):
            formatted_context = generator.format_context(fused_passages)

        # 6. LLM Generation (Greedy decoding)
        with tracker.time_stage("llm_generation_ms"):
            gen_result = generator.generate(query=query, passages=fused_passages)

    except Exception as e:
        success = 0
        error_msg = str(e)
        logger.error(f"Pipeline error for query '{query[:30]}...' on {cfg_name}: {e}")

    finally:
        total_e2e_ms = tracker.stop_request()

    # Calculate quality metrics
    retrieval_hit = compute_retrieval_hit(fused_passages, gold_answers)
    first_hit_rank = compute_first_hit_rank(fused_passages, gold_answers)

    gen_answer = gen_result.generated_answer if gen_result else ""
    ans_presence = compute_answer_presence(gen_answer, gold_answers)
    exact_match = compute_exact_match(gen_answer, gold_answers)
    token_f1 = compute_token_f1(gen_answer, gold_answers)
    faithfulness = compute_faithfulness(gen_answer, fused_passages)

    record = {
        "config_name": cfg_name,
        "selected_k": selected_k,
        "num_passages_retrieved": len(fused_passages),
        "query": query,
        "gold_answers": "|".join(gold_answers),
        "generated_answer": gen_answer,
        "success": success,
        "error_message": error_msg,
        # Quality metrics
        "retrieval_hit": retrieval_hit,
        "first_hit_rank": first_hit_rank if first_hit_rank is not None else -1,
        "answer_presence": ans_presence,
        "exact_match": exact_match,
        "token_f1": token_f1,
        "faithfulness": faithfulness,
        # Tokens
        "context_chars": len(gen_result.context_used) if gen_result else 0,
        "prompt_tokens": gen_result.prompt_tokens if gen_result else 0,
        "completion_tokens": gen_result.completion_tokens if gen_result else 0,
        "total_tokens": gen_result.total_tokens if gen_result else 0,
    }

    # Add stage latencies
    record.update(tracker.latencies.to_dict())
    return record


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAGE adaptive policy against fixed-K baselines.")
    parser.add_argument("--sample_size", type=int, default=5, help="Number of queries to benchmark (default: 5). Set to 0 for all.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warm-up queries to run (cold-start discard).")
    parser.add_argument("--policy_path", type=str, default=str(config.SAGE_MODEL_DIR / "sage_policy.pkl"))
    parser.add_argument("--output_dir", type=str, default=str(config.PROJECT_ROOT / "results" / "baseline_benchmarks"))
    parser.add_argument("--analysis_dir", type=str, default=str(config.PROJECT_ROOT / "results" / "latency_analysis"))
    parser.add_argument("--from_existing_results", action="store_true", help="Skip LLM generations and recompute metrics/plots from existing query_level_results.csv.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    analysis_dir = Path(args.analysis_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=================================================================")
    logger.info("PHASE 8.3 & 8.4: BASELINE BENCHMARKING AND LATENCY ANALYSIS")
    logger.info("=================================================================")

    query_csv_path = output_dir / "query_level_results.csv"

    if args.from_existing_results and query_csv_path.exists():
        logger.info(f"Loading existing query-level results from: {query_csv_path}")
        query_df = pd.read_csv(query_csv_path)
    else:
        # 1. Initialize Corpus & Hybrid Retriever
        logger.info("Loading prebuilt chunks and retrieval indexes...")
        chunker = DocumentChunker()
        chunks = chunker.load_chunks()
        retriever = HybridRetriever(rrf_k=config.RRF_K_CONSTANT, candidate_pool=config.RETRIEVAL_CANDIDATE_POOL)
        retriever.initialize(chunks)

        # 2. Initialize LLM Generator
        logger.info(f"Initializing Groq Generator (model={config.GROQ_MODEL}, temperature={config.GENERATION_TEMPERATURE})...")
        generator = GroqGenerator()

        # 3. Load Trained Adaptive Policy
        policy_path = Path(args.policy_path)
        if not policy_path.exists():
            raise FileNotFoundError(f"Trained policy artifact not found at {policy_path}. Please train SAGE policy first.")
        logger.info(f"Loading trained SAGE adaptive policy from: {policy_path}")
        policy = SAGEPolicy.load(policy_path)

        # 4. Load Evaluation Dataset
        loader = NQDataLoader()
        all_questions = loader.load_questions()
        answerable_questions = [q for q in all_questions if q.has_answer]
        logger.info(f"Loaded {len(all_questions)} total questions ({len(answerable_questions)} answerable).")

        if args.sample_size and args.sample_size > 0:
            eval_questions = answerable_questions[:args.sample_size]
            logger.info(f"Benchmarking on evaluation slice: {len(eval_questions)} queries.")
        else:
            eval_questions = answerable_questions
            logger.info(f"Benchmarking on full answerable set: {len(eval_questions)} queries.")

        # 5. Cold-Start Warm-Up Run
        if args.warmup > 0 and len(eval_questions) > 0:
            logger.info(f"Executing {args.warmup} warm-up query to eliminate cold-start transients...")
            warmup_q = eval_questions[0]
            for cfg in CONFIGS:
                _ = run_single_pipeline(
                    query=warmup_q.question,
                    gold_answers=warmup_q.short_answers,
                    cfg=cfg,
                    retriever=retriever,
                    generator=generator,
                    policy=policy,
                )
            logger.info("Warm-up complete. Steady-state benchmark starting.")

        # 6. Execute Steady-State Benchmark
        query_records: List[Dict[str, Any]] = []

        total_runs = len(eval_questions) * len(CONFIGS)
        pbar = tqdm(total=total_runs, desc="Benchmarking Configurations")

        for q in eval_questions:
            for cfg in CONFIGS:
                rec = run_single_pipeline(
                    query=q.question,
                    gold_answers=q.short_answers,
                    cfg=cfg,
                    retriever=retriever,
                    generator=generator,
                    policy=policy,
                )
                query_records.append(rec)
                pbar.update(1)

        pbar.close()

        # Convert to DataFrame
        query_df = pd.DataFrame(query_records)
        query_df.to_csv(query_csv_path, index=False)
        logger.info(f"Saved query-level results to: {query_csv_path}")

    # 7. Aggregate Metrics per Configuration
    summary_rows = []
    # Identify fixed_k30 for relative efficiency baseline
    k30_sub = query_df[query_df["config_name"] == "fixed_k30"]
    k30_mean_lat = k30_sub["total_end_to_end_ms"].mean() if len(k30_sub) > 0 else 1.0
    k30_mean_tokens = k30_sub["total_tokens"].mean() if len(k30_sub) > 0 else 1.0
    k30_mean_f1 = k30_sub["token_f1"].mean() if len(k30_sub) > 0 else 0.0

    for cfg in CONFIGS:
        cfg_name = cfg["name"]
        sub = query_df[query_df["config_name"] == cfg_name]
        n_samples = len(sub)

        lat_stats = calculate_distribution_stats(sub["total_end_to_end_ms"].values)
        tok_stats = calculate_distribution_stats(sub["total_tokens"].values)

        mean_k = float(sub["selected_k"].mean())
        recall_at_k = float(sub["retrieval_hit"].mean())
        exact_match = float(sub["exact_match"].mean())
        token_f1 = float(sub["token_f1"].mean())
        ans_presence = float(sub["answer_presence"].mean())
        faithfulness = float(sub["faithfulness"].mean())

        mean_prompt_tok = float(sub["prompt_tokens"].mean())
        mean_comp_tok = float(sub["completion_tokens"].mean())
        mean_gen_lat = float(sub["llm_generation_ms"].mean())
        mean_ret_lat = float((sub["dense_retrieval_ms"] + sub["sparse_retrieval_ms"] + sub["hybrid_fusion_ms"]).mean())

        # Relative efficiency vs K=30
        token_savings_pct = compute_relative_efficiency(k30_mean_tokens, tok_stats["mean"], higher_is_better=False)
        latency_improvement_pct = compute_relative_efficiency(k30_mean_lat, lat_stats["mean"], higher_is_better=False)
        quality_delta = float(token_f1 - k30_mean_f1)

        summary_rows.append({
            "config_name": cfg_name,
            "samples": n_samples,
            "success_rate": float(sub["success"].mean()),
            "avg_k": round(mean_k, 2),
            "recall_at_k": round(recall_at_k, 4),
            "answer_presence": round(ans_presence, 4),
            "exact_match": round(exact_match, 4),
            "token_f1": round(token_f1, 4),
            "faithfulness": round(faithfulness, 4),
            # Latency (ms)
            "mean_latency_ms": round(lat_stats["mean"], 2),
            "p50_latency_ms": round(lat_stats["p50"], 2),
            "p90_latency_ms": round(lat_stats["p90"], 2),
            "p95_latency_ms": round(lat_stats["p95"], 2),
            "p99_latency_ms": round(lat_stats["p99"], 2),
            "min_latency_ms": round(lat_stats["min"], 2),
            "max_latency_ms": round(lat_stats["max"], 2),
            "mean_retrieval_ms": round(mean_ret_lat, 2),
            "mean_generation_ms": round(mean_gen_lat, 2),
            # Token usage
            "avg_prompt_tokens": round(mean_prompt_tok, 1),
            "avg_completion_tokens": round(mean_comp_tok, 1),
            "avg_total_tokens": round(tok_stats["mean"], 1),
            # Relative Efficiency vs. Fixed K=30
            "token_savings_pct_vs_k30": round(token_savings_pct, 2),
            "latency_improvement_pct_vs_k30": round(latency_improvement_pct, 2),
            "quality_delta_vs_k30": round(quality_delta, 4),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = output_dir / "benchmark_results.csv"
    summary_json_path = output_dir / "benchmark_results.json"

    summary_df.to_csv(summary_csv_path, index=False)
    summary_df.to_json(summary_json_path, orient="records", indent=2)
    logger.info(f"Saved aggregated benchmark results to: {summary_csv_path} and {summary_json_path}")

    # 8. Latency Decomposition & Visualizations (Phase 8.4)
    logger.info("Performing latency bottleneck analysis and generating visualization suite...")
    analyzer = LatencyAnalyzer(query_results_df=query_df, output_dir=analysis_dir)
    stage_df = analyzer.compute_stage_summary()
    bottlenecks = analyzer.identify_bottlenecks()
    chart_paths = analyzer.generate_all_visualizations()

    # 9. Record Experiment Metadata
    meta_path = output_dir / "experiment_metadata.json"
    metadata = {
        "experiment_name": "SAGE_Baseline_and_Latency_Analysis",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "runtime_environment": {
            "os": platform.system(),
            "os_release": platform.release(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "models": {
            "llm": config.GROQ_MODEL,
            "llm_temperature": config.GENERATION_TEMPERATURE,
            "dense_retriever": config.DENSE_MODEL_NAME,
            "sparse_retriever": config.SPARSE_ALGORITHM,
            "policy_model": "RandomForestClassifier(n_estimators=100, max_depth=10)",
        },
        "evaluation_setup": {
            "total_queries_evaluated": len(query_df["query"].unique()),
            "warmup_queries": args.warmup,
            "slo_target_seconds": config.SLO_P95_LATENCY_SECONDS,
            "candidate_k_values": config.CANDIDATE_K_VALUES,
        },
        "bottleneck_analysis": bottlenecks,
        "generated_artifacts": {
            "query_level_csv": str(query_csv_path),
            "summary_csv": str(summary_csv_path),
            "summary_json": str(summary_json_path),
            "stage_latency_csv": str(analysis_dir / "stage_latency.csv"),
            "charts": [str(p) for p in chart_paths],
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved experiment metadata to: {meta_path}")

    # Print summary table to console
    print("\n" + "=" * 80)
    print("BENCHMARK RESULTS SUMMARY (Steady-State)")
    print("=" * 80)
    display_cols = [
        "config_name",
        "avg_k",
        "recall_at_k",
        "token_f1",
        "mean_latency_ms",
        "p95_latency_ms",
        "avg_total_tokens",
        "token_savings_pct_vs_k30",
        "latency_improvement_pct_vs_k30",
    ]
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 80)

    print("\nLATENCY BOTTLENECK ANALYSIS:")
    for cfg, binfo in bottlenecks.items():
        print(f"  [{cfg:<14}] Mean: {binfo['total_mean_ms']:7.1f} ms | "
              f"Primary: {binfo['primary_bottleneck']} ({binfo['primary_latency_ms']} ms, {binfo['primary_share_pct']}%) | "
              f"Secondary: {binfo['secondary_bottleneck']} ({binfo['secondary_latency_ms']} ms, {binfo['secondary_share_pct']}%)")
    print("\nAll charts saved under:", analysis_dir)


if __name__ == "__main__":
    main()


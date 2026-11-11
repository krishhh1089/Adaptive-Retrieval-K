"""
scripts/generate_oracle_dataset.py

Builds the quality-aware, SLO-constrained SAGE Oracle dataset across Natural Questions.

Generates three primary datasets in experiments/oracle_results/:
  1. oracle_candidate_evaluations.csv : All evaluated (query, candidate_k) rows.
  2. oracle_selected_labels.csv      : Selected K* (or infeasible status) per eligible query.
  3. oracle_dataset.csv              : Leakage-free RF training data (FEATURE_NAMES -> oracle_k).

Also writes experiments/oracle_results/oracle_excluded_queries.csv to preserve
missing-ground-truth status separately for unevaluable queries without fabricating scores.

Experiment metadata is written to artifacts/metadata/experiments/oracle_experiment_v2.json
strictly upon completion.

Backend Selection:
  Controlled by GENERATION_BACKEND env variable ("ollama" or "groq").
  Default is "ollama" (local inference, no rate-limit pacing).

Resume Cache Validation:
  When --resume is used, the cached backend and model are validated against the
  current configuration. If they differ, the script aborts to prevent silently
  mixing labels from different backends or models. Use --force to start fresh.
"""

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config
from src.adaptive.oracle import (
    CandidateEvaluation,
    OracleSelectionResult,
    OracleTrainingRow,
    QualityAwareOracle,
)
from src.data.chunker import DocumentChunker
from src.data.loader import NQDataLoader, QuestionRecord
from src.retrieval.features import FEATURE_NAMES
from src.retrieval.hybrid import HybridRetriever
from src.utils.logger import setup_logger

logger = setup_logger("generate_oracle")

# ---------------------------------------------------------------------------
# Resume cache validation helpers
# ---------------------------------------------------------------------------

_BACKEND_META_PATH = config.EXPERIMENTS_METADATA_DIR / "oracle_backend_identity.json"


def _write_backend_identity(backend: str, model_id: str) -> None:
    """Persists the active backend and model identity for resume validation."""
    _BACKEND_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_BACKEND_META_PATH, "w", encoding="utf-8") as f:
        json.dump({"backend": backend, "model_id": model_id}, f, indent=2)


def _validate_resume_identity(backend: str, model_id: str) -> None:
    """
    Validates that cached evaluations were generated with the same backend and model.
    Aborts if they differ to prevent silent mixing of Groq and local labels.
    """
    if not _BACKEND_META_PATH.exists():
        # No prior identity file — first run. Write and continue.
        _write_backend_identity(backend, model_id)
        return

    with open(_BACKEND_META_PATH, encoding="utf-8") as f:
        saved = json.load(f)

    saved_backend = saved.get("backend", "")
    saved_model = saved.get("model_id", "")

    if saved_backend != backend or saved_model != model_id:
        raise SystemExit(
            f"\n{'='*70}\n"
            f"RESUME VALIDATION FAILED — Backend or model mismatch.\n"
            f"  Cached evaluations: backend={saved_backend!r}, model={saved_model!r}\n"
            f"  Current config:     backend={backend!r},       model={model_id!r}\n\n"
            f"Mixing labels from different backends or models is not permitted.\n"
            f"Options:\n"
            f"  1. Revert to the original backend/model.\n"
            f"  2. Run with --force to discard cached evaluations and start fresh.\n"
            f"{'='*70}\n"
        )


# ---------------------------------------------------------------------------
# Generator factory
# ---------------------------------------------------------------------------

def _build_generator(backend: str):
    """Instantiates the appropriate generator based on GENERATION_BACKEND."""
    if backend == "ollama":
        from src.generation.local_generator import (
            LocalLLMGenerator,
            OllamaConnectionError,
            OllamaModelNotFoundError,
        )
        generator = LocalLLMGenerator()
        # Pre-flight checks
        if not generator.check_server():
            raise SystemExit(
                f"\nOllama server not reachable at {generator.base_url!r}.\n"
                "Start it with:  ollama serve\n"
            )
        if not generator.check_model_available():
            raise SystemExit(
                f"\nModel {generator.model!r} is not installed in Ollama.\n"
                f"Download it with:  ollama pull {generator.model}\n"
            )
        return generator, generator.model_id

    elif backend == "groq":
        from src.generation.generator import GroqGenerator
        generator = GroqGenerator(pace_for_rate_limit=True, sdk_max_retries=0)
        return generator, f"groq/{generator.model}"

    else:
        raise ValueError(
            f"Unknown GENERATION_BACKEND={backend!r}. "
            "Set GENERATION_BACKEND to 'ollama' or 'groq'."
        )


# ---------------------------------------------------------------------------
# Resume loader
# ---------------------------------------------------------------------------

def _load_existing_evaluations(csv_path: Path) -> Dict[Tuple[str, int], CandidateEvaluation]:
    """Loads previously completed non-error evaluations for safe resuming."""
    existing: Dict[Tuple[str, int], CandidateEvaluation] = {}
    if not csv_path.exists():
        return existing

    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        status = str(row.get("evaluation_status", ""))
        if status != "evaluated":
            continue
        qid = str(row["query_id"])
        k = int(row["candidate_k"])
        existing[(qid, k)] = CandidateEvaluation(
            query_id=qid,
            query=str(row["query"]),
            candidate_k=k,
            generated_answer="" if pd.isna(row.get("generated_answer")) else str(row.get("generated_answer")),
            ground_truth_answer="" if pd.isna(row.get("ground_truth_answer")) else str(row.get("ground_truth_answer")),
            answer_quality=None if pd.isna(row.get("answer_quality")) else float(row["answer_quality"]),
            exact_match=None if pd.isna(row.get("exact_match")) else int(row["exact_match"]),
            token_f1=None if pd.isna(row.get("token_f1")) else float(row["token_f1"]),
            answer_presence=None if pd.isna(row.get("answer_presence")) else int(row["answer_presence"]),
            retrieval_hit=None if pd.isna(row.get("retrieval_hit")) else int(row["retrieval_hit"]),
            retrieval_latency_ms=float(row["retrieval_latency_ms"]),
            generation_latency_ms=float(row["generation_latency_ms"]),
            end_to_end_latency_ms=float(row["end_to_end_latency_ms"]),
            slo_satisfied=bool(row["slo_satisfied"]),
            evaluation_status=status,
            prompt_tokens=0 if pd.isna(row.get("prompt_tokens")) else int(row["prompt_tokens"]),
            completion_tokens=0 if pd.isna(row.get("completion_tokens")) else int(row["completion_tokens"]),
            total_tokens=0 if pd.isna(row.get("total_tokens")) else int(row["total_tokens"]),
            error_message="" if pd.isna(row.get("error_message")) else str(row.get("error_message")),
        )
    return existing


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate quality-aware SAGE Oracle datasets.")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Optional limit on number of eligible queries to evaluate (0 = evaluate all eligible queries).",
    )
    parser.add_argument(
        "--slo-seconds",
        type=float,
        default=config.SLO_P95_LATENCY_SECONDS,
        help=f"End-to-end latency SLO in seconds (default: {config.SLO_P95_LATENCY_SECONDS}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing oracle_candidate_evaluations.csv without re-calling LLM for evaluated pairs.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite any existing Oracle output files and evaluate from scratch.",
    )
    parser.add_argument(
        "--max-consecutive-errors",
        type=int,
        default=5,
        help="Abort cleanly after this many consecutive candidate evaluation failures (default: 5).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 2 & 3: QUALITY-AWARE, SLO-CONSTRAINED ORACLE DATASET GENERATION")
    print("=" * 70)

    # --- Backend configuration ---
    backend = config.GENERATION_BACKEND.lower()
    print(f"\nGeneration backend : {backend.upper()}")

    generator, model_id = _build_generator(backend)
    print(f"Model ID           : {model_id}")
    print(f"Temperature        : {config.GENERATION_TEMPERATURE}")
    print(f"Max tokens         : {config.GENERATION_MAX_TOKENS}")

    # --- Resume cache validation ---
    if args.resume and not args.force:
        _validate_resume_identity(backend, model_id)
        print(f"Resume identity    : backend={backend!r} model={model_id!r} ✓")
    elif args.force:
        # Reset backend identity on --force
        _write_backend_identity(backend, model_id)
        print("Force mode: cleared cached identity, starting fresh.")
    else:
        # First run: record identity
        _write_backend_identity(backend, model_id)

    # 1. Load dataset and separate eligible vs missing ground truth
    loader = NQDataLoader()
    all_questions = loader.load_questions()
    eligible_all = [q for q in all_questions if QualityAwareOracle.is_query_eligible(q)]
    excluded_all = [q for q in all_questions if not QualityAwareOracle.is_query_eligible(q)]

    print(f"\nTotal dataset records          : {len(all_questions)}")
    print(f"Eligible (short answers exist) : {len(eligible_all)}")
    print(f"Excluded (missing ground truth): {len(excluded_all)}")

    eval_queries = eligible_all[: args.max_queries] if args.max_queries > 0 else eligible_all
    if args.max_queries > 0:
        print(f"Running bounded evaluation on  : {len(eval_queries)} eligible queries")

    # Save excluded queries separately without fabricating quality scores
    config.ORACLE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    excluded_path = config.ORACLE_RESULTS_DIR / "oracle_excluded_queries.csv"
    excluded_df = pd.DataFrame(
        [
            {
                "query_id": q.qid,
                "query": q.question,
                "evaluation_status": "missing_ground_truth",
                "reason": "missing_short_answer_annotations",
            }
            for q in excluded_all
        ]
    )
    excluded_df.to_csv(excluded_path, index=False)

    # 2. Initialize retrieval pipeline (architecture preserved)
    chunker = DocumentChunker()
    chunks = chunker.load_chunks()
    retriever = HybridRetriever(
        rrf_k=config.RRF_K_CONSTANT,
        candidate_pool=config.RETRIEVAL_CANDIDATE_POOL,
    )
    retriever.initialize(chunks)

    oracle = QualityAwareOracle(
        retriever=retriever,
        generator=generator,
        candidate_k_values=config.CANDIDATE_K_VALUES,
        slo_latency_seconds=args.slo_seconds,
    )

    # Warm-up dense encoder once so cold-start BGE-M3 load never contaminates latency
    print("\nWarming up BGE-M3 dense encoder (eliminating cold-start bias)...")
    _ = retriever.search(eval_queries[0].question if eval_queries else "warmup query", top_k=2)

    # 3. Check resume cache
    cand_eval_path = config.ORACLE_CANDIDATE_EVAL_PATH
    cached_map: Dict[Tuple[str, int], CandidateEvaluation] = {}
    if args.resume and not args.force:
        cached_map = _load_existing_evaluations(cand_eval_path)
        print(f"Resume mode active: loaded {len(cached_map)} completed candidate evaluations.")

    all_candidate_rows: List[CandidateEvaluation] = []
    seen_keys: Set[Tuple[str, int]] = set()
    selected_rows: List[OracleSelectionResult] = []
    training_rows: List[OracleTrainingRow] = []
    consecutive_errors = 0

    def _flush_all() -> None:
        combined = list(all_candidate_rows)
        for ck, cev in cached_map.items():
            if ck not in seen_keys:
                combined.append(cev)
        pd.DataFrame([e.to_dict() for e in combined]).to_csv(cand_eval_path, index=False)
        if selected_rows:
            pd.DataFrame([s.to_dict() for s in selected_rows]).to_csv(config.ORACLE_SELECTED_LABELS_PATH, index=False)
        if training_rows:
            pd.DataFrame([t.to_dict() for t in training_rows]).to_csv(config.ORACLE_DATASET_PATH, index=False)

    print(
        f"\nEvaluating {len(eval_queries)} queries across K={oracle.candidate_k_values} "
        f"(SLO <= {args.slo_seconds:.1f}s)..."
    )

    for q in tqdm(eval_queries, desc="Oracle Query Evaluation"):
        # Step A: Extract inference-time probe features (leakage-free)
        features = oracle.extract_probe_features(q)

        # Step B: Evaluate every candidate K
        q_evals: List[CandidateEvaluation] = []
        for k in oracle.candidate_k_values:
            key = (q.qid, int(k))
            if key in cached_map:
                ev = cached_map[key]
            else:
                ev = oracle.evaluate_candidate(q, candidate_k=int(k))
                if ev.evaluation_status == "evaluated":
                    consecutive_errors = 0
                elif ev.evaluation_status == "error":
                    consecutive_errors += 1
                    if consecutive_errors >= args.max_consecutive_errors:
                        _flush_all()
                        raise RuntimeError(
                            f"Aborting after {consecutive_errors} consecutive evaluation failures "
                            f"(last error at qid={q.qid}, K={k}: {ev.error_message})."
                        )
            q_evals.append(ev)
            all_candidate_rows.append(ev)
            seen_keys.add(key)

        # Step C: Apply quality-maximizing, SLO-constrained selection
        sel = oracle.select_oracle_k(
            query_id=q.qid,
            query=q.question,
            evaluations=q_evals,
            is_eligible=True,
        )
        selected_rows.append(sel)

        # Step D: Emit leakage-free supervised row only when feasible K* exists
        if sel.has_feasible_candidate and sel.selected_k is not None:
            training_rows.append(
                OracleTrainingRow(
                    query_id=q.qid,
                    features=features,
                    oracle_k=int(sel.selected_k),
                )
            )

        # Flush all 3 datasets incrementally for crash resilience
        _flush_all()

    # 4. Persist final 3 Oracle datasets
    cand_df = pd.DataFrame([e.to_dict() for e in all_candidate_rows])
    sel_df = pd.DataFrame([s.to_dict() for s in selected_rows])
    train_df = pd.DataFrame([t.to_dict() for t in training_rows])

    cand_df.to_csv(config.ORACLE_CANDIDATE_EVAL_PATH, index=False)
    sel_df.to_csv(config.ORACLE_SELECTED_LABELS_PATH, index=False)
    train_df.to_csv(config.ORACLE_DATASET_PATH, index=False)

    # Validate strict schema of training dataset
    if len(train_df) > 0:
        expected_cols = ["query_id", *FEATURE_NAMES, "oracle_k"]
        assert list(train_df.columns) == expected_cols, f"Schema mismatch: {list(train_df.columns)}"

    # Summary statistics
    n_feasible = int(sel_df["has_feasible_candidate"].sum()) if len(sel_df) > 0 else 0
    n_infeasible = len(sel_df) - n_feasible
    k_dist = (
        {str(int(k)): int(v) for k, v in train_df["oracle_k"].value_counts().sort_index().items()}
        if len(train_df) > 0
        else {}
    )

    # 5. Record Experiment Metadata upon completion
    config.EXPERIMENTS_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = config.EXPERIMENTS_METADATA_DIR / "oracle_experiment_v2.json"
    experiment_meta = {
        "experiment_name": "Quality_Aware_SLO_Constrained_Oracle",
        "version": "v2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": "Natural_Questions_100_Sample_v1",
        "candidate_k_values": list(oracle.candidate_k_values),
        "answer_quality_methodology": {
            "metric": "unigram_token_f1_with_abstention_guard",
            "abstention_score": 0.0,
            "missing_ground_truth_handling": "excluded_preserved_as_none",
            "tie_breaking_rule": "smallest_k_among_equivalent_max_quality",
        },
        "latency_slo_seconds": float(args.slo_seconds),
        "latency_slo_note": (
            "SLO measures local end-to-end inference latency (retrieval + local LLM generation). "
            "No rate-limit pacing is included. Results are hardware-dependent."
        ),
        "generation_configuration": {
            "backend": backend,
            "model_id": model_id,
            "temperature": config.GENERATION_TEMPERATURE,
            "max_tokens": config.GENERATION_MAX_TOKENS,
            "ollama_base_url": config.OLLAMA_BASE_URL if backend == "ollama" else None,
            "ollama_timeout_seconds": config.OLLAMA_TIMEOUT_SECONDS if backend == "ollama" else None,
        },
        "retrieval_configuration": {
            "dense_model": config.DENSE_MODEL_NAME,
            "dense_index": "FAISS_IndexFlatIP",
            "sparse_algorithm": config.SPARSE_ALGORITHM,
            "fusion": f"RRF_k{config.RRF_K_CONSTANT}",
            "candidate_pool_depth": config.RETRIEVAL_CANDIDATE_POOL,
        },
        "counts": {
            "total_dataset_queries": len(all_questions),
            "eligible_queries_total": len(eligible_all),
            "excluded_missing_gt_queries": len(excluded_all),
            "evaluated_queries": len(eval_queries),
            "candidate_evaluation_rows": len(cand_df),
            "queries_with_feasible_k": n_feasible,
            "queries_without_feasible_k": n_infeasible,
            "training_rows_emitted": len(train_df),
        },
        "oracle_k_distribution": k_dist,
        "outputs": {
            "candidate_evaluations_csv": str(config.ORACLE_CANDIDATE_EVAL_PATH.relative_to(PROJECT_ROOT)),
            "selected_labels_csv": str(config.ORACLE_SELECTED_LABELS_PATH.relative_to(PROJECT_ROOT)),
            "training_dataset_csv": str(config.ORACLE_DATASET_PATH.relative_to(PROJECT_ROOT)),
            "excluded_queries_csv": str(excluded_path.relative_to(PROJECT_ROOT)),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(experiment_meta, f, indent=2)

    print("\n" + "=" * 70)
    print("ORACLE GENERATION SUMMARY")
    print("=" * 70)
    print(f"Backend                        : {backend.upper()} ({model_id})")
    print(f"Evaluated Queries              : {len(eval_queries)} / {len(eligible_all)} eligible")
    print(f"Excluded (Missing GT)          : {len(excluded_all)}")
    print(f"Feasible Queries (<= {args.slo_seconds:.1f}s SLO) : {n_feasible}")
    print(f"Infeasible Queries (> {args.slo_seconds:.1f}s)   : {n_infeasible}")
    print(f"Candidate Evaluations Saved    : {config.ORACLE_CANDIDATE_EVAL_PATH} ({len(cand_df)} rows)")
    print(f"Selected Labels Saved          : {config.ORACLE_SELECTED_LABELS_PATH} ({len(sel_df)} rows)")
    print(f"RF Training Dataset Saved      : {config.ORACLE_DATASET_PATH} ({len(train_df)} rows)")
    print(f"Experiment Metadata Saved      : {meta_path}")
    print(f"Oracle K* Distribution         : {k_dist}")
    print("=" * 70)


if __name__ == "__main__":
    main()

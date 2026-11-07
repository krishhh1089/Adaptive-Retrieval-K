"""
scripts/build_oracle_dataset.py

Constructs the SAGE Oracle training dataset across all answerable queries.
Extracts 10-D feature vectors and computes optimal minimal budgets K*.
Saves output to:
- experiments/oracle_results/oracle_dataset.csv (full ML dataset with features + labels)
- experiments/oracle_labels/oracle_labels.csv (pure labels only)
"""

import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive.oracle import OracleLabeler
from src.data.chunker import DocumentChunker
from src.data.loader import NQDataLoader
from src.retrieval.hybrid import HybridRetriever
from src import config


def main():
    print("=" * 65)
    print("STEP 8.1: GENERATING SAGE ORACLE TRAINING DATASET")
    print("=" * 65)

    # 1. Load passages and initialize retriever
    chunker = DocumentChunker()
    chunks = chunker.load_chunks()
    print(f"\nLoaded {len(chunks)} passages from {config.CHUNKS_PATH.name}")

    retriever = HybridRetriever(
        rrf_k=config.RRF_K_CONSTANT, candidate_pool=config.RETRIEVAL_CANDIDATE_POOL
    )
    retriever.initialize(chunks)

    # 2. Load queries and filter for answerable records
    loader = NQDataLoader()
    all_questions = loader.load_questions()
    answerable_questions = [q for q in all_questions if q.has_answer]
    print(
        f"Total questions: {len(all_questions)} | "
        f"Answerable queries for Oracle training: {len(answerable_questions)}"
    )

    # 3. Instantiate Oracle Labeler
    labeler = OracleLabeler(
        retriever=retriever,
        candidate_k_values=config.CANDIDATE_K_VALUES,
        slo_latency_seconds=config.SLO_P95_LATENCY_SECONDS,
    )

    # 4. Generate Oracle labels and features
    print("\nExtracting features and computing Oracle K* labels...")
    rows = []
    for q in tqdm(answerable_questions, desc="Generating Oracle Samples"):
        sample = labeler.process_query(q)
        rows.append(sample.to_flat_dict())

    df = pd.DataFrame(rows)

    # 5. Save full dataset (features + labels) to oracle_results/
    config.ORACLE_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = config.ORACLE_RESULTS_DIR / "oracle_dataset.csv"
    jsonl_path = config.ORACLE_RESULTS_DIR / "oracle_dataset.jsonl"

    df.to_csv(csv_path, index=False)
    df.to_json(jsonl_path, orient="records", lines=True)

    # 6. Save clean labels to oracle_labels/
    config.ORACLE_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    labels_df = df[["qid", "question", "oracle_k"]]
    labels_csv_path = config.ORACLE_LABELS_DIR / "oracle_labels.csv"
    labels_jsonl_path = config.ORACLE_LABELS_DIR / "oracle_labels.jsonl"

    labels_df.to_csv(labels_csv_path, index=False)
    labels_df.to_json(labels_jsonl_path, orient="records", lines=True)

    print("\n" + "=" * 65)
    print("ORACLE DATASET GENERATION SUMMARY")
    print("=" * 65)
    print(f"Total Samples Generated: {len(df)}")
    print(f"Saved Full Dataset (features + labels):")
    print(f"  CSV:   {csv_path}")
    print(f"  JSONL: {jsonl_path}")
    print(f"Saved Pure Labels:")
    print(f"  CSV:   {labels_csv_path}")
    print(f"  JSONL: {labels_jsonl_path}")

    print("\nOracle K* Distribution:")
    k_counts = df["oracle_k"].value_counts().sort_index()
    for k_val, count in k_counts.items():
        pct = (count / len(df)) * 100
        print(f"  K* = {k_val:2d}: {count:2d} queries ({pct:5.1f}%)")

    print("\nSample Training Row:")
    sample_row = df.iloc[0].to_dict()
    for col in ["qid", "oracle_k", "dense_top1_score", "dense_score_drop_1_2", "bm25_top1_score", "jaccard_top2", "time_remaining_ms"]:
        print(f"  {col:25s}: {sample_row.get(col)}")

    print("\nStep 8.1 Complete! Ready for Phase 8.2 (Random Forest Training).")


if __name__ == "__main__":
    main()


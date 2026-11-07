"""
tests/test_hybrid.py

Verifies the HybridRetriever combining Dense (BGE-M3) and Sparse (BM25) via RRF (k=60).
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunker import DocumentChunker
from src.retrieval.hybrid import HybridRetriever
from src import config


def main():
    print("=== Step 6.1: Testing Hybrid Retrieval (Dense + BM25 + RRF) ===")

    # 1. Load preprocessed chunks
    chunker = DocumentChunker()
    chunks = chunker.load_chunks()
    print(f"Loaded {len(chunks)} passages from {config.CHUNKS_PATH.name}")

    # 2. Initialize HybridRetriever from saved indexes
    retriever = HybridRetriever(rrf_k=config.RRF_K_CONSTANT, candidate_pool=config.RETRIEVAL_CANDIDATE_POOL)
    retriever.initialize(chunks)

    # 3. Execute search
    test_query = "when is the last episode of season 8 of the walking dead"
    top_k = 5
    result = retriever.search(test_query, top_k=top_k)

    print(f"\nTest Query: '{test_query}'")
    print(f"Candidate Pool Depth: {config.RETRIEVAL_CANDIDATE_POOL} | Final Budget K: {top_k}")
    print("\n--- Latency Breakdown ---")
    print(f"  Dense Search:   {result.dense_latency_ms:.2f} ms")
    print(f"  Sparse Search:  {result.sparse_latency_ms:.2f} ms")
    print(f"  RRF Fusion:     {result.fusion_latency_ms:.2f} ms")
    print(f"  Total Latency:  {result.total_retrieval_latency_ms:.2f} ms")

    print(f"\n--- Top {top_k} Hybrid (RRF) Retrieved Passages ---")
    for rank, p in enumerate(result.passages, start=1):
        preview = p.text.replace("\n", " ")[:110]
        d_info = f"Rank {p.dense_rank} (score {p.dense_score:.3f})" if p.dense_rank else "Not in top 30"
        s_info = f"Rank {p.sparse_rank} (score {p.sparse_score:.2f})" if p.sparse_rank else "Not in top 30"
        print(f"  Rank {rank} | RRF Score: {p.rrf_score:.5f}")
        print(f"    Passage ID: [{p.chunk_id}] (Doc: {p.doc_id})")
        print(f"    Dense: {d_info} | Sparse: {s_info}")
        print(f"    Text: {preview}...\n")

    print("Step 6.1 Verification Complete!")


if __name__ == "__main__":
    main()


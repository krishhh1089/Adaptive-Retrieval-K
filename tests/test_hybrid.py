
"""
tests/test_hybrid.py

Tests Hybrid Retrieval (BGE-M3 + BM25 + RRF)
and measures cold-start versus warm retrieval latency.
"""

import sys
import time
import statistics
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunker import DocumentChunker
from src.retrieval.hybrid import HybridRetriever
from src import config


def main():

    print("=" * 65)
    print("STEP 6.1: HYBRID RETRIEVAL LATENCY BENCHMARK")
    print("=" * 65)

    # 1. Load existing chunks
    chunker = DocumentChunker()
    chunks = chunker.load_chunks()

    print(f"\nLoaded passages: {len(chunks)}")

    # 2. Initialize retriever using existing indexes
    retriever = HybridRetriever(
        rrf_k=config.RRF_K_CONSTANT,
        candidate_pool=config.RETRIEVAL_CANDIDATE_POOL
    )

    print("\nLoading existing retrieval indexes...")
    retriever.initialize(chunks)

    # 3. Test query
    test_query = (
        "when is the last episode of season 8 "
        "of the walking dead"
    )

    top_k = 5
    total_runs = 5

    print(f"\nQuery: {test_query}")
    print(f"Total searches: {total_runs}")

    # 4. Execute repeated searches using SAME retriever
    results = []
    wall_clock_times = []

    print("\nRunning latency benchmark...\n")

    for run in range(1, total_runs + 1):

        start = time.perf_counter()

        result = retriever.search(
            test_query,
            top_k=top_k
        )

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000

        results.append(result)
        wall_clock_times.append(elapsed_ms)

        label = "COLD START" if run == 1 else "WARM"

        print(
            f"Run {run} [{label}] | "
            f"Dense: {result.dense_latency_ms:.2f} ms | "
            f"Sparse: {result.sparse_latency_ms:.2f} ms | "
            f"RRF: {result.fusion_latency_ms:.2f} ms | "
            f"Total: {result.total_retrieval_latency_ms:.2f} ms | "
            f"Wall clock: {elapsed_ms:.2f} ms"
        )

    # 5. Separate cold and warm measurements
    cold_latency = wall_clock_times[0]

    warm_times = wall_clock_times[1:]

    warm_average = statistics.mean(warm_times)
    warm_median = statistics.median(warm_times)
    warm_min = min(warm_times)
    warm_max = max(warm_times)

    # 6. Component-wise warm latency
    warm_dense = statistics.mean(
        r.dense_latency_ms for r in results[1:]
    )

    warm_sparse = statistics.mean(
        r.sparse_latency_ms for r in results[1:]
    )

    warm_fusion = statistics.mean(
        r.fusion_latency_ms for r in results[1:]
    )

    print("\n" + "=" * 65)
    print("COLD START VS WARM LATENCY")
    print("=" * 65)

    print(f"Cold-start latency : {cold_latency:.2f} ms")
    print(f"Warm average       : {warm_average:.2f} ms")
    print(f"Warm median        : {warm_median:.2f} ms")
    print(f"Warm minimum       : {warm_min:.2f} ms")
    print(f"Warm maximum       : {warm_max:.2f} ms")

    print("\n--- Average Warm Component Latency ---")

    print(f"Dense retrieval    : {warm_dense:.2f} ms")
    print(f"BM25 retrieval     : {warm_sparse:.2f} ms")
    print(f"RRF fusion         : {warm_fusion:.2f} ms")

    print("\n" + "=" * 65)
    print("TOP 5 HYBRID RETRIEVED PASSAGES")
    print("=" * 65)

    # Display results from final warm search
    final_result = results[-1]

    for rank, passage in enumerate(
        final_result.passages,
        start=1
    ):

        preview = passage.text.replace("\n", " ")[:110]

        dense_info = (
            f"Rank {passage.dense_rank}"
            if passage.dense_rank
            else "Not in top 30"
        )

        sparse_info = (
            f"Rank {passage.sparse_rank}"
            if passage.sparse_rank
            else "Not in top 30"
        )

        print(f"\nRank {rank} | RRF: {passage.rrf_score:.5f}")
        print(f"Passage ID: {passage.chunk_id}")
        print(f"Dense: {dense_info} | Sparse: {sparse_info}")
        print(f"Text: {preview}...")

    print("\n" + "=" * 65)
    print("LATENCY BENCHMARK COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()

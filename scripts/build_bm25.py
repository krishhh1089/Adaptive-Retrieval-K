"""
scripts/build_bm25.py

Builds and verifies the BM25Okapi sparse retrieval index over chunks.jsonl.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunker import DocumentChunker
from src.retrieval.sparse import SparseBM25Retriever
from src import config


def main():
    print("=== Step 5.1: Building BM25 Sparse Index ===")
    
    # 1. Load the preprocessed chunks
    chunker = DocumentChunker()
    chunks = chunker.load_chunks()
    print(f"Loaded {len(chunks)} passages from {config.CHUNKS_PATH.name}")

    # 2. Build and persist the BM25 index
    retriever = SparseBM25Retriever()
    retriever.build_index(chunks)
    retriever.save()

    # 3. Reload from disk to verify persistence
    reloaded_retriever = SparseBM25Retriever()
    reloaded_retriever.load()

    # 4. Run sample query verification
    test_query = "when is the last episode of season 8 of the walking dead"
    results = reloaded_retriever.search(test_query, top_k=3)

    print("\n=== BM25 Index Verification ===")
    print(f"Total Passages Indexed: {len(reloaded_retriever.chunk_ids)}")
    print(f"Test Query: '{test_query}'")
    print("\nTop 3 BM25 Retrieved Results:")
    for rank, (idx, score) in enumerate(results, start=1):
        matched_chunk = chunks[idx]
        preview = matched_chunk.chunk_text.replace("\n", " ")[:120]
        print(f"  Rank {rank} (score: {score:.4f}): [{matched_chunk.chunk_id}] {preview}...")

    print("\nStep 5.1 Verification Complete!")


if __name__ == "__main__":
    main()


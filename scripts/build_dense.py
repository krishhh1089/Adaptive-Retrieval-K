"""
scripts/build_dense.py

Builds and verifies the BGE-M3 Dense FAISS IndexFlatIP over chunks.jsonl.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunker import DocumentChunker
from src.retrieval.dense import DenseBGERetriever
from src import config


def main():
    print("=== Step 5.2: Building BGE-M3 Dense FAISS Index ===")
    
    # 1. Load the preprocessed chunks
    chunker = DocumentChunker()
    chunks = chunker.load_chunks()
    print(f"Loaded {len(chunks)} passages from {config.CHUNKS_PATH.name}")

    # 2. Build and persist the FAISS index
    retriever = DenseBGERetriever()
    retriever.build_index(chunks, batch_size=16)
    retriever.save()

    # 3. Reload from disk to verify persistence
    reloaded_retriever = DenseBGERetriever()
    reloaded_retriever.load()

    # 4. Run sample query verification
    test_query = "when is the last episode of season 8 of the walking dead"
    results = reloaded_retriever.search(test_query, top_k=3)

    print("\n=== FAISS Dense Index Verification ===")
    print(f"Total Vectors in FAISS Index: {reloaded_retriever.faiss_index.ntotal}")
    print(f"Test Query: '{test_query}'")
    print("\nTop 3 Dense Retrieved Results:")
    for rank, (idx, score) in enumerate(results, start=1):
        matched_chunk = chunks[idx]
        preview = matched_chunk.chunk_text.replace("\n", " ")[:120]
        print(f"  Rank {rank} (cosine similarity: {score:.4f}): [{matched_chunk.chunk_id}] {preview}...")

    print("\nStep 5.2 Verification Complete!")


if __name__ == "__main__":
    main()


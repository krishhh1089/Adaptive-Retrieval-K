"""
tests/test_features.py

Verifies the SAGE feature extraction module (linguistic, retrieval, and consensus signals).
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.chunker import DocumentChunker
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.features import FeatureExtractor
from src import config


def main():
    print("=== Step 6.2: Testing SAGE Feature Extraction ===")

    # 1. Load chunks & hybrid retriever
    chunker = DocumentChunker()
    chunks = chunker.load_chunks()
    retriever = HybridRetriever(rrf_k=config.RRF_K_CONSTANT, candidate_pool=config.RETRIEVAL_CANDIDATE_POOL)
    retriever.initialize(chunks)

    # 2. Execute candidate retrieval
    test_query = "when is the last episode of season 8 of the walking dead"
    retrieval_result = retriever.search(test_query, top_k=30)

    # 3. Extract features
    features = FeatureExtractor.extract_features(test_query, retrieval_result)
    feat_dict = features.to_dict()

    print(f"\nQuery: '{test_query}'")
    print(f"Total Features Extracted: {len(feat_dict)}")
    print("\n--- Extracted Feature Vector ---")
    for k, v in feat_dict.items():
        if isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    # 4. Verify vector conversion for ML model input
    feat_array = features.to_array()
    print(f"\nNumPy Feature Array Shape: {feat_array.shape} (dtype={feat_array.dtype})")
    assert not any(v is None for v in feat_dict.values()), "Features contains None values!"
    print("\nStep 6.2 Verification Complete!")


if __name__ == "__main__":
    main()


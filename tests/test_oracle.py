"""
tests/test_oracle.py

Unit and integration tests for the SAGE Oracle Decision Rule and training sample generator.
"""

import sys
from pathlib import Path
import pytest

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.adaptive.oracle import OracleLabeler, OracleTrainingSample
from src.config import CANDIDATE_K_VALUES
from src.retrieval.hybrid import ScoredPassage


def test_oracle_check_hit():
    """Validates answer matching logic across passages."""
    passages = [
        ScoredPassage(
            chunk_index=0,
            chunk_id="c0",
            doc_id="d1",
            title="T",
            text="The walking dead season 8 finale aired on March 18, 2018.",
            rrf_score=0.5,
        ),
        ScoredPassage(
            chunk_index=1,
            chunk_id="c1",
            doc_id="d2",
            title="T",
            text="Another unrelated text about science.",
            rrf_score=0.3,
        ),
    ]

    # Matching hit
    assert OracleLabeler.check_hit(passages, ["March 18, 2018"]) is True
    assert OracleLabeler.check_hit(passages, ["march 18, 2018"]) is True  # case insensitive
    # Non-matching
    assert OracleLabeler.check_hit(passages, ["December 2019"]) is False
    # Empty edge cases
    assert OracleLabeler.check_hit([], ["March 18, 2018"]) is False
    assert OracleLabeler.check_hit(passages, []) is False


def test_oracle_decision_rule_minimal_k():
    """
    SAGE Decision Rule: K* must be the MINIMAL budget that achieves a hit.
    """
    labeler = OracleLabeler(retriever=None, candidate_k_values=[2, 3, 5, 7, 10, 15, 20, 25, 30])

    # Case 1: Answer is present at passage index 0 (Hit@2 is True) -> K* must be 2
    p_hit_at_1 = [
        ScoredPassage(0, "c0", "d1", "T", "Answer is Alpha", 0.5),
        ScoredPassage(1, "c1", "d1", "T", "Beta", 0.4),
        ScoredPassage(2, "c2", "d1", "T", "Gamma", 0.3),
    ]
    k_star, hits = labeler.compute_oracle_k(p_hit_at_1, ["Alpha"], total_latency_ms=100.0)
    assert k_star == 2
    assert hits[2] == 1
    assert hits[5] == 1

    # Case 2: Answer is only at passage index 4 (Hit@2=0, Hit@3=0, Hit@5=1) -> K* must be 5
    p_hit_at_5 = [
        ScoredPassage(0, "c0", "d1", "T", "Filler 1", 0.5),
        ScoredPassage(1, "c1", "d1", "T", "Filler 2", 0.4),
        ScoredPassage(2, "c2", "d1", "T", "Filler 3", 0.3),
        ScoredPassage(3, "c3", "d1", "T", "Filler 4", 0.2),
        ScoredPassage(4, "c4", "d1", "T", "Answer is Epsilon", 0.1),
    ]
    k_star_5, hits_5 = labeler.compute_oracle_k(p_hit_at_5, ["Epsilon"], total_latency_ms=100.0)
    assert k_star_5 == 5
    assert hits_5[2] == 0
    assert hits_5[3] == 0
    assert hits_5[5] == 1

    # Case 3: Answer never found anywhere -> K* must fallback to minimum budget (2)
    k_star_miss, hits_miss = labeler.compute_oracle_k(p_hit_at_5, ["NonExistent"], total_latency_ms=100.0)
    assert k_star_miss == 2
    assert all(h == 0 for h in hits_miss.values())


def test_oracle_live_sample():
    """Integration check generating Oracle labels for the first 3 dataset questions."""
    from src.data.chunker import DocumentChunker
    from src.data.loader import NQDataLoader
    from src.retrieval.hybrid import HybridRetriever
    from src import config

    chunker = DocumentChunker()
    chunks = chunker.load_chunks()
    retriever = HybridRetriever(
        rrf_k=config.RRF_K_CONSTANT, candidate_pool=config.RETRIEVAL_CANDIDATE_POOL
    )
    retriever.initialize(chunks)

    loader = NQDataLoader()
    questions = loader.load_questions()
    answerable_sample = [q for q in questions if q.has_answer][:3]

    labeler = OracleLabeler(retriever=retriever)

    for q in answerable_sample:
        sample = labeler.process_query(q)
        flat = sample.to_flat_dict()
        assert "oracle_k" in flat
        assert flat["oracle_k"] in CANDIDATE_K_VALUES
        assert len(sample.features.to_array()) == 10
        print(f"\nQID: {sample.qid} | Q: '{sample.question[:40]}...'")
        print(f"  Gold: {sample.gold_answers}")
        print(f"  Oracle K*: {sample.oracle_k}")
        print(f"  Hit Map: {sample.hit_at_candidate_k}")


if __name__ == "__main__":
    pytest.main(["-v", __file__])


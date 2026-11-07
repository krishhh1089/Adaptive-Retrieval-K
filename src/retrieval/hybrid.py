"""
src/retrieval/hybrid.py

Hybrid retrieval engine combining Dense (BGE-M3) and Sparse (BM25) via Reciprocal Rank Fusion (RRF).
Strictly separates candidate retrieval depth from the final selected context budget K.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time

from src.config import RETRIEVAL_CANDIDATE_POOL, RRF_K_CONSTANT
from src.data.chunker import Chunk
from src.retrieval.dense import DenseBGERetriever
from src.retrieval.sparse import SparseBM25Retriever
from src.utils.logger import setup_logger

logger = setup_logger("hybrid_retriever")


@dataclass
class ScoredPassage:
    chunk_index: int
    chunk_id: str
    doc_id: str
    title: str
    text: str
    rrf_score: float
    dense_rank: Optional[int] = None
    dense_score: Optional[float] = None
    sparse_rank: Optional[int] = None
    sparse_score: Optional[float] = None


@dataclass
class HybridRetrievalResult:
    query: str
    passages: List[ScoredPassage]
    dense_latency_ms: float
    sparse_latency_ms: float
    fusion_latency_ms: float
    total_retrieval_latency_ms: float


class HybridRetriever:
    """
    Coordinates Dense and BM25 retrievers and executes Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        dense_retriever: Optional[DenseBGERetriever] = None,
        sparse_retriever: Optional[SparseBM25Retriever] = None,
        chunks: Optional[List[Chunk]] = None,
        rrf_k: int = RRF_K_CONSTANT,
        candidate_pool: int = RETRIEVAL_CANDIDATE_POOL,
    ):
        self.dense_retriever = dense_retriever or DenseBGERetriever()
        self.sparse_retriever = sparse_retriever or SparseBM25Retriever()
        self.chunks = chunks or []
        self.rrf_k = rrf_k
        self.candidate_pool = candidate_pool

    def initialize(self, chunks: List[Chunk]) -> None:
        """
        Loads pre-built indexes from disk.
        """
        self.chunks = chunks
        logger.info("Initializing HybridRetriever indexes from disk...")
        self.dense_retriever.load()
        self.sparse_retriever.load()
        logger.info("Both Dense (FAISS) and Sparse (BM25) indexes loaded successfully.")

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_depth: Optional[int] = None,
    ) -> HybridRetrievalResult:
        """
        Executes hybrid retrieval:
        1. Retrieves candidate_depth passages from Dense and Sparse.
        2. Combines them using Reciprocal Rank Fusion.
        3. Returns the top_k fused passages with latency metrics.
        """
        pool_depth = candidate_depth or max(self.candidate_pool, top_k)

        # 1. Dense retrieval
        t0 = time.perf_counter()
        dense_results = self.dense_retriever.search(query, top_k=pool_depth)
        t_dense = (time.perf_counter() - t0) * 1000.0

        # 2. Sparse retrieval
        t0 = time.perf_counter()
        sparse_results = self.sparse_retriever.search(query, top_k=pool_depth)
        t_sparse = (time.perf_counter() - t0) * 1000.0

        # 3. Reciprocal Rank Fusion
        t0 = time.perf_counter()
        rrf_scores: Dict[int, float] = {}
        dense_meta: Dict[int, Tuple[int, float]] = {}  # idx -> (rank, score)
        sparse_meta: Dict[int, Tuple[int, float]] = {}

        for rank, (idx, score) in enumerate(dense_results, start=1):
            dense_meta[idx] = (rank, score)
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (self.rrf_k + rank))

        for rank, (idx, score) in enumerate(sparse_results, start=1):
            sparse_meta[idx] = (rank, score)
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + (1.0 / (self.rrf_k + rank))

        # Sort by RRF score descending
        fused_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        t_fusion = (time.perf_counter() - t0) * 1000.0

        # Truncate to desired final top_k
        selected_candidates = fused_sorted[:top_k]

        scored_passages: List[ScoredPassage] = []
        for idx, rrf_score in selected_candidates:
            chunk = self.chunks[idx] if self.chunks and idx < len(self.chunks) else None
            d_rank, d_score = dense_meta.get(idx, (None, None))
            s_rank, s_score = sparse_meta.get(idx, (None, None))

            scored_passages.append(
                ScoredPassage(
                    chunk_index=idx,
                    chunk_id=chunk.chunk_id if chunk else f"chunk_{idx}",
                    doc_id=chunk.doc_id if chunk else "",
                    title=chunk.document_title if chunk else "",
                    text=chunk.chunk_text if chunk else "",
                    rrf_score=rrf_score,
                    dense_rank=d_rank,
                    dense_score=d_score,
                    sparse_rank=s_rank,
                    sparse_score=s_score,
                )
            )

        return HybridRetrievalResult(
            query=query,
            passages=scored_passages,
            dense_latency_ms=t_dense,
            sparse_latency_ms=t_sparse,
            fusion_latency_ms=t_fusion,
            total_retrieval_latency_ms=t_dense + t_sparse + t_fusion,
        )


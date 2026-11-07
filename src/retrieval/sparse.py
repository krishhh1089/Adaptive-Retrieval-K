"""
src/retrieval/sparse.py

BM25 sparse lexical retriever module using BM25Okapi.
Preserves lowercased alphanumeric regex tokenization.
"""

from pathlib import Path
from typing import List, Tuple, Optional
import pickle
import re

from rank_bm25 import BM25Okapi

from src.config import BM25_INDEX_DIR
from src.data.chunker import Chunk
from src.utils.logger import setup_logger

logger = setup_logger("sparse_retriever")


class SparseBM25Retriever:
    """
    Handles BM25 sparse index creation, persistence, loading, and querying.
    """

    def __init__(self, index_dir: Path = BM25_INDEX_DIR):
        self.index_dir = index_dir
        self.index_file = self.index_dir / "bm25_index.pkl"
        self.tokenized_corpus_file = self.index_dir / "tokenized_corpus.pkl"

        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenizes text into lowercase alphanumeric words matching notebook Cell 25.
        """
        return re.findall(r"\b\w+\b", text.lower())

    def build_index(self, chunks: List[Chunk]) -> None:
        """
        Tokenizes chunks and constructs the BM25Okapi index.
        """
        logger.info(f"Tokenizing {len(chunks)} chunks for BM25...")
        self.chunk_ids = [c.chunk_id for c in chunks]
        tokenized_corpus = [self.tokenize(c.chunk_text) for c in chunks]

        logger.info("Building BM25Okapi index...")
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built with {len(tokenized_corpus)} passages.")

    def save(self) -> None:
        """
        Persists the BM25 index and chunk ID metadata to disk.
        """
        if self.bm25 is None:
            raise ValueError("Cannot save empty BM25 index. Call build_index first.")

        self.index_dir.mkdir(parents=True, exist_ok=True)
        with open(self.index_file, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids}, f)

        logger.info(f"BM25 index and metadata persisted to {self.index_file}")

    def load(self) -> None:
        """
        Loads the BM25 index and chunk ID metadata from disk.
        """
        if not self.index_file.exists():
            raise FileNotFoundError(f"BM25 index file not found at: {self.index_file}")

        with open(self.index_file, "rb") as f:
            data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.chunk_ids = data["chunk_ids"]

        logger.info(f"Loaded BM25 index from {self.index_file} ({len(self.chunk_ids)} passages)")

    def search(self, query: str, top_k: int = 30) -> List[Tuple[int, float]]:
        """
        Scores passages against the query and returns sorted (chunk_index, bm25_score) tuples.
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not initialized. Call build_index or load first.")

        query_tokens = self.tokenize(query)
        scores = self.bm25.get_scores(query_tokens)

        # Get indices of the top_k highest scores
        top_k = min(top_k, len(scores))
        top_indices = scores.argsort()[-top_k:][::-1]

        return [(int(idx), float(scores[idx])) for idx in top_indices]


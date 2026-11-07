
"""
src/retrieval/dense.py

Dense vector retriever using BGE-M3 and FAISS IndexFlatIP.

Features:
- BGE-M3 embedding generation
- Embedding persistence
- FAISS index serialization
- Cosine similarity search
- Detailed latency instrumentation
"""

from pathlib import Path
from typing import List, Tuple, Optional
import pickle
import time

import numpy as np
import faiss
import torch

from src.config import (
    DENSE_MODEL_NAME,
    DENSE_USE_FP16,
    EMBEDDINGS_DIR,
    FAISS_INDEX_DIR,
)
from src.data.chunker import Chunk
from src.utils.logger import setup_logger

logger = setup_logger("dense_retriever")


class DenseBGERetriever:
    """
    Manages dense embedding generation with BGE-M3
    and vector similarity search with FAISS.
    """

    def __init__(
        self,
        model_name: str = DENSE_MODEL_NAME,
        index_dir: Path = FAISS_INDEX_DIR,
        embeddings_dir: Path = EMBEDDINGS_DIR,
    ):
        self.model_name = model_name
        self.index_dir = index_dir
        self.embeddings_dir = embeddings_dir

        self.index_file = self.index_dir / "index.faiss"
        self.metadata_file = self.index_dir / "metadata.pkl"
        self.embeddings_file = (
            self.embeddings_dir / "bge_m3_embeddings.npy"
        )

        self.model = None
        self.faiss_index: Optional[faiss.IndexFlatIP] = None
        self.chunk_ids: List[str] = []

    def _init_model(self) -> None:
        """Lazily load BGE-M3 onto available hardware."""

        if self.model is None:
            import os
            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            model_slug = f"models--{self.model_name.replace('/', '--')}"
            if (cache_dir / model_slug).exists():
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"

            from FlagEmbedding import BGEM3FlagModel

            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            use_fp16 = (
                torch.cuda.is_available()
                and DENSE_USE_FP16
            )

            logger.info(
                f"Loading BGE-M3 model '{self.model_name}' "
                f"(device={device}, use_fp16={use_fp16})..."
            )

            self.model = BGEM3FlagModel(
                self.model_name,
                use_fp16=use_fp16,
                device=device,
            )

            logger.info("BGE-M3 model successfully loaded.")

    def encode_passages(
        self,
        texts: List[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Generate and cache passage embeddings."""

        if self.embeddings_file.exists():
            logger.info(
                f"Loading cached embeddings from "
                f"{self.embeddings_file.name}..."
            )

            embeddings = np.load(self.embeddings_file)

            if len(embeddings) == len(texts):
                logger.info(
                    f"Loaded {len(embeddings)} cached embeddings."
                )
                return embeddings

            logger.warning(
                "Cached embedding count mismatch. Recomputing..."
            )

        self._init_model()

        logger.info(
            f"Encoding {len(texts)} passages "
            f"in batches of {batch_size}..."
        )

        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=512,
        )

        embeddings = np.asarray(
            output["dense_vecs"],
            dtype=np.float32,
        )

        faiss.normalize_L2(embeddings)

        self.embeddings_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(self.embeddings_file, embeddings)

        logger.info(
            f"Saved {len(embeddings)} normalized embeddings "
            f"to {self.embeddings_file.name}"
        )

        return embeddings

    def build_index(
        self,
        chunks: List[Chunk],
        batch_size: int = 16,
    ) -> None:
        """Build FAISS index from document chunks."""

        texts = [c.chunk_text for c in chunks]
        self.chunk_ids = [c.chunk_id for c in chunks]

        embeddings = self.encode_passages(
            texts,
            batch_size=batch_size,
        )

        dimension = embeddings.shape[1]

        logger.info(
            f"Building FAISS IndexFlatIP "
            f"(dimension={dimension})..."
        )

        self.faiss_index = faiss.IndexFlatIP(dimension)
        self.faiss_index.add(embeddings)

        logger.info(
            f"FAISS index built with "
            f"{self.faiss_index.ntotal} vectors."
        )

    def save(self) -> None:
        """Save FAISS index and chunk metadata."""

        if self.faiss_index is None:
            raise ValueError(
                "FAISS index is empty. Call build_index first."
            )

        self.index_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        faiss.write_index(
            self.faiss_index,
            str(self.index_file),
        )

        with open(self.metadata_file, "wb") as f:
            pickle.dump(
                {"chunk_ids": self.chunk_ids},
                f,
            )

        logger.info(
            f"FAISS index saved: {self.index_file.name} "
            f"({self.faiss_index.ntotal} vectors)."
        )

    def load(self) -> None:
        """Load FAISS index and metadata from disk."""

        if (
            not self.index_file.exists()
            or not self.metadata_file.exists()
        ):
            raise FileNotFoundError(
                f"FAISS index or metadata missing in: "
                f"{self.index_dir}"
            )

        self.faiss_index = faiss.read_index(
            str(self.index_file)
        )

        with open(self.metadata_file, "rb") as f:
            metadata = pickle.load(f)

        self.chunk_ids = metadata["chunk_ids"]

        if len(self.chunk_ids) != self.faiss_index.ntotal:
            raise ValueError(
                "FAISS index and chunk metadata sizes do not match."
            )

        logger.info(
            f"Loaded FAISS index from {self.index_file.name} "
            f"({self.faiss_index.ntotal} vectors)."
        )

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode query and separately measure:
        model initialization, embedding generation,
        and vector normalization.
        """

        # 1. Model initialization
        start = time.perf_counter()

        self._init_model()

        model_load_time = time.perf_counter() - start

        # 2. Query embedding generation
        start = time.perf_counter()

        output = self.model.encode(
            [query],
            max_length=512,
        )

        q_vec = output["dense_vecs"]

        embedding_time = time.perf_counter() - start

        # 3. Convert and normalize
        start = time.perf_counter()

        q_vec = np.asarray(
            q_vec,
            dtype=np.float32,
        )

        faiss.normalize_L2(q_vec)

        normalization_time = time.perf_counter() - start

        logger.info(
            f"Model initialization: "
            f"{model_load_time * 1000:.2f} ms"
        )

        logger.info(
            f"Query embedding generation: "
            f"{embedding_time * 1000:.2f} ms"
        )

        logger.info(
            f"Query normalization: "
            f"{normalization_time * 1000:.2f} ms"
        )

        return q_vec

    def search(
        self,
        query: str,
        top_k: int = 30,
    ) -> List[Tuple[int, float]]:
        """
        Search FAISS and separately measure
        query encoding and FAISS search latency.
        """

        if self.faiss_index is None:
            raise ValueError(
                "FAISS index not initialized. "
                "Call build_index or load first."
            )

        # 1. Query encoding
        start = time.perf_counter()

        query_vec = self.encode_query(query)

        query_encoding_time = time.perf_counter() - start

        # 2. FAISS search only
        top_k = min(
            top_k,
            self.faiss_index.ntotal,
        )

        start = time.perf_counter()

        scores, indices = self.faiss_index.search(
            query_vec,
            top_k,
        )

        faiss_search_time = time.perf_counter() - start

        # 3. Total dense retrieval time
        total_time = (
            query_encoding_time + faiss_search_time
        )

        logger.info("=" * 50)
        logger.info("DENSE RETRIEVAL LATENCY BREAKDOWN")
        logger.info("=" * 50)

        logger.info(
            f"Query encoding: "
            f"{query_encoding_time * 1000:.2f} ms"
        )

        logger.info(
            f"FAISS search only: "
            f"{faiss_search_time * 1000:.2f} ms"
        )

        logger.info(
            f"Total dense retrieval: "
            f"{total_time * 1000:.2f} ms"
        )

        logger.info("=" * 50)

        return [
            (int(idx), float(score))
            for idx, score in zip(
                indices[0],
                scores[0],
            )
            if idx != -1
        ]

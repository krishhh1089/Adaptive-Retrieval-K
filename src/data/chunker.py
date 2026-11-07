"""
src/data/chunker.py

Recursive character chunking module for document passages.
Preserves chunk size (500) and overlap (100) from the baseline specification.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional
import json

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNKS_PATH,
    CHUNK_OVERLAP,
    CHUNK_SEPARATORS,
    CHUNK_SIZE,
)
from src.data.loader import Document
from src.utils.logger import setup_logger

logger = setup_logger("document_chunker")


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    document_title: str
    chunk_text: str


class DocumentChunker:
    """
    Splits documents into fixed-size overlapping text chunks using recursive boundary splitting.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        separators: Optional[List[str]] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or CHUNK_SEPARATORS

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            is_separator_regex=False,
        )

    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Splits a list of documents into chunks with deterministic IDs.
        """
        all_chunks: List[Chunk] = []

        for doc in documents:
            splits = self.text_splitter.split_text(doc.text)
            for i, split in enumerate(splits, start=1):
                chunk = Chunk(
                    chunk_id=f"{doc.doc_id}_chunk_{i}",
                    doc_id=doc.doc_id,
                    document_title=doc.title,
                    chunk_text=split,
                )
                all_chunks.append(chunk)

        logger.info(
            f"Chunked {len(documents)} documents into {len(all_chunks)} passages "
            f"(chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap})"
        )
        return all_chunks

    def save_chunks(self, chunks: List[Chunk], output_path: Path = CHUNKS_PATH) -> None:
        """
        Saves chunks to a JSONL file.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for c in chunks:
                f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
        logger.info(f"Persisted {len(chunks)} chunks to {output_path.name}")

    def load_chunks(self, input_path: Path = CHUNKS_PATH) -> List[Chunk]:
        """
        Loads chunks from a JSONL file.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Chunks file not found at: {input_path}")

        chunks: List[Chunk] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                chunks.append(
                    Chunk(
                        chunk_id=data["chunk_id"],
                        doc_id=data["doc_id"],
                        document_title=data.get("document_title", ""),
                        chunk_text=data["chunk_text"],
                    )
                )

        logger.info(f"Loaded {len(chunks)} chunks from {input_path.name}")
        return chunks


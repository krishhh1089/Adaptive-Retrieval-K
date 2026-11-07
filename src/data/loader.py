"""
src/data/loader.py

Dataset loader for Natural Questions sample corpus and queries.
Preserves ground-truth annotations separately to avoid label leakage.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
import json

from src.config import QUESTION_RECORDS_PATH, UNIQUE_DOCUMENTS_PATH
from src.utils.logger import setup_logger

logger = setup_logger("data_loader")


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


@dataclass
class QuestionRecord:
    qid: str
    question: str
    target_doc_id: str
    short_answers: List[str] = field(default_factory=list)
    has_answer: bool = False


class NQDataLoader:
    """
    Loads and validates the Natural Questions sample dataset and corpus.
    """

    def __init__(
        self,
        documents_path: Path = UNIQUE_DOCUMENTS_PATH,
        questions_path: Path = QUESTION_RECORDS_PATH,
    ):
        self.documents_path = documents_path
        self.questions_path = questions_path

    def load_documents(self) -> List[Document]:
        """
        Loads unique Wikipedia documents from unique_documents.jsonl.
        """
        if not self.documents_path.exists():
            raise FileNotFoundError(f"Documents file not found: {self.documents_path}")

        documents: List[Document] = []
        with open(self.documents_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                doc = Document(
                    doc_id=data.get("doc_id", f"doc_{line_idx}"),
                    title=data.get("document_title", ""),
                    text=data.get("document_text", ""),
                )
                documents.append(doc)

        logger.info(f"Loaded {len(documents)} unique documents from {self.documents_path.name}")
        return documents

    def load_questions(self) -> List[QuestionRecord]:
        """
        Loads question records and safely parses annotations without ground-truth leakage.
        """
        if not self.questions_path.exists():
            raise FileNotFoundError(f"Questions file not found: {self.questions_path}")

        records: List[QuestionRecord] = []
        with open(self.questions_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                question_text = data.get("question", "").strip()
                doc_id = data.get("doc_id", "")
                annotations = data.get("annotations", {})

                # Extract all non-empty short answers safely
                extracted_answers: List[str] = []
                short_answers_list = annotations.get("short_answers", [])
                for sa in short_answers_list:
                    texts = sa.get("text", [])
                    for t in texts:
                        clean_t = t.strip()
                        if clean_t and clean_t not in extracted_answers:
                            extracted_answers.append(clean_t)

                records.append(
                    QuestionRecord(
                        qid=f"q_{line_idx}",
                        question=question_text,
                        target_doc_id=doc_id,
                        short_answers=extracted_answers,
                        has_answer=len(extracted_answers) > 0,
                    )
                )

        logger.info(
            f"Loaded {len(records)} questions ({sum(1 for r in records if r.has_answer)} answerable) "
            f"from {self.questions_path.name}"
        )
        return records


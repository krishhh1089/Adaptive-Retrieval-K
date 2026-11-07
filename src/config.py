"""
src/config.py

Centralized configuration for Load-Aware SAGE research project.
Maintains single source of truth for paths, hyperparameters, and experimental settings.
"""

from pathlib import Path
import os
from dotenv import load_dotenv

# Load local environment variables from .env file
load_dotenv()

# ==========================================
# 1. Project Root and Path Resolution
# ==========================================
# Resolve project root dynamically (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Artifact Directories
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATASET_DIR = ARTIFACTS_DIR / "dataset"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"
INDEXES_DIR = ARTIFACTS_DIR / "indexes"
FAISS_INDEX_DIR = INDEXES_DIR / "faiss"
BM25_INDEX_DIR = INDEXES_DIR / "bm25"
MODELS_DIR = ARTIFACTS_DIR / "models"
SAGE_MODEL_DIR = MODELS_DIR / "random_forest"
METADATA_DIR = ARTIFACTS_DIR / "metadata"

# Dataset Files (Existing Artifacts)
NQ_CLEAN_SAMPLE_PATH = DATASET_DIR / "nq_clean_sample.jsonl"
NQ_TEST_SAMPLE_PATH = DATASET_DIR / "nq_test_sample.jsonl"
QUESTION_RECORDS_PATH = DATASET_DIR / "question_records.jsonl"
UNIQUE_DOCUMENTS_PATH = DATASET_DIR / "unique_documents.jsonl"
CHUNKS_PATH = DATASET_DIR / "chunks.jsonl"

# Experiment Output Directories
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
BASELINE_RESULTS_DIR = EXPERIMENTS_DIR / "baseline_results"
LOAD_AWARE_RESULTS_DIR = EXPERIMENTS_DIR / "load_aware_results"
ORACLE_LABELS_DIR = EXPERIMENTS_DIR / "oracle_labels"
ORACLE_RESULTS_DIR = EXPERIMENTS_DIR / "oracle_results"
LOGS_DIR = EXPERIMENTS_DIR / "logs"

# Ensure all writeable directories exist
for directory in [
    EMBEDDINGS_DIR,
    FAISS_INDEX_DIR,
    BM25_INDEX_DIR,
    SAGE_MODEL_DIR,
    METADATA_DIR,
    BASELINE_RESULTS_DIR,
    LOAD_AWARE_RESULTS_DIR,
    ORACLE_LABELS_DIR,
    ORACLE_RESULTS_DIR,
    LOGS_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. Dataset & Chunking Parameters
# ==========================================
# Preserved exactly from notebook cell 16
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
CHUNK_SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    ""
]

# ==========================================
# 3. Retrieval Parameters
# ==========================================
DENSE_MODEL_NAME = "BAAI/bge-m3"
DENSE_EMBEDDING_DIM = 1024
DENSE_USE_FP16 = True

SPARSE_ALGORITHM = "BM25Okapi"

# Reciprocal Rank Fusion constant (standard default = 60)
RRF_K_CONSTANT = 60

# Candidate pool depth retrieved before adaptive budgeting
RETRIEVAL_CANDIDATE_POOL = 30

# Candidate K values explored for adaptive selection
CANDIDATE_K_VALUES = [2, 3, 5, 7, 10, 15, 20, 25, 30]

# ==========================================
# 4. LLM Generation Parameters
# ==========================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GENERATION_TEMPERATURE = 0.0  # Greedy decoding for reproducible research evaluation
GENERATION_MAX_TOKENS = 256
MAX_RETRIES = 3
REQUEST_TIMEOUT_SECONDS = 30.0

# ==========================================
# 5. Original SAGE Baseline Parameters
# ==========================================
# Preserved from paper configuration: 100 trees, max depth 10
RANDOM_FOREST_N_ESTIMATORS = 100
RANDOM_FOREST_MAX_DEPTH = 10
RANDOM_STATE = 42

# Target Service Level Objective (P95 latency in seconds)
SLO_P95_LATENCY_SECONDS = 5.0

# ==========================================
# 6. Workload Simulation & Monitoring
# ==========================================
TELEMETRY_INTERVAL_SECONDS = 0.1
DEFAULT_CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]


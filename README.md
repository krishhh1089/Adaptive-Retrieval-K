# Adaptive-Retrieval-K

Adaptive-Retrieval-K is a research project on **predicting how many passages to retrieve for each query**. It aims to use query features and retrieval signals to choose a retrieval budget \(K\),balancing answer-relevant context against retrieval cost.

The project is not a load-aware SAGE system, Its focus is query-level adaptive retrieval budgeting; workload-aware routing and service-load prediction are outside the project scope.

## Project status

The repository currently contains the data preparation and retrieval foundation for this work:

- Natural Questions sample-data loading and passage chunking
- Dense retrieval with BGE-M3 embeddings and a FAISS index
- Sparse retrieval with BM25
- Hybrid ranking with Reciprocal Rank Fusion (RRF)
- A hybrid retrieval script that reports candidate depth, final \(K\), and retrieval latency

The adaptive budget prediction model and its end-to-end evaluation are **not implemented yet**. The current retrieval components provide signals and a baseline for that next stage; this README does not claim trained-model results.

## Intended approach

For each query, the planned workflow is:

1. Extract query features and signals from an initial retrieval run.
2. Predict a retrieval budget \(K\) for that query.
3. Return the top \(K\) passages from the ranked candidates.
4. Evaluate the quality and cost of the selected budget against fixed-budget baselines.

The configured candidate budgets are `2, 3, 5, 7, 10, 15, 20, 25, 30`. They are experimental settings, not predictions from a trained model.

## Current retrieval pipeline

```text
Natural Questions documents
            |
            v
  500-character chunks
            |
       +----+----+
       |         |
   BGE-M3      BM25
    + FAISS   lexical
       |         |
       +----+----+
            |
           RRF
            |
   top-K passages
```

Dense and sparse retrieval each produce ranked candidates. RRF combines their rankings (configured with `k = 60`), and the hybrid retriever returns the requested top `K` passages. Candidate retrieval depth and final output budget are separate parameters.

## Repository layout

```text
src/
  config.py                 # Paths and retrieval/experiment settings
  data/
    loader.py               # Natural Questions document and question loader
    chunker.py              # Document chunking and JSONL persistence
  retrieval/
    dense.py                # BGE-M3 embeddings and FAISS search
    sparse.py               # BM25 indexing and search
    hybrid.py               # Dense + sparse retrieval and RRF fusion
  adaptive/                 # Reserved for adaptive budget prediction
  evaluation/               # Reserved for evaluation utilities

scripts/
  build_dense.py            # Build and verify the dense index
  build_bm25.py             # Build and verify the BM25 index
  test_hybrid.py            # Run a hybrid retrieval smoke test

artifacts/
  dataset/                  # Input JSONL data and generated chunks
  embeddings/               # Cached dense embeddings
  indexes/                  # Persisted FAISS and BM25 indexes

experiments/
  logs/                     # Experiment logs

notebooks/                  # Existing research notebooks
```

## Setup

Use Python 3.10 or later, then create and activate a virtual environment.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The data loader expects the following files under `artifacts/dataset/`:

```text
unique_documents.jsonl
question_records.jsonl
chunks.jsonl
```

`unique_documents.jsonl` and `question_records.jsonl` are Natural Questions-derived inputs. `chunks.jsonl` is the chunked corpus consumed by the indexing scripts. The repository does not include a data-download or chunk-generation command, so provide the input files and prepare `chunks.jsonl` before building indexes.

Chunk size and overlap are configured as 500 and 100 characters, respectively, in `src/config.py`.

## Build and test retrieval

From the repository root, build both indexes:

```bash
python scripts/build_dense.py
python scripts/build_bm25.py
```

The dense step downloads or loads the configured `BAAI/bge-m3` model through FlagEmbedding, creates normalized passage embeddings, and writes the FAISS index. The BM25 step builds and persists its lexical index.

After both indexes are available, run the hybrid retrieval smoke test:

```bash
python scripts/test_hybrid.py
```

It uses a sample query and prints the top passages, dense and sparse rankings, RRF scores, and measured retrieval latencies. These timings are observational output, not benchmark results.

## Configuration

Retrieval settings, artifact paths, candidate budgets, and model names are centralized in `src/config.py`. Generated embeddings and indexes are stored under `artifacts/`.

## Planned work

- Define query features and retrieval signals for budget prediction.
- Create labeled examples or targets for selecting an effective \(K\).
- Train and evaluate a budget prediction model against fixed-\(K\) baselines.
- Report retrieval quality, selected budgets, and retrieval cost with reproducible experiments.

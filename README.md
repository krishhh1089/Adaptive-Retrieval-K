# Load-Aware SAGE

**Dynamic Retrieval Budget Allocation under Concurrent RAG Workloads**

Inspired by the research paper *SAGE: SLO-Aware Adaptive Retrieval for Production RAG Systems (2026)*, **Load-Aware SAGE** optimizes retrieval-augmented generation (RAG) by dynamically predicting query-level retrieval budgets \(K \in \{2, 3, 5, 7, 10, 15, 20, 25, 30\}\) and throttling under workload pressure to enforce strict latency Service Level Objectives (\(P95 \le 5.0\text{s}\)).

---

## Key Research Highlights

* **Adaptive Retrieval Policy (\(\pi(q) \rightarrow K\))**: Replaces static retrieval budgets with a lightweight Random Forest model that predicts the minimal sufficient passage count for each query.
* **Lightweight \(k=2\) Retrieval Probe**: Extracts early retrieval signals (dense/sparse confidence scores, score margins, and sparse-dense agreement) with negligible overhead (\(<14\text{ ms}\)).
* **Dramatic Efficiency Gains**: Adaptive SAGE achieves an **84.88% token reduction** and **84.79% latency reduction** over fixed \(K=30\) retrieval while preserving answer quality.
* **Strict SLO Compliance**: Under steady-state execution, Adaptive SAGE achieves a \(P95\) latency of **4.28s**, strictly meeting the \(P95 \le 5.0\text{s}\) SLO threshold, while Fixed \(K=30\) severely violates it at **29.63s**.

---

## Project Status

| Phase | Description | Status |
|:---|:---|:---:|
| **Phase 1–2** | Environment, Configuration, and Path Resolution | Completed |
| **Phase 3–4** | Dataset Ingestion & Document Chunking (NQ Wikipedia) | Completed |
| **Phase 5** | Indexing: Dense (`BGE-M3` + `FAISS IndexFlatIP`) & Sparse (`BM25Okapi`) | Completed |
| **Phase 6** | Research-Aligned 10-Feature Probe Extraction | Completed |
| **Phase 7** | LLM Generation Engine (`Groq` `openai/gpt-oss-120b`, greedy decoding) | Completed |
| **Phase 8.1** | SAGE Oracle Decision Rule & Dataset Generation (\(K^*\) labels) | Completed |
| **Phase 8.2** | Random Forest Policy Training (\(n=100\), \(\text{max\_depth}=10\)) | Completed |
| **Phase 8.3–8.4** | Baseline Benchmarking & Stage-Wise Latency Profiling | Completed |
| **Phase 9** | Real-Time Workload & Telemetry Monitor (\(L\)) | *Upcoming* |
| **Phase 10** | Load-Aware Adaptive Controller (\(\pi(q, L) = (s, k, b)\)) | *Upcoming* |
| **Phase 11** | Concurrency Evaluation Sweeps (\(C \in \{1, 2, 4, 8, 16\}\)) | *Upcoming* |

---

## Empirical Benchmark Findings (Steady-State)

Evaluated across Fixed-\(K\) baselines and the trained Adaptive SAGE Policy under identical conditions on Natural Questions:

| Configuration | Avg \(K\) | Recall@K | Mean Latency | **P95 Latency** | Avg Tokens | **Token Savings vs. K=30** | **Latency Impv. vs. K=30** | SLO Status (\(\le 5.0\text{s}\)) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Fixed \(K=2\)** | 2.0 | 40.0% | 3,385.4 ms | 5,170.2 ms | 579.8 | 84.86% | 87.18% | Borderline |
| **Fixed \(K=5\)** | 5.0 | 40.0% | 5,343.2 ms | 7,003.5 ms | 901.4 | 76.46% | 79.77% | Violated |
| **Fixed \(K=10\)** | 10.0 | 60.0% | 11,503.3 ms | 12,260.4 ms | 1,509.8 | 60.57% | 56.44% | Violated |
| **Fixed \(K=30\)** | 30.0 | 60.0% | 26,406.6 ms | 29,625.5 ms | 3,829.0 | 0.00% | 0.00% | Violated |
| **Adaptive SAGE** | **2.0** | **40.0%** | **4,015.7 ms** | **4,284.0 ms** | **578.8** | **84.88%** | **84.79%** | **Passed** |

### Latency Bottleneck Breakdown
Decomposed monotonic profiling revealed:
* **Primary Bottleneck**: LLM Generation via Groq API accounts for **89.05% to 98.49%** of request latency. Fixed \(K=30\) forces generation of ~4,000 prompt tokens, driving latency to 26.0s. Adaptive SAGE trims prompt context to ~384 tokens, dropping generation to 3.65s (**7.1x speedup**).
* **Secondary Bottleneck**: Dense query vector encoding (`BGE-M3` on CPU) takes **~300–340 ms** (5.7% to 9.5%).
* **Sparse Retrieval**: BM25 inverted index scoring takes **~47–50 ms** (~1%).
* **SAGE Policy Overhead**: Probe feature extraction (0.10 ms) + Random Forest inference (13.49 ms) totals **<14 ms**, introducing negligible overhead.

---

## Repository Structure

```text
Load-Aware-SAGE/
├── src/
│   ├── config.py                 # Central configuration, paths, and hyperparameters
│   ├── adaptive/
│   │   ├── oracle.py             # SAGE Oracle Decision Rule for optimal minimal K*
│   │   └── policy.py             # Random Forest Adaptive Retrieval Policy model
│   ├── data/
│   │   ├── loader.py             # Natural Questions data loader & record models
│   │   └── chunker.py            # RecursiveCharacterTextSplitter (500-char chunks)
│   ├── evaluation/
│   │   ├── latency_tracker.py    # Fine-grained monotonic stage latency tracker
│   │   ├── benchmark_metrics.py  # EM, Token F1, Recall@K, Faithfulness, Percentiles
│   │   └── latency_analysis.py   # Latency bottleneck analyzer & Matplotlib charts
│   ├── generation/
│   │   └── generator.py          # Groq LLM client (openai/gpt-oss-120b, greedy decoding)
│   ├── retrieval/
│   │   ├── dense.py              # BGE-M3 embeddings & FAISS IndexFlatIP search
│   │   ├── sparse.py             # BM25Okapi index & lexical search
│   │   ├── hybrid.py             # Reciprocal Rank Fusion (RRF, k=60)
│   │   └── features.py           # 10 SAGE research-aligned probe features
│   └── utils/
│       └── logger.py             # Colorized console & file logger
│
├── scripts/
│   ├── build_dense.py            # Generates BGE-M3 passage embeddings & FAISS index
│   ├── build_bm25.py             # Builds and serializes BM25 lexical index
│   ├── build_oracle_dataset.py   # Extracts probe features & computes Oracle K* labels
│   ├── train_sage_policy.py      # Trains the Random Forest policy (100 trees, depth 10)
│   └── evaluate_baselines.py     # End-to-end benchmarking of Fixed-K vs. Adaptive SAGE
│
├── tests/
│   ├── test_features.py          # 36 tests: Probe features, bounds, NaN guards
│   ├── test_oracle.py            # 3 tests: Decision rule & hit computation
│   ├── test_policy.py            # 11 tests: Policy training, predict, snapping, IO
│   ├── test_evaluation.py        # 8 tests: Metrics, latency tracking, plot generators
│   ├── test_hybrid.py            # Hybrid retrieval smoke tests
│   └── test_generator.py         # Generation engine smoke tests
│
├── artifacts/
│   ├── dataset/                  # Source questions, documents, and 8,826 chunks
│   ├── embeddings/               # Precomputed BGE-M3 embeddings (1024-dim, float32)
│   ├── indexes/                  # Serialized FAISS & BM25 indexes
│   ├── models/                   # Serialized Random Forest policy (sage_policy.pkl)
│   └── metadata/                 # Versioned provenance metadata (v1 JSON files)
│
├── experiments/
│   ├── oracle_labels/            # Pure query-level oracle target labels (CSV/JSONL)
│   ├── oracle_results/           # Full ML training dataset (features + labels)
│   └── logs/                     # Detailed execution logs
│
├── results/
│   ├── baseline_benchmarks/      # Query-level and aggregated benchmark CSVs/JSONs
│   └── latency_analysis/         # Stage latency CSV and 6 generated comparison plots
│
└── docs/
    └── baseline_benchmarking_and_latency_analysis.md  # Detailed research report
```

---

## Installation & Setup

### 1. Prerequisites
* Python 3.10+ (tested on Python 3.12.4)
* Groq API Key (for LLM generation)

### 2. Virtual Environment
```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

```bash
# Linux / macOS
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b
```

---

## Verification & Execution Guide

### Run Unit Tests
To verify all modules and edge-case guards (58 tests):
```powershell
.\venv\Scripts\python.exe -m pytest -v
```

### 1. Build Retrieval Indexes
```powershell
.\venv\Scripts\python.exe scripts/build_bm25.py
.\venv\Scripts\python.exe scripts/build_dense.py
```

### 2. Generate Oracle Training Dataset
Computes the minimal optimal budget \(K^* = \min \{K \mid \text{Hit@}K=1 \text{ and } \text{Latency} \le T_{\text{SLO}}\}\):
```powershell
.\venv\Scripts\python.exe scripts/build_oracle_dataset.py
```
Outputs:
* `experiments/oracle_results/oracle_dataset.csv` (features + labels)
* `experiments/oracle_labels/oracle_labels.csv` (pure labels)

### 3. Train the SAGE Adaptive Policy
Trains the Random Forest model and persists weights to `artifacts/models/`:
```powershell
.\venv\Scripts\python.exe scripts/train_sage_policy.py
```

### 4. Run Baseline Benchmarking & Latency Analysis
Evaluates Fixed-\(K\) (\(K \in \{2, 5, 10, 30\}\)) against Adaptive SAGE:
```powershell
# Quick 5-query validation with warm-up run
.\venv\Scripts\python.exe scripts/evaluate_baselines.py --sample_size 5 --warmup 1

# Full benchmark across all queries
.\venv\Scripts\python.exe scripts/evaluate_baselines.py --sample_size 0 --warmup 1

# Re-compute metrics and plots from existing query data
.\venv\Scripts\python.exe scripts/evaluate_baselines.py --from_existing_results
```
All comparative charts and stage breakdowns will be saved under `results/latency_analysis/`.

---

## Research Feature Set

The SAGE feature extractor extracts 10 signals using a lightweight \(k=2\) retrieval probe:

| Feature Name | Description | Source |
|:---|:---|:---|
| `query_char_len` | Total character length of query string | Query |
| `query_word_count` | Whitespace-tokenized word count | Query |
| `query_type` | Question category integer ID (who, what, when, where, why, which, how, numeric, other) | Query |
| `query_complexity` | Complexity heuristic score (0=Simple, 1=Medium, 2=Complex) | Query |
| `dense_top1_score` | Similarity score of the Top-1 dense candidate | Dense Probe |
| `dense_score_drop_1_2` | Difference between dense Top-1 and Top-2 scores (\(s_1 - s_2\)) | Dense Probe |
| `bm25_top1_score` | Score of the highest-ranked BM25 lexical candidate | Sparse Probe |
| `bm25_score_drop_1_2` | Difference between BM25 Top-1 and Top-2 scores | Sparse Probe |
| `jaccard_top2` | Jaccard similarity between Dense Top-2 and BM25 Top-2 passage IDs | Agreement |
| `time_remaining_ms` | Remaining request latency budget (\(\max(0, T_{\text{SLO}} - \Delta t)\)) | Latency |

---

## Citation & References

* **SAGE Paper**: *SAGE: SLO-Aware Adaptive Retrieval for Production RAG Systems (2026)*
* **Embedding Model**: `BAAI/bge-m3`
* **Dataset**: Google Natural Questions (`natural_questions`)

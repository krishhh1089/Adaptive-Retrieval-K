# Load-Aware SAGE

**Dynamic Retrieval Budget Allocation under Concurrent RAG Workloads**

Inspired by the research paper *SAGE: SLO-Aware Adaptive Retrieval for Production RAG Systems (2026)*, **Load-Aware SAGE** optimizes retrieval-augmented generation (RAG) by dynamically predicting query-level retrieval budgets \(K \in \{2, 3, 5, 7, 10, 15, 20, 25, 30\}\) and throttling under workload pressure to enforce strict latency Service Level Objectives (\(P95 \le 5.0\text{s}\)).

---

## Key Research Highlights

* **Adaptive Retrieval Policy (\(\pi(q) \rightarrow K\))**: Replaces static retrieval budgets with a lightweight Random Forest model trained to predict query-level passage budgets.
* **Quality-Aware, SLO-Constrained Oracle**: Selects \(K^* = \arg\max_K Q(q, K)\) subject to \(L(q, K) \le 5.0\text{s}\), using answer quality (Token F1 with explicit abstention handling) and smallest-\(K\) tie-breaking rather than passage containment alone.
* **Lightweight \(k=2\) Retrieval Probe**: Extracts early retrieval signals (dense/sparse confidence scores, score margins, and sparse-dense overlap) with negligible inference overhead.
* **Leakage-Free Training Pipeline**: Strictly separates candidate generation/evaluation telemetry from model input features (`FEATURE_NAMES`) and enforces query-grouped train/validation splits.

---

## Project Status

| Phase | Description | Status |
|:---|:---|:---:|
| **Phase 1–2** | Environment, Configuration, and Path Resolution | Completed |
| **Phase 3–4** | Dataset Ingestion & Document Chunking (NQ Wikipedia) | Completed |
| **Phase 5** | Indexing: Dense (`BGE-M3` + `FAISS IndexFlatIP`) & Sparse (`BM25Okapi`) | Completed |
| **Phase 6** | Research-Aligned 10-Feature Probe Extraction | Completed |
| **Phase 7** | LLM Generation Engine (`Groq` `openai/gpt-oss-120b`, greedy decoding) | Completed |
| **Phase 8.1** | Quality-Aware Oracle Pipeline (\(K^*\) labels, 3 separated datasets) | Completed (41/41 eligible queries, 369 candidate evaluations) |
| **Phase 8.2** | Random Forest Policy Training (\(n=100\), \(\text{max\_depth}=10\), grouped split) | Completed (Train=32, Val=9; Val Acc=0.6667, MAE=1.5556) |
| **Phase 8.3–8.4** | Baseline Benchmarking & Stage-Wise Latency Profiling Harness | Harness Ready (Ready for re-benchmark on v2 policy) |
| **Phase 9** | Real-Time Workload & Telemetry Monitor (\(L\)) | *Upcoming* |
| **Phase 10** | Load-Aware Adaptive Controller (\(\pi(q, L) = (s, k, b)\)) | *Upcoming* |
| **Phase 11** | Concurrency Evaluation Sweeps (\(C \in \{1, 2, 4, 8, 16\}\)) | *Upcoming* |

---

## Repository Structure

```text
Load-Aware-SAGE/
├── src/
│   ├── config.py                 # Central configuration, paths, and hyperparameters
│   ├── adaptive/
│   │   ├── oracle.py             # Quality-aware, SLO-constrained SAGE Oracle (argmax Q s.t. L <= SLO)
│   │   └── policy.py             # Random Forest Adaptive Retrieval Policy model
│   ├── data/
│   │   ├── loader.py             # Natural Questions data loader & record models
│   │   └── chunker.py            # RecursiveCharacterTextSplitter (500-char chunks)
│   ├── evaluation/
│   │   ├── latency_tracker.py    # Fine-grained monotonic stage latency tracker
│   │   ├── benchmark_metrics.py  # EM, Token F1, Answer Quality, Recall@K, Faithfulness
│   │   └── latency_analysis.py   # Latency bottleneck analyzer & Matplotlib charts
│   ├── generation/
│   │   └── generator.py          # Groq LLM client (openai/gpt-oss-120b, TPM pacing & telemetry)
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
│   ├── generate_oracle_dataset.py # Evaluates all candidate K values & builds 3 Oracle datasets
│   ├── train_sage_policy.py      # Validates features, grouped-splits, & trains Random Forest
│   └── evaluate_baselines.py     # End-to-end benchmarking of Fixed-K vs. Adaptive SAGE
│
├── tests/
│   ├── test_features.py          # 36 tests: Probe features, bounds, NaN guards
│   ├── test_oracle.py            # 7 tests: Quality-aware Oracle policy, ties, infeasibility, leakage
│   ├── test_policy.py            # 12 tests: Policy training, validation guards, serialization
│   ├── test_evaluation.py        # 8 tests: Metrics, latency tracking, plot generators
│   ├── test_hybrid.py            # Hybrid retrieval smoke tests
│   └── test_generator.py         # Generation engine smoke tests
│
├── artifacts/
│   ├── dataset/                  # Source questions, documents, and 8,826 chunks
│   ├── embeddings/               # Precomputed BGE-M3 embeddings (1024-dim, float32)
│   ├── indexes/                  # Serialized FAISS & BM25 indexes
│   ├── models/random_forest/     # Serialized Random Forest policy (sage_policy.pkl)
│   └── metadata/                 # Versioned provenance metadata (dataset, chunks, embeddings, retrieval, experiments, models)
│
└── experiments/
    ├── baseline_results/         # Baseline evaluation outputs
    ├── load_aware_results/       # Future load-aware outputs
    ├── oracle_results/           # Candidate evaluations, selected labels, RF dataset, excluded audit
    └── logs/                     # Detailed execution logs
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
To verify all modules, leakage guards, Oracle policy rules, and evaluation metrics (63 tests):
```powershell
.\venv\Scripts\python.exe -m pytest tests/test_features.py tests/test_oracle.py tests/test_policy.py tests/test_evaluation.py -v
```

### 1. Build Retrieval Indexes
```powershell
.\venv\Scripts\python.exe scripts/build_bm25.py
.\venv\Scripts\python.exe scripts/build_dense.py
```

### 2. Generate Quality-Aware Oracle Dataset
Evaluates all candidate budgets \(K \in \{2, 3, 5, 7, 10, 15, 20, 25, 30\}\) on eligible short-answer queries and selects \(K^* = \arg\max_K Q(q, K)\) s.t. \(L(q, K) \le 5.0\text{s}\):
```powershell
# Quick validation run (e.g., 6 queries)
.\venv\Scripts\python.exe scripts/generate_oracle_dataset.py --max-queries 6 --resume

# Complete all 41 eligible queries (resumes safely without re-running cached pairs)
.\venv\Scripts\python.exe scripts/generate_oracle_dataset.py --resume
```
Outputs under `experiments/oracle_results/`:
* `oracle_candidate_evaluations.csv`: Every evaluated `(query_id, candidate_k)` outcome & latency.
* `oracle_selected_labels.csv`: Selected `K*`, feasibility flags, and selection rationale per eligible query.
* `oracle_dataset.csv`: Leakage-free supervised training data (`FEATURE_NAMES -> oracle_k`).
* `oracle_excluded_queries.csv`: Unannotated queries preserved separately with `missing_ground_truth` status.

### 3. Train the SAGE Adaptive Policy
Validates leakage guards, enforces query-grouped train/validation splits (`query_id` disjoint), trains `RandomForestClassifier`, and records metrics:
```powershell
.\venv\Scripts\python.exe scripts/train_sage_policy.py
```
Persists `artifacts/models/random_forest/sage_policy.pkl`, `experiments/oracle_results/policy_evaluation_results.json`, and `artifacts/metadata/models/models_v2.json`.

### 4. Run Baseline Benchmarking & Latency Analysis
Evaluates Fixed-\(K\) (\(K \in \{2, 5, 10, 30\}\)) against Adaptive SAGE:
```powershell
# Quick 5-query validation with warm-up run
.\venv\Scripts\python.exe scripts/evaluate_baselines.py --sample_size 5 --warmup 1

# Full benchmark across all queries
.\venv\Scripts\python.exe scripts/evaluate_baselines.py --sample_size 0 --warmup 1
```

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

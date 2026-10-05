# Development Roadmap

The project is developed incrementally so each RAG component can be understood, tested, and validated before introducing the next layer.

---

## Phase 0 — Project Foundation ✅

- Python project
- Poetry
- FastAPI
- health endpoint
- Ruff
- Mypy
- Pytest
- repository structure
- Git/GitHub workflow

---

## Phase 1 — Document Ingestion ✅

- PDF validation
- PyMuPDF extraction
- page-level representation
- PDF metadata
- SHA-256 identity
- conservative normalisation
- encrypted PDF detection
- unusable/non-text PDF handling
- page-level provenance

---

## Phase 2 — Document Chunking ✅

- tokenizer abstraction
- Hugging Face tokenizer
- overlapping token windows
- chunk metadata
- deterministic chunk identity
- page-local provenance
- configurable size and overlap

Baseline:

```text
384 tokens
64-token overlap
```

---

## Phase 3 — Embeddings ✅

- Sentence Transformers
- `BAAI/bge-small-en-v1.5`
- document/query embeddings
- normalised vectors
- batched inference
- provider abstraction

---

## Phase 4 — Vector Storage ✅

- persistent local Qdrant
- cosine collection
- deterministic point IDs
- chunk payload persistence
- document replacement/deletion
- compatibility validation
- provenance reconstruction

---

## Phase 5 — Dense Semantic Retrieval ✅

- query embedding
- nearest-neighbour search
- top-k retrieval
- cosine scoring
- document filtering
- optional threshold
- typed results
- retrieval tests
- smoke testing

---

## Phase 6 — Local LLM ✅

- LLM provider abstraction
- Ollama
- local Qwen baseline
- structured chat messages
- token controls
- token accounting
- inference diagnostics
- runtime error translation

---

## Phase 7 — End-to-End RAG ✅

```text
question
 ↓
retrieval
 ↓
context construction
 ↓
local generation
 ↓
grounded answer
```

Also includes context budgeting, evidence labels, redundancy filtering, whole-chunk selection, insufficient-evidence fallback, and typed `RAGResponse`.

---

## Phase 8 — Citation Integrity ✅

- `[S1]`, `[S2]`, ... protocol
- citation parser
- character offsets
- source mapping
- unknown-ID rejection
- missing-citation validation
- trusted provenance
- typed `CitedRAGResponse`

Boundary: citation identity validation does not prove claim-level semantic entailment.

---

## Phase 9 — RAG API ✅

Implemented routes:

```text
GET    /health
POST   /documents
GET    /documents
DELETE /documents/{document_id}
POST   /search
POST   /query
```

Also includes request/response schemas, service boundaries, document indexing, registry operations, error mapping, runtime composition, lifecycle cleanup, API tests, and local HTTP smoke testing.

---

## Phase 10 — User Interface ✅

- React + TypeScript
- Vite
- central API client
- document upload/selection/deletion
- corpus-wide and document-scoped retrieval
- Ask mode
- Search mode
- cited-answer rendering
- evidence inspector
- generation diagnostics
- loading/error/empty states
- keyboard submission
- stale-request protection
- frontend tests

---

## Phase 11 — Hybrid Retrieval ✅

### Lexical Retrieval

- BM25 retriever
- lexical tokenizer
- shared persisted chunk corpus
- lexical unit tests

Baseline:

```text
k1 = 1.5
b = 0.75
```

### Qdrant Chunk Corpus

- persisted chunk enumeration
- document-scoped chunk retrieval
- full `DocumentChunk` reconstruction
- separate `ChunkCorpus` protocol
- unchanged dense `VectorStore` protocol

### Reciprocal Rank Fusion

Historical Phase 11 formula:

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

Historical baseline:

```text
k = 60
```

### Hybrid Retriever

```text
Dense candidates
+
Lexical candidates
 ↓
RRF
 ↓
Final ranked evidence
```

Candidate multiplier:

```text
2
```

### Controlled Smoke Comparison

The six-query comparison established dense/lexical complementarity but did not claim higher Hybrid accuracy.

---

## Phase 12 — Cross-Encoder Reranking ✅

### Reranking Layer

Implemented:

```text
Hybrid retrieval
 ↓
expanded candidate pool
 ↓
cross-encoder scoring
 ↓
final top-k
```

Historical Phase 12 baseline:

```text
model:
  cross-encoder/ms-marco-MiniLM-L6-v2

candidate pool:
  20

final top_k:
  5
```

### Architecture

- `Reranker` protocol
- local Sentence Transformers cross-encoder scorer
- deterministic `CrossEncoderReranker`
- `RerankingRetriever`
- candidate expansion
- final rank reconstruction
- unchanged Hybrid control

### Retrieval Mode

Added:

```text
hybrid_reranked
```

Available through `/search`, `/query`, and the React workspace.

### Controlled Comparison

The six-query Phase 12 experiment showed useful semantic promotions as well as clear regressions.

Phase 12 therefore established reranking behaviour and integration, not higher retrieval accuracy.

---

## Phase 13 — Formal Evaluation ✅

Phase 13 moved the project from qualitative retrieval inspection to measured relevance-labelled evaluation.

### 13.1 Metric Primitives ✅

Implemented:

```text
Precision@K
Recall@K
MRR
nDCG
```

Also added typed per-query/aggregate metrics and comparison models.

### 13.2 Candidate Pooling ✅

- pooled candidates across Dense, Lexical, Hybrid, and Hybrid + Reranking
- top-20 diagnostic traces
- separate review and trace artifacts

### 13.3 Graded Qrels ✅

Relevance scale:

```text
0 = irrelevant
1 = marginal
2 = useful
3 = direct
```

Expanded qrels:

```text
18 queries
527 judgments
53 binary-relevant judgments at threshold >= 2
```

Review process: model-assisted and blind to system identity/rank.

### 13.4 Split-Aware Evaluation ✅

```text
12 development queries
6 holdout queries
```

Development queries could be used for diagnosis/tuning.

Holdout evaluation required explicit confirmation and remained sealed until retrieval configuration was frozen.

### 13.5 Initial Formal Benchmark ✅

Initial development aggregate:

| Mode | P@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| Dense | 0.2500 | 0.4514 | 0.5556 | 0.4551 |
| Lexical | 0.3667 | 0.6319 | 0.7222 | 0.6264 |
| Hybrid | 0.3500 | 0.5903 | 0.5306 | 0.4951 |
| Hybrid + Reranking | 0.3167 | 0.5625 | 0.5958 | 0.4801 |

The benchmark showed that more complex retrieval was not automatically better.

### 13.6 Improvement / Degradation Ledger ✅

Evaluation records both aggregate metrics and query-level improvements, degradations, mixed outcomes, and unchanged cases.

The project explicitly preserves failure cases instead of reporting only aggregate gains.

### 13.7 Expanded Benchmark ✅

Three papers:

```text
Attention Is All You Need
Language Models are Unsupervised Multitask Learners
Scaling Laws for Neural Language Models
```

All 18 expanded questions are document-scoped.

### 13.8 Evidence-Driven Retrieval Tuning ✅

#### Weighted RRF

An 84-configuration development-only sweep selected:

```text
RRF k = 5
dense weight = 1.00
lexical weight = 1.25
candidate multiplier = 2
```

The chosen configuration prioritised a zero-regression development diagnostic over the absolute highest aggregate score.

Hybrid development improvement:

```text
P@5      0.3500 → 0.3833
Recall   0.5903 → 0.6736
MRR      0.5306 → 0.5750
nDCG     0.4951 → 0.5619
```

#### Reranker pool diagnosis

The first attempt to reduce reranking depth changed one shared pool size from 20 to 10.

That also reduced nested Dense/BM25 candidate depth from 40 to 20, so the implementation did not match the offline experiment.

This intermediate configuration was retained as a documented degradation/architecture lesson.

#### Candidate/rerank depth decoupling

Final configuration:

```text
candidate_pool_size = 20
rerank_pool_size    = 10
```

Resulting path:

```text
Dense 40 + BM25 40
      ↓
weighted RRF
      ↓
Hybrid 20
      ↓
CrossEncoder scores top 10
      ↓
final top 5
```

Final development reranked result:

```text
P@5      0.3833
Recall   0.7153
MRR      0.6903
nDCG     0.5987
```

Relative to the initial expanded reranker:

```text
P@5      +21.1%
Recall   +27.2%
MRR      +15.9%
nDCG     +24.7%
```

### 13.9 Frozen Holdout Evaluation ✅

The final six-query holdout was evaluated only after freezing retrieval configuration.

| Mode | P@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| Dense | 0.4000 | 0.7667 | 0.4778 | 0.4806 |
| Lexical | 0.3667 | 0.6917 | 0.6806 | 0.5083 |
| Hybrid | 0.3667 | 0.6917 | 0.6667 | 0.5397 |
| **Hybrid + Reranking** | **0.4667** | **0.8500** | **0.9167** | **0.7284** |

Holdout `Hybrid → Hybrid + Reranking` outcomes:

```text
4 improvements
1 no change
1 degradation
```

The Natural Questions degradation remains a frozen failure case and is not used for post-hoc parameter tuning.

### 13.10 Documentation ✅

- final retrieval architecture recorded
- initial, intermediate, and final development runs retained
- holdout result retained
- full Phase 13 technical narrative documented
- failure cases and claim boundaries documented
- project-story source bank added for future public write-ups

---

## Phase 14 — Research Features ← NEXT

Potential:

- structured paper summaries
- methodology extraction
- findings extraction
- limitations extraction
- cross-paper comparison
- evidence tables
- research-gap analysis
- multi-document synthesis

Phase 14 should build on the now-measured retrieval stack rather than changing retrieval parameters using the opened holdout.

---

## Phase 15 — Observability

Potential telemetry:

- retrieval mode/config
- retrieved and selected chunks
- retrieval/reranking scores
- token usage
- latency
- truncation
- citation outcomes

Private source text should not be unnecessarily logged.

---

## Phase 16 — Containerisation

Potential services:

```text
FastAPI
Qdrant
Ollama
Frontend
```

---

## Phase 17 — Demo / Deployment

Potential:

- reproducible local demo
- setup automation
- container orchestration
- deployment docs
- public-safe sample corpus
- portfolio demo

---

## Phase 18 — Benchmarking / Technical Report

Potential outputs:

- expanded retrieval benchmark
- RAG answer evaluation report
- abstention benchmark
- citation-support benchmark
- latency/resource measurements
- architecture diagrams
- experiment tables
- trade-offs and failure analysis
- reproducibility instructions

---

# Current Position

```text
Phase 0   Foundation              ✅
Phase 1   Ingestion               ✅
Phase 2   Chunking                ✅
Phase 3   Embeddings              ✅
Phase 4   Vector Store            ✅
Phase 5   Dense Retrieval         ✅
Phase 6   Local LLM               ✅
Phase 7   RAG                     ✅
Phase 8   Citations               ✅
Phase 9   API                     ✅
Phase 10  Frontend                ✅
Phase 11  Hybrid Retrieval        ✅
Phase 12  Reranking               ✅
Phase 13  Evaluation              ✅
Phase 14  Research Features       ← NEXT
Phase 15  Observability
Phase 16  Containerisation
Phase 17  Demo / Deployment
Phase 18  Benchmark / Report
```

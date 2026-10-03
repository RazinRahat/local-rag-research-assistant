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

Boundary: citation identity validation does not yet prove claim-level semantic entailment.

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

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

Baseline:

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

### Retrieval Routing

Supported modes:

```text
dense
lexical
hybrid
```

Dense remains the default.

The same router is used by `/search` and `/query`.

### Frontend

Implemented:

```text
Dense | Lexical | Hybrid
```

Changing retrieval mode invalidates existing and in-flight research results.

### Controlled Smoke Comparison

Six-query whole-corpus comparison:

```text
top_k = 5
```

Average top-five Jaccard overlap:

```text
Dense ↔ Lexical    ≈ 0.214
Dense ↔ Hybrid     ≈ 0.478
Lexical ↔ Hybrid   ≈ 0.507
```

A useful exact-term example was `Adam optimizer`, where dense ranked the actual optimizer-method passage second while lexical and hybrid promoted it to rank one.

Phase 11 demonstrates complementary retrieval behaviour.

It does not claim that hybrid retrieval has higher accuracy.

Detailed findings:

```text
docs/hybrid-retrieval.md
docs/experiments.md
```

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

Baseline:

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
- unchanged Phase 11 hybrid control

### Retrieval Mode

Added:

```text
hybrid_reranked
```

Available through:

```text
POST /search
POST /query
React workspace
```

Dense remains the default.

### Frontend

Implemented:

```text
Dense | Lexical | Hybrid | Reranked
```

The frontend preserves strategy-specific score semantics and invalidates stale research results when retrieval mode changes.

### Controlled Comparison

Six fixed queries were run through:

```text
hybrid
hybrid_reranked
```

Measured:

```text
Mean top-5 Jaccard:           ≈ 0.409
Top-1 agreement:              1 / 6
Hybrid median latency:        ≈ 35.1 ms
Reranked median latency:      ≈ 108.9 ms
Added latency:                ≈ 73.8 ms
```

Reranking produced useful semantic promotions as well as clear regressions.

Phase 12 therefore establishes reranking behaviour and integration, not higher retrieval accuracy.

Detailed findings:

```text
docs/reranking.md
docs/experiments.md
```

---

## Phase 13 — Evaluation

### Retrieval

```text
Recall@K
Precision@K
MRR
nDCG
```

Compare dense, lexical, hybrid, hybrid + reranking, candidate depth, top-k, BM25 parameters, RRF parameters, embedding models, and chunking.

### Answer Evaluation

Potential dimensions:

- faithfulness
- answer relevance
- context relevance
- completeness
- abstention quality

### Citation Evaluation

Separate citation identity validity from semantic claim support.

---

## Phase 14 — Research Features

Potential:

- summaries
- methodology extraction
- findings
- limitations
- cross-paper comparisons
- evidence tables
- research-gap analysis
- multi-document synthesis

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

- retrieval benchmark
- RAG evaluation report
- latency/resource measurements
- architecture diagrams
- experiment tables
- trade-offs
- limitations
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
Phase 13  Evaluation              ← NEXT
Phase 14  Research Features
Phase 15  Observability
Phase 16  Containerisation
Phase 17  Demo / Deployment
Phase 18  Benchmark / Report
```

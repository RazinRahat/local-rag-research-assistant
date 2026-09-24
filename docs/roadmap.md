# Development Roadmap

The project is being developed incrementally so each RAG component can be understood, tested, and validated before the next dependency is introduced.

## Phase 0 — Project Foundation ✅

* Python environment
* Poetry dependency management
* FastAPI application
* Health endpoint
* Ruff
* Mypy
* Pytest
* Repository structure
* Git/GitHub workflow

## Phase 1 — Document Ingestion ✅

* PDF validation
* PyMuPDF extraction
* Page-level document representation
* PDF metadata
* SHA-256 document identity
* Conservative text normalisation
* Encrypted PDF detection
* Page-level provenance preservation

## Phase 2 — Document Chunking ✅

* Tokenizer abstraction
* Overlapping token windows
* Chunk metadata
* Deterministic chunk identity
* Page-local provenance
* Configurable chunk size and overlap
* Model-aware Hugging Face tokenisation

## Phase 3 — Embeddings ✅

* Sentence Transformers integration
* BGE-small-en-v1.5 baseline
* Query and document embeddings
* Normalised vectors
* Batched inference
* Model-aware chunking
* Embedding provider abstraction
* Semantic similarity smoke testing

## Phase 4 — Vector Storage ✅## Phase 5 — Semantic Retrieval ✅

- Query embedding
- Dense nearest-neighbour search
- Top-k retrieval
- Similarity scoring
- Typed retrieval results
- Source provenance reconstruction
- Optional score thresholds
- Document-scoped retrieval
- Retrieval unit tests
- Qdrant search tests
- Manual semantic search validation

## Phase 6 — Local LLM 🚧

Current focus:

- Local model runtime
- LLM provider abstraction
- Prompt construction
- Generation configuration
- Local inference smoke testing

## Phase 7 — End-to-End RAG

Planned pipeline:

```text
question
 ↓
query embedding
 ↓
retrieve evidence
 ↓
build context
 ↓
generate answer
 ↓
grounded response
```

## Phase 8 — Citations

Planned:

* Chunk-to-source citation mapping
* Document and page references
* Inspectable supporting evidence
* Citation validation

## Phase 9 — RAG API

Planned endpoints include:

```text
POST   /documents
GET    /documents
POST   /query
POST   /search
DELETE /documents/{id}
GET    /health
```

## Phase 10 — User Interface

Planned:

* Lightweight research interface
* Document management
* Question answering
* Source inspection
* Retrieval evidence display

A more polished frontend may follow once the underlying RAG pipeline is stable.

## Phase 11 — Hybrid Retrieval

Planned:

* BM25 lexical retrieval
* Dense retrieval
* Reciprocal Rank Fusion
* Metadata filtering
* Dense vs sparse retrieval comparison

## Phase 12 — Reranking

Planned:

* Cross-encoder reranking
* Candidate expansion
* Relevance rescoring
* Top-k reduction
* Retrieval-quality comparison

## Phase 13 — Evaluation

Retrieval metrics:

* Recall@K
* Precision@K
* Mean Reciprocal Rank (MRR)
* nDCG

Answer-level evaluation:

* Faithfulness
* Answer relevance
* Context relevance
* Citation correctness

Evaluation should make it possible to identify whether failures originate from retrieval, ranking, context construction, or generation.

## Phase 14 — Research Features

Potential features:

* Paper summaries
* Methodology extraction
* Finding extraction
* Limitation extraction
* Cross-paper comparisons
* Evidence tables
* Research-gap analysis
* Multi-document synthesis

## Phase 15 — Observability

Potential telemetry:

* Query
* Retrieved chunks
* Similarity scores
* Reranking scores
* Context size
* Token usage
* Latency
* Model configuration
* Final response

The purpose is to make RAG failures inspectable rather than opaque.

## Phase 16 — Containerisation

Potential services:

```text
FastAPI
Qdrant
Local Model Runtime
Frontend
```

The current local Qdrant implementation can later be migrated to a standalone Qdrant service.

## Phase 17 — Demo / Deployment

Provide a reproducible way to run and demonstrate the complete system.

Potential goals include:

* Simple local startup
* Reproducible environment
* Example research corpus
* Demonstration queries
* Portfolio-ready interface

## Phase 18 — Benchmarking and Technical Report

Compare system configurations rather than relying on intuition.

Potential experiments include:

* Chunk size
* Chunk overlap
* Chunking strategy
* Embedding model
* Retrieval algorithm
* Top-k
* Similarity threshold
* Reranking
* Context size
* Batch size
* Latency
* Memory usage

The final project should demonstrate not only that RAG works, but why particular design choices were selected.

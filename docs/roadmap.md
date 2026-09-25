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
* PDF metadata extraction
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
* Embedding provider abstraction
* Model-aware chunking
* Semantic similarity smoke testing

## Phase 4 — Vector Storage ✅

* Persistent Qdrant vector database
* Local persistent development mode
* Collection creation and validation
* Cosine vector space
* Persistent chunk payloads
* Deterministic Qdrant point identifiers
* Vector insertion and upserts
* Document-level deletion
* Document replacement and re-indexing
* Duplicate prevention
* Document metadata filtering
* Persistence across application restarts
* In-memory Qdrant testing
* Vector-store abstraction

## Phase 5 — Semantic Retrieval ✅

* Query embedding
* Dense nearest-neighbour search
* Top-k retrieval
* Similarity scoring
* Typed retrieval results
* Source provenance reconstruction
* Optional score thresholds
* Document-scoped retrieval
* Retriever-compatible architecture
* Retrieval unit tests
* Qdrant search tests
* Manual semantic-search validation

## Phase 6 — Local LLM ✅

* Ollama local model runtime
* Qwen3.5 development baseline
* LLM provider abstraction
* Typed chat messages
* Typed generation results
* Configurable generation settings
* Explicit runtime context window
* Output-token limits
* Controlled temperature and sampling
* Non-streaming baseline
* Thinking-output control
* Token accounting
* Model-loading metrics
* Prompt-processing metrics
* Generation latency metrics
* Domain-specific generation errors
* Mock HTTP provider tests
* Local inference smoke-testing workflow

## Phase 7 — End-to-End RAG ✅

* Retriever abstraction
* RAG orchestration service
* Retrieved-evidence context construction
* Explicit model-context budgeting
* Reserved generation space
* Context safety margin
* Chat-template-aware token estimation
* Runtime prompt-token comparison
* Whole-chunk evidence selection
* Retrieval-order preservation
* Duplicate evidence removal
* Highly overlapping evidence filtering
* Source labelling
* Retrieved-evidence delimiters
* Grounding instructions
* Retrieved-document instruction isolation
* Insufficient-evidence fallback
* No-evidence generation bypass
* Typed `EvidenceBlock`
* Typed `ContextBuildResult`
* Typed `RAGResponse`
* Context-builder unit tests
* End-to-end orchestration tests
* Full local RAG smoke-testing workflow

Current dense-RAG pipeline:

```text
question
 ↓
query embedding
 ↓
semantic retrieval
 ↓
retrieved evidence
 ↓
context construction
 ↓
grounded prompt
 ↓
local LLM
 ↓
RAG response
```

## Phase 8 — Citations 🚧

Current focus:

* source-reference representation
* model-facing source identifiers
* citation parsing
* citation-to-evidence mapping
* citation validation
* rejection of unknown or invented source identifiers
* document/page citation rendering
* inspectable supporting evidence
* typed cited-answer representation
* citation unit tests
* end-to-end citation smoke testing

Planned provenance:

```text
Generated Claim
      ↓
Source Reference
      ↓
EvidenceBlock
      ↓
DocumentChunk
      ↓
Page
      ↓
Source Document
```

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

Potential API responsibilities:

* document ingestion
* document indexing
* semantic search
* RAG questions
* cited responses
* document deletion
* service health

## Phase 10 — User Interface

Planned:

* lightweight research interface
* document management
* question answering
* source inspection
* retrieval-evidence display
* citation interaction

A more polished frontend may follow once the underlying RAG pipeline and citation model are stable.

## Phase 11 — Hybrid Retrieval

Planned:

* BM25 lexical retrieval
* dense retrieval
* Reciprocal Rank Fusion
* metadata filtering
* dense vs sparse retrieval comparison
* hybrid retrieval evaluation

The current dense retriever will remain the baseline for comparison.

## Phase 12 — Reranking

Planned:

* cross-encoder reranking
* candidate expansion
* relevance rescoring
* top-k reduction
* reranked vs non-reranked retrieval comparison

## Phase 13 — Evaluation

### Retrieval Metrics

* Recall@K
* Precision@K
* Mean Reciprocal Rank (MRR)
* nDCG

### Answer-Level Evaluation

* faithfulness
* answer relevance
* context relevance
* citation correctness

### Context and Pipeline Diagnostics

Potential measurements include:

* retrieved candidate count
* selected evidence count
* redundant chunks removed
* chunks skipped for context budget
* estimated prompt tokens
* actual prompt tokens
* token-estimation error
* context truncation rate
* retrieval latency
* generation latency

Evaluation should make it possible to identify whether failures originate from retrieval, ranking, context construction, generation, or citation handling.

## Phase 14 — Research Features

Potential features:

* paper summaries
* methodology extraction
* finding extraction
* limitation extraction
* cross-paper comparisons
* evidence tables
* research-gap analysis
* multi-document synthesis

These features should use the existing retrieval, provenance, citation, and evaluation layers rather than bypassing them with standalone prompting.

## Phase 15 — Observability

Potential telemetry:

* query
* retrieval configuration
* retrieved chunks
* similarity scores
* reranking scores
* selected evidence
* context size
* estimated prompt tokens
* actual prompt tokens
* token usage
* model loading time
* prompt-processing time
* generation latency
* model configuration
* final response
* citations
* insufficient-evidence state

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

The current Ollama boundary also allows model-runtime infrastructure to remain separate from application logic.

## Phase 17 — Demo / Deployment

Provide a reproducible way to run and demonstrate the complete system.

Potential goals include:

* simple local startup
* reproducible environment
* example research corpus
* demonstration queries
* cited answer examples
* source inspection
* portfolio-ready interface

## Phase 18 — Benchmarking and Technical Report

Compare system configurations rather than relying on intuition.

Potential experiments include:

* chunk size
* chunk overlap
* chunking strategy
* embedding model
* retrieval algorithm
* dense vs hybrid retrieval
* top-k
* similarity threshold
* reranking
* overlap-deduplication threshold
* context safety margin
* context size
* output-token allowance
* evidence ordering
* generation configuration
* latency
* memory use
* citation correctness

The final project should demonstrate not only that RAG works, but why particular design choices were selected.

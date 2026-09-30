# Development Roadmap

The project is being developed incrementally so each RAG component can be understood, tested, and validated before the next dependency is introduced. Completed phases describe implemented capabilities; future phases are proposals, not claims of current functionality.

## Phase 0 — Project Foundation ✅

- Python environment and Poetry dependency management
- FastAPI application and health endpoint
- Ruff, Mypy, and Pytest
- Repository structure and Git/GitHub workflow

## Phase 1 — Document Ingestion ✅

- PDF validation and PyMuPDF extraction
- Page-level document representation and PDF metadata
- SHA-256 document identity
- Conservative text normalisation
- Encrypted PDF detection and provenance preservation

## Phase 2 — Document Chunking ✅

- Tokenizer abstraction and overlapping token windows
- Typed chunk metadata and deterministic chunk identity
- Page-local provenance and configurable size/overlap
- Model-aware Hugging Face tokenisation

## Phase 3 — Embeddings ✅

- Sentence Transformers integration and BGE-small-en-v1.5 baseline
- Query/document embeddings and vector normalisation
- Batched inference, provider abstraction, and model-aware chunking
- Semantic similarity smoke testing

## Phase 4 — Vector Storage ✅

- Persistent local Qdrant and vector-store abstraction
- Collection creation and compatibility validation
- Cosine vector space, payloads, and deterministic point IDs
- Vector insertion, document deletion, and re-indexing/replacement
- Document metadata filtering and persistence tests
- Isolated in-memory Qdrant tests

## Phase 5 — Semantic Retrieval ✅

- Query embedding and dense nearest-neighbour search
- Top-k retrieval, similarity scores, and typed results
- Source provenance reconstruction, score thresholds, and document filtering
- Retriever abstraction, unit tests, Qdrant tests, and manual search validation

## Phase 6 — Local LLM ✅

- Ollama runtime with a Qwen3.5 development baseline
- `LLMProvider`, typed chat messages, and `GenerationResult`
- Configurable context/output budgets and inference settings
- Non-streaming baseline and thinking-output control
- Token accounting, load/prefill/generation timing, and domain exceptions
- Mock HTTP tests and local inference smoke-testing workflow

## Phase 7 — End-to-End RAG ✅

- RAG orchestration (`Retriever` → `ContextBuilder` → `LLMProvider`)
- Explicit prompt budget, reserved output, and safety margin
- Chat-template-aware token counting and estimated-vs-actual token tracking
- Whole-chunk selection in retrieval order
- Identical/highly overlapping evidence filtering
- Temporary source labels and evidence delimiters
- Grounding instructions and untrusted-document isolation
- No-evidence generation bypass and typed `RAGResponse`
- Context-builder tests, orchestration tests, and full local RAG smoke-test workflow

## Phase 8 — Citation Integrity ✅

- Strict model-facing citation instructions using `[S1]`, `[S2]`, …
- Independent `citations/` layer after RAG generation
- `CitationReference` with source ID and character offsets
- `CitationSource` mapping to trusted `EvidenceBlock`
- `CitationValidationResult` preserving references and unique sources
- `CitedRAGResponse` containing both original RAG output and validated citations
- Citation parser for recognised `[S#]` syntax
- Rejection of unknown recognised source identifiers
- Rejection of citation-free evidence-backed answers
- Detection of duplicate source IDs in supplied evidence
- Citation-free handling of explicit insufficient-evidence responses
- Provenance access through source chunk, document, filename, and page
- Parser, validator, and service unit tests
- Real local cited-RAG smoke-testing workflow

**Boundary of Phase 8:** source-reference identity is validated; universal malformed-format detection, per-claim citation coverage, and semantic claim-to-evidence entailment are **not yet** implemented. Display-ready UI source rendering follows in the API/UI phases.

The current cited-RAG path is:

```text
question
 ↓
semantic retrieval
 ↓
selected EvidenceBlock[] with S1, S2, ...
 ↓
bounded grounded prompt
 ↓
local LLM
 ↓
RAGResponse with inline [S#] references
 ↓
CitationService: parse → validate → map
 ↓
CitedRAGResponse
```

## Phase 9 — RAG API 🚧

Current focus:

- FastAPI request/response schemas and dependency wiring
- Application lifecycle and resource management for Qdrant/Ollama
- Document upload, validation, ingestion, indexing, listing, and deletion
- Semantic search and cited-question endpoints
- Request validation and appropriate domain-to-HTTP error mapping
- Exposure of answer, citation references, unique sources, and relevant diagnostics
- API tests with lightweight fake providers and isolated storage
- Avoiding accidental disclosure of private source content in logs or exceptions

Planned routes (subject to refinement):

```text
POST   /documents
GET    /documents
DELETE /documents/{id}
POST   /search
POST   /query
GET    /health
```

The existing health endpoint is already implemented; the full RAG routes are part of this phase.

## Phase 10 — User Interface

Planned:

- Lightweight research interface and document management
- Question answering and source inspection
- Cited answer rendering and citation interaction
- Retrieved-evidence display
- More polished frontend once API and citation contracts are stable

## Phase 11 — Hybrid Retrieval

Planned:

- BM25 lexical retrieval and current dense retrieval
- Reciprocal Rank Fusion and metadata filtering
- Dense vs sparse vs hybrid comparison
- Hybrid retrieval evaluation against the dense baseline

## Phase 12 — Reranking

Planned:

- Cross-encoder reranking
- Candidate expansion, relevance rescoring, top-k reduction
- Ranked vs unranked retrieval-quality comparison

## Phase 13 — Evaluation

Retrieval metrics:

- Recall@K, Precision@K, Mean Reciprocal Rank (MRR), nDCG

Answer-level evaluation:

- Faithfulness, answer relevance, context relevance, citation correctness
- Citation identity validity **separately** from semantic support of claims

Context/pipeline diagnostics:

- Retrieved and selected evidence counts; redundant/budget-skipped chunks
- Estimated/actual prompt tokens and estimation error
- Context truncation, retrieval latency, and generation latency

Evaluation should distinguish retrieval, context selection, generation, and citation failures rather than obscuring them behind a single RAG score.

## Phase 14 — Research Features

Potential features:

- Paper summaries; methodology, findings, and limitations extraction
- Cross-paper comparisons; evidence tables; research-gap analysis
- Multi-document synthesis using existing evidence/citation paths

## Phase 15 — Observability

Potential telemetry:

- Query and retrieval configuration
- Retrieved/selected chunks and scores
- Reranking scores, context/token usage, runtime timings
- Model configuration, final answer, citations, and insufficient-evidence state
- Citation validation failure category (without unnecessarily logging private source text)

## Phase 16 — Containerisation

Potential services:

```text
FastAPI
Qdrant
Local Model Runtime
Frontend
```

The current Qdrant local-mode adapter can later be configured for a standalone server; the Ollama API remains behind its provider interface.

## Phase 17 — Demo / Deployment

- Reproducible local startup and example (redistributable) corpus
- Demonstration questions and cited answers
- Inspectable source evidence and a portfolio-ready interface

## Phase 18 — Benchmarking and Technical Report

Compare configurations rather than relying on intuition. Potential dimensions include chunk size/overlap/strategy, embedding model, retrieval algorithm, top-k, thresholds, hybrid retrieval, reranking, context budget, deduplication, model settings, latency, memory, citation validity, and claim-level citation correctness.

The final project should demonstrate not only that RAG works, but why particular design choices were selected.

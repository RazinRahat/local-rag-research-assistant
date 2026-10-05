# System Architecture

## Purpose

The Local RAG Research Assistant is designed as a modular, local-first Retrieval-Augmented Generation system for research documents.

Rather than treating RAG as one black-box operation, the application separates ingestion, chunking, embeddings, persistence, retrieval, reranking, evaluation, context construction, generation, citations, API transport, and frontend presentation into independently testable components.

---

# High-Level Architecture

```text
                            User
                             │
                             ▼
                      React Workspace
                             │
                             ▼
                          FastAPI
                             │
                             ▼
                      ResearchService
                             │
           ┌─────────────────┼──────────────────┐
           │                 │                  │
           ▼                 ▼                  ▼
      Document Flow    RetrievalRouter     Citation Service
           │                 │
           ▼       ┌─────────┼─────────┬──────────────┐
        Qdrant     ▼         ▼         ▼              ▼
                Dense     Lexical   Hybrid      Hybrid Reranked
                  │          │         │              │
                  │          │    Dense + BM25        │
                  │          │         │              │
                  │          │         ▼              │
                  │          │   Weighted RRF ────────┘
                  │          │                        │
                  │          │                        ▼
                  │          │                  Hybrid top 20
                  │          │                        ↓
                  │          │                    top 10
                  │          │                        ↓
                  │          │                  Cross-Encoder
                  └──────────┴────────────────────────┘
                             │
                             ▼
                        ContextBuilder
                             │
                             ▼
                         Ollama/Qwen
                             │
                             ▼
                          RAGResponse
                             │
                             ▼
                       CitationService
                             │
                             ▼
                      CitedRAGResponse
```

Formal retrieval evaluation is a separate boundary around the retrieval subsystem rather than part of runtime answer generation.

---

# Ingestion

```text
PDF
 ↓
PyMuPDF
 ↓
ParsedDocument
 ↓
DocumentPage[]
```

Responsibilities include validation, page-level text extraction, metadata extraction, SHA-256 identity, encrypted-document detection, conservative normalisation, and page-boundary preservation.

---

# Chunking

```text
DocumentPage[]
      ↓
Tokenizer
      ↓
DocumentChunk[]
```

Current baseline:

```text
384 content tokens
64-token overlap
```

Each chunk preserves document identity, filename, page number, chunk positions, token range, and source text.

---

# Embeddings

```text
DocumentChunk
      ↓
EmbeddingProvider
      ↓
ChunkEmbedding
```

Current implementation:

```text
SentenceTransformerEmbedder
BAAI/bge-small-en-v1.5
384 dimensions
```

Vectors are normalised before dense retrieval.

---

# Vector Storage

```text
ChunkEmbedding[]
       ↓
VectorStore
       ↓
Qdrant
```

Stored points contain vector data, chunk text, provenance, and embedding metadata.

The core vector-store boundary supports collection creation, compatibility checks, replacement, deletion, search, counts, and lifecycle cleanup.

Qdrant also implements separate document-registry and chunk-corpus roles needed by document management and lexical retrieval.

---

# Retrieval Architecture

```text
                            Query
                              │
                              ▼
                       RetrievalRouter
                              │
        ┌─────────────────────┼──────────────────────┬──────────────────────┐
        ▼                     ▼                      ▼                      ▼
      Dense                Lexical                Hybrid             Hybrid Reranked
        │                     │                      │                      │
        ▼                     ▼             ┌────────┴────────┐             │
SemanticRetriever       BM25Retriever        ▼                 ▼            │
        │                     │            Dense             BM25           │
        ▼                     ▼              │                 │            │
EmbeddingProvider       ChunkCorpus           └───────┬─────────┘            │
        │                     │                      ▼                      │
        ▼                     ▼                Weighted RRF ────────────────┘
   VectorStore              Qdrant                   │
        │                                            ▼
        ▼                                      RetrievalResult[]
      Qdrant
```

All strategies satisfy the same `Retriever` protocol.

---

## Dense Retrieval

```text
query
 ↓
BGE embedding
 ↓
Qdrant cosine search
 ↓
RetrievalResult[]
```

Dense remains the default.

---

## Lexical Retrieval

```text
query
 ↓
lexical tokenisation
 ↓
BM25
 ↓
ChunkCorpus
 ↓
RetrievalResult[]
```

Current baseline:

```text
k1 = 1.5
b = 0.75
```

---

## Weighted Hybrid Retrieval

```text
Dense candidates
+
Lexical candidates
 ↓
Weighted Reciprocal Rank Fusion
 ↓
RetrievalResult[]
```

Frozen Phase 13 Hybrid configuration:

```text
RRF rank constant = 5
dense weight = 1.00
lexical weight = 1.25
candidate multiplier = 2
```

RRF is used because dense cosine and BM25 scores are not directly comparable.

The generic RRF primitive remains reusable; the Hybrid retriever owns the tuned branch weights and production RRF constant.

---

# Reranking Architecture

`hybrid_reranked` wraps Hybrid with a second-stage cross-encoder.

The final architecture separates **first-stage candidate generation** from **second-stage reranking depth**:

```text
candidate_pool_size = 20
rerank_pool_size    = 10
```

For a default final `top_k=5` request:

```text
RerankingRetriever
requests Hybrid top 20
        ↓
Hybrid candidate_multiplier = 2
        ↓
Dense top 40 + BM25 top 40
        ↓
weighted RRF
        ↓
Hybrid top 20
        ↓
retain ranks 1–10
        ↓
CrossEncoderReranker
        ↓
final top 5
```

The cross-encoder model is:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

This decoupling was introduced after Phase 13 showed that reducing a single shared pool parameter also reduced nested Hybrid branch depth, unintentionally changing candidate generation.

---

# Qdrant Interface Roles

```text
QdrantVectorStore
├── VectorStore
├── DocumentRegistry
└── ChunkCorpus
```

These responsibilities remain represented by separate protocols.

---

# Retrieval Routing

`RetrievalConfig` supports:

```text
dense
lexical
hybrid
hybrid_reranked
```

`RetrievalRouter` selects the corresponding implementation.

The same router is passed to both `ResearchService` and `RAGService`.

---

# Context Construction

```text
RetrievalResult[]
      ↓
ContextBuilder
      ↓
EvidenceBlock[]
      ↓
ChatMessage[]
```

The context layer manages finite model context, reserved output budget, safety margin, evidence ordering, duplicate removal, overlap filtering, whole-chunk inclusion, source labels, delimiters, and grounding instructions.

---

# Token Accounting

```text
ChatMessage[]
      ↓
chat template
      ↓
formatted prompt
      ↓
tokenizer
      ↓
estimated tokens
```

The RAG response can also preserve runtime token counts returned by Ollama.

---

# Generation

```text
ChatMessage[]
      ↓
LLMProvider
      ↓
OllamaProvider
      ↓
Qwen
      ↓
GenerationResult
```

Generation remains independent from retrieval implementation.

---

# RAG Orchestration

```text
Question
   ↓
Retriever
   ↓
RetrievalResult[]
   ↓
ContextBuilder
   ↓
ChatMessage[]
   ↓
LLMProvider
   ↓
GenerationResult
   ↓
RAGResponse
```

---

# Evidence Representation

```text
EvidenceBlock
├── source_id
├── retrieval_rank
├── score
└── DocumentChunk
```

Score meaning depends on retrieval mode:

```text
dense              → cosine similarity
lexical            → BM25 score
hybrid             → weighted RRF score
hybrid_reranked    → cross-encoder relevance score
```

---

# Citation Integrity

```text
Generated Answer
      ↓
Citation Parser
      ↓
[S1], [S2], ...
      ↓
Citation Validator
      ↓
Known EvidenceBlock
      ↓
CitationSource
```

The application, not the model, owns document/page/chunk provenance.

Citation identity validation does not prove semantic support for every generated claim.

---

# Insufficient Evidence

```text
No Evidence
    ↓
Skip LLM
    ↓
Insufficient-Evidence Response
```

Nearest-neighbour retrieval can still return weak candidates for unrelated questions, so out-of-scope/negative-query evaluation remains a future RAG evaluation task.

---

# API Boundary

```text
HTTP
 ↓
FastAPI
 ↓
ResearchService
 ↓
Domain Components
```

Routes:

```text
GET    /health
POST   /documents
GET    /documents
DELETE /documents/{document_id}
POST   /search
POST   /query
```

---

# Frontend Boundary

```text
React
 ↓
ResearchAPIClient
 ↓
FastAPI
```

The browser does not access Qdrant, BGE, Ollama, reranker internals, or tokenizer internals directly.

---

# Async Safety

Search and RAG hooks track request sequences.

Changing document scope or retrieval mode invalidates older requests so stale responses cannot overwrite current UI state.

---

# Local-First Design

```text
PDF
 ↓
Local parsing
 ↓
Local chunking
 ↓
Local embeddings
 ↓
Local Qdrant
 ↓
Local retrieval
 ↓
Local reranking
 ↓
Local context building
 ↓
Local Ollama/Qwen
 ↓
Local cited answer
```

---

# Evaluation Architecture

Retrieval, generation, citation identity, and citation semantic support are treated as separate evaluation problems.

Phase 13 formal retrieval evaluation uses:

```text
3 papers
18 document-scoped queries
12 development
6 holdout
527 graded judgments
```

Pipeline:

```text
fixed query set
      ↓
retrieval modes
      ↓
pooled candidates
      ├───────────────┐
      ▼               ▼
blind review        trace
      │               │
      ▼               │
     qrels            │
      └───────┬───────┘
              ▼
        metric runner
              ↓
Precision / Recall / MRR / nDCG
              ↓
regression diagnostics
```

The development split may be used for tuning. The holdout split is inspected only after retrieval configuration is frozen.

Recall is pooled recall because relevance judgments are based on pooled candidates rather than exhaustive corpus annotation.

---

# Phase 13 Architectural Lesson

One of the most useful Phase 13 findings was architectural rather than purely metric-based.

The first attempt to reduce cross-encoder depth changed a single `candidate_pool_size` from 20 to 10. Because the wrapped Hybrid retriever expands its own candidates, this also changed Dense/BM25 depth from 40 to 20.

The intended experiment was:

```text
retrieve deeply
then rerank fewer candidates
```

but the code actually tested:

```text
retrieve less deeply
and rerank fewer candidates
```

Introducing `rerank_pool_size` made those concerns explicit and testable.

This is retained in the architecture because it is a general lesson: configuration values should map to one clear responsibility when nested retrieval stages amplify each other.

---

# Current State

Completed:

```text
PDF ingestion
Chunking
Embeddings
Qdrant persistence
Dense retrieval
Local generation
RAG context construction
Citation integrity
FastAPI
React UI
BM25 lexical retrieval
Weighted RRF hybrid retrieval
Cross-encoder reranking
Retrieval routing
Formal graded retrieval evaluation
Development tuning
Frozen holdout evaluation
```

Next:

```text
Phase 14 — Research Features
```

Later evaluation expands into answer faithfulness, abstention quality, and claim-level citation support rather than continuing to optimise the existing 12-query development set.

# System Architecture

## Purpose

The Local RAG Research Assistant is designed as a modular, local-first Retrieval-Augmented Generation system for research documents.

Rather than treating RAG as one black-box operation, the application separates ingestion, chunking, embeddings, persistence, retrieval, context construction, generation, citations, API transport, frontend presentation, and evaluation into independently testable components.

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
           ┌─────────────────┼─────────────────┐
           │                 │                 │
           ▼                 ▼                 ▼
      Document Flow    RetrievalRouter    Citation Service
           │                 │
           ▼        ┌────────┼────────┐
        Qdrant      ▼        ▼        ▼
                  Dense   Lexical   Hybrid
                    │        │        │
                    │        │    Dense + BM25
                    │        │        │
                    │        │        ▼
                    │        │       RRF
                    └────────┴────────┘
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

---

# Retrieval Architecture

```text
                            Query
                              │
                              ▼
                      RetrievalRouter
                              │
           ┌──────────────────┼──────────────────┐
           ▼                  ▼                  ▼
        Dense              Lexical             Hybrid
           │                  │                  │
           ▼                  ▼          ┌───────┴───────┐
SemanticRetriever       BM25Retriever     ▼               ▼
           │                  │        Dense            BM25
           ▼                  ▼           │               │
EmbeddingProvider       ChunkCorpus        └───────┬───────┘
           │                  │                   ▼
           ▼                  ▼                  RRF
      VectorStore           Qdrant                 │
           │                                        ▼
           ▼                                RetrievalResult[]
         Qdrant
```

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

## Hybrid Retrieval

```text
Dense candidates
+
Lexical candidates
 ↓
Reciprocal Rank Fusion
 ↓
RetrievalResult[]
```

Current baseline:

```text
RRF rank constant = 60
candidate multiplier = 2
```

RRF is used because dense cosine and BM25 scores are not directly comparable.

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

`RetrievalConfig` contains the requested strategy:

```text
dense
lexical
hybrid
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
dense    → cosine similarity
lexical  → BM25 score
hybrid   → RRF score
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

---

# Insufficient Evidence

```text
No Evidence
    ↓
Skip LLM
    ↓
Insufficient-Evidence Response
```

Nearest-neighbour retrieval can still return weak candidates for unrelated questions, so weak-evidence detection remains an evaluation problem.

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

The browser does not access Qdrant, BGE, Ollama, or tokenizer internals directly.

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
Local context building
 ↓
Local Ollama/Qwen
 ↓
Local cited answer
```

---

# Evaluation Boundary

Retrieval, generation, and citation behaviour are evaluated separately.

Planned retrieval metrics:

```text
Recall@K
Precision@K
MRR
nDCG
```

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
RRF hybrid retrieval
Retrieval routing
Dense/lexical/hybrid comparison
```

Next:

```text
Cross-encoder reranking
```

Then:

```text
Formal labelled evaluation
```

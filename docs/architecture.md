# System Architecture

## Purpose

The Local RAG Research Assistant is designed as a modular, local-first Retrieval-Augmented Generation system for working with research documents.

Rather than treating RAG as a single black-box operation, the application separates document ingestion, chunking, embeddings, persistence, retrieval, generation, citations, and evaluation into independently testable components.

The intended full architecture is:

```text
                           User
                            │
                            ▼
                     Application UI
                            │
                            ▼
                         FastAPI
                            │
       ┌────────────────────┼─────────────────────┐
       │                    │                     │
       ▼                    ▼                     ▼
   Ingestion            Retrieval             Evaluation
       │                    │                     │
       ▼                    ▼                     ▼
 PDF Parsing          Query Embedding        Retrieval Metrics
       │                    │               Answer Evaluation
 Normalisation             Search            Citation Checks
       │                    │
       ▼                    ▼
   Chunking          ┌───────────────┐
       │             │ Vector Store  │
       ▼             └───────┬───────┘
  Embeddings                 │
       │                 Candidates
       ▼                     │
 Vector Store                ▼
                       Hybrid Retrieval
                             │
                             ▼
                          Reranker
                             │
                             ▼
                       Context Builder
                             │
                             ▼
                          Local LLM
                             │
                             ▼
                    Grounded RAG Response
                             │
                             ▼
                    Citations + Evidence
```

The project is being implemented incrementally so each layer can be understood and validated before additional orchestration is introduced.

---

## Design Boundaries

The system deliberately separates its major responsibilities.

### Ingestion

The ingestion layer converts external research documents into the application's canonical representation.

```text
PDF
 ↓
ParsedDocument
```

The current implementation uses PyMuPDF for PDF parsing.

The ingestion layer is responsible for:

* validating supported documents
* extracting page-level text
* extracting PDF metadata
* computing deterministic document identity
* preserving page boundaries
* normalising extracted text
* detecting encrypted or non-text PDFs

Later document formats should be able to enter through this boundary without changing retrieval or generation logic.

---

### Chunking

The chunking layer converts parsed pages into retrieval-sized textual units.

```text
ParsedDocument
      ↓
DocumentPage[]
      ↓
DocumentChunk[]
```

The current baseline uses overlapping token windows aligned with the embedding model's tokenizer.

Each chunk preserves:

* document identity
* filename
* page number
* chunk position
* token range
* source text

This maintains the provenance required for later retrieval and citation generation.

---

### Embeddings

The embedding layer converts chunks and natural-language queries into dense vector representations.

```text
DocumentChunk
      ↓
EmbeddingProvider
      ↓
ChunkEmbedding
```

The current implementation uses:

```text
BAAI/bge-small-en-v1.5
```

through Sentence Transformers.

The embedding provider is abstracted so models can later be compared or replaced without rewriting storage or retrieval logic.

Document and query embeddings are normalised before similarity search.

---

### Vector Storage

The vector-store layer persists document embeddings and their source metadata.

```text
ChunkEmbedding[]
       ↓
VectorStore
       ↓
Qdrant
```

Each stored point contains:

```text
vector
+
chunk text
+
document provenance
+
embedding metadata
```

The current implementation uses persistent local Qdrant storage.

The vector-store boundary supports:

* collection creation and validation
* embedding compatibility checks
* document insertion
* document replacement
* document deletion
* persistent storage
* metadata filtering
* vector search

The application does not expose Qdrant-specific objects outside this layer.

---

### Retrieval

The retrieval layer converts a natural-language question into ranked source evidence.

```text
Question
   ↓
Query Embedding
   ↓
Vector Search
   ↓
Top-K Matches
   ↓
RetrievalResult[]
```

The current baseline implements dense semantic retrieval using BGE query embeddings and Qdrant cosine search.

Retrieval supports:

* configurable top-k
* similarity scores
* document-scoped search
* optional score thresholds
* typed ranked results
* reconstruction of source provenance

The planned retrieval stack will later evolve toward:

```text
Dense Retrieval
       +
BM25 Retrieval
       ↓
Hybrid Fusion
       ↓
Reranking
```

Dense retrieval therefore acts as the baseline against which future retrieval strategies can be measured.

---

### Generation

The generation layer provides language-model inference independently of retrieval.

```text
ChatMessage[]
      ↓
LLMProvider
      ↓
OllamaProvider
      ↓
Local Qwen Model
      ↓
GenerationResult
```

The current implementation uses Ollama as the local model runtime and Qwen3.5 as the development baseline.

The generation boundary is responsible for:

* model invocation
* message transport
* inference configuration
* response validation
* token accounting
* model-loading metrics
* prompt-processing metrics
* generation timing
* runtime error translation

The generation layer is deliberately not responsible for:

* retrieval
* evidence selection
* context construction
* grounding
* citations

Those belong to higher RAG layers.

---

### Context Construction

Context construction is the next major integration boundary.

Retrieved chunks should not simply be concatenated without control.

The context layer will manage:

* model context budget
* evidence selection
* retrieval ordering
* source labelling
* duplicate evidence
* overlapping chunks
* relevance constraints
* reserved generation space
* grounding instructions

Conceptually:

```text
Total Runtime Context
│
├── system instructions
├── user question
├── retrieved evidence
└── reserved answer budget
```

This layer will connect retrieval and generation in Phase 7.

---

### Citations

Every final citation should correspond to real retrieved evidence.

The intended provenance chain is:

```text
Answer claim
 ↓
Retrieved evidence
 ↓
DocumentChunk
 ↓
page_number
 ↓
source document
```

Because provenance is preserved through ingestion, chunking, storage, and retrieval, later citation construction should not need to infer or invent source information.

---

### Evaluation

Evaluation remains independent from generation.

The project will evaluate retrieval separately from answer generation so failures can be diagnosed accurately.

For example:

```text
bad retrieved evidence
        ↓
retrieval failure

good evidence
+
unsupported answer
        ↓
generation / grounding failure
```

Planned retrieval metrics include:

* Recall@K
* Precision@K
* Mean Reciprocal Rank
* nDCG

Planned answer-level evaluation includes:

* faithfulness
* answer relevance
* context relevance
* citation correctness

---

# Current Architecture

As of Phase 6, the system contains three major working pipelines.

## Indexing Pipeline

```text
Research PDF
      │
      ▼
   PyMuPDF
      │
      ▼
ParsedDocument
      │
      ▼
DocumentPage[]
      │
      ▼
Model-Aware Tokenizer
      │
      ▼
DocumentChunk[]
      │
      ▼
Sentence Transformer
      │
      ▼
ChunkEmbedding[]
      │
      ▼
    Qdrant
```

This pipeline transforms research documents into persistent searchable semantic representations.

---

## Retrieval Pipeline

```text
Natural-Language Query
          │
          ▼
  EmbeddingProvider
          │
          ▼
     Query Vector
          │
          ▼
     VectorStore
          │
          ▼
      Qdrant Search
          │
          ▼
  RetrievalResult[]
```

The retrieval pipeline can already function independently as a semantic research search engine.

---

## Generation Pipeline

```text
ChatMessage[]
      │
      ▼
   LLMProvider
      │
      ▼
 OllamaProvider
      │
      ▼
 Ollama Runtime
      │
      ▼
 Local Qwen Model
      │
      ▼
GenerationResult
```

The generation pipeline can independently produce local model responses and exposes token and latency information for later observability.

---

## Phase 7 Integration

The next system layer connects retrieval and generation.

```text
                     Question
                        │
              ┌─────────┴─────────┐
              ▼                   │
      Semantic Retrieval          │
              │                   │
              ▼                   │
      Retrieved Evidence          │
              │                   │
              └─────────┬─────────┘
                        ▼
                 Context Builder
                        │
                        ▼
                 Grounded Prompt
                        │
                        ▼
                   LLMProvider
                        │
                        ▼
                    RAG Answer
```

This is the point where the project becomes a complete Retrieval-Augmented Generation system.

---

# Provider Abstractions

The project uses explicit interfaces around external infrastructure.

## Embeddings

```text
EmbeddingProvider
       │
       └── SentenceTransformerEmbedder
```

Potential future providers include:

```text
EmbeddingProvider
├── SentenceTransformerEmbedder
├── OllamaEmbeddingProvider
└── RemoteEmbeddingProvider
```

---

## Vector Storage

```text
VectorStore
    │
    └── QdrantVectorStore
```

This keeps retrieval independent from Qdrant-specific APIs.

---

## Generation

```text
LLMProvider
    │
    └── OllamaProvider
```

Potential future implementations include:

```text
LLMProvider
├── OllamaProvider
├── OpenAIProvider
├── AnthropicProvider
└── other local runtimes
```

The future RAG orchestration layer should depend only on these interfaces rather than specific vendors.

---

# Provenance Across the Pipeline

One of the central design goals is to preserve source identity through every transformation.

```text
Research PDF
     ↓
ParsedDocument
     ↓
DocumentPage
     ↓
DocumentChunk
     ↓
ChunkEmbedding
     ↓
Qdrant Point
     ↓
RetrievalResult
     ↓
Future RAG Evidence
     ↓
Citation
```

At no stage should the system lose the relationship between retrieved information and the original document.

---

# Local-First Design

The default architecture is intended to support an entirely local workflow:

```text
Research Document
       ↓
Local Parsing
       ↓
Local Chunking
       ↓
Local Embeddings
       ↓
Local Vector Store
       ↓
Local Retrieval
       ↓
Local LLM
       ↓
Answer
```

This can be valuable when working with:

* research papers
* unpublished work
* internal documents
* confidential technical information
* private knowledge bases

The local-first design also keeps the infrastructure independently reproducible rather than requiring hosted model providers.

---

# Testing Boundaries

Each major component is designed to be testable without loading every other subsystem.

Examples:

```text
Ingestion tests
    → generated PDFs

Chunking tests
    → synthetic document pages

Embedding service tests
    → fake embedding provider

Vector-store tests
    → in-memory Qdrant

Retrieval tests
    → fake embedder + fake vector store

Generation tests
    → HTTPX mock transport
```

Real models and persistent infrastructure are then tested separately through smoke tests.

This keeps the normal test suite:

* fast
* deterministic
* offline-friendly
* isolated

while still validating real integration behaviour separately.

---

# Framework Philosophy

The early project intentionally avoids beginning with LangChain, LlamaIndex, or another high-level RAG orchestration framework.

The aim is to understand directly:

* what text is extracted
* how documents are represented
* how chunks are created
* which tokenizer determines chunk size
* how embeddings are produced
* what metadata is stored
* how vector search behaves
* what similarity scores represent
* how source provenance survives retrieval
* how the LLM is invoked
* how context limits affect generation
* where hallucinations can originate

Higher-level frameworks may be explored later once these mechanics are understood and measurable.

---

# Architectural Direction

The completed project is expected to evolve approximately toward:

```text
                     Research Documents
                            │
                            ▼
                        Ingestion
                            │
                            ▼
                         Chunking
                            │
                            ▼
                        Embeddings
                            │
                            ▼
                       Vector Store
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
        Dense Retrieval              BM25 Retrieval
              │                           │
              └─────────────┬─────────────┘
                            ▼
                      Hybrid Fusion
                            │
                            ▼
                         Reranker
                            │
                            ▼
                    Context Builder
                            │
                            ▼
                       Local LLM
                            │
                            ▼
                    Grounded Answer
                            │
                            ▼
                   Citations + Evidence
                            │
                            ▼
                        Evaluation
```

Each additional component should extend the existing architecture rather than bypassing the interfaces already established.

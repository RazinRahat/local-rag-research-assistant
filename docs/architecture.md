docs/architecture.md
System Architecture
Purpose

The Local RAG Research Assistant is designed as a modular, local-first Retrieval-Augmented Generation system for working with research documents.

Rather than treating RAG as a single black-box operation, the application separates document processing, retrieval, ranking, generation, citation, and evaluation into independently testable components.

The intended full architecture is:

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
Design Boundaries

The project intentionally separates several domains.

Ingestion

Responsible for converting external document formats into the application's canonical document representation.

PDF
 ↓
ParsedDocument

Later parsers should be able to support additional formats without changing retrieval or generation logic.

Chunking

Responsible for converting document pages into retrieval-sized units.

ParsedDocument
 ↓
DocumentChunk[]

Chunking preserves document and page provenance.

Embeddings

Responsible for converting textual chunks and queries into dense vector representations.

DocumentChunk
 ↓
Embedding Model
 ↓
ChunkEmbedding

The embedding provider is abstracted so models can later be compared or replaced.

Vector Storage

Will persist vectors and searchable metadata.

The vector store should contain enough provenance to recover:

vector
 ↓
chunk
 ↓
page
 ↓
document
Retrieval

Will identify potentially relevant chunks for a query.

The planned retrieval stack includes:

Dense Retrieval
       +
BM25 Retrieval
       ↓
Hybrid Fusion
       ↓
Reranking
Context Construction

Retrieved chunks should not simply be concatenated without control.

The context layer will eventually manage:

token budgets
relevance thresholds
duplicate evidence
ordering
neighbouring context
maximum context size
Generation

The local language model will receive the question and retrieved evidence and will be instructed to answer from that evidence.

If the supplied evidence is insufficient, the system should prefer saying so rather than inventing unsupported information.

Citations

Every answer citation should correspond to real retrieved evidence.

The intended provenance chain is:

Answer claim
 ↓
Retrieved chunk
 ↓
DocumentChunk
 ↓
page_number
 ↓
source document
Evaluation

Evaluation is intentionally independent from generation.

The project will measure retrieval separately from answer generation so failures can be diagnosed rather than simply labelled "the RAG system failed."

Current Architecture

As of Phase 3:

PDF
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
Hugging Face Tokenizer
 │
 ▼
DocumentChunk[]
 │
 ▼
Sentence Transformer
 │
 ▼
ChunkEmbedding[]

The current pipeline runs locally and preserves provenance across every transformation.

Provider Abstractions

The project uses interfaces at external model boundaries.

Current example:

EmbeddingProvider
       │
       └── SentenceTransformerEmbedder

Future possibilities include:

EmbeddingProvider
├── SentenceTransformerEmbedder
├── OllamaEmbeddingProvider
└── RemoteEmbeddingProvider

A similar approach may later be used for LLM providers and vector stores.

Local-First Design

The eventual default architecture is intended to support:

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
Local LLM
       ↓
Answer

This can be valuable for research papers, internal documents, unpublished work, corporate knowledge, and other information that users may not want uploaded to an external model provider.

Framework Philosophy

The early project intentionally avoids starting with LangChain or similar orchestration frameworks.

The aim is to understand:

what text is being embedded
what metadata is stored
how chunks are constructed
what similarity scores mean
how retrieval selects evidence
why reranking changes results
what context the LLM actually receives
how hallucinations can originate upstream

Higher-level frameworks may be explored later once these mechanics are understood.
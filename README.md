# Local RAG Research Assistant

A local-first research assistant for querying and analysing research documents using Retrieval-Augmented Generation (RAG).

The project is being built component by component to explore the engineering behind document processing, semantic retrieval, local language generation, context construction, grounded answers, citations, and RAG evaluation without hiding the core pipeline behind a high-level RAG framework.

## Current Status

**Phase 7 complete — end-to-end local RAG.**

```text
PDF
 ↓
Page Extraction
 ↓
Text Normalisation
 ↓
Model-Aware Chunking
 ↓
Local Embeddings
 ↓
Persistent Vector Store
 ↓
Semantic Retrieval
 ↓
Context Construction
 ↓
Local LLM
 ↓
Grounded RAG Response
 ↓
Citations                 ← next
```

The system can now ingest and index research documents, retrieve semantically relevant evidence, construct a bounded evidence context, and generate a local answer through Ollama.

Formal citation validation and user-facing source references are the next stage.

## Current Features

* PDF validation and page-level text extraction
* PDF metadata extraction
* SHA-256 document fingerprinting
* Conservative text normalisation
* Page provenance preservation
* Encrypted and non-text page detection
* Typed document and chunk models
* Configurable token-aware chunking
* Model-aware Hugging Face tokenisation
* Deterministic chunk identifiers
* Local BGE embedding generation
* Normalised document and query embeddings
* Batched embedding inference
* Persistent Qdrant vector storage
* Collection and embedding compatibility validation
* Document-level vector replacement and deletion
* Persistent chunk text and provenance metadata
* Dense semantic retrieval
* Configurable top-k search
* Document-scoped retrieval
* Optional retrieval score thresholds
* Typed ranked retrieval results
* Local LLM inference through Ollama
* Provider-independent LLM interface
* Configurable generation parameters
* Generation token accounting
* Model load and inference timing metrics
* Bounded RAG context construction
* Chat-template-aware prompt token estimation
* Whole-chunk evidence selection
* Retrieval-order preservation
* Duplicate and highly overlapping evidence filtering
* Explicit source labels inside prompts
* Retrieved-evidence prompt boundaries
* Grounding instructions
* Insufficient-evidence fallback when retrieval returns no usable evidence
* Typed end-to-end RAG responses
* Estimated vs runtime prompt-token tracking
* Strict type checking and automated tests

## Architecture

The project now implements a complete local dense-RAG baseline:

```text
                         Research PDF
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
                           Qdrant
                              │
                              │
User Question ────────────────┤
        │                     │
        ▼                     ▼
 Query Embedding       Stored Vectors
        │                     │
        └──────────┬──────────┘
                   ▼
           Semantic Retrieval
                   │
                   ▼
            Ranked Evidence
                   │
                   ▼
            Context Builder
                   │
        ┌──────────┼───────────┐
        │          │           │
    token      evidence     source
    budget     filtering    labels
        │          │           │
        └──────────┼───────────┘
                   ▼
            Grounded Prompt
                   │
                   ▼
              LLMProvider
                   │
                   ▼
             Ollama / Qwen
                   │
                   ▼
              RAGResponse
```

Every retrieved evidence block retains its original document, page, chunk identity, retrieval score, and rank.

The RAG response also exposes the evidence supplied to the model rather than returning only generated text.

See [System Architecture](docs/architecture.md) and [End-to-End RAG Pipeline](docs/rag-pipeline.md) for the deeper design.

## Tech Stack

* Python 3.13
* FastAPI
* Pydantic
* PyMuPDF
* Hugging Face Transformers
* Sentence Transformers
* BGE embeddings
* NumPy
* Qdrant
* Ollama
* Qwen3.5
* HTTPX
* Poetry
* Pytest
* Ruff
* Mypy

Additional citation, retrieval, evaluation, and research-specific capabilities will be introduced incrementally.

## Project Structure

```text
src/research_assistant/
├── main.py
├── config.py
├── ingestion/
├── chunking/
├── embeddings/
├── vector_store/
├── retrieval/
├── generation/
└── rag/

tests/
├── ingestion/
├── chunking/
├── embeddings/
├── vector_store/
├── retrieval/
├── generation/
└── rag/

docs/
├── architecture.md
├── ingestion.md
├── chunking.md
├── embeddings.md
├── vector-store.md
├── retrieval.md
├── generation.md
├── rag-pipeline.md
├── experiments.md
└── roadmap.md
```

## Development

Install dependencies:

```bash
poetry install
```

Run the FastAPI development server:

```bash
poetry run uvicorn research_assistant.main:app --reload --app-dir src
```

Run the quality gate:

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy
poetry run pytest
```

All checks should pass before changes are committed.

## Local Model Runtime

The current generation baseline uses Ollama with a local Qwen model.

Pull the configured model:

```bash
ollama pull qwen3.5:9b
```

Inspect locally available models:

```bash
ollama list
```

The application communicates with Ollama through its local HTTP API.

Model files are managed outside the repository and are not committed to Git.

## Local Documents and Data

Research documents can be placed in:

```text
data/documents/
```

The persistent local vector store is created under:

```text
data/vector_store/
```

Document contents, generated vector stores, environment files, and local model artifacts are excluded from version control.

The vector store contains document-derived text and metadata and should therefore be treated as potentially private data rather than merely as disposable cache.

## Documentation

Deeper implementation details are available in:

* [System Architecture](docs/architecture.md)
* [Document Ingestion](docs/ingestion.md)
* [Document Chunking](docs/chunking.md)
* [Embeddings](docs/embeddings.md)
* [Vector Storage](docs/vector-store.md)
* [Semantic Retrieval](docs/retrieval.md)
* [Local Generation](docs/generation.md)
* [End-to-End RAG Pipeline](docs/rag-pipeline.md)
* [Experiments](docs/experiments.md)
* [Development Roadmap](docs/roadmap.md)

## Roadmap

**Completed**

* Project foundation
* PDF ingestion
* Token-aware chunking
* Local embedding generation
* Persistent vector storage
* Semantic retrieval
* Local LLM integration
* End-to-end RAG orchestration

**Next**

* Grounded citations
* Inspectable supporting evidence
* Citation validation

**Later**

* RAG API and user interface
* Hybrid retrieval
* Reranking
* RAG evaluation and benchmarking
* Research-specific synthesis
* Observability
* Containerisation
* Reproducible demo and deployment

## Design Principles

* **Local first:** keep research documents, vector storage, retrieval, and model inference on the user's machine where practical.
* **Preserve provenance:** evidence should remain traceable from the generated answer back to the original document and page.
* **Understand before abstracting:** implement and inspect the core pipeline before introducing high-level RAG frameworks.
* **Separate responsibilities:** parsing, chunking, embeddings, persistence, retrieval, context construction, and generation remain independently testable.
* **Bound the context:** retrieved evidence should fit an explicit model-context budget rather than being concatenated without limits.
* **Treat retrieved text as data:** document contents are evidence, not executable model instructions.
* **Measure instead of guess:** retrieval, context, generation, and later citation decisions should increasingly be driven by controlled experiments.

---

🚧 Under active development.

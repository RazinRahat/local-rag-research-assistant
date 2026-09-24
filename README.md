# Local RAG Research Assistant

A local-first research assistant for querying and analysing research documents using Retrieval-Augmented Generation (RAG).

The project is being built component by component to explore the engineering behind document processing, semantic retrieval, grounded generation, citations, and RAG evaluation without hiding the core pipeline behind a high-level RAG framework.

## Current Status

**Phase 5 complete — semantic retrieval.**

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
Local LLM              ← next
 ↓
Grounded Answer + Citations
```

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
* Typed ranked retrieval results
* Strict type checking and automated tests

## Architecture

The current pipeline converts research PDFs into searchable semantic representations while preserving their source information:

```text
Research PDF
     │
     ▼
ParsedDocument
     │
     ▼
DocumentPage[]
     │
     ▼
DocumentChunk[]
     │
     ▼
Embedding Model
     │
     ▼
ChunkEmbedding[]
     │
     ▼
   Qdrant
     │
     ▼
Query Embedding
     │
     ▼
Semantic Search
     │
     ▼
RetrievalResult[]
```

Each retrieved result retains its source document, page number, chunk identity, original text, similarity score, and ranking.

See [System Architecture](docs/architecture.md) for the broader design.

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
* Poetry
* Pytest
* Ruff
* Mypy

Additional generation, evaluation, and orchestration infrastructure will be introduced incrementally.

## Project Structure

```text
src/research_assistant/
├── main.py
├── config.py
├── ingestion/
├── chunking/
├── embeddings/
├── vector_store/
└── retrieval/

tests/
├── ingestion/
├── chunking/
├── embeddings/
├── vector_store/
└── retrieval/

docs/
├── architecture.md
├── ingestion.md
├── chunking.md
├── embeddings.md
├── vector-store.md
├── retrieval.md
├── experiments.md
└── roadmap.md
```

## Development

Install dependencies:

```bash
poetry install
```

Run the API:

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

## Local Documents and Data

Research documents can be placed in:

```text
data/documents/
```

The persistent local vector store is created under:

```text
data/vector_store/
```

Document contents, generated stores, model artifacts, and environment files are excluded from version control.

## Documentation

Deeper implementation details are available in:

* [System Architecture](docs/architecture.md)
* [Document Ingestion](docs/ingestion.md)
* [Document Chunking](docs/chunking.md)
* [Embeddings](docs/embeddings.md)
* [Vector Storage](docs/vector-store.md)
* [Semantic Retrieval](docs/retrieval.md)
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

**Next**

* Local LLM integration
* End-to-end RAG querying
* Grounded citations

**Later**

* Hybrid retrieval and reranking
* RAG evaluation and benchmarking
* Research-specific synthesis
* API and user interface
* Observability and containerisation

## Design Principles

* **Local first:** keep research documents on the user's machine where possible.
* **Preserve provenance:** retrieved evidence should remain traceable to its source document and page.
* **Understand before abstracting:** implement the core pipeline before introducing high-level RAG frameworks.
* **Measure instead of guess:** retrieval and chunking decisions should eventually be driven by experiments.

---

🚧 Under active development.

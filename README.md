# Local RAG Research Assistant

A local-first research assistant for querying and analysing research documents using Retrieval-Augmented Generation (RAG).

The project is being built component by component to explore the engineering behind document processing, semantic retrieval, grounded generation, citations, and RAG evaluation without hiding the core pipeline behind a high-level RAG framework.

## Current Status

**Phase 3 complete — local document embeddings.**

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
Vector Store        ← next
 ↓
Retrieval
 ↓
Reranking
 ↓
Local LLM
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
* Strict type checking and automated tests

## Architecture

The current pipeline converts research PDFs into semantic representations while preserving their source information:

```text
PDF
 │
 ▼
ParsedDocument
 │
 └── DocumentPage[]
          │
          ▼
     DocumentChunk[]
          │
          ▼
     Embedding Model
          │
          ▼
     ChunkEmbedding[]
```

Each embedded chunk retains its document identity, page number, text, and chunk identifier so later retrieval results can be traced back to their original source.

See [System Architecture](docs/architecture.md) for the deeper design.

## Tech Stack

* Python 3.13
* FastAPI
* Pydantic
* PyMuPDF
* Hugging Face Transformers
* Sentence Transformers
* BGE embeddings
* NumPy
* Poetry
* Pytest
* Ruff
* Mypy

Additional retrieval and generation infrastructure will be introduced incrementally.

## Project Structure

```text
src/research_assistant/
├── main.py
├── config.py
├── ingestion/
├── chunking/
└── embeddings/

tests/
├── ingestion/
├── chunking/
└── embeddings/

docs/
├── architecture.md
├── ingestion.md
├── chunking.md
├── embeddings.md
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

## Local Documents

Research documents can be placed in:

```text
data/documents/
```

Document contents, generated stores, model artifacts, and environment files are excluded from version control.

## Documentation

Deeper implementation details are available in:

* [System Architecture](docs/architecture.md)
* [Document Ingestion](docs/ingestion.md)
* [Document Chunking](docs/chunking.md)
* [Embeddings](docs/embeddings.md)
* [Experiments](docs/experiments.md)
* [Development Roadmap](docs/roadmap.md)

## Roadmap

**Completed**

* Project foundation
* PDF ingestion
* Token-aware chunking
* Local embedding generation

**Next**

* Persistent vector storage
* Semantic retrieval
* End-to-end RAG querying

**Later**

* Hybrid retrieval and reranking
* Grounded citations
* RAG evaluation and benchmarking
* Research-specific synthesis
* Local LLM integration
* API and user interface
* Observability and containerisation

## Design Principles

* **Local first:** keep research documents on the user's machine where possible.
* **Preserve provenance:** retrieved evidence should remain traceable to its source document and page.
* **Understand before abstracting:** implement the core pipeline before introducing high-level RAG frameworks.
* **Measure instead of guess:** retrieval and chunking decisions should eventually be driven by experiments.

---

🚧 Under active development.

# Local RAG Research Assistant

A local-first research assistant for querying and analysing research documents using Retrieval-Augmented Generation (RAG).

The project is being built component by component to explore the engineering behind document processing, retrieval, grounded generation, citations, and RAG evaluation without hiding the core pipeline behind a high-level framework.

## Current Status

**Phase 2 complete — document ingestion and token-aware chunking.**

```text
PDF
 ↓
Page Extraction
 ↓
Text Normalisation
 ↓
Token-Aware Chunking
 ↓
Embeddings        ← next
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

* PDF validation and page-level extraction
* PDF metadata extraction
* SHA-256 document fingerprinting
* Text normalisation
* Page provenance preservation
* Detection of encrypted and non-text pages
* Typed internal document models
* Token-aware document chunking
* Configurable chunk size and overlap
* Deterministic chunk identifiers
* Page-level chunk provenance
* Automated tests and strict static type checking

## Architecture

The project currently converts research PDFs into structured retrieval units:

```text
PDF
 │
 ▼
ParsedDocument
 │
 ├── DocumentMetadata
 │
 └── DocumentPage[]
          │
          ▼
     Tokenizer
          │
          ▼
    DocumentChunk[]
```

Each chunk retains its source document, page number, token range, and position within the document so later retrieval results can be traced back to their original source.

## Tech Stack

* Python 3.13
* FastAPI
* Pydantic
* PyMuPDF
* tiktoken
* Poetry
* Pytest
* Ruff
* Mypy

RAG infrastructure will be introduced incrementally as the project develops.

## Project Structure

```text
src/research_assistant/
├── main.py
├── config.py
├── ingestion/
│   ├── models.py
│   ├── normalizer.py
│   └── pdf_loader.py
└── chunking/
    ├── models.py
    ├── tokenizer.py
    └── chunker.py

tests/
├── ingestion/
└── chunking/
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

Run the quality checks:

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

Document contents, generated data, environment files, and local vector stores are excluded from version control.

## Roadmap

**Completed**

* Project foundation
* PDF ingestion
* Document representation
* Token-aware chunking

**Next**

* Local embedding generation
* Vector storage
* Semantic retrieval
* End-to-end RAG querying

**Later**

* Hybrid retrieval and reranking
* Grounded source citations
* RAG evaluation and retrieval benchmarking
* Research-specific synthesis features
* Local LLM integration
* API and user interface
* Observability and containerisation

## Design Approach

The project follows three main principles:

* **Local first:** research documents should remain on the user's machine where possible.
* **Preserve provenance:** retrieved information should remain traceable to its original document and page.
* **Measure instead of guess:** chunking, retrieval, and later RAG components will be evaluated experimentally rather than selected only by intuition.

---

🚧 Under active development.

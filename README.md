# Local RAG Research Assistant

A local-first research assistant for retrieving, analysing, and synthesising information from research documents using Retrieval-Augmented Generation (RAG).

The project is being built incrementally from first principles to explore the engineering behind modern RAG systems, including document ingestion, chunking, embeddings, retrieval, reranking, grounded generation, citations, and evaluation.

> **Current stage:** Phase 1 — Document Ingestion complete.
> **Next:** Phase 2 — Document Chunking.

---

## Overview

Most RAG tutorials begin by connecting a document loader, vector database, and LLM through a high-level framework.

This project takes a different approach.

The core RAG pipeline is being implemented component by component so that each stage can be inspected, tested, evaluated, and improved independently.

The eventual pipeline will look like:

```text
Research Documents
        │
        ▼
Document Ingestion
        │
        ▼
Text Normalisation
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
        ▼
Retrieval
        │
        ├── Dense Retrieval
        ├── BM25
        └── Hybrid Retrieval
        │
        ▼
Reranking
        │
        ▼
Context Construction
        │
        ▼
Local LLM
        │
        ▼
Grounded Answer
        │
        └── Source Citations
```

The final system is intended to run primarily on local infrastructure so research documents can be processed without requiring them to leave the user's machine.

---

## Project Goals

The project aims to support:

* Local research-document processing
* Page-aware document ingestion
* Configurable document chunking
* Local embedding generation
* Dense semantic retrieval
* BM25 keyword retrieval
* Hybrid retrieval
* Cross-encoder reranking
* Local LLM inference
* Grounded question answering
* Traceable source citations
* Multi-document synthesis
* Research-paper comparison
* Evidence extraction
* RAG evaluation
* Retrieval benchmarking
* Observability and debugging
* FastAPI-based application interfaces

---

## Current Capabilities

### Document Ingestion

The current ingestion pipeline supports:

* PDF file validation
* Page-level text extraction with PyMuPDF
* Human-readable page numbering
* PDF metadata extraction
* SHA-256 document fingerprinting
* Duplicate-document identity support
* Conservative Unicode and whitespace normalisation
* Page provenance preservation
* Empty-page preservation
* Detection of pages containing extractable text
* Page dimensions and rotation metadata
* Detection of encrypted PDFs
* Typed internal document models
* Automated ingestion tests

Documents are converted into an internal representation before they are used by later RAG components.

```text
PDF
 │
 ▼
PyMuPDF
 │
 ▼
Text Extraction
 │
 ▼
Normalisation
 │
 ▼
ParsedDocument
 │
 ├── DocumentMetadata
 │
 └── DocumentPage[]
```

This keeps the remainder of the application independent from the underlying PDF parsing library.

---

## Document Model

A parsed document is represented conceptually as:

```text
ParsedDocument
│
├── metadata
│   ├── document_id
│   ├── file_name
│   ├── file_size_bytes
│   ├── sha256
│   ├── page_count
│   ├── text_page_count
│   ├── title
│   ├── author
│   └── other PDF metadata
│
└── pages
    ├── DocumentPage(page_number=1)
    ├── DocumentPage(page_number=2)
    ├── DocumentPage(page_number=3)
    └── ...
```

Page boundaries are intentionally preserved so later chunks and retrieved evidence can retain accurate document provenance.

This will eventually allow answers to reference sources such as:

```text
[paper.pdf, p. 7]
```

rather than providing citations without a traceable relationship to the original document.

---

## Tech Stack

### Current

* Python 3.13
* FastAPI
* Pydantic
* Pydantic Settings
* PyMuPDF
* Poetry
* Pytest
* Pytest Coverage
* Ruff
* Mypy
* HTTPX

### Planned

The RAG infrastructure will be introduced incrementally.

Expected technologies include:

* Sentence Transformers
* BGE / Nomic embedding models
* ChromaDB
* Qdrant
* BM25
* Cross-encoder rerankers
* Ollama
* RAGAS / custom evaluation tooling
* Docker
* Streamlit or Next.js

High-level RAG frameworks may be explored later, after the underlying pipeline has been implemented directly.

---

## Project Structure

```text
local-rag-research-assistant/
│
├── README.md
├── pyproject.toml
├── poetry.lock
├── .gitignore
├── .python-version
│
├── src/
│   └── research_assistant/
│       ├── __init__.py
│       ├── main.py
│       ├── config.py
│       │
│       └── ingestion/
│           ├── __init__.py
│           ├── exceptions.py
│           ├── models.py
│           ├── normalizer.py
│           └── pdf_loader.py
│
├── tests/
│   ├── __init__.py
│   ├── test_health.py
│   │
│   └── ingestion/
│       ├── __init__.py
│       ├── test_normalizer.py
│       └── test_pdf_loader.py
│
├── data/
│   ├── documents/
│   ├── processed/
│   └── vector_store/
│
└── scripts/
```

Document data, generated vector stores, environment variables, and local model artifacts are excluded from Git.

---

## Development

### Requirements

* Python 3.13
* Poetry

Check the Python version:

```bash
python3 --version
```

Install dependencies:

```bash
poetry install
```

Check the Poetry environment:

```bash
poetry env info
```

---

## Running the API

Start the development server:

```bash
poetry run uvicorn research_assistant.main:app --reload --app-dir src
```

The API will run at:

```text
http://127.0.0.1:8000
```

Health endpoint:

```text
GET /health
```

Expected response:

```json
{
  "status": "ok",
  "service": "local-rag-research-assistant"
}
```

Interactive API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## Quality Checks

The project currently uses Ruff, Mypy, and Pytest as its development quality gate.

### Linting

```bash
poetry run ruff check .
```

### Formatting

```bash
poetry run ruff format --check .
```

To automatically format code:

```bash
poetry run ruff format .
```

### Static Type Checking

```bash
poetry run mypy
```

### Tests

```bash
poetry run pytest
```

All checks should pass before changes are committed.

---

## Working With Documents

Local research documents can be placed inside:

```text
data/documents/
```

For example:

```text
data/documents/research_paper.pdf
```

The contents of this directory are intentionally excluded from Git.

This prevents research papers, private documents, and potentially copyrighted source material from accidentally being committed to the repository.

---

## Development Roadmap

| Phase | Component                                            | Status     |
| ----- | ---------------------------------------------------- | ---------- |
| 0     | Project foundation, environment, FastAPI and tooling | ✅ Complete |
| 1     | PDF ingestion and document representation            | ✅ Complete |
| 2     | Document chunking                                    | 🔜 Next    |
| 3     | Embedding generation                                 | Planned    |
| 4     | Vector storage                                       | Planned    |
| 5     | Semantic retrieval                                   | Planned    |
| 6     | Local LLM integration                                | Planned    |
| 7     | End-to-end RAG pipeline                              | Planned    |
| 8     | Source citations                                     | Planned    |
| 9     | FastAPI RAG endpoints                                | Planned    |
| 10    | User interface                                       | Planned    |
| 11    | BM25 and hybrid retrieval                            | Planned    |
| 12    | Reranking                                            | Planned    |
| 13    | RAG evaluation                                       | Planned    |
| 14    | Research-oriented features                           | Planned    |
| 15    | Observability                                        | Planned    |
| 16    | Containerisation                                     | Planned    |
| 17    | Demo / deployment                                    | Planned    |
| 18    | Retrieval benchmarking and technical report          | Planned    |

---

## Phase 2 — Document Chunking

The next stage will transform page-level document text into smaller retrieval units.

The first chunking implementation will focus on:

```text
ParsedDocument
       │
       ▼
DocumentPage[]
       │
       ▼
Token-Aware Chunker
       │
       ▼
DocumentChunk[]
```

Each chunk will retain provenance such as:

```text
document_id
file_name
page_number
chunk_index
text
token_count
```

The project will initially establish a simple chunking baseline before experimenting with more advanced strategies such as:

* Fixed token windows
* Overlapping chunks
* Recursive chunking
* Sentence-aware chunking
* Section-aware chunking
* Semantic chunking

These strategies will later be compared using retrieval metrics instead of selecting chunk sizes purely by intuition.

---

## Planned Evaluation

The project will eventually evaluate retrieval using metrics such as:

```text
Recall@K
Precision@K
Mean Reciprocal Rank (MRR)
nDCG
```

It will also investigate answer-level properties such as:

```text
faithfulness
answer relevance
context relevance
citation correctness
```

The goal is to measure RAG performance rather than relying only on qualitative demonstrations.

---

## Design Principles

The project follows several principles throughout development:

**Understand before abstracting.**
Core RAG components are implemented directly before introducing high-level orchestration frameworks.

**Preserve provenance.**
Document identity, page information, and source relationships should survive every stage of the pipeline.

**Evaluate components independently.**
Parsing, chunking, retrieval, reranking, and generation should be measurable separately.

**Prefer explicit failure over silent degradation.**
Unsupported, encrypted, malformed, or otherwise problematic documents should produce understandable failures.

**Keep private data local where possible.**
Research documents and generated document stores are excluded from version control and are intended to support local processing.

---

## Status

🚧 **Under active development**

Current milestone:

```text
Phase 1 ✅
Document Ingestion

        ↓

Phase 2 🔜
Document Chunking
```

The project is being expanded incrementally toward a complete, evaluated local RAG research assistant.

# Local RAG Research Assistant

A local-first retrieval-augmented generation system for querying,
analysing, and synthesising research documents.

## Status

🚧 Under active development.

## Goals

- Local document processing
- Dense retrieval
- Hybrid retrieval
- Reranking
- Grounded answer generation
- Source citations
- RAG evaluation
- Local LLM inference

## Tech Stack

- Python
- FastAPI
- PyMuPDF
- Pydantic
- Pytest
- Ruff
- Mypy

Additional RAG infrastructure will be added incrementally.

## Current Capabilities

- PDF validation
- Page-level text extraction
- PDF metadata extraction
- Document fingerprinting with SHA-256
- Conservative text normalization
- Page provenance preservation

## Development

Install dependencies:

```bash
poetry install


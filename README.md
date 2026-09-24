# Local RAG Research Assistant

A local-first research assistant for querying and analysing research documents using Retrieval-Augmented Generation (RAG).

The project is being built component by component to explore the engineering behind document processing, semantic retrieval, local language generation, grounded answers, citations, and RAG evaluation without hiding the core pipeline behind a high-level RAG framework.

## Current Status

**Phase 6 complete — local LLM integration.**

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
Local LLM
 ↓
End-to-End RAG          ← next
 ↓
Grounded Answer + Citations
```

The retrieval and generation systems currently work independently. Phase 7 will connect retrieved evidence to the local LLM through an explicit context-construction and grounding layer.

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
* Local LLM inference through Ollama
* Provider-independent LLM interface
* Configurable generation parameters
* Explicit local context-window configuration
* Generation token accounting
* Model load and inference timing metrics
* Mocked HTTP tests for local model integration
* Strict type checking and automated tests

## Architecture

The project currently contains two independently testable pipelines.

### Retrieval

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

### Generation

```text
ChatMessage[]
     │
     ▼
 LLMProvider
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

Phase 7 will connect the two:

```text
Question
   │
   ├──────────────► Semantic Retrieval
   │                       │
   │                       ▼
   │                 Retrieved Evidence
   │                       │
   └──────────────┬────────┘
                  ▼
            Context Builder
                  │
                  ▼
              Local LLM
                  │
                  ▼
           Grounded Answer
```

Each retrieved chunk retains its source document, page number, chunk identity, text, similarity score, and rank so later generated answers can remain traceable to their evidence.

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
* Qdrant
* Ollama
* Qwen3.5
* HTTPX
* Poetry
* Pytest
* Ruff
* Mypy

Additional RAG orchestration, citation, evaluation, and research-specific capabilities will be introduced incrementally.

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
└── generation/

tests/
├── ingestion/
├── chunking/
├── embeddings/
├── vector_store/
├── retrieval/
└── generation/

docs/
├── architecture.md
├── ingestion.md
├── chunking.md
├── embeddings.md
├── vector-store.md
├── retrieval.md
├── generation.md
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

The current local generation baseline uses Ollama.

Ensure Ollama is installed and running, then pull the configured model:

```bash
ollama pull qwen3.5:9b
```

Check locally available models:

```bash
ollama list
```

The application currently communicates with Ollama through its local HTTP API.

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

The vector store contains document-derived text and metadata and should therefore be treated as potentially private data rather than merely as a disposable cache.

## Documentation

Deeper implementation details are available in:

* [System Architecture](docs/architecture.md)
* [Document Ingestion](docs/ingestion.md)
* [Document Chunking](docs/chunking.md)
* [Embeddings](docs/embeddings.md)
* [Vector Storage](docs/vector-store.md)
* [Semantic Retrieval](docs/retrieval.md)
* [Local Generation](docs/generation.md)
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

**Next**

* End-to-end RAG pipeline
* Context construction
* Grounding behaviour
* Source citations

**Later**

* Hybrid retrieval and reranking
* RAG evaluation and benchmarking
* Research-specific synthesis
* API and user interface
* Observability
* Containerisation
* Reproducible demo and deployment

## Design Principles

* **Local first:** keep research documents and model inference on the user's machine where practical.
* **Preserve provenance:** retrieved evidence should remain traceable to its original document and page.
* **Understand before abstracting:** implement and inspect the core pipeline before introducing high-level RAG frameworks.
* **Separate responsibilities:** parsing, chunking, embeddings, persistence, retrieval, and generation remain independently testable.
* **Measure instead of guess:** retrieval, generation, and later RAG decisions should increasingly be driven by controlled experiments.

---

🚧 Under active development.

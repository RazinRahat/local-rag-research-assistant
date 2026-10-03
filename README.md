# Local RAG Research Assistant

A local-first research assistant for querying, analysing, and citing research documents using Retrieval-Augmented Generation (RAG).

The project is built component by component so the mechanics behind ingestion, chunking, embeddings, retrieval, context construction, local generation, citations, hybrid search, and later evaluation remain explicit and inspectable rather than being hidden behind a high-level RAG framework.

## Current Status

**Phase 12 complete — local research workspace with selectable dense, lexical, hybrid, and cross-encoder-reranked retrieval.**

```text
Research PDF
     ↓
Page Extraction & Normalisation
     ↓
Model-Aware Chunking
     ↓
Local BGE Embeddings
     ↓
Persistent Qdrant Storage
     ↓
Retrieval Router
     │
     ├── Dense Retrieval
     ├── BM25 Lexical Retrieval
     ├── Hybrid Retrieval
     │        ↓
     │ Reciprocal Rank Fusion
     └── Hybrid + Reranking
              ↓
       Cross-Encoder Reranker
     ↓
Bounded Context Construction
     ↓
Local LLM (Ollama / Qwen)
     ↓
Grounded RAG Response
     ↓
Citation Parsing & Validation
     ↓
FastAPI
     ↓
React Research Workspace
     ↓
Interactive Answer + Evidence Inspection
```

The system can ingest and index research PDFs, preserve page and chunk provenance, retrieve evidence using multiple retrieval strategies, construct a token-bounded prompt, generate a local evidence-grounded answer, validate recognised citation identities, and allow the user to inspect the exact retrieved evidence associated with each validated source.

Citation identity validation does **not** yet establish claim-level semantic entailment.

Formal labelled retrieval evaluation, answer-quality evaluation, abstention evaluation, and citation-support evaluation remain future work.

---

## Current Features

### Document Pipeline

- PDF validation using PyMuPDF
- Page-level text extraction
- PDF metadata extraction
- SHA-256 document identity
- Conservative text normalisation
- Encrypted and unusable PDF detection
- Page-level provenance preservation
- Typed document and chunk models
- Configurable token-aware chunking
- Model-aligned Hugging Face tokenisation
- Deterministic chunk identities
- Configurable token overlap
- Local document and query embeddings
- Normalised BGE embeddings
- Persistent Qdrant storage
- Document replacement and deletion
- Stored source text and provenance metadata

Current chunking baseline:

```text
chunk size:     384 tokens
chunk overlap:   64 tokens
```

Current embedding baseline:

```text
BAAI/bge-small-en-v1.5
384 dimensions
```

### Retrieval

The system exposes four retrieval strategies:

```text
Dense
Lexical
Hybrid
Hybrid + Reranking
```

#### Dense Retrieval

```text
query
 ↓
BGE query embedding
 ↓
Qdrant cosine search
 ↓
ranked chunks
```

Dense remains the default retrieval strategy.

#### Lexical Retrieval

```text
query
 ↓
lexical tokenisation
 ↓
BM25
 ↓
persisted DocumentChunk corpus
 ↓
ranked chunks
```

Current BM25 baseline:

```text
k1 = 1.5
b  = 0.75
```

#### Hybrid Retrieval

Hybrid retrieval combines dense and lexical rankings using Reciprocal Rank Fusion.

```text
               Query
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
 Dense Retrieval      BM25 Retrieval
       │                   │
       └─────────┬─────────┘
                 ▼
                 RRF
                 │
                 ▼
         Ranked Evidence
```

Current hybrid baseline:

```text
RRF rank constant:      60
candidate multiplier:    2
final top_k:             5
```

Raw cosine and BM25 scores are not averaged because they use incompatible numerical scales.

#### Cross-Encoder Reranking

Phase 12 adds a second-stage reranker after hybrid retrieval.

```text
Dense + BM25
     ↓
    RRF
     ↓
Top 20 Candidates
     ↓
cross-encoder/ms-marco-MiniLM-L6-v2
     ↓
Final Top 5
```

The Phase 11 `hybrid` mode remains unchanged as a control.

The new `hybrid_reranked` mode uses the cross-encoder relevance score for the final ranking. These scores are not calibrated probabilities and are not numerically comparable with cosine, BM25, or RRF scores.

### Retrieval Modes

Retrieval mode can be selected through both:

```text
POST /search
POST /query
```

Supported values:

```text
dense
lexical
hybrid
hybrid_reranked
```

Dense remains the backward-compatible default.

### Retrieval and RAG

- Dense semantic retrieval using BGE + Qdrant
- BM25 lexical retrieval
- Reciprocal Rank Fusion
- Selectable dense, lexical, hybrid, and cross-encoder-reranked retrieval
- Candidate expansion and second-stage reranking
- Configurable top-k retrieval
- Document-scoped retrieval
- Dense-only optional cosine score thresholds
- Typed `RetrievalResult`
- Full chunk provenance reconstruction
- Local Qwen generation through Ollama
- Explicit context budgeting
- Chat-template-aware token estimation
- Whole-chunk evidence selection
- Duplicate and high-overlap evidence filtering
- Evidence-order preservation
- Grounding instructions
- Source-labelled evidence delimiters
- Explicit insufficient-evidence fallback
- Typed `RAGResponse`
- Prompt-token diagnostics
- Local generation diagnostics

### Citation Integrity

- Strict `[S1]`, `[S2]`, ... citation protocol
- Citation parsing with character offsets
- Mapping from prompt-local source IDs to trusted evidence
- Rejection of unknown recognised source IDs
- Rejection of missing citations in evidence-backed answers
- Document/page/chunk provenance recovered from stored evidence
- Generated filenames and page numbers are not trusted
- Typed `CitedRAGResponse`
- Interactive source inspection in the frontend

Citation identity validation should not be confused with semantic entailment.

### API

FastAPI exposes:

```text
GET    /health
POST   /documents
GET    /documents
DELETE /documents/{document_id}
POST   /search
POST   /query
```

The application layer provides typed request and response models, document indexing, retrieval strategy selection, document management, cited RAG question answering, error mapping, and runtime lifecycle cleanup.

### Frontend

The React research workspace provides:

- indexed document listing
- PDF upload
- document replacement
- document deletion
- corpus-wide scope
- document-specific scope
- Ask mode
- Search mode
- Dense / Lexical / Hybrid / Reranked selector
- ranked evidence cards
- retrieval-score display
- cited RAG answers
- clickable validated citations
- evidence provenance inspection
- generation diagnostics
- insufficient-evidence presentation
- loading and error states
- `Cmd/Ctrl + Enter` submission
- stale-response protection

Changing document scope or retrieval strategy invalidates older in-flight search/RAG responses.

---

## Phase 11 Hybrid Retrieval Experiment

Phase 11 includes a controlled smoke comparison using the same six queries with:

```text
top_k = 5

Dense
Lexical
Hybrid
```

Average top-five Jaccard overlap:

```text
Dense ↔ Lexical    ≈ 0.214
Dense ↔ Hybrid     ≈ 0.478
Lexical ↔ Hybrid   ≈ 0.507
```

Top-result agreement:

```text
Dense ↔ Lexical    2 / 6
Dense ↔ Hybrid     3 / 6
Lexical ↔ Hybrid   4 / 6
```

The important finding is not that one retriever "won".

Instead, dense and lexical retrieval produced meaningfully different candidate sets while hybrid retrieval incorporated evidence from both.

### Example: `Adam optimizer`

For:

```text
Adam optimizer
```

dense retrieval ranked a bibliography passage containing the Adam paper first.

The actual Transformer optimizer-method section appeared second.

BM25 promoted the actual optimizer section to rank 1.

Hybrid retrieval also promoted that section to rank 1.

This is a concrete example of lexical retrieval complementing semantic retrieval.

### Important Claim Boundary

The six-query experiment demonstrates retrieval behaviour and complementarity.

It does **not** establish that hybrid retrieval has higher retrieval accuracy.

Formal labelled evaluation will be introduced later using metrics such as:

```text
Recall@K
Precision@K
MRR
nDCG
```

See:

- [Hybrid Retrieval](docs/hybrid-retrieval.md)
- [Experiments](docs/experiments.md)

---

## Phase 12 Cross-Encoder Reranking Experiment

Phase 12 reused the same six-query set to compare the frozen hybrid baseline against `hybrid_reranked`.

Configuration:

```text
top_k = 5
repeats = 3
warm-up = enabled
candidate pool = 20
cross-encoder = cross-encoder/ms-marco-MiniLM-L6-v2
```

Measured:

```text
Mean top-5 Jaccard overlap:   ≈ 0.409
Top-1 agreement:              1 / 6
Hybrid median latency:        ≈ 35.1 ms
Reranked median latency:      ≈ 108.9 ms
Added end-to-end latency:     ≈ 73.8 ms
```

The reranker produced useful semantic promotions on some explanatory queries, but also clear regressions on others. For example, it promoted the direct recurrence/parallelisation explanation for a semantic paraphrase, while the short query `Adam optimizer` caused it to over-promote a less useful scaling-law passage.

Phase 12 therefore establishes reranking behaviour and integration, not higher retrieval accuracy.

See:

- [Cross-Encoder Reranking](docs/reranking.md)
- [Experiments](docs/experiments.md)

---

## Architecture

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
                 ┌──────────────┴──────────────┐
                 │                             │
                 ▼                             ▼
            Document Pipeline             RetrievalRouter
                 │                             │
                 ▼             ┌─────────┬─────────┬──────────┬──────────────────┐
               Qdrant            ▼         ▼         ▼          ▼
                              Dense     Lexical    Hybrid   Hybrid Reranked
                                │          │         │          │
                                │          │    Dense + BM25    │
                                │          │         │          │
                                │          │         ▼          │
                                │          │        RRF ────────┘
                                │          │                    │
                                │          │                    ▼
                                │          │              Cross-Encoder
                                └──────────┴────────────────────┘
                                             │
                                             ▼
                                      ContextBuilder
                                             │
                                             ▼
                                      Ollama / Qwen
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

## Tech Stack

### Backend

- Python 3.13
- FastAPI
- Pydantic
- HTTPX
- PyMuPDF
- Hugging Face Transformers
- Sentence Transformers
- `BAAI/bge-small-en-v1.5`
- NumPy
- Qdrant
- Ollama
- Qwen3.5
- Poetry
- Pytest
- Ruff
- Mypy

### Frontend

- React
- TypeScript
- Vite
- Vitest
- Testing Library
- jsdom

---

## Project Structure

```text
src/research_assistant/
├── api/
├── ingestion/
├── chunking/
├── embeddings/
├── vector_store/
├── retrieval/
├── generation/
├── rag/
└── citations/

tests/
├── api/
├── ingestion/
├── chunking/
├── embeddings/
├── vector_store/
├── retrieval/
├── generation/
├── rag/
└── citations/

frontend/
└── src/
    ├── api/
    ├── documents/
    ├── search/
    └── query/

scripts/
├── compare_retrieval_modes.py
└── compare_reranking_modes.py

experiments/
├── retrieval_queries.json
└── results/

docs/
├── architecture.md
├── api.md
├── frontend.md
├── ingestion.md
├── chunking.md
├── embeddings.md
├── vector-store.md
├── retrieval.md
├── hybrid-retrieval.md
├── reranking.md
├── generation.md
├── rag-pipeline.md
├── citations.md
├── experiments.md
└── roadmap.md
```

---

## Development

Install backend dependencies:

```bash
poetry install
```

Run the backend quality gate:

```bash
poetry run ruff check .
poetry run ruff format --check .
poetry run mypy
poetry run pytest
```

Start FastAPI:

```bash
poetry run uvicorn \
  research_assistant.main:app \
  --app-dir src \
  --host 127.0.0.1 \
  --port 8000
```

Install frontend dependencies:

```bash
cd frontend
npm install
```

Run the frontend quality gate:

```bash
npm run lint
npm run typecheck
npm run test
npm run build
```

Start Vite:

```bash
npm run dev
```

Open:

```text
http://localhost:5173
```

---

## Local Model Runtime

Ensure the configured Ollama model is available:

```bash
ollama pull qwen3.5:9b
ollama list
```

Search mode does not require LLM generation.

Ask mode requires the configured local model.

---

## Local Documents and Data

Local documents, vector storage, environment files, generated model files, and raw experiment result dumps should not be committed.

Qdrant stores source text and provenance metadata, so the local vector store should be treated as potentially private research data.

---

## Documentation

- [System Architecture](docs/architecture.md)
- [RAG API](docs/api.md)
- [Frontend Research Workspace](docs/frontend.md)
- [Document Ingestion](docs/ingestion.md)
- [Document Chunking](docs/chunking.md)
- [Embeddings](docs/embeddings.md)
- [Vector Storage](docs/vector-store.md)
- [Retrieval](docs/retrieval.md)
- [Hybrid Retrieval](docs/hybrid-retrieval.md)
- [Cross-Encoder Reranking](docs/reranking.md)
- [Local Generation](docs/generation.md)
- [End-to-End RAG Pipeline](docs/rag-pipeline.md)
- [Citation Integrity](docs/citations.md)
- [Experiments](docs/experiments.md)
- [Development Roadmap](docs/roadmap.md)

---

## Roadmap

**Completed through Phase 12:** foundation, ingestion, token-aware chunking, local embeddings, persistent Qdrant storage, dense retrieval, local LLM integration, end-to-end RAG, citation integrity, FastAPI, React workspace, BM25 lexical retrieval, Reciprocal Rank Fusion, selectable hybrid retrieval, and local cross-encoder reranking.

**Next:** Phase 13 — formal relevance-labelled retrieval and RAG evaluation.

**After that:** research-specific features, observability, containerisation, deployment, and broader benchmarking.

---

## Design Principles

- **Local first:** keep documents, vectors, retrieval, and model inference local where practical.
- **Preserve provenance:** trace evidence to its document, page, and chunk.
- **Understand before abstracting:** expose RAG mechanics before adding high-level frameworks.
- **Separate responsibilities:** ingestion, embeddings, storage, retrieval, generation, citations, HTTP transport, and UI remain explicit boundaries.
- **Bound context:** treat model context as a finite resource.
- **Treat retrieved text as data:** document text does not become application instructions.
- **Validate model output:** only application-known source IDs can become trusted citations.
- **Keep identity separate from entailment:** citation identity is not claim-level proof.
- **Measure instead of guess:** do not claim retrieval improvements without evaluation evidence.

---

🚧 Under active development. Phase 12 cross-encoder reranking is complete; Phase 13 labelled evaluation is next.

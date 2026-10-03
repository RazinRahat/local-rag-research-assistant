# RAG API

## Purpose

FastAPI exposes the Local RAG Research Assistant through a typed HTTP application boundary.

```text
HTTP
 ↓
FastAPI
 ↓
ResearchService
 ↓
Existing Domain Components
```

---

# Routes

```text
GET    /health
POST   /documents
GET    /documents
DELETE /documents/{document_id}
POST   /search
POST   /query
```

---

# `GET /health`

Checks API availability.

```bash
curl http://127.0.0.1:8000/health
```

---

# `POST /documents`

Uploads and indexes a PDF.

```bash
curl \
  -X POST \
  http://127.0.0.1:8000/documents \
  -F "file=@paper.pdf"
```

Pipeline:

```text
UploadFile
   ↓
LocalPDFIndexer
   ↓
PDF Ingestion
   ↓
Chunking
   ↓
Embeddings
   ↓
Qdrant
```

Example response:

```json
{
  "document_id": "<sha256>",
  "file_name": "paper.pdf",
  "chunk_count": 42
}
```

---

# `GET /documents`

Lists indexed documents.

```bash
curl http://127.0.0.1:8000/documents
```

Example:

```json
[
  {
    "document_id": "<sha256>",
    "file_name": "paper.pdf",
    "chunk_count": 42
  }
]
```

---

# `DELETE /documents/{document_id}`

Deletes one indexed document.

```bash
curl \
  -X DELETE \
  http://127.0.0.1:8000/documents/<document_id>
```

Successful deletion:

```text
204 No Content
```

---

# `POST /search`

Runs retrieval without invoking the language model.

Supported strategies:

```text
dense
lexical
hybrid
```

Dense is the default.

## Dense Search

```bash
curl \
  -X POST \
  http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does self-attention work?",
    "top_k": 5,
    "mode": "dense"
  }'
```

## Lexical Search

```bash
curl \
  -X POST \
  http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Adam optimizer",
    "top_k": 5,
    "mode": "lexical"
  }'
```

## Hybrid Search

```bash
curl \
  -X POST \
  http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does self-attention work?",
    "top_k": 5,
    "mode": "hybrid"
  }'
```

Search pipeline:

```text
SearchRequest
     ↓
RetrievalConfig
     ↓
RetrievalRouter
     │
     ├── dense   → SemanticRetriever
     ├── lexical → BM25Retriever
     └── hybrid  → HybridRetriever → RRF
     ↓
RetrievalResult[]
```

---

# Retrieval Options

Shared options:

```text
top_k
score_threshold
document_id
mode
```

Supported mode values:

```text
dense
lexical
hybrid
```

Default:

```text
dense
```

`top_k` is constrained to 1–20 at the HTTP layer.

`document_id` optionally scopes retrieval to one indexed document.

`score_threshold` is supported only for dense retrieval because cosine, BM25, and RRF scores have different semantics.

---

# `POST /query`

Runs cited RAG.

Example:

```bash
curl \
  -X POST \
  http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Why does the Transformer avoid recurrent computation?",
    "top_k": 5,
    "mode": "hybrid"
  }'
```

Pipeline:

```text
QueryRequest
     ↓
RetrievalConfig
     ↓
RetrievalRouter
     ↓
Configured Retriever
     ↓
RetrievalResult[]
     ↓
ContextBuilder
     ↓
Ollama / Qwen
     ↓
RAGResponse
     ↓
CitationService
     ↓
CitedRAGResponse
```

---

# Response Shape

Conceptually:

```text
CitedRAGResponse
├── rag
│   ├── question
│   ├── answer
│   ├── evidence
│   ├── retrieved_count
│   ├── used_evidence_count
│   ├── token diagnostics
│   └── generation diagnostics
└── citations
    ├── references
    └── sources
```

---

# Request Validation

Pydantic validates:

- query/question length
- whitespace-only input
- `top_k`
- SHA-256 document IDs
- retrieval modes
- score-threshold compatibility
- unknown fields

---

# Error Handling

Examples:

```text
invalid PDF
→ client error

missing document
→ 404

invalid request
→ 422

Ollama unavailable
→ 503

invalid generation/citation response
→ 502
```

---

# Testing

API tests cover request validation, document upload/list/delete, dense/lexical/hybrid modes, the dense default, invalid strategy values, threshold compatibility, route orchestration, and error mapping.

---

# Running the API

```bash
ollama list
```

```bash
poetry run uvicorn \
  research_assistant.main:app \
  --app-dir src \
  --host 127.0.0.1 \
  --port 8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

# Current Limitations

The API remains a local development baseline.

It does not yet include authentication, multi-user isolation, public deployment hardening, streaming generation, background indexing jobs, distributed workers, production observability, or rate limiting.

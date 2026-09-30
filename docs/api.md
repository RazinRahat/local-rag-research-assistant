# RAG API

## Purpose

Phase 9 exposes the existing Local RAG Research Assistant through a typed FastAPI application layer.

The API does not reimplement ingestion, retrieval, generation, or citation logic. Instead, it provides an HTTP boundary around the components developed in Phases 1–8.

```text
HTTP Request
     ↓
FastAPI
     ↓
ResearchAPI
     ↓
ResearchService
     ↓
Existing RAG Components
```

The current API is intended as a local, single-user development baseline.

---

## API Architecture

```text
                           Client
                              │
                              ▼
                           FastAPI
                              │
                ┌─────────────┼─────────────┐
                │             │             │
                ▼             ▼             ▼
            Documents       Search         Query
                │             │             │
                └─────────────┼─────────────┘
                              ▼
                       ResearchService
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
      LocalPDFIndexer   SemanticRetriever   RAGService
             │                │                │
             ▼                │                ▼
        Ingestion             │          ContextBuilder
             │                │                │
             ▼                │                ▼
         Chunking             │          Ollama / Qwen
             │                │                │
             ▼                │                ▼
        Embeddings            │           RAGResponse
             │                │                │
             └──────► Qdrant ◄┘                ▼
                                        CitationService
                                               │
                                               ▼
                                        CitedRAGResponse
```

FastAPI is therefore a transport layer, not the location of RAG business logic.

---

## Application Boundaries

### `ResearchAPI`

`ResearchAPI` defines the operations required by the HTTP layer:

```text
add_document()
list_documents()
delete_document()
search()
query()
close()
```

Routes depend on this protocol rather than directly constructing concrete model, retrieval, or storage implementations.

This allows route tests to use lightweight fake services without loading Qdrant, Sentence Transformers, or Ollama.

### `ResearchService`

`ResearchService` coordinates the existing application components.

Its responsibilities are limited to:

- delegating document indexing
- listing and deleting indexed documents
- invoking semantic retrieval
- invoking `RAGService`
- validating generated citations
- releasing shared runtime resources

It does not parse PDFs, construct embeddings, call Qdrant directly for retrieval, or implement citation parsing.

### `LocalPDFIndexer`

`LocalPDFIndexer` connects the existing Phase 1–4 indexing pipeline:

```text
BinaryIO upload
      ↓
temporary PDF
      ↓
load_pdf()
      ↓
ParsedDocument
      ↓
chunk_document()
      ↓
DocumentChunk[]
      ↓
embed_chunks()
      ↓
ChunkEmbedding[]
      ↓
VectorStore.replace_document()
      ↓
StoredDocument
```

Uploaded content is streamed to a temporary local file because the existing ingestion boundary accepts a filesystem path.

The Phase 1 loader remains responsible for SHA-256 document identity.

---

## Runtime Composition

`api/runtime.py` is the composition root for the local application.

It creates and connects:

```text
HuggingFaceTokenizer
SentenceTransformerEmbedder
QdrantVectorStore
SemanticRetriever
OllamaProvider
HuggingFaceChatTokenCounter
ContextBuilder
RAGService
CitationService
LocalPDFIndexer
ResearchService
```

This is the only layer that needs to know which concrete implementations are used together.

The rest of the system continues to depend on explicit protocols and typed models.

---

## Application Lifecycle

The complete research stack is created lazily.

```text
FastAPI starts
     ↓
ResearchService not yet constructed
     ↓
GET /health
     ↓
lightweight response
```

The first endpoint requiring the research service triggers runtime composition.

```text
research request
     ↓
get_api_service()
     ↓
build_local_service()
     ↓
shared ResearchService
```

Subsequent requests reuse the same local service instance.

FastAPI lifespan handling closes the shared Ollama and Qdrant resources when the application shuts down.

The current implementation is intentionally designed for a local single-process workflow. Multi-process workers sharing the same embedded Qdrant path are outside the current baseline.

---

## Endpoints

### `GET /health`

Returns lightweight application status.

Example response:

```json
{
  "status": "ok",
  "service": "local-rag-research-assistant"
}
```

This endpoint does not require model inference.

---

### `POST /documents`

Uploads and indexes one PDF document.

Example:

```bash
curl \
  -X POST \
  http://127.0.0.1:8000/documents \
  -F "file=@paper.pdf"
```

The indexing path is:

```text
UploadFile
   ↓
LocalPDFIndexer
   ↓
PDF ingestion
   ↓
chunking
   ↓
embedding
   ↓
Qdrant replacement/indexing
```

Example response shape:

```json
{
  "document_id": "<sha256>",
  "file_name": "paper.pdf",
  "chunk_count": 12
}
```

The actual chunk count depends on the document and configured tokenizer/chunking settings.

Current upload constraints include:

- PDF filename requirement
- non-empty input
- ingestion validation
- configurable upload-size limit
- at least one indexable text chunk

The application does not currently provide OCR for scanned/image-only PDFs.

---

### `GET /documents`

Lists documents represented in the vector store.

Example:

```bash
curl \
  http://127.0.0.1:8000/documents
```

Qdrant stores individual chunks rather than a separate document table.

The API therefore derives document summaries by scrolling stored chunk payloads and grouping them by `document_id`.

Example response:

```json
[
  {
    "document_id": "<sha256>",
    "file_name": "paper.pdf",
    "chunk_count": 12
  }
]
```

Only document metadata required by this endpoint is loaded during enumeration; embedding vectors are not required.

---

### `DELETE /documents/{document_id}`

Deletes all stored chunks belonging to one document.

The document ID must use the application's 64-character lowercase SHA-256 representation.

Example:

```bash
curl \
  -X DELETE \
  http://127.0.0.1:8000/documents/<document_id>
```

A successful deletion returns:

```text
204 No Content
```

A missing document is translated to an HTTP 404 response.

---

### `POST /search`

Runs semantic retrieval without invoking the language model.

Example:

```bash
curl \
  -X POST \
  http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How does self-attention work?",
    "top_k": 5
  }'
```

The path is:

```text
SearchRequest
     ↓
RetrievalConfig
     ↓
SemanticRetriever
     ↓
BGE query embedding
     ↓
Qdrant cosine search
     ↓
RetrievalResult[]
```

This endpoint is useful for inspecting retrieval independently from generation.

Supported options include:

- `top_k`
- optional `score_threshold`
- optional `document_id`

The HTTP model currently constrains `top_k` to the range 1–20.

---

### `POST /query`

Runs the complete cited-RAG pipeline.

Example:

```bash
curl \
  -X POST \
  http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Why does the Transformer avoid recurrence?",
    "top_k": 5
  }'
```

The pipeline is:

```text
QueryRequest
     ↓
RetrievalConfig
     ↓
SemanticRetriever
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

The response exposes the existing typed Phase 7–8 data rather than flattening it into an opaque answer string.

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

This allows future clients to display the answer and inspect its supporting evidence separately.

---

## Request Validation

HTTP request validation is handled with Pydantic.

Shared retrieval options include:

```text
top_k
score_threshold
document_id
```

Unexpected fields are rejected.

Whitespace-only questions and search queries are rejected before invoking the retrieval pipeline.

Document IDs supplied through request models use the application's SHA-256 format.

Validation therefore occurs before expensive embedding or model inference where practical.

---

## Error Handling

Expected application failures are translated into explicit HTTP responses.

Examples include:

```text
unsupported upload
    → client error

invalid/unusable PDF
    → client error

missing indexed document
    → 404

Ollama unavailable
    → 503

invalid model/citation response
    → 502
```

The API does not expose raw Qdrant, PyMuPDF, or HTTPX exceptions as its public interface.

Unexpected infrastructure or programming failures are not silently converted into successful responses.

---

## Testing Strategy

Phase 9 continues the project's layered testing strategy.

### Request model tests

Validate:

- defaults
- whitespace handling
- retrieval-option conversion
- invalid `top_k`
- invalid document IDs
- unknown fields

### Application-service tests

Use fake implementations of:

- document indexer
- document registry
- retriever
- RAG service
- LLM resource

This validates orchestration without loading local models.

### PDF indexer tests

Use:

- generated real PDFs
- real ingestion
- real chunking
- fake tokenizer/embedder/vector store where appropriate

The real Phase 1–4 indexing chain is additionally checked through a local smoke test.

### Route tests

Use a fake `ResearchAPI`.

This verifies:

- upload routing
- document listing
- deletion
- search requests
- query requests
- HTTP validation

without initialising the full local runtime.

### Vector-store tests

Document enumeration is tested using isolated Qdrant storage.

Existing Phase 4 retrieval/storage tests remain independent from the Phase 9 `DocumentRegistry` boundary.

### End-to-end smoke test

The final local smoke test exercises:

```text
HTTP
 ↓
FastAPI
 ↓
ResearchService
 ↓
real ingestion
 ↓
real chunking
 ↓
real embeddings
 ↓
real Qdrant
 ↓
real retrieval
 ↓
real context construction
 ↓
real Ollama/Qwen generation
 ↓
real citation validation
 ↓
HTTP response
```

Smoke-test success verifies integration behavior. It is not treated as a formal accuracy, retrieval-quality, or latency benchmark.

---

## Running the API

Ensure Ollama and the configured model are available:

```bash
ollama list
```

Start FastAPI:

```bash
poetry run uvicorn \
  research_assistant.main:app \
  --app-dir src \
  --host 127.0.0.1 \
  --port 8000
```

Open Swagger:

```text
http://127.0.0.1:8000/docs
```

The API currently listens on localhost by default in the documented development workflow.

---

## Current Limitations

The Phase 9 API should be understood as a local engineering baseline.

It does not yet provide:

- authentication or user accounts
- public deployment hardening
- multiple-user isolation
- background indexing jobs
- streaming generation
- a separate document metadata database
- OCR for image-only PDFs
- rate limiting
- distributed workers
- transactionality across multiple persistent systems
- production observability
- calibrated retrieval thresholds
- semantic claim-to-citation entailment checking

Qdrant currently acts as both vector persistence and the source used to reconstruct indexed-document summaries.

These limitations are intentional boundaries for the current phase rather than hidden production claims.

---

## Next Phase

Phase 10 introduces the user interface.

The API now provides stable boundaries for:

```text
document management
semantic search
cited question answering
source evidence inspection
```

The frontend can therefore consume the RAG system without directly depending on Python model, storage, or inference components.
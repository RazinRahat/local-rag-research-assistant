# Frontend Research Workspace

## Purpose

Phase 10 adds a browser interface to the Local RAG Research Assistant.

The frontend is intentionally a client of the Phase 9 FastAPI application rather than a second implementation of the RAG pipeline.

```text
User
 ↓
React UI
 ↓
ResearchAPIClient
 ↓
FastAPI
 ↓
ResearchService
 ↓
existing ingestion / retrieval / RAG components
```

The goal is to make the local research workflow inspectable without hiding the backend mechanics that earlier phases established.

---

## Technology

The frontend uses:

```text
React
TypeScript
Vite
Vitest
Testing Library
jsdom
```

Vite was selected instead of a larger application framework because the current application does not require server-side rendering, authenticated routing, or server components. The research UI is a local HTTP client of FastAPI, so a lightweight SPA keeps the browser layer focused on research interaction rather than framework infrastructure.

The frontend intentionally does not introduce Redux, a component library, Three.js, or another orchestration framework at this stage.

---

## Development Boundary

During local development:

```text
Browser :5173
   ↓ /api/*
Vite proxy
   ↓ remove /api prefix
FastAPI :8000
```

The proxy is configured in `frontend/vite.config.ts`.

Example:

```text
Browser request
POST /api/query

Forwarded request
POST http://127.0.0.1:8000/query
```

This keeps frontend code independent from the backend development origin and avoids requiring broad CORS configuration for the normal local workflow.

---

## Frontend Structure

The Phase 10 structure is approximately:

```text
frontend/
├── src/
│   ├── api/
│   │   ├── client.ts
│   │   ├── client.test.ts
│   │   ├── index.ts
│   │   └── models.ts
│   ├── documents/
│   │   ├── DocumentSidebar.tsx
│   │   ├── DocumentSidebar.test.tsx
│   │   ├── index.ts
│   │   ├── useDocuments.ts
│   │   └── useDocuments.test.tsx
│   ├── search/
│   │   ├── SearchResults.tsx
│   │   ├── SearchResults.test.tsx
│   │   ├── index.ts
│   │   ├── useSearch.ts
│   │   └── useSearch.test.tsx
│   ├── query/
│   │   ├── AnswerWithCitations.tsx
│   │   ├── AnswerWithCitations.test.tsx
│   │   ├── CitedAnswer.css
│   │   ├── CitedAnswer.tsx
│   │   ├── CitedAnswer.test.tsx
│   │   ├── EvidenceInspector.tsx
│   │   ├── EvidenceInspector.test.tsx
│   │   ├── index.ts
│   │   ├── useResearchQuery.ts
│   │   └── useResearchQuery.test.tsx
│   ├── App.css
│   ├── App.test.tsx
│   ├── App.tsx
│   ├── index.css
│   ├── main.tsx
│   └── setupTests.ts
├── package.json
├── tsconfig.json
└── vite.config.ts
```

---

## API Client

All browser HTTP access is centralised through `ResearchAPIClient`.

The client owns:

- endpoint paths
- JSON request serialisation
- JSON response parsing
- multipart PDF upload construction
- HTTP failure detection
- FastAPI error-detail translation
- document deletion handling

React components do not call `fetch()` directly.

```text
React component
      ↓
feature hook
      ↓
ResearchAPIClient
      ↓
HTTP
```

This prevents transport concerns from being duplicated across UI components.

### Upload Handling

PDF uploads use browser `FormData`.

The client deliberately does not manually set:

```text
Content-Type: multipart/form-data
```

The browser must generate the multipart boundary itself.

---

## Frontend API Models

The frontend mirrors the backend JSON contract with TypeScript interfaces.

Current major models include:

- `StoredDocument`
- `DocumentChunk`
- `RetrievalResult`
- `SearchRequest`
- `SearchResponse`
- `QueryRequest`
- `EvidenceBlock`
- `RAGResponse`
- `CitationReference`
- `CitationSource`
- `CitationValidationResult`
- `CitedRAGResponse`
- `GenerationResult`

The transport models retain backend `snake_case` field names. This avoids a second mapping layer while the project is still evolving rapidly and makes HTTP debugging direct.

---

## Document Management

`useDocuments()` owns server-backed document state.

```text
GET /documents
      ↓
documents[]
```

The hook supports:

- initial document loading
- refresh
- upload
- delete
- upload/deletion loading states
- API-error presentation

The browser treats document identity as the backend-provided SHA-256 ID.

Re-uploading the same document replaces the existing UI entry rather than creating a duplicate because the backend uses content-derived document identity and replacement semantics.

### Scope Selection

The research workspace supports:

```text
All documents
```

or one specific indexed document.

The selected document ID is passed to `/search` and `/query` as `document_id` when a specific document is active.

---

## Semantic Search

Search mode calls:

```text
POST /search
```

It does not invoke the language model.

The UI exposes returned retrieval data directly:

- rank
- raw similarity score
- filename
- page number
- chunk text
- token count
- chunk position

Similarity is presented as the backend score, for example:

```text
0.812
```

The frontend does not convert that value into a relevance percentage because no score calibration has been established.

The displayed result heading is tied to the query that actually produced the result set. Editing the textarea after a completed search does not relabel existing results.

---

## Cited RAG Answers

Ask mode calls:

```text
POST /query
```

The returned `CitedRAGResponse` contains both the RAG response and validated citation information.

The UI exposes:

- answer text
- original question
- retrieved evidence count
- used evidence count
- validated citation-reference count
- actual prompt token count
- context truncation state
- validated citation sources
- generation diagnostics when available
- insufficient-evidence state

The browser does not parse arbitrary model-generated filenames or page numbers. Provenance comes from trusted application data.

---

## Citation Rendering

Phase 8 returns character offsets for recognised citation references.

The frontend uses these offsets to replace only validated occurrences with interactive controls.

```text
answer text
   ↓
CitationReference
   ↓
[S1] button
```

Unknown source IDs are not turned into interactive trusted citations.

### Unicode Offsets

Python indexes Unicode strings by code point, while JavaScript strings use UTF-16 code units for normal indexing. The renderer therefore operates over:

```ts
Array.from(answer)
```

before applying `start_index` and `end_index` values.

This preserves offset alignment for non-BMP characters such as emoji appearing before a citation.

---

## Evidence Inspector

Selecting a citation reveals the corresponding trusted `CitationSource.evidence`.

The inspector exposes:

- source ID
- document filename
- page number
- retrieval rank
- raw similarity score
- exact retrieved chunk text
- token count
- document ID
- chunk ID
- global chunk index
- page-local chunk index
- token range
- character count

The evidence inspector uses the evidence already returned for the RAG request.

It does not run a new vector search when a citation is clicked.

```text
CitedRAGResponse
      ↓
CitationSource[]
      ↓
selected source
      ↓
EvidenceInspector
```

This preserves the exact evidence provenance of the generation request.

Citation identity validation should not be confused with semantic entailment. A mapped source ID proves that the identifier corresponds to evidence supplied to the model; claim-level support evaluation remains future work.

---

## Async Request Safety

Search and RAG requests can complete out of order, particularly when local generation is slow.

Both `useSearch()` and `useResearchQuery()` maintain a request sequence.

```text
request A starts
request B starts
request B completes
request A completes later
```

Only the current request is allowed to update visible results.

Changing document scope or clearing research state also invalidates an older in-flight response.

This prevents stale answers from appearing under the wrong document scope.

---

## Interaction and Accessibility

Current interaction behaviour includes:

- semantic button labels for document selection and deletion
- hidden but labelled PDF file input
- `aria-pressed` state for selectable modes/documents/citations
- alert regions for failures
- polite loading-status regions
- accessible evidence-inspector regions
- `Cmd + Enter` on macOS and `Ctrl + Enter` elsewhere for submission
- normal Enter retained for multiline textarea input
- research execution disabled when the indexed library is empty
- destructive deletion requires explicit confirmation

The current work is an accessibility-conscious baseline rather than a formal accessibility certification.

---

## Error and Empty States

The browser explicitly represents:

- document-library loading
- document upload progress
- document deletion progress
- empty document library
- document API failures
- semantic-search loading
- semantic-search failure
- no search results
- RAG generation loading
- RAG API failure
- insufficient evidence

FastAPI errors are translated by the central API client before reaching feature hooks.

---

## Testing Strategy

The frontend test suite is split by responsibility.

### API client

Validates HTTP request/response behaviour without rendering React.

### Feature hooks

Validates server-backed state, error handling, request construction, and stale-response protection.

### Components

Validate accessible rendering and interaction for documents, search results, cited answers, and evidence inspection.

### App integration

Validates interactions across feature boundaries, including:

- upload appearing in the library
- document scope propagation
- search execution
- RAG execution
- citation selection
- empty-library guards
- keyboard submission
- selected-document deletion and scope reset

The frontend test suite deliberately does not evaluate scientific correctness, embedding quality, answer faithfulness, or citation entailment. Those are evaluation concerns rather than UI concerns.

---

## Quality Gate

From `frontend/`:

```bash
npm run lint
npm run typecheck
npm run test
npm run build
```

The frontend should pass all four before the manual end-to-end smoke workflow.

---

## Running Locally

### Backend

From the repository root:

```bash
poetry run uvicorn \
  research_assistant.main:app \
  --app-dir src \
  --host 127.0.0.1 \
  --port 8000
```

### Frontend

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5173
```

Ollama must also be available for Ask mode. Search mode only requires the embedding/retrieval stack.

---

## Phase 10 Smoke Workflow

The completed manual Phase 10 smoke workflow exercises:

```text
load application
    ↓
upload real PDF
    ↓
verify indexed document
    ↓
repeat upload / replacement
    ↓
document-scoped search
    ↓
corpus-wide search
    ↓
keyboard submission
    ↓
cited RAG answer
    ↓
inline citation inspection
    ↓
source-card inspection
    ↓
unsupported question path
    ↓
stale-request protection
    ↓
delete document
```

This smoke workflow establishes integration behaviour. It is not a retrieval benchmark or answer-quality benchmark.

---

## Current Limitations

The frontend does not yet provide:

- authentication
- multi-user sessions
- streaming generation
- browser-persisted workspace state
- public deployment hardening
- background upload/indexing jobs
- hybrid retrieval controls
- reranker controls
- formal evaluation dashboards
- claim-level citation-support scoring
- observability dashboards

These are later project phases rather than assumptions hidden inside the current interface.

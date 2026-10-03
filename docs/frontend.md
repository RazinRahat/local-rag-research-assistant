# Frontend Research Workspace

## Purpose

The frontend provides a browser workspace for the Local RAG Research Assistant.

It remains an API client rather than a second implementation of the RAG pipeline.

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
RAG Components
```

---

# Technology

```text
React
TypeScript
Vite
Vitest
Testing Library
jsdom
```

---

# Development Boundary

```text
Browser :5173
   ↓
/api/*
   ↓
Vite proxy
   ↓
FastAPI :8000
```

---

# Main Workspace

The interface supports document management, Ask mode, Search mode, document scope, retrieval strategy, and evidence inspection.

---

# Document Management

The sidebar supports:

- indexed document loading
- PDF upload
- document selection
- whole-corpus selection
- document deletion
- loading/upload/delete states
- errors

Document selection controls retrieval scope.

---

# Research Actions

The workspace exposes:

```text
Ask
Search
```

Search calls `POST /search`.

Ask calls `POST /query`.

---

# Retrieval Strategies

Phase 11 adds:

```text
Dense
Lexical
Hybrid
```

Dense is selected by default.

The chosen strategy applies to both Ask and Search.

---

# Request Behaviour

Dense can rely on the backend default:

```json
{
  "query": "attention mechanism",
  "top_k": 5
}
```

Lexical:

```json
{
  "query": "Adam optimizer",
  "top_k": 5,
  "mode": "lexical"
}
```

Hybrid:

```json
{
  "query": "attention mechanism",
  "top_k": 5,
  "mode": "hybrid"
}
```

Document scope optionally adds:

```json
{
  "document_id": "<sha256>"
}
```

---

# Score Presentation

```text
Dense
→ cosine similarity

Lexical
→ BM25

Hybrid
→ Reciprocal Rank Fusion
```

The frontend therefore avoids describing every score as semantic similarity.

Evidence inspection uses the neutral label:

```text
Retrieval score
```

Scores are not converted into percentages.

---

# Search Results

Search cards expose:

- rank
- retrieval score
- filename
- page
- chunk text
- token count
- chunk position

The displayed query is the one that actually produced the returned result set.

---

# Cited RAG Answers

The frontend displays generated answer, question, evidence counts, token diagnostics, truncation state, citation references, citation sources, generation diagnostics, and insufficient-evidence state.

---

# Citation Rendering

Citation references use validated character offsets returned by the backend.

```text
answer
 ↓
CitationReference
 ↓
validated [S1]
 ↓
interactive source control
```

Unknown source identifiers are not promoted into trusted citations.

---

# Evidence Inspector

The evidence inspector displays:

- source ID
- filename
- page
- retrieval rank
- retrieval score
- chunk text
- token count
- document ID
- chunk ID
- chunk indices
- token range
- character count

Clicking a citation does not trigger a new search.

It displays the evidence associated with the original answer.

---

# Async Request Safety

Search and RAG hooks maintain request sequences.

Changing document scope or retrieval strategy clears current research state and invalidates older requests.

This prevents stale dense results from appearing after the user switches to hybrid, and similar race conditions.

---

# Keyboard Interaction

`Cmd/Ctrl + Enter` submits the current research action.

Normal Enter remains available for multiline input.

---

# Accessibility

The workspace uses semantic buttons, labels, grouped controls, `aria-pressed`, loading announcements, error states, keyboard interaction, and provenance context.

---

# Testing

Frontend tests cover:

- API client behavior
- document state
- search hook
- query hook
- retrieval-mode forwarding
- dense default
- stale responses
- search-result rendering
- citations
- evidence inspection
- App integration
- upload/search
- keyboard behavior
- scope changes

---

# Running the Frontend

```bash
cd frontend
npm install
```

```bash
npm run lint
npm run typecheck
npm run test
npm run build
```

```bash
npm run dev
```

Open:

```text
http://localhost:5173
```

---

# Current Boundary

The browser does not directly access Qdrant, BGE, BM25 internals, Ollama, tokenizer internals, or citation-parser internals.

---

# Future Work

Potential improvements include richer retrieval comparison views, evaluation dashboards, streaming generation, experiment history, research tables, paper comparison workflows, and stronger responsive polish.

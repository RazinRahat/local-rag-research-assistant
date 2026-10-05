# Local RAG Research Assistant

A local-first research assistant for querying, analysing, and citing research documents using Retrieval-Augmented Generation (RAG).

The project is built component by component so ingestion, chunking, embeddings, persistence, retrieval, context construction, local generation, citations, hybrid search, reranking, and evaluation remain explicit and inspectable instead of being hidden behind a high-level RAG framework.

## Current Status

**Phase 13 complete — the retrieval stack now has relevance-labelled development and holdout evaluation, evidence-driven hybrid tuning, and a frozen cross-encoder reranking configuration.**

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
     ├── Weighted Hybrid Retrieval
     │        ↓
     │ Weighted Reciprocal Rank Fusion
     └── Hybrid + Reranking
              ↓
       Hybrid candidate pool
              ↓
       Cross-Encoder rerank subset
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

The system can ingest and index research PDFs, preserve page and chunk provenance, retrieve evidence using multiple strategies, construct a token-bounded prompt, generate a local evidence-grounded answer, validate recognised citation identities, and expose the exact evidence behind each validated source.

Citation identity validation does **not** establish claim-level semantic entailment. Phase 13 evaluates retrieval quality; answer faithfulness, abstention quality, and claim-level citation support remain separate future evaluation tasks.

---

## Current Features

### Document Pipeline

- PDF validation using PyMuPDF
- page-level text extraction
- PDF metadata extraction
- SHA-256 document identity
- conservative text normalisation
- encrypted and unusable PDF detection
- page-level provenance preservation
- typed document and chunk models
- configurable token-aware chunking
- model-aligned Hugging Face tokenisation
- deterministic chunk identities
- configurable token overlap
- local document and query embeddings
- normalised BGE embeddings
- persistent Qdrant storage
- document replacement and deletion
- stored source text and provenance metadata

Current chunking baseline:

```text
chunk size:      384 tokens
chunk overlap:    64 tokens
```

Current embedding baseline:

```text
BAAI/bge-small-en-v1.5
384 dimensions
```

### Retrieval

The system exposes four retrieval strategies:

```text
dense
lexical
hybrid
hybrid_reranked
```

Dense remains the backward-compatible default.

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

Hybrid retrieval combines dense and lexical rankings with **weighted Reciprocal Rank Fusion**:

```text
score(d) = Σ w_i / (k + rank_i(d))
```

Frozen Phase 13 configuration:

```text
RRF rank constant:      5
Dense weight:           1.00
Lexical weight:         1.25
candidate multiplier:   2
```

Raw cosine and BM25 scores are not averaged because their numerical scales are not comparable.

#### Cross-Encoder Reranking

The frozen reranking path separates first-stage candidate depth from the number of passages scored by the cross-encoder:

```text
Dense top 40 + BM25 top 40
             ↓
       weighted RRF
             ↓
       Hybrid top 20
             ↓
       keep top 10
             ↓
cross-encoder/ms-marco-MiniLM-L6-v2
             ↓
        Final top 5
```

Current reranking configuration:

```text
candidate_pool_size = 20
rerank_pool_size    = 10
final top_k         = 5
```

The cross-encoder score is a relevance score, not a calibrated probability, and must not be compared numerically with cosine, BM25, or RRF scores.

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

### Retrieval and RAG

- dense semantic retrieval using BGE + Qdrant
- BM25 lexical retrieval
- weighted Reciprocal Rank Fusion
- selectable dense, lexical, hybrid, and cross-encoder-reranked retrieval
- separate first-stage candidate depth and rerank depth
- configurable top-k retrieval
- document-scoped retrieval
- dense-only optional cosine score thresholds
- typed `RetrievalResult`
- full chunk provenance reconstruction
- local Qwen generation through Ollama
- explicit context budgeting
- chat-template-aware token estimation
- whole-chunk evidence selection
- duplicate and high-overlap evidence filtering
- evidence-order preservation
- grounding instructions
- source-labelled evidence delimiters
- explicit insufficient-evidence fallback
- typed `RAGResponse`
- prompt-token diagnostics
- local generation diagnostics

### Citation Integrity

- strict `[S1]`, `[S2]`, ... citation protocol
- citation parsing with character offsets
- mapping from prompt-local source IDs to trusted evidence
- rejection of unknown recognised source IDs
- rejection of missing citations in evidence-backed answers
- document/page/chunk provenance recovered from stored evidence
- generated filenames and page numbers are not trusted
- typed `CitedRAGResponse`
- interactive source inspection in the frontend

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

### Frontend

The React research workspace provides indexed-document management, corpus-wide and document-scoped search, Ask/Search modes, Dense/Lexical/Hybrid/Reranked selection, ranked evidence cards, cited answers, clickable validated citations, provenance inspection, diagnostics, error/empty states, keyboard submission, and stale-response protection.

---

## Phase 11 — Dense + Lexical Complementarity

The first controlled retrieval comparison used six whole-corpus queries and `top_k=5`.

Average top-five Jaccard overlap:

```text
Dense ↔ Lexical    ≈ 0.214
Dense ↔ Hybrid     ≈ 0.478
Lexical ↔ Hybrid   ≈ 0.507
```

The useful finding was not that one retriever won. Dense and lexical retrieval produced materially different evidence, and RRF incorporated both signals.

For `Adam optimizer`, dense retrieval ranked a bibliography passage first while BM25 promoted the actual Transformer optimizer-method passage. This established complementarity, not formal accuracy superiority.

---

## Phase 12 — Cross-Encoder Behaviour

A six-query comparison added `cross-encoder/ms-marco-MiniLM-L6-v2` after Hybrid retrieval.

```text
Mean top-5 Jaccard overlap:   ≈ 0.409
Top-1 agreement:              1 / 6
Hybrid median latency:        ≈ 35.1 ms
Reranked median latency:      ≈ 108.9 ms
Added end-to-end latency:     ≈ 73.8 ms
```

The reranker made useful semantic promotions but also clear regressions. Phase 12 therefore established integration and behaviour, not higher retrieval accuracy.

---

## Phase 13 — Formal Retrieval Evaluation

Phase 13 converted the earlier qualitative comparisons into a relevance-labelled benchmark.

Benchmark design:

```text
papers:                 3
queries:                18 document-scoped
  development:          12
  holdout:               6
relevance judgments:   527
relevance scale:         0–3
binary relevance:       grade >= 2
evaluation top_k:        5
```

The relevance review was **model-assisted and blind to system identity/rank**. It should not be described as an independently human-labelled gold standard. Recall is pooled recall because qrels were created from pooled retrieved candidates rather than exhaustive corpus annotation.

### Initial expanded development benchmark

| Mode | P@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| Dense | 0.2500 | 0.4514 | 0.5556 | 0.4551 |
| Lexical | 0.3667 | 0.6319 | 0.7222 | 0.6264 |
| Hybrid | 0.3500 | 0.5903 | 0.5306 | 0.4951 |
| Hybrid + Reranking | 0.3167 | 0.5625 | 0.5958 | 0.4801 |

The result was important because the more complex pipelines were **not automatically better**. BM25 was the strongest aggregate development baseline.

### Evidence-driven Hybrid tuning

An offline development-only sweep evaluated 84 weighted-RRF configurations. The absolute best aggregate configuration was rejected because it introduced several query-level regressions. The selected configuration instead prioritised a zero-regression development profile:

```text
k = 5
dense weight = 1.00
lexical weight = 1.25
candidate multiplier = 2
```

It improved Hybrid development metrics to:

| Metric | Before | After | Relative change |
|---|---:|---:|---:|
| P@5 | 0.3500 | 0.3833 | +9.5% |
| Recall@5 | 0.5903 | 0.6736 | +14.1% |
| MRR | 0.5306 | 0.5750 | +8.4% |
| nDCG@5 | 0.4951 | 0.5619 | +13.5% |

The production implementation reproduced the offline prediction exactly.

### Reranking lesson: one parameter was controlling two depths

The first reranker experiment suggested that scoring only the strongest 10 Hybrid candidates could improve quality. A direct implementation changed `candidate_pool_size` from 20 to 10. That improved aggregate results, but it also silently reduced the nested Dense/BM25 branch depth from 40 to 20.

That meant the code was no longer testing the same architecture as the offline experiment.

The fix was to separate:

```text
candidate_pool_size = 20   # Hybrid candidate generation depth
rerank_pool_size    = 10   # Cross-encoder scoring depth
```

This preserved deep first-stage retrieval while limiting the cross-encoder to the strongest Hybrid subset.

### Final development result

| Mode | P@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| Dense | 0.2500 | 0.4514 | 0.5556 | 0.4551 |
| Lexical | 0.3667 | 0.6319 | **0.7222** | **0.6264** |
| Hybrid | **0.3833** | 0.6736 | 0.5750 | 0.5619 |
| Hybrid + Reranking | **0.3833** | **0.7153** | 0.6903 | 0.5987 |

Relative to the initial expanded development reranker, the final reranked pipeline improved:

```text
P@5      +21.1%
Recall   +27.2%
MRR      +15.9%
nDCG     +24.7%
```

Lexical retrieval still had the highest development MRR and nDCG. That trade-off is retained rather than tuned away.

### Frozen holdout result

After the development configuration was frozen, the six holdout queries were evaluated once without further parameter tuning.

| Mode | P@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| Dense | 0.4000 | 0.7667 | 0.4778 | 0.4806 |
| Lexical | 0.3667 | 0.6917 | 0.6806 | 0.5083 |
| Hybrid | 0.3667 | 0.6917 | 0.6667 | 0.5397 |
| **Hybrid + Reranking** | **0.4667** | **0.8500** | **0.9167** | **0.7284** |

On this small frozen holdout, Hybrid + Reranking was best on all four aggregate metrics. Across the six holdout queries, reranking produced four improvements, one unchanged result, and one degradation relative to Hybrid.

The degradation matters: on the GPT-2 Natural Questions query, Hybrid retrieved both relevant passages, but the cross-encoder dropped the passage containing the complete 4.1% exact-match result from the final top five. This remains a documented failure case rather than a reason to tune on the holdout.

See [Phase 13 Evaluation](docs/phase13-evaluation.md) and [Experiments](docs/experiments.md) for the full sequence, including failed experiments and claim boundaries.

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
                  ┌─────────────┴─────────────┐
                  │                           │
                  ▼                           ▼
            Document Pipeline           RetrievalRouter
                  │                           │
                  ▼          ┌─────────┬─────────┬──────────┬──────────────────┐
                Qdrant        ▼         ▼         ▼          ▼
                            Dense     Lexical    Hybrid   Hybrid Reranked
                              │          │         │          │
                              │          │    Dense + BM25    │
                              │          │         │          │
                              │          │         ▼          │
                              │          │   Weighted RRF ─────┘
                              │          │                    │
                              │          │                    ▼
                              │          │            Hybrid top 20
                              │          │                    ↓
                              │          │              keep top 10
                              │          │                    ↓
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
- `cross-encoder/ms-marco-MiniLM-L6-v2`
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
├── reranking/
├── evaluation/
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
├── reranking/
├── evaluation/
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
├── compare_reranking_modes.py
├── build_relevance_pool.py
├── evaluate_retrieval.py
└── build_retrieval_ledger.py

experiments/
├── retrieval_queries.json
├── retrieval_queries_phase13_expansion.json
├── relevance_judgments_phase13_expansion.json
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
├── phase13-evaluation.md
├── project-story.md
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

---

## Local Model Runtime

Ensure the configured Ollama model is available:

```bash
ollama pull qwen3.5:9b
ollama list
```

Search/evaluation modes do not require LLM generation. Ask mode requires the configured local model.

---

## Local Documents and Experiment Data

Local documents, vector storage, environment files, generated model files, and raw experiment result dumps should not be committed.

Qdrant stores source text and provenance metadata, so the local vector store should be treated as potentially private research data.

Raw Phase 13 traces and evaluation JSON files remain local under `experiments/results/`; derived conclusions and reproducible configuration belong in documentation.

---

## Documentation

- [System Architecture](docs/architecture.md)
- [Retrieval](docs/retrieval.md)
- [Phase 13 Evaluation](docs/phase13-evaluation.md)
- [Project Story / Post Source Bank](docs/project-story.md)
- [Experiments](docs/experiments.md)
- [Development Roadmap](docs/roadmap.md)
- [RAG API](docs/api.md)
- [Frontend Research Workspace](docs/frontend.md)
- [Document Ingestion](docs/ingestion.md)
- [Document Chunking](docs/chunking.md)
- [Embeddings](docs/embeddings.md)
- [Vector Storage](docs/vector-store.md)
- [Hybrid Retrieval](docs/hybrid-retrieval.md)
- [Cross-Encoder Reranking](docs/reranking.md)
- [Local Generation](docs/generation.md)
- [End-to-End RAG Pipeline](docs/rag-pipeline.md)
- [Citation Integrity](docs/citations.md)

---

## Roadmap

**Completed through Phase 13:** foundation, ingestion, token-aware chunking, local embeddings, persistent Qdrant storage, dense retrieval, local LLM integration, end-to-end RAG, citation integrity, FastAPI, React workspace, BM25 lexical retrieval, weighted hybrid retrieval, cross-encoder reranking, graded qrels, development tuning, and a frozen holdout retrieval evaluation.

**Next:** Phase 14 — research-specific features such as structured summaries, methodology/findings extraction, evidence tables, cross-paper comparison, and multi-document synthesis.

Later phases cover observability, containerisation, demo/deployment, and broader benchmarking/technical reporting.

---

## Design Principles

- **Local first:** keep documents, vectors, retrieval, and model inference local where practical.
- **Preserve provenance:** trace evidence to its document, page, and chunk.
- **Understand before abstracting:** expose RAG mechanics before adding high-level frameworks.
- **Separate responsibilities:** ingestion, embeddings, storage, retrieval, reranking, evaluation, generation, citations, HTTP transport, and UI remain explicit boundaries.
- **Bound context:** treat model context as a finite resource.
- **Treat retrieved text as data:** document text does not become application instructions.
- **Validate model output:** only application-known source IDs can become trusted citations.
- **Keep identity separate from entailment:** citation identity is not claim-level proof.
- **Measure instead of guess:** a working feature is not automatically a better feature.
- **Keep failed experiments:** regressions are evidence about the system, not noise to hide.
- **Protect the holdout:** freeze configuration before final holdout inspection and do not tune on holdout failures.

---

🚧 Under active development. Phase 13 retrieval evaluation is complete; Phase 14 research features are next.

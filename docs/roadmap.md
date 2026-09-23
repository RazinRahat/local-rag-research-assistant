Development Roadmap

The project is being developed incrementally so each RAG component can be tested before the next dependency is introduced.

Phase 0 — Project Foundation ✅
Python environment
Poetry dependency management
FastAPI application
health endpoint
Ruff
Mypy
Pytest
repository structure
Git/GitHub workflow
Phase 1 — Document Ingestion ✅
PDF validation
PyMuPDF extraction
page-level document representation
PDF metadata
SHA-256 document identity
text normalisation
encrypted PDF detection
provenance preservation
Phase 2 — Document Chunking ✅
tokenizer abstraction
overlapping token windows
chunk metadata
deterministic chunk identity
page-local provenance
configurable chunk size and overlap
Phase 3 — Embeddings ✅
Hugging Face model tokenisation
Sentence Transformers integration
BGE-small-en-v1.5 baseline
query and document embeddings
normalised vectors
batched inference
model-aware chunking
Phase 4 — Vector Storage

Planned:

persistent vector database
collections
chunk payloads
vector insertion
document replacement
duplicate handling
metadata filtering
Phase 5 — Semantic Retrieval

Planned:

query embedding
nearest-neighbour search
top-k retrieval
similarity scores
retrieval result models
retrieval tests
Phase 6 — Local LLM

Planned:

local model runtime
provider abstraction
prompt construction
generation configuration
Phase 7 — End-to-End RAG

Planned:

question
 ↓
retrieve
 ↓
build context
 ↓
generate
 ↓
answer
Phase 8 — Citations

Planned:

chunk-to-source citation mapping
document/page references
inspectable supporting evidence
Phase 9 — RAG API

Planned endpoints include:

POST /documents
GET  /documents
POST /query
POST /search
DELETE /documents/{id}
GET  /health
Phase 10 — User Interface

Initial lightweight interface followed by a more polished frontend if useful.

Phase 11 — Hybrid Retrieval
BM25
dense retrieval
reciprocal-rank fusion
metadata filtering
Phase 12 — Reranking
cross-encoder reranking
candidate expansion
top-k reduction
Phase 13 — Evaluation

Retrieval metrics:

Recall@K
Precision@K
MRR
nDCG

Answer-level evaluation:

faithfulness
answer relevance
context relevance
citation correctness
Phase 14 — Research Features

Potential features:

paper summaries
methodology extraction
finding extraction
limitation extraction
cross-paper comparisons
evidence tables
research-gap analysis
Phase 15 — Observability

Record information such as:

query
retrieved chunks
similarity scores
reranking scores
token usage
latency
model configuration
final response
Phase 16 — Containerisation

Potential services:

API
Vector Database
Local Model Runtime
Frontend
Phase 17 — Demo / Deployment

Provide a reproducible way to run and demonstrate the project.

Phase 18 — Benchmarking and Technical Report

Compare system configurations rather than relying on intuition.

Potential experiments:

chunk size
chunk overlap
chunking method
embedding model
retrieval algorithm
top-k
reranking
context size
latency
memory use

The final project should demonstrate not only that RAG works, but why particular design choices were selected.
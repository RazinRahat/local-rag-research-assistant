# Experiments

This document records controlled experiments and engineering observations used to evaluate design choices in the RAG pipeline.

The objective is to replace intuition-driven configuration with reproducible measurements wherever practical.

Results should be recorded with enough information to reproduce the experiment, including:

- dataset or document corpus
- query set
- relevant configuration
- model versions
- retrieval settings
- generation settings
- metrics
- observations
- known limitations

Smoke tests and qualitative observations should be clearly distinguished from formal evaluation.

---

## Current Baselines

The project currently has working implementation baselines for:

- document ingestion
- token-aware chunking
- local embedding generation
- persistent vector storage
- dense semantic retrieval
- local language generation
- bounded context construction
- end-to-end local RAG orchestration
- citation parsing and trusted evidence mapping
- typed FastAPI application and document-management boundary

The current end-to-end baseline is approximately:

```text
Research PDF
      ↓
PyMuPDF ingestion
      ↓
model-aware chunking
      ↓
384-token chunks
64-token overlap
      ↓
BAAI/bge-small-en-v1.5
      ↓
384-dimensional normalised embeddings
      ↓
Qdrant cosine retrieval
      ↓
top-k evidence
      ↓
bounded context construction
      ↓
Qwen3.5 through Ollama
      ↓
RAG response
      ↓
strict [S#] citation parsing and evidence validation
      ↓
CitedRAGResponse
      ↓
FastAPI response
```

These settings are working baselines.

They should not be interpreted as experimentally established optima.

---

## Experiment Recording Template

Future experiments should use a consistent structure.

### Experiment

**Question**

What engineering decision is being tested?

**Configuration**

```text
Dataset:
Queries:
Chunk size:
Chunk overlap:
Embedding model:
Retrieval method:
Top-k:
Score threshold:
Context size:
Safety margin:
Generation model:
Generation configuration:
```

**Metrics**

List the measurements relevant to the experiment.

**Results**

Record measured values only.

**Observations**

Document qualitative behaviour, failure modes, and unexpected outcomes.

**Conclusion**

State only what the measurements support.

**Follow-Up**

Record the next comparison or unresolved question.

---

## Phase 1–4 Baseline Observations

The first four phases established the indexing pipeline:

```text
PDF
 ↓
ParsedDocument
 ↓
DocumentChunk[]
 ↓
ChunkEmbedding[]
 ↓
Persistent Vector Store
```

The main engineering objective during these phases was correctness and provenance rather than benchmarking.

Important baseline properties include:

- page-level provenance is preserved
- chunks remain page-local
- chunk identifiers are deterministic
- the embedding tokenizer is model-aware
- embeddings remain attached to source chunks
- Qdrant payloads preserve original chunk text and metadata
- document replacement prevents repeated indexing from blindly accumulating stale chunks

Formal comparison of alternative parsing, chunking, and embedding approaches remains future work.

---

## Phase 5 — Dense Retrieval Baseline

Phase 5 established dense semantic retrieval.

The current baseline uses:

```text
Query
 ↓
BGE query embedding
 ↓
Qdrant cosine search
 ↓
Top-K RetrievalResult[]
```

Initial validation focuses on whether semantically relevant passages are returned for natural-language questions.

Useful qualitative query categories include:

- direct questions using terminology from the source
- paraphrased questions
- conceptually related questions with limited lexical overlap
- unrelated or out-of-domain questions

The dense baseline should eventually be evaluated against labelled query-to-evidence relationships.

Planned retrieval metrics include:

- Recall@K
- Precision@K
- Mean Reciprocal Rank
- nDCG

No universal similarity threshold has been selected.

A score threshold should only become a default after testing its precision-recall trade-off on a relevant evaluation set.

---

## Phase 6 — Local Generation Baseline

Phase 6 established local generation independently from retrieval.

The current generation baseline uses:

```text
Ollama
+
Qwen3.5
```

with a conservative research-assistant configuration.

Relevant generation measurements include:

- prompt tokens
- completion tokens
- model load duration
- prompt-evaluation duration
- generation duration
- total duration
- approximate generated tokens per second

First-request latency and subsequent-request latency should be recorded separately because model loading can materially affect the first request.

Generation settings such as:

- temperature
- top-p
- context size
- output-token allowance

should eventually be compared under controlled conditions rather than changed based only on subjective preference.

---

## Phase 7 — RAG Integration Baseline

Phase 7 connects retrieval and generation.

The current pipeline is:

```text
Question
   ↓
Dense Retrieval
   ↓
RetrievalResult[]
   ↓
ContextBuilder
   ↓
EvidenceBlock[]
   ↓
Grounded ChatMessage[]
   ↓
Local LLM
   ↓
RAGResponse
```

The Phase 7 baseline adds several observable variables that should be retained during experiments.

These include:

```text
retrieved_count
used_evidence_count
skipped_redundant
skipped_for_budget
estimated_prompt_tokens
actual_prompt_tokens
context_truncated
insufficient_evidence
```

This allows failures to be inspected at more than one point in the pipeline.

---

### Context Budget Baseline

The current local generation configuration uses an explicit runtime context budget.

Conceptually:

```text
Runtime Context
-
Reserved Output Tokens
-
Safety Margin
=
Prompt Budget
```

The prompt budget includes:

- system instructions
- user question
- source metadata
- retrieved evidence
- evidence delimiters
- chat-template formatting

Retrieved chunks are considered in retrieval order and included only when the resulting prompt remains within the available budget.

The baseline keeps whole chunks rather than truncating them arbitrarily.

Future comparisons may investigate:

- smaller and larger context windows
- alternative output reservations
- alternative safety margins
- sentence-aware trimming
- context compression
- neighbouring-context expansion

---

### Chat Token Estimation

The RAG context builder estimates prompt length before invoking the local model.

The current estimation path is:

```text
ChatMessage[]
      ↓
Qwen chat template
      ↓
formatted chat text
      ↓
Qwen tokenizer
      ↓
estimated_prompt_tokens
```

During implementation, an initial assumption that \`apply_chat_template(tokenize=True)\` would always return a plain \`list[int]\` proved too strict for the installed Transformers/tokenizer behaviour.

The implementation was changed to:

```text
apply_chat_template(
    tokenize=False
)
      ↓
formatted string
      ↓
tokenizer.encode(
    add_special_tokens=False,
    truncation=False
)
      ↓
token count
```

This avoids relying on a version-specific return shape while retaining chat-template-aware counting.

This implementation change should be treated as an engineering boundary correction rather than an experiment result.

---

### Estimated vs Actual Prompt Tokens

The RAG response preserves:

```text
estimated_prompt_tokens
actual_prompt_tokens
```

The estimator is used before inference.

Ollama reports the runtime prompt-token count after generation.

A useful derived measurement is:

```text
token_estimation_error
=
actual_prompt_tokens
-
estimated_prompt_tokens
```

Future experiments should record this across multiple query and context sizes.

Relevant questions include:

- Is the error consistently small?
- Is the estimator systematically high or low?
- Does error increase with context length?
- Is the current safety margin unnecessarily large?
- Is the safety margin sufficient for all tested prompts?

No numerical conclusions should be recorded until measurements have been collected.

---

### Evidence Selection

The current context builder considers evidence in dense-retrieval order.

Potential measurements include:

- number of retrieved chunks
- number retained
- number removed as redundant
- number skipped for budget
- source-document diversity
- page diversity

Future experiments should investigate whether simple retrieval-order inclusion is sufficient or whether context quality improves through:

- reranking
- diversification
- neighbouring-context expansion
- parent-child retrieval
- evidence compression

---

### Overlap Deduplication

Chunks currently use overlap during indexing.

This can cause retrieved candidates to contain repeated text.

The context builder removes identical chunks and can filter highly overlapping chunks from the same document page.

The overlap-deduplication threshold should eventually be treated as an experimental variable.

Potential comparison:

```text
No deduplication
vs
Conservative deduplication
vs
Aggressive deduplication
```

Possible measurements include:

- prompt token savings
- unique evidence coverage
- answer faithfulness
- answer completeness

No threshold is currently claimed to be optimal.

---

### Grounding Smoke Tests

RAG integration should be inspected using several categories of questions.

#### Directly Answerable Query

A question whose answer appears clearly in the indexed corpus.

Inspect:

- retrieved evidence
- context-selected evidence
- generated answer
- unsupported additions

#### Paraphrased Query

A semantically equivalent question using different wording from the source.

This helps inspect whether dense retrieval finds relevant evidence beyond exact lexical matching.

#### Out-of-Domain Query

A question unrelated to the indexed corpus.

Inspect:

- similarity scores
- whether retrieval still returns weak candidates
- whether a configured threshold removes them
- whether the model claims insufficient evidence
- whether the model answers from prior knowledge despite grounding instructions

These tests are qualitative integration checks.

They are not substitutes for formal faithfulness evaluation.

---

### Insufficient-Evidence Behaviour

If retrieval returns no usable evidence, the RAG service bypasses the language model and returns an explicit insufficient-evidence response.

This behaviour can be tested deterministically.

The harder case is:

```text
retrieval returns weak but non-empty evidence
```

Dense nearest-neighbour search can produce mathematically closest chunks even when the corpus does not contain the answer.

Future evaluation should therefore examine:

- retrieval score distributions
- threshold selection
- false-positive evidence
- no-answer questions
- model behaviour when evidence is weak

This will be important before claiming reliable hallucination reduction.

---

## Phase 8 — Citation Integrity Baseline

Phase 8 adds citation parsing and source-identity validation as an independent operation after a completed `RAGResponse`.

The model is instructed to write source references as `[S1]`, `[S2]`, …, using only the temporary IDs supplied in the selected context. The `CitationService` then extracts recognised tokens and maps them back to the exact `EvidenceBlock` values used for generation.

```text
Grounded answer with [S#] references
             ↓
       parse_citations()
             ↓
       CitationReference[]
             ↓
 validate against selected EvidenceBlock[]
             ↓
       CitationSource[]
             ↓
        CitedRAGResponse
```

The Phase 8 unit-test baseline covers strict reference extraction, character offsets, multiple and repeated references, irrelevant brackets, source mapping, unknown recognised IDs, missing citations in evidence-backed answers, and citation-free explicit insufficient-evidence responses. A manual local smoke test checks source metadata on an actual cited RAG answer.

### Citation-Integrity Invariants

- A recognised `[S#]` must resolve to evidence supplied for that individual generation request.
- Repeated inline references are preserved; the unique cited-source list is deduplicated.
- Stored provenance—not text generated by the LLM—supplies document ID, filename, page number, chunk ID, and evidence text.
- Unknown recognised IDs such as `[S99]` are rejected when not in the supplied evidence.
- Citation-free evidence-backed answers fail validation; explicit insufficient-evidence responses are exempt.

**Limitations:** this parser recognises only strict syntax. Unsupported formats such as `[S1, S2]` are ignored, not comprehensively diagnosed; a response with at least one recognised citation could also contain unrecognised malformed references. Phase 8 does not yet prove that a real cited chunk supports the nearby factual assertion or that every sentence has a citation.

### Future Citation Measurements

A representative, labelled set should distinguish syntax/provenance integrity from semantic claim support. Candidate measures include:

- proportion of generated outputs containing recognised citations
- count/rate of recognised unknown source IDs
- proportion of responses violating the expected citation protocol
- references per answer and unique sources per answer
- duplicate references and source-distribution patterns
- claim-level citation coverage (future annotation)
- semantic support of claims by cited passages (future annotation)
- citation validity versus citation correctness

No observed pass percentages or benchmark values are recorded here merely because the implementation and local smoke tests succeeded.

---

## Phase 9 — API Integration Baseline

Phase 9 exposes the working local RAG pipeline through FastAPI while preserving the existing component boundaries.

The implemented API baseline is:

```text
HTTP request
     ↓
FastAPI
     ↓
ResearchService
     ↓
indexing / retrieval / RAG pipeline
     ↓
typed response
     ↓
HTTP JSON
```

The implemented routes are:

```text
GET    /health
POST   /documents
GET    /documents
DELETE /documents/{document_id}
POST   /search
POST   /query
```

### API Integration Checks

Phase 9 uses several levels of validation.

**Request-model tests** verify:

- default retrieval settings
- whitespace handling
- invalid top-k values
- invalid document IDs
- rejection of unknown fields
- conversion to domain `RetrievalConfig`

**Application-service tests** verify delegation for:

- document indexing
- document listing
- document deletion
- semantic retrieval
- RAG generation
- citation validation
- runtime resource cleanup

**Indexer tests** exercise:

```text
generated PDF
    ↓
real ingestion
    ↓
real chunking
    ↓
embedding boundary
    ↓
vector-store boundary
```

without requiring the production embedding model for every unit test.

**Vector-store integration tests** verify that Qdrant document enumeration reconstructs document summaries from stored chunk payloads and that document replacement updates the derived chunk count correctly.

**Route tests** verify HTTP behaviour using a fake `ResearchAPI`, keeping normal API tests independent from BGE, persistent Qdrant data, and Ollama.

**Real local smoke tests** additionally exercise the actual indexing pipeline and the complete HTTP path through the local models and persistent infrastructure.

### Phase 9 Invariants

- `/search` performs retrieval without invoking the language model.
- `/query` returns the existing `CitedRAGResponse` rather than bypassing citation validation.
- FastAPI routes do not implement ingestion, retrieval, generation, or citation logic directly.
- `GET /health` remains lightweight.
- Shared local infrastructure is composed in one runtime boundary.
- Qdrant and Ollama resources are closed during application shutdown.
- Document enumeration does not require loading stored embedding vectors.
- Phase 1–8 component tests remain valid after introducing the HTTP layer.

### Current API Limitations

The successful smoke tests establish integration behaviour, not production performance.

No claims are currently made about:

- request throughput
- concurrent-user capacity
- upload latency distributions
- indexing latency distributions
- retrieval latency distributions
- generation latency distributions at the HTTP level
- maximum practical corpus size
- public-service security
- multi-process behaviour
- multi-user isolation

The current API uses a local single-process architecture and embedded persistent Qdrant storage.

These are engineering baseline choices, not experimentally established deployment optima.

### Future API Measurements

Once observability is introduced, useful measurements may include:

- request latency by endpoint
- PDF indexing time
- embedding time
- retrieval time
- context-construction time
- model load and generation time
- end-to-end query latency
- request failure category
- indexed document/chunk counts
- memory use
- repeated-request warm vs cold latency

These measurements should preserve the existing principle of separating retrieval, context, generation, citation, and infrastructure failures rather than collapsing them into a single end-to-end score.

---

## Phase 10 — Frontend Integration Baseline

Phase 10 adds a browser research workspace over the Phase 9 API. The purpose of the Phase 10 checks is integration validation rather than retrieval-quality or model-quality benchmarking.

The implemented browser baseline supports:

```text
PDF upload / document lifecycle
        ↓
corpus scope selection
        ↓
semantic search
        ↓
cited RAG question answering
        ↓
validated citation inspection
        ↓
trusted evidence provenance
```

### Frontend Automated Checks

The frontend suite validates the following boundaries independently:

**HTTP client**

- health/document/search/query request construction
- multipart document upload without manually overriding the generated boundary
- FastAPI error-detail translation
- 204 document deletion handling

**Document state**

- initial document loading
- upload upsert by document identity
- deletion from local state
- error presentation
- document selection and destructive-action confirmation

**Semantic search**

- document-scoped and corpus-wide request construction
- ranked-result rendering
- score display without interpreting similarity as probability
- loading, empty, and failure states
- stale-response rejection when newer state supersedes an in-flight request

**Cited RAG**

- document-scoped query construction
- answer, insufficient-evidence, loading, and failure states
- generation diagnostics rendering
- stale-response rejection

**Citation/evidence inspection**

- only known validated source IDs become interactive controls
- citation character ranges remain aligned with Unicode code-point indexing
- inline references and source cards select the same trusted evidence
- document/page/chunk provenance is presented from `EvidenceBlock` data
- citation inspection does not require a new retrieval request

**Application integration**

- document upload appears without a page refresh
- selected-document scope is propagated to search and RAG requests
- deletion of the selected document returns scope to the whole corpus
- empty libraries disable research execution
- `Cmd/Ctrl + Enter` submits while normal Enter remains available for multiline input

### Real Frontend ↔ Backend Smoke Workflow

A manual local smoke workflow was completed across the real browser-facing stack. The workflow exercised:

```text
React / Vite
    ↓
ResearchAPIClient
    ↓
FastAPI
    ↓
ResearchService
    ↓
PyMuPDF / chunking / BGE / Qdrant
    ↓
retrieval / context construction
    ↓
Ollama / Qwen
    ↓
citation validation
    ↓
React answer + evidence inspection
```

The workflow covered:

- initial document-library loading
- real PDF upload and indexing
- duplicate-upload replacement semantics
- document-scoped semantic search
- corpus-wide semantic search
- keyboard submission
- real cited RAG generation
- inline citation selection
- source-card selection
- exact evidence/provenance inspection
- an unsupported-question path
- stale-request/scope-change behaviour
- result/question provenance after textarea edits
- delete confirmation, deletion, and scope reset

No numerical retrieval-quality, latency, throughput, memory, or accuracy claims are inferred from this smoke workflow. Exact values should be recorded only when captured as part of a controlled experiment or benchmark.

### Phase 10 Invariants

- The browser communicates with the research system only through the Phase 9 HTTP API.
- React components do not query Qdrant or Ollama directly.
- Semantic Search does not invoke the language model.
- Citation inspection uses evidence already returned for the generation request.
- Similarity scores are displayed as raw retrieval scores rather than calibrated relevance probabilities.
- Changing corpus scope invalidates displayed search/RAG state.
- Stale asynchronous responses cannot overwrite newer scope/request state.
- Citation identity validation remains distinct from claim-level citation support.

### Current Frontend Limitations

The completed frontend is a local research-workspace baseline. No claims are currently made about:

- public internet deployment
- authentication or account isolation
- concurrent multi-user behaviour
- streaming token rendering
- browser-session persistence
- accessibility certification
- mobile-first product optimisation
- calibrated confidence scores
- formal answer correctness
- formal claim-to-source entailment

These are future engineering or evaluation concerns rather than missing evidence that should be filled with assumptions.

---

## Planned Retrieval Experiments

Planned retrieval comparisons include:

- chunk size
- chunk overlap
- chunking strategy
- embedding model
- retrieval top-k
- similarity threshold
- dense vs hybrid retrieval
- BM25 contribution
- Reciprocal Rank Fusion
- reranking

Relevant metrics include:

```text
Recall@K
Precision@K
MRR
nDCG
```

---

## Planned Context Experiments

Planned context-construction comparisons include:

- context size
- reserved output allowance
- safety margin
- overlap-deduplication threshold
- evidence ordering
- evidence diversity
- whole chunks vs trimmed chunks
- neighbouring-context inclusion
- context compression

Relevant measurements may include:

- evidence coverage
- prompt-token usage
- context truncation rate
- token-estimation error
- answer faithfulness
- answer completeness

---

## Planned Generation Experiments

Potential comparisons include:

- local model size
- generation temperature
- top-p
- output-token budget
- context-window allocation
- reasoning/thinking configuration
- prompt variants

Relevant measurements may include:

- faithfulness
- answer relevance
- latency
- token usage
- memory use

Generation experiments should use the same retrieval evidence when comparing model configurations so retrieval differences do not confound the comparison.

---

## Planned Citation Evaluation

Phase 8 provides a working citation-integrity baseline (see above). Formal citation evaluation is still planned. It should distinguish **reference identity validity** (does `[S2]` exist in the supplied context?) from **semantic correctness** (does source S2 actually support the assertion it accompanies?).

Further experiments can assess malformed-protocol detection, unknown-ID rejection, source-to-page correctness, missing-citation rates, claim-level coverage, and annotation of unsupported citations. Citation behaviour should be compared under a fixed retrieval set when experimenting with generation instructions, so changes can be attributed more clearly.

The citation layer should be evaluated independently from retrieval and generation wherever practical.
---

## Planned End-to-End Evaluation

Later evaluation should distinguish at least four failure classes:

```text
Retrieval Failure
    → relevant evidence not retrieved
Context Failure
    → relevant evidence retrieved but not supplied to model
Generation Failure
    → correct evidence supplied but answer unsupported or incorrect
Citation Identity Failure
    → recognised source ID is missing or unmapped
Citation Support Failure
    → source ID resolves but evidence does not support the associated claim
```

This decomposition is central to the project.

A single overall RAG score would make these failures harder to diagnose.

---

## Benchmarking Principles

The project should follow several principles during experimentation.

**Change one major variable at a time.**

Avoid changing chunking, embeddings, retrieval settings, and generation model simultaneously when trying to understand causality.

**Preserve the baseline.**

New techniques should be compared against the simplest working implementation.

**Record failures.**

Unexpected or poor results are useful engineering evidence and should not be discarded.

**Separate qualitative inspection from formal metrics.**

A few impressive example answers are not evidence of general retrieval or RAG quality.

**Do not invent conclusions before measurements exist.**

Configuration choices should be described as baselines until controlled experiments justify stronger claims.

---

## Planned Technical Report

The final project should be able to explain not only:

> Does the RAG system work?

but also:

> Why were these architectural and retrieval choices selected?

The technical report should eventually compare:

- ingestion behaviour
- chunking strategies
- embedding models
- retrieval methods
- score thresholds
- hybrid retrieval
- reranking
- context construction
- token budgeting
- local model performance
- answer faithfulness
- citation correctness
- latency
- memory usage

The final system should therefore be supported by measured engineering decisions rather than only a working demonstration.

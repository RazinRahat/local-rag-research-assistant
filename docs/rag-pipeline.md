# End-to-End RAG Pipeline

## Purpose

Phase 7 connects the independently implemented retrieval and generation systems into the project's first complete Retrieval-Augmented Generation pipeline.

Before this phase, the application had two separate capabilities.

### Retrieval

```text
Question
   ↓
Query Embedding
   ↓
Semantic Retrieval
   ↓
RetrievalResult[]
```

### Generation

```text
ChatMessage[]
   ↓
LLMProvider
   ↓
Local LLM
   ↓
GenerationResult
```

Phase 7 introduces an orchestration and context-construction layer between them:

```text
Question
   ↓
Semantic Retrieval
   ↓
Retrieved Evidence
   ↓
Context Builder
   ↓
Grounded Prompt
   ↓
Local LLM
   ↓
RAGResponse
```

This makes the system a functioning local dense-RAG baseline while preserving clear component boundaries.

---

## Why an Explicit RAG Layer

Retrieval results should not be concatenated directly and passed to a language model.

A production-oriented RAG system has to make deliberate decisions about:

* evidence selection
* context-window limits
* retrieval ordering
* overlapping chunks
* duplicate evidence
* source identity
* prompt structure
* weak or missing evidence
* grounding instructions
* prompt injection inside retrieved documents

These concerns belong above retrieval and generation.

The project therefore introduces a dedicated RAG layer rather than hiding these decisions inside either the retriever or LLM provider.

---

## RAG Architecture

The current orchestration path is:

```text
                        User Question
                             │
                             ▼
                         Retriever
                             │
                             ▼
                    RetrievalResult[]
                             │
                             ▼
                     ContextBuilder
                             │
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
        Token Budget     Deduplication   Source Labels
              │              │              │
              └──────────────┼──────────────┘
                             ▼
                       ChatMessage[]
                             │
                             ▼
                         LLMProvider
                             │
                             ▼
                      GenerationResult
                             │
                             ▼
                         RAGResponse
```

The orchestration service depends on application interfaces:

```text
Retriever
LLMProvider
ChatTokenCounter
```

rather than concrete implementations such as Qdrant or Ollama.

---

## Retriever Abstraction

The RAG layer depends on a `Retriever` protocol.

Conceptually:

```text
Retriever
    │
    └── SemanticRetriever
```

The interface accepts a natural-language query and retrieval configuration and returns typed ranked evidence.

This means future retrieval systems can replace the dense baseline without redesigning RAG orchestration.

Potential future implementations include:

```text
Retriever
├── SemanticRetriever
├── HybridRetriever
└── RerankingRetriever
```

---

## Context Construction

The `ContextBuilder` converts ranked retrieval results into a bounded prompt.

Its responsibilities include:

```text
retrieval results
      ↓
remove redundant evidence
      ↓
assign source labels
      ↓
test candidate against token budget
      ↓
retain highest-ranked evidence that fits
      ↓
construct grounded messages
```

The context builder does not generate text.

It prepares evidence for generation.

---

## Context Budget

The local LLM has a finite runtime context.

The Phase 6 baseline uses:

```text
Runtime context        8192
Reserved output         512
Safety margin           512
───────────────────────────
Maximum prompt budget  7168
```

The prompt budget includes more than document text.

It must contain:

```text
system instructions
+
user question
+
source metadata
+
evidence delimiters
+
retrieved text
+
chat-template control tokens
```

The system therefore evaluates the complete formatted conversation rather than estimating only the size of the selected chunks.

---

## Why a Safety Margin Exists

The application estimates prompt length with the corresponding Hugging Face tokenizer and chat template, while generation is executed by Ollama.

These two paths should be closely aligned but are not treated as guaranteed to produce perfectly identical token counts.

The context builder therefore reserves a configurable safety margin.

The application also records:

```text
estimated_prompt_tokens
actual_prompt_tokens
```

so the difference can be measured after generation.

This allows the estimator to be validated empirically rather than assumed to be exact.

---

## Chat-Template-Aware Token Estimation

Plain text token counting is not sufficient for chat models.

A request contains role and message-boundary tokens in addition to visible text.

Conceptually:

```text
system message
      ↓
chat template
      ↓
system control tokens
system text
message boundary
user control tokens
user text
assistant generation header
```

The Phase 7 token counter therefore applies the Qwen chat template before counting tokens.

The implemented process is:

```text
ChatMessage[]
      ↓
apply_chat_template(
    tokenize=False,
    add_generation_prompt=True
)
      ↓
formatted chat text
      ↓
tokenizer.encode(
    add_special_tokens=False,
    truncation=False
)
      ↓
token count
```

`add_special_tokens=False` is important because the rendered chat template already contains the model's required control tokens.

`truncation=False` is equally important: the application is measuring whether the request fits, not silently shortening an oversized prompt.

---

## Token-Counter Boundary Lesson

During the first real RAG smoke test, the implementation initially assumed that:

```text
apply_chat_template(tokenize=True)
```

would always return a plain:

```text
list[int]
```

The installed Transformers/tokenizer combination returned a broader tokenizer output shape, causing the runtime guard to reject an otherwise valid result.

The implementation was changed to a more explicit two-stage process:

```text
render chat template as text
        ↓
encode rendered text
        ↓
count resulting token IDs
```

This avoids depending on version-specific return shapes while preserving the correct chat-template semantics.

This is an example of why third-party library boundaries are validated explicitly instead of weakening type checking globally.

---

## Whole-Chunk Evidence Selection

The baseline context builder includes only complete retrieved chunks.

It does not cut arbitrary chunks midway to squeeze additional text into the prompt.

Conceptually:

```text
Chunk 1 fits     → include
Chunk 2 fits     → include
Chunk 3 too big  → skip
Chunk 4 fits     → may still include
```

Preserving whole retrieval units avoids creating misleading fragments through arbitrary truncation.

Future experiments may investigate:

* sentence-aware trimming
* context compression
* parent-child retrieval
* neighbouring chunk expansion

but those are not part of the baseline.

---

## Retrieval Ordering

Evidence is considered in retrieval rank order.

```text
Rank 1
 ↓
Rank 2
 ↓
Rank 3
 ↓
...
```

This means the most relevant dense-retrieval candidates receive first access to the available context budget.

The RAG layer does not independently rerank results.

Reranking will be introduced later as a separate retrieval stage.

---

## Evidence Deduplication

Token-overlap chunking can cause related retrieval results to contain repeated text.

The context builder therefore detects:

* identical chunk IDs
* highly overlapping chunks from the same document page

The current baseline uses a conservative overlap threshold.

Ordinary neighbouring chunks created by the normal overlap window remain eligible.

Only strongly redundant evidence is removed.

This avoids wasting limited model context on near-duplicate material while preserving useful neighbouring information.

---

## Source Labels

Every selected evidence block receives a local prompt source identifier:

```text
S1
S2
S3
...
```

A rendered block looks conceptually like:

```text
<SOURCE id="S1" file="paper.pdf" page="3">
Retrieved document text...
</SOURCE>
```

The source label is not yet a final citation.

It provides a stable identity linking prompt evidence to the existing retrieval provenance.

Phase 8 will build formal citation handling on top of this structure.

---

## Provenance

Evidence already contains the provenance established earlier in the pipeline.

```text
EvidenceBlock
├── source_id
├── retrieval_rank
├── score
└── DocumentChunk
    ├── chunk_id
    ├── document_id
    ├── file_name
    ├── page_number
    ├── chunk_index
    └── text
```

The resulting provenance chain is:

```text
RAG answer
    ↓
EvidenceBlock
    ↓
RetrievalResult
    ↓
DocumentChunk
    ↓
page
    ↓
source document
```

Phase 8 will make this relationship user-facing through validated citations.

---

## Grounding Instructions

The system prompt instructs the local model to:

* answer from supplied evidence
* avoid unsupported outside information
* say when the evidence is insufficient
* avoid inventing references or page numbers
* distinguish evidence from cautious interpretation
* treat retrieved document text as source material rather than instructions

These instructions establish grounding behaviour.

They do not prove that every generated answer is faithful.

Formal faithfulness evaluation comes later.

---

## Retrieved Documents as Untrusted Data

Document contents may themselves contain instructions.

For example, a retrieved PDF could include text such as:

```text
Ignore previous instructions...
```

That text is evidence, not application policy.

The system prompt explicitly tells the model to treat retrieved evidence as source material and to ignore instructions contained inside it.

The prompt also uses explicit source boundaries to reinforce the separation between:

```text
application instructions
```

and:

```text
retrieved document content
```

This reduces one prompt-injection path, though adversarial-RAG behaviour will require broader testing later.

---

## Insufficient Evidence

If retrieval returns no usable results, the RAG service does not call the language model.

Instead:

```text
no retrieval evidence
       ↓
skip generation
       ↓
return insufficient-evidence response
```

This prevents one obvious failure mode where the model would otherwise answer entirely from prior knowledge.

However, dense vector search can still return mathematically nearest chunks for an unrelated question.

Therefore:

```text
some retrieved chunks
```

does not automatically mean:

```text
sufficient evidence
```

Reliable weak-evidence detection will require retrieval evaluation and threshold calibration.

---

## Score Thresholds

The retrieval layer already supports an optional score threshold.

Phase 7 does not introduce a universal default.

A value such as:

```text
0.6
```

may behave differently depending on:

* embedding model
* corpus
* query style
* chunk strategy
* document domain

Threshold selection should therefore be based on labelled retrieval experiments rather than intuition.

---

## Typed RAG Response

The orchestration layer returns a `RAGResponse` rather than only generated text.

Conceptually:

```text
RAGResponse
├── question
├── answer
├── evidence[]
├── retrieved_count
├── used_evidence_count
├── estimated_prompt_tokens
├── actual_prompt_tokens
├── context_truncated
├── insufficient_evidence
└── generation
```

This makes the RAG pipeline inspectable.

A caller can determine:

* what was retrieved
* what was actually passed to the model
* whether context was truncated
* how much prompt space was used
* how much the token estimator differed from the runtime
* how generation performed

This information will later support the API, user interface, evaluation, and observability layers.

---

## Estimated vs Actual Prompt Tokens

Phase 7 intentionally retains both values:

```text
estimated_prompt_tokens
actual_prompt_tokens
```

The estimator is used before model invocation to enforce the context budget.

The Ollama response reports the runtime prompt-token count after generation.

Comparing them provides a useful diagnostic:

```text
actual - estimated
```

Repeated observations can later determine whether the current safety margin is unnecessarily large, appropriately conservative, or insufficient.

No formal benchmark conclusion is made yet.

---

## Test Strategy

The Phase 7 test suite isolates orchestration from heavyweight infrastructure.

### Context Builder Tests

A fake token counter verifies:

* retrieval-order preservation
* duplicate removal
* context-budget enforcement
* truncation state

### RAG Service Tests

Fake implementations of the retriever and LLM verify:

```text
query
 ↓
retrieval
 ↓
context construction
 ↓
generation
 ↓
RAGResponse
```

without loading:

* BGE
* Qdrant
* Ollama
* Qwen

The service also verifies that the LLM is not invoked when retrieval produces no evidence.

### Real Smoke Test

The manual integration test uses:

```text
BGE
+
persistent Qdrant
+
ContextBuilder
+
Qwen tokenizer
+
Ollama
+
Qwen
```

to validate the complete local pipeline.

The smoke test confirms integration behaviour but is not treated as formal RAG evaluation.

---

## Current Limitations

The baseline RAG pipeline does not yet provide:

* validated answer citations
* claim-to-source alignment
* hallucination detection
* calibrated weak-evidence thresholds
* hybrid retrieval
* reranking
* context compression
* sentence-aware evidence trimming
* query rewriting
* multi-hop retrieval
* source diversity optimisation
* structured answer schemas
* streaming generation
* formal faithfulness evaluation

These remain separate later phases.

---

## Grounded Does Not Mean Guaranteed

The project currently provides grounding mechanisms:

```text
retrieval
+
bounded evidence context
+
explicit source boundaries
+
grounding instructions
+
no-evidence bypass
```

This should not be described as a guarantee that hallucinations cannot occur.

The appropriate current claim is:

> The model is instructed to answer from retrieved evidence, and the application exposes the evidence supplied to generation.

Whether answers are actually faithful must be measured during evaluation.

---

## Next Step

Phase 8 will turn preserved provenance into validated user-facing citations.

The architecture will evolve toward:

```text
Question
   ↓
Retrieval
   ↓
Context Construction
   ↓
Local LLM
   ↓
Generated Answer
   ↓
Citation Parsing / Mapping
   ↓
Validated Sources
   ↓
Answer + Citations + Evidence
```

The objective is not merely to display source labels, but to ensure every returned citation corresponds to evidence that actually entered the RAG context.

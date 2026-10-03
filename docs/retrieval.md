# Retrieval

## Purpose

The retrieval layer converts natural-language queries into ranked source evidence.

The system currently supports:

```text
dense
lexical
hybrid
```

Each strategy returns the same typed `RetrievalResult` and preserves original document provenance.

---

## Common Retrieval Contract

```text
query
+
RetrievalConfig
 ↓
Retriever
 ↓
RetrievalResult[]
```

`RetrievalResult` contains:

```text
rank
score
DocumentChunk
```

This keeps downstream RAG and citation logic independent from retrieval strategy.

---

# Dense Retrieval

```text
Natural-Language Query
        ↓
EmbeddingProvider
        ↓
BGE Query Embedding
        ↓
VectorStore
        ↓
Qdrant Cosine Search
        ↓
RetrievalResult[]
```

Current embedding model:

```text
BAAI/bge-small-en-v1.5
```

Embedding dimension:

```text
384
```

Dense retrieval is useful for conceptual similarity, paraphrases, and semantically related wording.

Dense remains the default strategy.

---

## Dense Scores

Dense retrieval returns cosine similarity.

The project does not define one global default similarity threshold.

Threshold usefulness depends on the embedding model, corpus, query distribution, chunking strategy, and retrieval objective.

---

# Lexical Retrieval

```text
Natural-Language Query
        ↓
Lexical Tokenisation
        ↓
BM25
        ↓
Persisted Chunk Corpus
        ↓
RetrievalResult[]
```

Current baseline:

```text
k1 = 1.5
b = 0.75
```

The implementation uses a lightweight Unicode-aware tokenizer with case normalisation.

The current baseline does not yet introduce stemming, learned sparse vectors, query expansion, or phrase retrieval.

---

## Why Lexical Retrieval

Dense retrieval may be weaker when retrieval depends strongly on exact lexical identity.

Examples include:

```text
dataset names
model names
acronyms
algorithm names
equations
rare identifiers
technical phrases
```

BM25 provides a complementary retrieval signal for these cases.

---

# Shared Qdrant Corpus

Lexical retrieval uses the same persisted chunks already stored in Qdrant.

```text
QdrantVectorStore
├── VectorStore
│   └── dense vector persistence/search
├── DocumentRegistry
│   └── document listing/count/deletion
└── ChunkCorpus
    └── lexical chunk enumeration
```

`ChunkCorpus` remains separate from the core vector-store protocol.

---

# Reciprocal Rank Fusion

Dense cosine similarity and BM25 relevance scores use different numerical scales.

Hybrid retrieval therefore uses rank rather than raw score magnitude.

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

Current baseline:

```text
k = 60
```

---

# Hybrid Retrieval

```text
                 Query
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
 Dense Retriever        BM25 Retriever
        │                     │
        ▼                     ▼
 Dense Ranking          Lexical Ranking
        │                     │
        └──────────┬──────────┘
                   ▼
                  RRF
                   │
                   ▼
          RetrievalResult[]
```

The hybrid retriever depends only on the `Retriever` protocol.

---

## Candidate Expansion

Current baseline:

```text
candidate_multiplier = 2
```

With:

```text
top_k = 5
```

the hybrid retriever requests up to 10 dense and 10 lexical candidates before final fusion.

The multiplier is not claimed to be optimal.

---

# Retrieval Routing

Available modes:

```text
dense
lexical
hybrid
```

`RetrievalRouter` chooses the configured strategy.

Dense remains the default for backward compatibility.

The same router is used by both `ResearchService` and `RAGService`.

---

# Document-Scoped Retrieval

Global retrieval:

```text
query
 ↓
all indexed documents
```

Document-scoped retrieval:

```text
query
 ↓
document_id
 ↓
one indexed document
```

---

# Score Semantics

```text
dense
→ cosine similarity

lexical
→ BM25 relevance

hybrid
→ RRF score
```

Those scores must not be compared numerically across modes.

---

## Score Thresholds

`score_threshold` is supported only for dense retrieval.

```text
dense cosine threshold
≠
BM25 threshold
≠
RRF threshold
```

---

# Phase 11 Comparison

A six-query smoke comparison was run using:

```text
top_k = 5
scope = all indexed documents

dense
lexical
hybrid
```

Average top-five Jaccard overlap:

```text
Dense ↔ Lexical    ≈ 0.214
Dense ↔ Hybrid     ≈ 0.478
Lexical ↔ Hybrid   ≈ 0.507
```

The result demonstrates that dense and lexical retrieval provide materially different candidate signals.

---

## Exact-Term Example

For:

```text
Adam optimizer
```

dense retrieval ranked a bibliography passage first.

The actual optimizer-method passage appeared second.

BM25 and hybrid retrieval promoted the method passage to rank one.

---

## Semantic Example

For:

```text
Why can the model process sequence positions in parallel?
```

dense retrieval returned the Transformer discussion explaining the sequential limitation of recurrence and the greater parallelisation enabled by attention.

This demonstrates why semantic retrieval remains important.

---

# Current Limitations

The retrieval system does not yet include:

- cross-encoder reranking
- learned sparse retrieval
- phrase-aware lexical retrieval
- stemming
- query rewriting
- query expansion
- HyDE
- parent-child retrieval
- metadata filtering beyond document identity
- diversity-aware retrieval
- empirically tuned BM25 parameters
- empirically tuned RRF parameters
- labelled retrieval evaluation

The BM25 baseline can also score chunks containing only some terms from a multi-term query.

---

# Testing Strategy

Tests cover:

- dense query validation
- dense top-k handling
- document filtering
- BM25 lexical ranking
- lexical tokenisation
- empty corpus handling
- RRF mathematics
- deterministic fusion
- duplicate detection
- hybrid candidate expansion
- router strategy selection
- Qdrant chunk-corpus enumeration
- API retrieval-mode propagation

---

# Evaluation Direction

Formal labelled retrieval evaluation will use:

```text
Recall@K
Precision@K
MRR
nDCG
```

Future comparisons will include:

```text
Dense
Lexical
Hybrid
Hybrid + Reranking
```

---

# Next Step

Phase 12 introduces reranking.

```text
Dense Retrieval
       +
BM25 Retrieval
       ↓
      RRF
       ↓
Candidate Set
       ↓
Cross-Encoder
       ↓
Reranked Evidence
```

The Phase 11 baseline should remain frozen during the first reranking experiments.

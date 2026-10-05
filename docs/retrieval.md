# Retrieval

## Purpose

The retrieval layer converts natural-language queries into ranked source evidence while preserving original document, page, chunk, and text provenance.

The system currently supports:

```text
dense
lexical
hybrid
hybrid_reranked
```

Every strategy returns the same typed `RetrievalResult`, so downstream RAG and citation logic remain independent from retrieval implementation.

---

# Common Retrieval Contract

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
384 dimensions
normalised vectors
```

Dense retrieval is useful for conceptual similarity, paraphrases, and semantically related wording.

Dense remains the default strategy for backward compatibility.

## Dense scores

Dense retrieval returns cosine similarity.

The project does not define one global similarity threshold. Threshold usefulness depends on the embedding model, corpus, query distribution, chunking strategy, and retrieval objective.

`score_threshold` is therefore supported only for Dense retrieval.

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

Current BM25 baseline:

```text
k1 = 1.5
b  = 0.75
```

The implementation uses a lightweight Unicode-aware tokenizer with case normalisation.

BM25 provides a complementary signal when retrieval depends strongly on exact lexical identity, including model names, acronyms, algorithms, dataset names, equations, rare identifiers, and technical phrases.

The current lexical baseline does not yet include stemming, phrase-aware retrieval, query expansion, or learned sparse vectors.

---

# Shared Qdrant Corpus

Lexical retrieval enumerates the same persisted `DocumentChunk` objects already stored in Qdrant.

```text
QdrantVectorStore
├── VectorStore
│   └── dense vector persistence/search
├── DocumentRegistry
│   └── document listing/count/deletion
└── ChunkCorpus
    └── lexical chunk enumeration
```

`ChunkCorpus` remains separate from the core dense `VectorStore` protocol.

---

# Weighted Reciprocal Rank Fusion

Dense cosine similarity and BM25 relevance scores use incompatible numerical scales.

Hybrid retrieval therefore fuses **ranks**, not raw score magnitudes.

The generic fusion function supports weighted RRF:

```text
RRF(d) = Σ w_i / (k + rank_i(d))
```

The generic `RRFConfig` keeps a conventional default rank constant for reusable equal-weight fusion, while the production Hybrid configuration supplies the Phase 13 tuned settings:

```text
Hybrid RRF k = 5
Dense weight = 1.00
Lexical weight = 1.25
```

This separation keeps the generic fusion primitive backward-compatible while allowing the Hybrid retriever to own retrieval-specific tuning.

---

# Hybrid Retrieval

```text
                 Query
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
  Dense Retriever       BM25 Retriever
         │                   │
         ▼                   ▼
   Dense Ranking       Lexical Ranking
         │                   │
         └─────────┬─────────┘
                   ▼
            Weighted RRF
                   │
                   ▼
           RetrievalResult[]
```

The Hybrid retriever depends only on the shared `Retriever` protocol.

## Candidate expansion

Current configuration:

```text
candidate_multiplier = 2
```

For a normal standalone Hybrid request with:

```text
top_k = 5
```

Hybrid asks each branch for:

```text
Dense top 10
BM25 top 10
```

before fusion and final top-5 selection.

Candidate expansion is different when Hybrid is wrapped by the reranking pipeline; that nested path is described below.

---

# Cross-Encoder Reranking

`hybrid_reranked` is a second-stage retrieval mode.

The final Phase 13 architecture deliberately separates two depth controls:

```text
candidate_pool_size = 20
rerank_pool_size    = 10
```

Their meanings are different:

```text
candidate_pool_size
→ how many Hybrid results the reranking wrapper asks the first-stage retriever to produce

rerank_pool_size
→ how many of those strongest Hybrid results are sent to the cross-encoder
```

For the default final top-5 request:

```text
RerankingRetriever requests Hybrid top 20
                 ↓
Hybrid candidate_multiplier = 2
                 ↓
Dense top 40 + BM25 top 40
                 ↓
weighted RRF
                 ↓
Hybrid top 20
                 ↓
keep Hybrid ranks 1–10
                 ↓
cross-encoder/ms-marco-MiniLM-L6-v2
                 ↓
final top 5
```

This design was introduced after a Phase 13 experiment showed that using one parameter for both depths changed first-stage candidate generation when the intention was only to restrict cross-encoder scoring.

The cross-encoder uses query-passage pairs and returns a relevance score. The final score is not a calibrated probability.

---

# Retrieval Routing

Available modes:

```text
dense
lexical
hybrid
hybrid_reranked
```

`RetrievalRouter` chooses the configured strategy.

The same router is used by both `ResearchService` and `RAGService`.

Dense remains the default for backward compatibility.

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

Phase 13 formal evaluation uses document-scoped queries so the benchmark tests evidence ranking rather than ambiguity about which paper is being referenced.

---

# Score Semantics

```text
dense
→ cosine similarity

lexical
→ BM25 relevance

hybrid
→ weighted RRF score

hybrid_reranked
→ cross-encoder relevance score
```

These scores must not be compared numerically across modes.

`score_threshold` remains dense-only.

---

# Historical Phase 11 Comparison

The initial six-query smoke comparison showed complementary candidate sets:

```text
Dense ↔ Lexical    ≈ 0.214 top-5 Jaccard
Dense ↔ Hybrid     ≈ 0.478
Lexical ↔ Hybrid   ≈ 0.507
```

For `Adam optimizer`, Dense ranked a bibliography passage first while BM25 and Hybrid promoted the actual optimizer-method passage.

This established behavioural complementarity, not formal superiority.

---

# Phase 13 Formal Evaluation

The expanded benchmark uses:

```text
3 papers
18 document-scoped queries
12 development
6 holdout
527 graded judgments
0–3 relevance scale
binary relevance >= 2
top_k = 5
```

The labels were created through a model-assisted blind review process. Recall is pooled recall rather than exhaustive corpus recall.

## Initial development result

| Mode | P@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| Dense | 0.2500 | 0.4514 | 0.5556 | 0.4551 |
| Lexical | 0.3667 | 0.6319 | 0.7222 | 0.6264 |
| Hybrid | 0.3500 | 0.5903 | 0.5306 | 0.4951 |
| Hybrid + Reranking | 0.3167 | 0.5625 | 0.5958 | 0.4801 |

The important finding was that BM25 initially outperformed the more complex modes on the development benchmark.

## Tuned Hybrid

An 84-configuration development-only RRF sweep selected:

```text
k = 5
dense weight = 1.00
lexical weight = 1.25
```

The selection prioritised a no-regression development profile rather than the single highest aggregate nDCG configuration.

Final Hybrid development metrics:

```text
P@5      0.3833
Recall   0.6736
MRR      0.5750
nDCG     0.5619
```

## Final reranked development metrics

After separating candidate generation depth from rerank depth:

```text
P@5      0.3833
Recall   0.7153
MRR      0.6903
nDCG     0.5987
```

Compared with the initial development reranker, the final configuration improved P@5 by 21.1%, Recall@5 by 27.2%, MRR by 15.9%, and nDCG@5 by 24.7%.

## Frozen holdout

The final six-query holdout result was:

| Mode | P@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| Dense | 0.4000 | 0.7667 | 0.4778 | 0.4806 |
| Lexical | 0.3667 | 0.6917 | 0.6806 | 0.5083 |
| Hybrid | 0.3667 | 0.6917 | 0.6667 | 0.5397 |
| **Hybrid + Reranking** | **0.4667** | **0.8500** | **0.9167** | **0.7284** |

Reranking improved four holdout queries, left one unchanged, and degraded one relative to Hybrid.

The main holdout degradation was the GPT-2 Natural Questions query, where the cross-encoder retained a partially relevant result but dropped the passage containing the full 4.1% exact-match result from the final top five.

See `docs/phase13-evaluation.md` for the full experiment sequence.

---

# Current Limitations

The retrieval system still does not include:

- learned sparse retrieval
- phrase-aware lexical retrieval
- stemming
- query rewriting
- query expansion
- HyDE
- parent-child retrieval
- metadata filtering beyond document identity
- diversity-aware retrieval
- tuned BM25 parameters
- large-scale multi-corpus benchmark coverage
- exhaustive relevance annotation
- negative-query/abstention evaluation

The current benchmark is small, uses pooled relevance judgments, and should not be used to claim universal parameter optimality.

---

# Testing Strategy

Tests cover:

- dense query validation and top-k handling
- document filtering
- BM25 lexical ranking and tokenisation
- empty corpus handling
- RRF mathematics and deterministic fusion
- weighted-RRF validation
- duplicate detection
- Hybrid candidate expansion
- reranking candidate depth
- separate rerank depth
- nested branch expansion
- cross-encoder ordering
- router strategy selection
- Qdrant chunk-corpus enumeration
- API retrieval-mode propagation
- evaluation metrics, aggregation, splits, and diagnostics

---

# Next Evaluation Directions

Retrieval evaluation should now expand to new data rather than continue tuning the same development set.

Useful next comparisons include larger corpora, negative queries, BM25 preprocessing variants, chunking variants, embedding models, learned sparse retrieval, and latency/resource benchmarking.

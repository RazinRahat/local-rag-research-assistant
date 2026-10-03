# Hybrid Retrieval

## Purpose

Phase 11 extends the original dense semantic retrieval baseline with BM25 lexical retrieval and Reciprocal Rank Fusion.

The objective is not to replace dense retrieval.

Instead, the system exposes three explicit retrieval strategies:

```text
Dense
Lexical
Hybrid
```

This allows retrieval strategies to remain independently inspectable and later comparable through formal evaluation.

---

## Architecture

```text
                              Query
                                │
                                ▼
                       RetrievalRouter
                                │
              ┌─────────────────┼─────────────────┐
              │                 │                 │
              ▼                 ▼                 ▼
            Dense            Lexical           Hybrid
              │                 │                 │
              ▼                 ▼          ┌──────┴──────┐
     SemanticRetriever    BM25Retriever     ▼             ▼
              │                 │         Dense          BM25
              ▼                 ▼           │             │
        BGE Embedding      ChunkCorpus       └──────┬──────┘
              │                 │                  ▼
              ▼                 ▼                 RRF
       Qdrant Search      Qdrant Payloads          │
              │                 │                  ▼
              ▼                 ▼          RetrievalResult[]
      RetrievalResult[]   RetrievalResult[]
```

All three strategies return the same typed `RetrievalResult`.

Downstream components therefore remain independent from retrieval implementation.

---

## Dense Retrieval

Dense retrieval remains the original Phase 5 baseline.

```text
query
 ↓
BGE query embedding
 ↓
Qdrant cosine search
 ↓
ranked chunks
```

Current embedding model:

```text
BAAI/bge-small-en-v1.5
```

Dense retrieval is useful for conceptual similarity, paraphrases, and natural-language questions whose wording may differ from the source.

Dense remains the default mode.

---

## Lexical Retrieval

The lexical branch uses BM25.

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

Current baseline:

```text
k1 = 1.5
b  = 0.75
```

These values are engineering baselines and have not yet been tuned against labelled relevance data.

BM25 is particularly useful for:

- exact terminology
- dataset names
- model names
- acronyms
- technical identifiers
- algorithm names
- rare phrases

---

## Shared Corpus

The lexical retriever uses the same persisted chunks already stored in Qdrant.

```text
QdrantVectorStore
├── VectorStore
│   └── dense vector retrieval
├── DocumentRegistry
│   └── document lifecycle
└── ChunkCorpus
    └── lexical corpus enumeration
```

The core `VectorStore` protocol remains focused on vector persistence and search.

Lexical corpus access is represented through a separate narrow protocol.

---

## Why Reciprocal Rank Fusion

Dense retrieval returns cosine similarity.

BM25 returns its own lexical relevance score.

Example:

```text
Dense score:  0.79
BM25 score:   8.41
```

Those values are not directly comparable.

The hybrid retriever therefore uses Reciprocal Rank Fusion:

```text
RRF(d) = Σ 1 / (k + rank_i(d))
```

Current baseline:

```text
k = 60
```

RRF rewards chunks that rank highly across multiple retrieval branches without inventing a raw-score normalization scheme.

---

## Candidate Expansion

Current baseline:

```text
candidate_multiplier = 2
```

Example:

```text
final top_k = 5

Dense candidates   = 10
Lexical candidates = 10

        ↓
       RRF

final results = 5
```

The multiplier is not claimed to be optimal.

---

## Retrieval Modes

Supported modes:

```text
dense
lexical
hybrid
```

Mode selection is carried through `RetrievalConfig`.

`RetrievalRouter` maps that configuration to the correct retriever.

The same router is used by:

```text
POST /search
POST /query
```

Dense remains the default for backward compatibility.

---

## Score Semantics

`RetrievalResult.score` has different semantics depending on mode.

```text
Dense
→ cosine similarity

Lexical
→ BM25 score

Hybrid
→ RRF score
```

Scores should never be numerically compared across modes.

---

# Phase 11 Comparison

## Experiment

A controlled smoke comparison used:

```text
queries:       6
top_k:         5
scope:         all indexed documents
modes:
  - dense
  - lexical
  - hybrid
```

The same query and final top-k were used for every strategy.

Query categories included:

- exact terminology
- rare terminology
- semantic paraphrase
- technical concept
- dataset question
- architecture question

---

## Candidate-Set Overlap

Average top-five Jaccard overlap:

```text
Dense ↔ Lexical    ≈ 0.214
Dense ↔ Hybrid     ≈ 0.478
Lexical ↔ Hybrid   ≈ 0.507
```

Top-result agreement:

```text
Dense ↔ Lexical    2 / 6
Dense ↔ Hybrid     3 / 6
Lexical ↔ Hybrid   4 / 6
```

The important observation is:

```text
Dense and lexical retrieval
produced materially different candidate sets.

Hybrid retrieval incorporated evidence
from both branches.
```

These are descriptive overlap measurements, not retrieval-accuracy metrics.

---

## Case Study — Adam Optimizer

Query:

```text
Adam optimizer
```

Dense retrieval placed a bibliography chunk containing the Adam paper at rank 1.

The actual Transformer optimizer-method section appeared at dense rank 2.

BM25 promoted the actual optimizer section to rank 1.

Hybrid retrieval also promoted that section to rank 1.

This is a concrete example of lexical retrieval complementing dense retrieval.

---

## Case Study — Label Smoothing

Query:

```text
label smoothing
```

All three modes returned the direct Transformer label-smoothing section at rank 1.

Lower-ranked lexical results also exposed a baseline limitation: BM25 can assign relevance to a chunk containing only part of a multi-term query.

Possible future variables include phrase handling, matched-term coverage, stemming, query preprocessing, and learned sparse retrieval.

These are intentionally not tuned during Phase 11.

---

## Case Study — Semantic Paraphrase

Query:

```text
Why can the model process sequence positions in parallel?
```

Dense retrieval returned the Transformer passage explaining that recurrent models impose sequential computation and that the Transformer enables greater parallelisation.

This demonstrates the value of semantic retrieval for conceptual questions.

---

## Case Study — Multi-Head Attention

Query:

```text
How does multi-head attention work?
```

Dense retrieval ranked the direct explanatory passage first.

Hybrid retrieval promoted a closely related implementation/formula passage involving `MultiHead(Q, K, V)` and parallel attention heads.

This demonstrates that RRF can alter ranking order even when both branches retrieve relevant material.

Formal relevance evaluation is required before judging whether the reordering is better.

---

## Whole-Corpus Ambiguity

Query:

```text
What datasets were used to evaluate the model?
```

was executed with:

```text
document_id = null
```

Several indexed papers contain models and evaluation datasets.

Dense and lexical retrieval therefore interpreted the question differently.

Their top-five result sets had zero shared chunks for this query.

This is useful as a retrieval-diversity stress test, not as proof that either retriever was incorrect.

---

## Smoke-Test Latency

Observed mean request times:

```text
Dense      ≈ 28.3 ms
Lexical    ≈ 22.0 ms
Hybrid     ≈ 31.9 ms
```

The first dense request took approximately 79 ms.

Excluding that first request, the remaining dense requests averaged approximately 18.2 ms.

These values are integration diagnostics, not performance benchmarks.

---

# Phase 11 Findings

Phase 11 demonstrates that:

1. Dense and lexical retrieval provide different candidate signals.
2. BM25 can promote exact technical evidence that dense retrieval ranks lower.
3. Dense retrieval remains valuable for semantic paraphrases.
4. RRF combines both branches without comparing incompatible score scales.
5. Hybrid retrieval can change result ordering when branch rankings disagree.
6. All retrieval strategies preserve the same chunk provenance contract.
7. Retrieval mode can be selected through both the API and frontend.

---

# What Phase 11 Does Not Claim

Phase 11 does not claim that hybrid retrieval is universally better than dense or lexical retrieval.

It also does not claim that the following values are optimal:

```text
BM25 k1 = 1.5
BM25 b = 0.75
RRF k = 60
candidate multiplier = 2
top_k = 5
```

Those questions require labelled retrieval evaluation.

---

# Reusable Project Talking Points

## Short Explanation

> I extended a dense BGE/Qdrant RAG retriever with BM25 lexical search and selectable hybrid retrieval. Because cosine and BM25 scores use incompatible scales, I combined their ranks using Reciprocal Rank Fusion rather than averaging scores.

## Engineering Example

> One useful failure case was the query "Adam optimizer". Dense retrieval ranked the Adam bibliography reference first and the actual optimizer-method section second. BM25 promoted the method section to rank one, and hybrid retrieval preserved that promotion.

## Architecture Explanation

> Dense retrieval remains the default instead of being silently replaced. A retrieval router lets both search and RAG explicitly select dense, lexical, or hybrid retrieval while preserving the same downstream evidence and citation contracts.

## Evaluation Explanation

> I ran the same six-query set through dense, BM25, and hybrid retrieval and compared the candidate sets. Dense and lexical retrieval had about 0.214 average top-five Jaccard overlap, showing that they contributed meaningfully different candidates.

## RRF Explanation

> I used Reciprocal Rank Fusion because cosine similarity and BM25 relevance scores cannot be directly compared. Rank fusion avoids inventing a normalization scheme before there is evaluation data to justify one.

## Responsible Evaluation Explanation

> I deliberately did not claim hybrid retrieval was better after a six-query smoke test. The experiment established complementary behaviour, while formal accuracy claims are deferred until I can measure Recall@K, MRR, and nDCG against labelled evidence.

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
  Reranking
       ↓
Top Evidence
```

The Phase 11 retrieval baseline should remain frozen during the initial reranking comparison.

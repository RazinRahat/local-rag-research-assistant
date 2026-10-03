# Cross-Encoder Reranking

## Purpose

Phase 12 adds a second-stage cross-encoder reranker after the Phase 11 hybrid retrieval pipeline.

The objective is to separate:

```text
candidate discovery
```

from:

```text
candidate relevance scoring
```

The Phase 11 hybrid baseline remains unchanged so the effect of reranking can be inspected independently.

---

# Architecture

```text
Query
  │
  ▼
HybridRetriever
  │
  ├── Dense Retrieval
  │
  └── BM25 Retrieval
         │
         ▼
        RRF
         │
         ▼
  Top 20 Candidates
         │
         ▼
 Cross-Encoder
         │
         ▼
   Final Top 5
```

The reranking layer depends on the existing `Retriever` abstraction rather than embedding dense, BM25, Qdrant, or RRF logic directly.

---

# Reranker

Current baseline model:

```text
cross-encoder/ms-marco-MiniLM-L6-v2
```

The model scores query-passage pairs directly.

Unlike dense retrieval, where the query and passage are encoded independently, the cross-encoder processes the query and passage together.

This provides a stronger interaction signal at the cost of additional inference latency.

---

# Candidate Expansion

Current Phase 12 baseline:

```text
final top_k:          5
reranking pool:      20
```

The reranking retriever therefore asks the hybrid retriever for up to 20 candidates before reducing them to the requested final top-k.

Because the Phase 11 hybrid retriever itself uses:

```text
candidate_multiplier = 2
```

a final request for five reranked results approximately produces:

```text
Dense top 40
+
BM25 top 40
      ↓
     RRF
      ↓
Hybrid top 20
      ↓
Cross-Encoder
      ↓
Final top 5
```

The value `20` is an engineering baseline, not an experimentally established optimum.

---

# Retrieval Modes

The application now exposes four retrieval strategies:

```text
dense
lexical
hybrid
hybrid_reranked
```

Dense remains the default.

`hybrid` remains the frozen Phase 11 control.

`hybrid_reranked` explicitly adds cross-encoder rescoring rather than silently changing the meaning of hybrid retrieval.

The same mode is available through:

```text
POST /search
POST /query
React research workspace
```

---

# Score Semantics

`RetrievalResult.score` has mode-specific semantics:

```text
dense
→ cosine similarity

lexical
→ BM25 relevance

hybrid
→ Reciprocal Rank Fusion score

hybrid_reranked
→ cross-encoder relevance score
```

These values must not be compared numerically across retrieval modes.

Cross-encoder outputs are not treated as calibrated probabilities.

Negative reranking scores are valid.

---

# Testing

Phase 12 adds tests around:

```text
CrossEncoder configuration
pair scoring
reranking order
score validation
duplicate candidates
candidate expansion
final top-k reduction
Retriever compatibility
Hybrid → Reranking integration
retrieval routing
API mode validation
frontend mode forwarding
strategy-aware score presentation
```

The previous dense, lexical, and hybrid retrieval tests remain regression controls.

---

# Controlled Phase 12 Comparison

The reranking comparison reused the same six-query set introduced in Phase 11.

Configuration:

```text
queries:                  6
final top_k:              5
scope:                    all indexed documents
repetitions:              3
warm-up:                  enabled

Hybrid:
  RRF k:                  60
  candidate multiplier:    2

Reranking:
  model:
    cross-encoder/ms-marco-MiniLM-L6-v2
  candidate pool:         20
```

The experiment compared:

```text
hybrid
vs
hybrid_reranked
```

No relevance labels were used.

The experiment therefore measures ranking behaviour, candidate-set changes, and runtime diagnostics rather than retrieval accuracy.

---

# Aggregate Results

Measured results:

```text
Mean top-5 Jaccard overlap:       ≈ 0.409

Top-1 agreement:                  1 / 6
Top-1 agreement rate:             ≈ 16.7%

Mean hybrid median latency:       ≈ 35.1 ms
Mean reranked median latency:     ≈ 108.9 ms
Added end-to-end latency:         ≈ 73.8 ms
```

Reranking therefore substantially changed the final evidence set and ordering.

The reranked pipeline was approximately three times as slow as the normal hybrid path under this local Phase 12 configuration.

The additional latency includes both:

```text
larger hybrid candidate retrieval
+
cross-encoder inference
```

and should not be interpreted as pure cross-encoder execution time.

---

# Query-Level Findings

## Exact Technical Term — `Adam optimizer`

Hybrid retrieval ranked the Transformer passage containing the actual optimizer configuration first.

Cross-encoder reranking instead promoted a scaling-law passage containing the phrase `adam-optimized training runs` from hybrid rank 4 to reranked rank 1.

The directly relevant Transformer optimizer evidence no longer appeared at the top.

This is an important failure case.

It suggests that short or underspecified keyword-style queries can cause the current cross-encoder to reward lexical or topical compatibility without necessarily selecting the most useful passage for the user's likely information need.

---

## Rare Term — `label smoothing`

Both modes returned the same Transformer label-smoothing passage at rank 1.

Four of the five final chunks were shared.

```text
Top-5 Jaccard ≈ 0.667
Top-1 unchanged
```

The Phase 11 hybrid pipeline already handled this exact rare-term query effectively.

Reranking did not materially improve the top result.

---

## Semantic Paraphrase

Query:

```text
Why can the model process sequence positions in parallel?
```

Hybrid retrieval placed the direct discussion of recurrence and parallelisation at rank 2.

Cross-encoder reranking promoted that passage to rank 1.

The reranked top passage directly explains that recurrent computation is inherently sequential and that the Transformer avoids that constraint.

This is a qualitatively promising example of second-stage semantic reranking.

---

## Technical Concept

Query:

```text
How does multi-head attention work?
```

Hybrid retrieval ranked a formula-heavy multi-head-attention passage first.

Cross-encoder reranking promoted the explanatory section describing query/key/value projections, parallel attention heads, concatenation, and representation subspaces from hybrid rank 3 to reranked rank 1.

The result is qualitatively better aligned with an explanatory natural-language question.

However, the top-five candidate sets differed substantially:

```text
Top-5 Jaccard = 0.25
```

Formal relevance labels are still required before claiming a retrieval-quality improvement.

---

## Dataset Question

Query:

```text
What datasets were used to evaluate the model?
```

The top result changed between two different papers in the corpus.

The hybrid result prioritised a scaling-paper evaluation-dataset passage.

The reranked result prioritised a GPT-2 evaluation passage.

Because the query refers only to `the model` while retrieval is corpus-wide, the question itself is underspecified.

This demonstrates that retrieval errors can originate from query ambiguity rather than ranking alone.

Document-scoped evaluation should therefore be included in later experiments.

---

## Architecture / Recurrence

Query:

```text
Why does the Transformer avoid recurrent computation?
```

Hybrid retrieval ranked the direct Transformer passage explaining the sequential limitation of recurrent computation first.

Cross-encoder reranking moved that passage to rank 2 and promoted a later conclusion/table passage to rank 1.

```text
Top-5 Jaccard = 0.25
```

This is another useful reranking failure case.

The candidate remains relevant to the Transformer, but it is less directly explanatory than the Phase 11 hybrid top result.

---

# Interpretation

Phase 12 demonstrates that a cross-encoder is not simply a universally better final ranking layer.

Observed behaviour includes:

```text
semantic promotion
exact-term stability
candidate replacement
large rank movement
query-ambiguity sensitivity
and clear reranking regressions
```

The cross-encoder appears particularly promising for explanatory natural-language queries such as the semantic-paraphrase and multi-head-attention examples.

The same baseline can behave poorly for short technical keyword queries or where several papers contain related terminology.

---

# Latency Trade-Off

The Phase 12 pipeline increased mean median retrieval latency from approximately:

```text
35.1 ms
```

to:

```text
108.9 ms
```

for an added:

```text
73.8 ms
```

under the current local configuration.

This remains fast enough for interactive local use in the tested environment, but latency optimisation has not yet been performed.

Potential future variables include:

```text
candidate pool depth
cross-encoder model
batch size
device
document scope
hybrid branch depth
```

These should be tuned only after retrieval-quality metrics are available.

---

# What Phase 12 Establishes

Phase 12 establishes that:

```text
1. Cross-encoder reranking is integrated end to end.
2. Hybrid and reranked retrieval remain separately selectable.
3. Reranking materially changes final evidence selection.
4. The reranker can promote semantically direct passages.
5. The reranker can also produce meaningful regressions.
6. Candidate expansion introduces a measurable latency cost.
7. Score semantics remain explicit across retrieval modes.
8. Formal relevance-labelled evaluation is now necessary.
```

---

# What Phase 12 Does Not Claim

Phase 12 does not establish that:

```text
hybrid_reranked > hybrid
```

in retrieval accuracy.

It also does not establish that:

```text
candidate_pool_size = 20
```

is optimal.

The six-query comparison is a controlled engineering experiment, not a labelled benchmark.

---

# Reusable Project Talking Points

## Architecture

> I implemented a two-stage retrieval pipeline where dense and BM25 results are fused with Reciprocal Rank Fusion, expanded to a candidate pool, and then rescored using a local cross-encoder. I kept the original hybrid mode intact so reranking could be compared against a frozen control.

## Evaluation Discipline

> I deliberately did not call reranking an improvement after a qualitative smoke test. Across six fixed queries it changed five of six top results, but some changes were clearly useful while others were regressions. That became the motivation for building a labelled retrieval evaluation layer next.

## Useful Success Case

> For a semantic question about why sequence positions can be processed in parallel, hybrid retrieval placed the direct recurrence explanation second. The cross-encoder promoted it to first.

## Useful Failure Case

> For the short query `Adam optimizer`, the cross-encoder over-promoted a passage that merely discussed `adam-optimized training runs`, while hybrid retrieval had ranked the actual Transformer optimizer configuration first. That showed why a reranker still needs measured evaluation rather than blind trust.

## Performance Trade-Off

> Under the Phase 12 local configuration, median retrieval latency increased from roughly 35 ms for hybrid retrieval to 109 ms with reranking. The extra cost includes deeper candidate retrieval as well as cross-encoder inference.

---

# Next Step

Phase 13 introduces relevance-labelled evaluation.

The retrieval comparison will become:

```text
Dense
Lexical
Hybrid
Hybrid + Reranking
       ↓
labelled query/evidence relationships
       ↓
Recall@K
Precision@K
MRR
nDCG
```

This will determine whether observed rank changes actually improve retrieval quality rather than merely changing the ordering.
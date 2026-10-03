# Experiments

## Purpose

This document records controlled experiments and observations used to evaluate engineering decisions in the Local RAG Research Assistant.

The project intentionally separates implementation success from measured quality improvement.

A working feature is not automatically treated as a better feature.

---

# Current Baselines

## Chunking

```text
chunk size:      384 tokens
overlap:          64 tokens
```

## Dense Embeddings

```text
model: BAAI/bge-small-en-v1.5
dimension: 384
normalised: yes
```

## BM25

```text
k1: 1.5
b: 0.75
```

## Reciprocal Rank Fusion

```text
rank constant: 60
```

## Hybrid Retrieval

```text
candidate multiplier: 2
final top_k: 5
```

These values are baselines, not claimed optima.

---

# Phase 11 Retrieval Comparison

## Goal

The Phase 11 comparison asks:

```text
Do dense and lexical retrieval
actually produce different evidence?

Does RRF combine those signals
without silently replacing the dense baseline?
```

It does not attempt to establish final retrieval accuracy.

---

## Configuration

```text
queries:       6
top_k:         5
scope:         all indexed documents

modes:
  dense
  lexical
  hybrid
```

Query categories included:

- exact terminology
- rare terminology
- semantic paraphrase
- technical concept
- dataset/evaluation question
- architecture question

---

## Recorded Data

Each request records:

- retrieval mode
- query
- elapsed request time
- rank
- strategy-specific score
- document ID
- filename
- page number
- chunk ID
- chunk index
- chunk text
- token count

The comparison also calculates pairwise Jaccard overlap between top-k candidate sets.

---

# Result-Set Overlap

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

Interpretation:

```text
Dense and lexical retrieval are not
producing redundant candidate sets.

Hybrid retrieval incorporates evidence
from both branches.
```

These values measure overlap, not correctness.

---

# Query Observations

## Exact-Term Query

Query:

```text
Adam optimizer
```

Dense retrieval ranked a bibliography chunk containing the Adam paper first.

The actual Transformer optimizer-method section appeared second.

BM25 promoted the actual optimizer section to rank 1.

Hybrid retrieval also ranked the method section first.

This is a concrete example where lexical evidence changed ranking in a useful-looking way.

Formal evaluation is still required before generalising that observation.

---

## Rare-Term Query

Query:

```text
label smoothing
```

Dense, lexical, and hybrid retrieval all ranked the direct Transformer label-smoothing passage first.

Lower-ranked lexical results showed that simple BM25 can still score chunks that contain only one of the query terms.

This motivates later experiments involving:

- term coverage
- phrase matching
- lexical preprocessing
- learned sparse retrieval

No tuning is performed yet.

---

## Semantic-Paraphrase Query

Query:

```text
Why can the model process sequence positions in parallel?
```

Dense retrieval surfaced the Transformer passage explaining that recurrence requires sequential computation and that the Transformer enables greater parallelisation.

This demonstrates why the dense branch remains useful.

---

## Technical-Concept Query

Query:

```text
How does multi-head attention work?
```

Dense retrieval ranked the direct explanation first.

Hybrid retrieval promoted a closely related implementation/formula passage.

This demonstrates that fusion can materially alter ranking order.

It does not establish that the hybrid ranking is more relevant.

---

## Whole-Corpus Dataset Query

Query:

```text
What datasets were used to evaluate the model?
```

The request used:

```text
document_id = null
```

Multiple indexed papers contain different models and evaluation datasets.

Dense and lexical retrieval therefore returned completely different top-five candidate sets.

Their top-five overlap for this query was:

```text
0
```

This is useful as a candidate-diversity stress test.

The query itself is ambiguous at whole-corpus scope.

---

## Architecture Query

Query:

```text
Why does the Transformer avoid recurrent computation?
```

Dense retrieval returned the Transformer passage describing the sequential nature of recurrence and greater parallelisation through attention at rank 1.

Hybrid retrieval also retained strongly relevant Transformer evidence.

---

# Latency Observation

Observed mean request durations:

```text
Dense      ≈ 28.3 ms
Lexical    ≈ 22.0 ms
Hybrid     ≈ 31.9 ms
```

Median durations were approximately:

```text
Dense      ≈ 19.9 ms
Lexical    ≈ 21.8 ms
Hybrid     ≈ 31.8 ms
```

The first dense request took approximately 79 ms.

Excluding that request:

```text
Dense ≈ 18.2 ms average
```

These are diagnostic observations only.

The experiment did not include controlled warm-up, repeated trials, confidence intervals, load testing, machine isolation, or concurrency testing.

---

# Phase 11 Conclusions

The experiment supports the following observations:

1. Dense and lexical retrieval provide different candidate signals.
2. Exact terminology can benefit from BM25.
3. Semantic paraphrases continue to benefit from dense retrieval.
4. RRF combines ranked results without comparing incompatible raw score scales.
5. Hybrid retrieval changes ranking when dense and lexical branches disagree.
6. Hybrid retrieval preserves common evidence when the branches strongly agree.
7. The initial BM25 baseline has limitations for partial multi-term matches.
8. Whole-corpus queries can be ambiguous when the target paper/model is unspecified.

The experiment does not prove that hybrid retrieval has higher retrieval accuracy.

---

# Phase 12 — Cross-Encoder Reranking Comparison

Phase 12 compared the frozen Phase 11 hybrid baseline against a second-stage cross-encoder reranker.

## Configuration

```text
queries:                  6
scope:                    all indexed documents
final top_k:              5
repeats:                  3
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
hybrid_reranked
```

The six Phase 11 queries were reused without changing the query set.

---

## Aggregate Measurements

```text
Mean top-5 Jaccard:             ≈ 0.409
Top-1 agreement:                1 / 6
Top-1 agreement rate:           ≈ 16.7%

Hybrid median latency:          ≈ 35.1 ms
Reranked median latency:        ≈ 108.9 ms
Added end-to-end latency:       ≈ 73.8 ms
```

The latency difference includes both deeper hybrid candidate retrieval and cross-encoder inference.

It should not be interpreted as pure cross-encoder execution time.

---

## Query-Level Observations

### `Adam optimizer`

Hybrid retrieval ranked the Transformer passage containing the actual optimizer configuration first.

The cross-encoder instead promoted a scaling-law passage containing the phrase `adam-optimized training runs`.

That passage moved from hybrid rank 4 to reranked rank 1.

This is an important reranking failure case.

It suggests that a short technical keyword query can cause the current reranker to reward topical or lexical compatibility without necessarily selecting the most useful passage for the likely information need.

---

### `label smoothing`

Both modes returned the direct Transformer label-smoothing passage at rank 1.

Four of the five final chunks were shared.

```text
Top-5 Jaccard ≈ 0.667
Top-1 unchanged
```

The Phase 11 hybrid pipeline already handled this exact rare-term query effectively.

---

### Semantic Paraphrase

Query:

```text
Why can the model process sequence positions in parallel?
```

Hybrid retrieval placed the direct recurrence/parallelisation explanation at rank 2.

Cross-encoder reranking promoted that passage to rank 1.

This is a qualitatively promising example of second-stage semantic reranking.

---

### Multi-Head Attention

Query:

```text
How does multi-head attention work?
```

The reranker promoted the explanatory section describing query/key/value projections, parallel attention heads, concatenation, and representation subspaces to rank 1.

It had been hybrid rank 3.

The top-five candidate sets still differed substantially:

```text
Top-5 Jaccard = 0.25
```

This is a useful qualitative success case, but it is not yet a measured accuracy improvement.

---

### Dataset Query

Query:

```text
What datasets were used to evaluate the model?
```

The top result changed between different papers in the corpus.

Because the question refers only to `the model` while the experiment searches the whole corpus, the query is underspecified.

This shows that retrieval failures can originate from query ambiguity as well as ranking.

Document-scoped evaluation should therefore be included in later experiments.

---

### Architecture / Recurrence

Query:

```text
Why does the Transformer avoid recurrent computation?
```

Hybrid retrieval ranked the direct explanation of recurrent sequential computation first.

Cross-encoder reranking moved that passage to rank 2 and promoted a later conclusion/table passage to rank 1.

The top-five overlap was:

```text
0.25
```

This is another useful reranking failure case.

---

## Phase 12 Conclusion

The experiment shows that reranking materially changes final evidence selection.

Observed behaviours include:

- semantic promotion
- exact-term stability
- candidate replacement
- large rank movement
- sensitivity to ambiguous whole-corpus queries
- meaningful reranking regressions

The experiment does **not** establish that `hybrid_reranked` has higher retrieval accuracy than `hybrid`.

No relevance labels were used.

Phase 13 will introduce labelled query-to-evidence relationships so ranking changes can be measured using:

```text
Recall@K
Precision@K
MRR
nDCG
```

Raw generated reports remain local under:

```text
experiments/results/
```

The Phase 12 comparison utility is:

```text
scripts/compare_reranking_modes.py
```

---

# Formal Retrieval Evaluation

Phase 13 will introduce labelled query-to-evidence relationships.

Planned metrics:

```text
Recall@K
Precision@K
Mean Reciprocal Rank
nDCG
```

Possible experimental variables:

- dense vs lexical vs hybrid
- retrieval top-k
- candidate multiplier
- BM25 `k1`
- BM25 `b`
- RRF rank constant
- embedding model
- document scope
- cross-encoder reranking
- chunk size
- chunk overlap

---

# Context Experiments

Future context-construction comparisons may include:

- context size
- reserved generation budget
- token-estimation safety margin
- evidence ordering
- overlap-deduplication threshold
- evidence diversity
- neighbouring chunks
- whole chunks vs trimmed chunks
- context compression

---

# Generation Experiments

Potential generation comparisons include:

- local model size
- temperature
- top-p
- output-token budget
- context-window allocation
- prompt variants
- reasoning configuration

Generation experiments should use fixed retrieval evidence where possible so retrieval changes do not confound generation comparisons.

---

# Citation Evaluation

Citation evaluation should distinguish citation identity validity from semantic claim support.

Future evaluation should measure:

- citation protocol compliance
- unknown citation IDs
- missing citations
- citation count
- source diversity
- claim-level citation coverage
- semantic support
- unsupported claims
- citation correctness

---

# Reproducibility

Comparison utilities:

```text
scripts/compare_retrieval_modes.py
scripts/compare_reranking_modes.py
experiments/retrieval_queries.json
```

Raw generated reports are stored locally under:

```text
experiments/results/
```

Raw result dumps should generally not be committed because they may contain substantial retrieved source text and document identifiers.

Derived observations belong in this documentation instead.

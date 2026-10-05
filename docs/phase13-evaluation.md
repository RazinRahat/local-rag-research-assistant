# Phase 13 — Retrieval Evaluation and Evidence-Driven Tuning

## 13.1 Why Phase 13 existed

The project had reached a point where qualitative examples were no longer sufficient.

Questions such as these needed measured answers:

```text
Is Hybrid actually better than Dense or BM25?
Does reranking help more often than it hurts?
Which configuration improves retrieval without hiding regressions?
Do development gains survive on unseen queries?
```

Phase 13 introduced metric primitives, pooled candidate collection, blind relevance review, qrels, split-aware evaluation, regression diagnostics, offline tuning, live verification, and a final frozen holdout run.

---

## 13.2 Benchmark design

Three indexed research papers were used:

```text
Attention Is All You Need
Language Models are Unsupervised Multitask Learners
Scaling Laws for Neural Language Models
```

The expanded benchmark contained:

```text
18 document-scoped queries

12 development queries
  → available for diagnosis and tuning

6 holdout queries
  → sealed until retrieval configuration was frozen
```

Negative/out-of-scope queries were intentionally deferred to a later RAG abstention evaluation because retrieval relevance and answer abstention are different evaluation problems.

---

## 13.3 Pool, review, trace, and qrels

Phase 13 explicitly separated four artifacts:

```text
Pool
→ union of candidate passages retrieved by the evaluated systems

Review
→ blind relevance assignment; system identity/rank hidden

Trace
→ diagnostic system rankings and scores

Qrels
→ final query-to-chunk relevance labels used by the evaluator
```

The relevance review used a 0–3 scale:

```text
0 = irrelevant
1 = marginal / context-only
2 = useful relevant evidence
3 = direct answer evidence
```

Final expanded qrels contained:

```text
queries:                 18
judgments:              527
non-zero judgments:      72
binary-relevant (>=2):   53
```

The review was model-assisted and blind to retrieval ranks/system identity. It should not be described as an independently human-labelled gold standard.

The evaluator uses `relevance >= 2` for binary metrics and the full 0–3 grades for nDCG.

Because qrels came from pooled retrieved candidates rather than exhaustive annotation of every corpus chunk, reported Recall@5 is **pooled recall**.

---

## 13.4 Metrics

Formal retrieval evaluation uses:

```text
Precision@5
Recall@5
MRR
nDCG@5
```

Each retrieval mode retains its own score semantics:

```text
Dense               → cosine similarity
Lexical             → BM25 relevance score
Hybrid              → weighted RRF score
Hybrid + Reranking  → cross-encoder relevance score
```

Those raw scores are not compared across modes.

---

## 13.5 Initial expanded development benchmark

The first 12-query development run produced:

| Mode | P@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| Dense | 0.2500 | 0.4514 | 0.5556 | 0.4551 |
| Lexical | 0.3667 | 0.6319 | **0.7222** | **0.6264** |
| Hybrid | 0.3500 | 0.5903 | 0.5306 | 0.4951 |
| Hybrid + Reranking | 0.3167 | 0.5625 | 0.5958 | 0.4801 |

This was the first major Phase 13 finding:

> More retrieval stages did not automatically produce better retrieval.

BM25 was the strongest aggregate development baseline. Hybrid underperformed Lexical, and the initial reranker reduced Precision, Recall, and nDCG relative to Hybrid.

That result changed the project from feature-building into diagnosis.

---

# Phase 13.8A — Diagnosing and Tuning Hybrid Fusion

## Problem

The original Hybrid configuration used standard equal-weight RRF with `k=60`:

```text
Dense contribution   = 1 / (60 + rank)
Lexical contribution = 1 / (60 + rank)
```

With such a large constant, early ranks are relatively flat. For example:

```text
rank 1  → 1/61
rank 10 → 1/70
```

A rank-1 result contributes only modestly more than a rank-10 result.

This mattered because the expanded development set showed several cases where BM25 had very strong exact-match evidence that Hybrid did not promote enough.

## Offline sweep

The development-only trace was used to simulate 84 RRF configurations across:

```text
branch depth
RRF rank constant
lexical weight
```

Dense weight remained fixed at 1.0.

The simulator first reproduced the live `k=60`, equal-weight Hybrid rankings for all 12 development queries, giving confidence that the offline model matched production behaviour.

## Highest aggregate configuration was rejected

The highest aggregate nDCG configuration was:

```text
branch depth = 20
k = 1
dense weight = 1.0
lexical weight = 3.0
```

It achieved approximately:

```text
P@5      0.3833
Recall   0.6806
MRR      0.7389
nDCG     0.6380
```

But query-level diagnostics showed:

```text
7 improvements
3 degradations
2 mixed
```

The project deliberately rejected this configuration because maximum mean score was not enough to justify several regressions.

## Selected zero-regression configuration

The selected configuration was:

```text
branch depth = 10
k = 5
dense weight = 1.0
lexical weight = 1.25
```

Offline development result:

```text
P@5      0.3833
Recall   0.6736
MRR      0.5750
nDCG     0.5619

4 improvements
0 degradations
0 mixed
8 unchanged
```

This was chosen because it was the strongest nDCG result among configurations with no diagnostic regressions.

## Live verification

The production implementation reproduced those aggregate values exactly.

Hybrid improved from the initial development benchmark by:

| Metric | Before | After | Relative change |
|---|---:|---:|---:|
| P@5 | 0.3500 | 0.3833 | +9.5% |
| Recall@5 | 0.5903 | 0.6736 | +14.1% |
| MRR | 0.5306 | 0.5750 | +8.4% |
| nDCG@5 | 0.4951 | 0.5619 | +13.5% |

A concrete example was the Transformer optimizer/schedule query: the original Hybrid top five contained no binary-relevant passage, while the tuned configuration brought the direct optimizer passage into rank 5.

---

# Phase 13.8B — Reranking Diagnosis

## New problem after Hybrid improved

After weighted-RRF tuning, the reranker improved automatically because it received a better first-stage ranking, but the aggregate relationship still looked like this:

```text
Hybrid
P@5      0.3833
Recall   0.6736
MRR      0.5750
nDCG     0.5619

Hybrid + Reranking
P@5      0.3333
Recall   0.6042
MRR      0.6792
nDCG     0.5312
```

The reranker raised MRR but reduced Precision, Recall, and nDCG.

The interpretation was useful:

> The cross-encoder was good at moving a relevant passage earlier, but sometimes too aggressive when allowed to reorder a deep candidate set.

## Dev-only trace and pool sweep

A post-RRF development trace confirmed that all current top-20 Hybrid and reranked candidates already had qrels, so no new review cycle was required.

An offline sweep simulated reranking only the first `N` candidates from a fixed Hybrid top-20 ranking.

The strongest conservative setting was:

```text
retrieve Hybrid top 20
rerank only Hybrid top 10
return final top 5
```

Offline prediction:

```text
P@5      0.3833
Recall   0.7153
MRR      0.6903
nDCG     0.5987

5 diagnostic improvements
0 degradations
7 unchanged
```

---

# Failed Implementation: Coupled Candidate Depth

The first implementation changed:

```text
candidate_pool_size: 20 → 10
```

At first glance that looked equivalent to “rerank only ten candidates”. It was not.

Because `RerankingRetriever` asks Hybrid for `candidate_pool_size`, and Hybrid itself uses `candidate_multiplier=2`, the nested pipeline changed from:

```text
candidate_pool_size = 20
→ Hybrid top 20
→ Dense top 40 + BM25 top 40
```

to:

```text
candidate_pool_size = 10
→ Hybrid top 10
→ Dense top 20 + BM25 top 20
```

So one configuration value was controlling two different concepts:

```text
first-stage candidate generation depth
and
second-stage cross-encoder depth
```

The live development result still improved over the previous reranker:

```text
P@5      0.3500
Recall   0.6458
MRR      0.6903
nDCG     0.5690
```

But it did not reproduce the offline prediction, and the GPT-2 model-scale query lost useful evidence.

This failure was retained because it exposed an architectural coupling rather than just a bad hyperparameter.

---

# Corrected Architecture: Decoupled Retrieval and Rerank Depth

The fix introduced two explicit controls:

```text
candidate_pool_size = 20
rerank_pool_size    = 10
```

The final nested pipeline became:

```text
Dense top 40
     +
BM25 top 40
     ↓
weighted RRF
     ↓
Hybrid top 20
     ↓
retain Hybrid top 10
     ↓
CrossEncoder scores 10
     ↓
final top 5
```

This implementation reproduced the intended offline prediction exactly.

---

# Final Development Result

| Mode | P@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| Dense | 0.2500 | 0.4514 | 0.5556 | 0.4551 |
| Lexical | 0.3667 | 0.6319 | **0.7222** | **0.6264** |
| Hybrid | **0.3833** | 0.6736 | 0.5750 | 0.5619 |
| Hybrid + Reranking | **0.3833** | **0.7153** | 0.6903 | 0.5987 |

Relative to the initial expanded development reranker:

```text
P@5      +21.1%
Recall   +27.2%
MRR      +15.9%
nDCG     +24.7%
```

A combined metric/top-1 diagnostic comparison between the initial and final reranked development systems showed:

```text
5 improvements
0 degradations
7 unchanged
```

The final development result still had an important trade-off: BM25 retained the highest MRR and nDCG, while the reranked pipeline had the highest Recall and joint-best Precision.

That trade-off was accepted instead of continuing to tune on the same 12 queries.

---

# Phase 13.9 — Frozen Holdout Evaluation

After tuning stopped, the configuration was frozen:

```text
Hybrid
  candidate_multiplier = 2
  RRF k = 5
  dense weight = 1.0
  lexical weight = 1.25

Reranking
  candidate_pool_size = 20
  rerank_pool_size = 10
  model = cross-encoder/ms-marco-MiniLM-L6-v2

Evaluation
  top_k = 5
  relevance threshold = 2
```

The six holdout queries were then evaluated without additional parameter tuning.

## Holdout aggregate

| Mode | P@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| Dense | 0.4000 | 0.7667 | 0.4778 | 0.4806 |
| Lexical | 0.3667 | 0.6917 | 0.6806 | 0.5083 |
| Hybrid | 0.3667 | 0.6917 | 0.6667 | 0.5397 |
| **Hybrid + Reranking** | **0.4667** | **0.8500** | **0.9167** | **0.7284** |

On this six-query holdout, Hybrid + Reranking was strongest on all four aggregate metrics.

Relative to Hybrid on the same holdout:

```text
P@5      +27.3%
Recall   +22.9%
MRR      +37.5%
nDCG     +35.0%
```

Query-level `hybrid → hybrid_reranked` outcomes were:

```text
4 improvements
1 no change
1 degradation
```

The stronger holdout score should not be interpreted as proof of broad generalisation. Six holdout queries remain a small sample.

---

# Holdout Success Cases

## GPT-2 translation

Plain Hybrid failed to retrieve either binary-relevant WMT-14 French-English result in its top five:

```text
P@5      0.00
Recall   0.00
MRR      0.00
nDCG     0.030
```

The reranker recovered both passages, including the 11.5 BLEU result:

```text
P@5      0.40
Recall   1.00
MRR      0.50
nDCG     0.614
```

## Positional embedding comparison

Hybrid already retrieved both relevant passages, but ranked a context-only passage first.

The reranker moved the two direct result passages to ranks 1 and 2:

```text
Hybrid nDCG          0.708
Reranked nDCG        1.000
Hybrid MRR           0.500
Reranked MRR         1.000
```

## Scaling-law overfitting relation

Reranking improved:

```text
Recall   0.75 → 1.00
nDCG     0.568 → 0.841
```

## Compute allocation

Reranking improved:

```text
Recall   0.40 → 0.60
MRR      0.50 → 1.00
nDCG     0.415 → 0.692
```

---

# Holdout Failure Case — Natural Questions

The Natural Questions query is deliberately retained as the main frozen holdout failure case.

Hybrid retrieved both relevant passages:

```text
P@5      0.40
Recall   1.00
MRR      1.00
nDCG     0.637
```

One passage contained the complete headline result: 4.1% exact-match accuracy, comparison to the 1% baseline, and 63.1% accuracy among the most-confident 1% of answers.

After reranking:

```text
P@5      0.20
Recall   0.50
MRR      1.00
nDCG     0.343
```

The cross-encoder kept a partially relevant Natural Questions passage at rank 1 but pushed the more complete 4.1% result outside the final top five.

This is evidence that the final reranker is not uniformly better at every query. It also demonstrates why holdout failures must be documented rather than tuned away after inspection.

---

# What Phase 13 Established

Phase 13 supports these claims:

1. Dense, lexical, and reranked retrieval have meaningfully different strengths.
2. Equal-weight RRF with `k=60` was not the best development configuration for this benchmark.
3. A modest lexical weight and smaller RRF constant improved Hybrid without development regressions in the selected tuning diagnostic.
4. Cross-encoder reranking benefits from strong first-stage candidate generation but can become too aggressive over deeper candidate sets.
5. First-stage candidate depth and reranker scoring depth should be represented as separate configuration concepts.
6. The final frozen reranked system transferred successfully to the six-query holdout and led aggregate holdout metrics.
7. The final system still has query-level failure cases.

Phase 13 does **not** establish:

- exhaustive corpus recall
- universal superiority of the chosen parameters
- independent human-labelled ground truth
- answer faithfulness
- claim-level citation entailment
- abstention quality on out-of-scope questions
- production-scale latency or throughput
- broad generalisation beyond this small three-paper benchmark

---

# Reproducibility and Experiment Artifacts

Important local artifacts:

```text
relevance_judgments_phase13_expansion.json
relevance_pool_trace_20261005T025115Z.json
relevance_pool_trace_20261005T153632Z.json

phase13_rrf_grid.csv
phase13_rrf_offline_tuning_report.json
phase13_reranker_pool_sweep.csv
phase13_reranker_pool_sweep_report.json

retrieval_evaluation_dev_20261005T092458Z.json
  → initial expanded development baseline

retrieval_evaluation_dev_20261005T150647Z.json
  → weighted-RRF verification, old reranker depth

retrieval_evaluation_dev_20261005T161520Z.json
  → failed coupled-depth implementation

retrieval_evaluation_dev_20261005T164314Z.json
  → final frozen development configuration

retrieval_evaluation_holdout_20261005T164733Z.json
  → one-time frozen holdout result
```

Raw result dumps should remain local because they contain retrieved source text and document identifiers. Derived metrics, decisions, failures, and configuration belong in committed documentation.

---

# Future Evaluation

Retrieval evaluation is now a foundation for later experiments rather than an excuse to keep tuning the same development set.

Future work should expand into:

- a larger multi-paper benchmark
- negative/out-of-scope queries
- abstention quality
- answer faithfulness
- answer completeness
- context relevance
- claim-level citation support
- unsupported-claim detection
- citation coverage/correctness
- controlled latency/resource benchmarking
- broader embedding/BM25/chunking comparisons on new evaluation data

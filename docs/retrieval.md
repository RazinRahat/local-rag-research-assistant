# Semantic Retrieval

## Purpose

The retrieval layer converts a natural-language question into ranked evidence from the indexed research corpus.

Before this stage, the system could:

```text
PDF
 ↓
Parse
 ↓
Chunk
 ↓
Embed
 ↓
Persist
```

but it could not yet search those stored representations in a reusable application-level way.

Phase 5 adds:

```text
Natural-Language Query
        ↓
Query Embedding
        ↓
Vector Search
        ↓
Top-K Matches
        ↓
RetrievalResult[]
```

This is the first point where the system functions as a semantic research search engine rather than only an indexing pipeline.

---

## Retrieval Architecture

The retrieval path is deliberately split across three responsibilities.

```text
SemanticRetriever
       │
       │ natural-language query
       ▼
EmbeddingProvider
       │
       │ query vector
       ▼
VectorStore
       │
       │ nearest stored vectors
       ▼
VectorSearchResult[]
       │
       ▼
SemanticRetriever
       │
       ▼
RetrievalResult[]
```

Each layer has a distinct role.

### Embedding Provider

Responsible for converting the query into the same semantic vector space used for document chunks.

```text
"What datasets were used?"
        ↓
BGE query embedding
        ↓
384-dimensional vector
```

### Vector Store

Responsible for nearest-neighbour search over persisted vectors.

The vector-store interface does not know how a query was generated.

It only receives:

```text
query vector
top-k
optional threshold
optional document filter
```

### Semantic Retriever

Coordinates query embedding and vector search and converts the storage results into ranked retrieval-domain objects.

---

## Why Retrieval Is a Separate Layer

The project does not call Qdrant directly from FastAPI or future generation code.

Instead:

```text
future API / RAG service
        ↓
SemanticRetriever
        ↓
VectorStore
        ↓
Qdrant
```

This avoids leaking database-specific APIs into the rest of the application.

It also means future changes such as:

* another vector database
* hybrid search
* reranking
* query rewriting
* metadata-aware retrieval

can be introduced without redesigning the entire application.

---

## Query Embeddings

Queries are embedded using the same embedding provider used for document vectors.

The baseline model is:

```text
BAAI/bge-small-en-v1.5
```

Document passages and queries are handled through separate provider methods:

```text
embed_documents()
embed_query()
```

This leaves room for retrieval models that apply different instructions or encoding behaviour to queries and passages.

The current query vector is normalised before search.

---

## Semantic Similarity

The Qdrant collection uses cosine distance.

Because the BGE embeddings are normalised:

```text
||query|| ≈ 1
||document|| ≈ 1
```

semantic similarity is closely related to the dot product used during the earlier NumPy smoke tests.

Conceptually:

```text
query vector
      │
      ├── Chunk A → high similarity
      ├── Chunk B → medium similarity
      └── Chunk C → low similarity
```

Higher-ranking chunks should represent text that is semantically closer to the query.

---

## Top-K Retrieval

The retriever supports configurable `top_k`.

Example:

```text
Corpus size: 4,000 chunks
top_k: 5
```

The retrieval layer returns only the five highest-ranked matches.

Conceptually:

```text
1. Chunk 843   score 0.82
2. Chunk 104   score 0.79
3. Chunk 991   score 0.75
4. Chunk 622   score 0.71
5. Chunk 301   score 0.68
```

`top_k` is intentionally configurable because it creates a trade-off.

Too low:

```text
relevant evidence may be missed
```

Too high:

```text
weak or irrelevant evidence may be introduced
```

The current default is a baseline rather than an experimentally established optimum.

---

## Retrieval Result Model

The retrieval domain returns typed results.

Conceptually:

```text
RetrievalResult
├── rank
├── score
└── chunk
    ├── chunk_id
    ├── document_id
    ├── file_name
    ├── page_number
    ├── text
    └── chunk metadata
```

The result therefore contains both:

```text
semantic relevance
```

and:

```text
source provenance
```

This is critical for later grounded generation and citation construction.

---

## Provenance Reconstruction

Stored Qdrant points contain payload metadata.

During retrieval, the Qdrant adapter reconstructs the original `DocumentChunk`.

The provenance chain is therefore:

```text
Qdrant match
     ↓
payload
     ↓
DocumentChunk
     ↓
page number
     ↓
source document
```

Later, a generated answer can cite evidence such as:

```text
paper.pdf, p. 7
```

without having to guess where a retrieved passage came from.

---

## Storage Boundary Validation

Qdrant payloads are treated as external persistence data rather than trusted domain objects.

The adapter explicitly validates required payload fields before constructing a `DocumentChunk`.

This prevents storage-specific data from silently bypassing application invariants.

Conceptually:

```text
Qdrant payload
      ↓
validate fields
      ↓
DocumentChunk
```

rather than:

```text
arbitrary payload
      ↓
application
```

---

## Document-Scoped Retrieval

The retrieval layer supports optional filtering by `document_id`.

Global search:

```text
query
 ↓
all indexed documents
```

Document-scoped search:

```text
query
 ↓
document_id filter
 ↓
one research paper
```

This will later support workflows such as:

```text
Search all papers
```

versus:

```text
Ask only about this paper
```

without creating a separate vector collection for every document.

---

## Score Thresholds

The vector-store search interface supports an optional similarity threshold.

Conceptually:

```text
Top result     0.83
Second         0.76
Third          0.62
Fourth         0.41
```

A threshold could remove weaker results.

However, the system currently does not define a default threshold.

A value such as:

```text
0.5
```

cannot automatically be treated as universally meaningful.

Useful thresholds depend on factors such as:

* embedding model
* query distribution
* corpus
* chunking strategy
* retrieval objective

Threshold selection should therefore be evaluated empirically rather than chosen by intuition.

---

## Test Strategy

The retrieval layer is tested at two levels.

### Vector-Store Search Tests

In-memory Qdrant is populated with deliberately simple vectors such as:

```text
(1, 0, 0)
(0, 1, 0)
(0, 0, 1)
```

A query such as:

```text
(1, 0, 0)
```

should rank the first vector highest.

This tests actual Qdrant search behaviour without involving a language model.

### Retriever Unit Tests

The semantic retriever uses fake implementations of:

```text
EmbeddingProvider
VectorStore
```

This verifies:

* query validation
* top-k handling
* result ranking
* domain-model conversion

without loading BGE or Qdrant.

This separation keeps normal tests fast and deterministic.

---

## Real-Model Smoke Testing

The retrieval pipeline is also tested with:

```text
BGE query embedding
+
persistent Qdrant vectors
+
real research-paper chunks
```

Example questions include:

```text
How does multi-head attention work?

Why can Transformers process sequence positions in parallel?

What positional encoding is used?

What datasets were used to evaluate the model?
```

The returned passages are manually inspected for semantic relevance.

This confirms end-to-end behaviour but is not yet treated as formal retrieval evaluation.

---

## Comparison With Manual Similarity Search

During the embedding phase, semantic similarity was calculated directly in NumPy:

```text
query_vector · chunk_vector
```

Phase 5 moves that search into Qdrant.

Comparing both approaches provides a useful sanity check:

```text
same query
+
same embeddings
+
same similarity geometry
```

should produce broadly consistent top results.

This helps verify that persistence and database querying have not changed the expected semantic behaviour.

---

## Dense Retrieval vs Keyword Retrieval

The current system implements dense semantic retrieval.

Dense retrieval is useful because it can match semantically related text even when exact terminology differs.

For example:

```text
Query:
"Why are recurrent neural networks unnecessary?"

Document:
"The model dispenses entirely with recurrence..."
```

A semantic embedding model can recognise the relationship even when literal word overlap is limited.

However, dense retrieval can be weaker for:

* exact identifiers
* unusual acronyms
* model names
* equations
* rare terminology
* exact lexical matches

This is one reason hybrid retrieval is planned later.

---

## Current Limitations

The current retrieval system does not yet include:

* BM25
* sparse vectors
* hybrid retrieval
* reciprocal-rank fusion
* reranking
* query expansion
* query rewriting
* HyDE
* parent-child retrieval
* metadata filters beyond document identity
* empirically tuned score thresholds
* labelled retrieval evaluation
* diversification of highly similar chunks

These limitations are intentional.

The dense retriever serves as a clear baseline that future retrieval strategies can be compared against.

---

## Future Retrieval Architecture

The planned retrieval stack will evolve toward:

```text
                        Query
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
       Dense Retrieval             BM25 Search
             │                         │
             └────────────┬────────────┘
                          ▼
               Reciprocal Rank Fusion
                          │
                          ▼
                   Candidate Set
                          │
                          ▼
                   Cross-Encoder
                     Reranking
                          │
                          ▼
                   Top Evidence
```

The current semantic retriever forms the dense branch of that future architecture.

---

## Evaluation Direction

Formal retrieval evaluation will be introduced later using labelled question-to-evidence relationships.

Planned metrics include:

```text
Recall@K
Precision@K
MRR
nDCG
```

This will allow decisions about:

* chunk size
* overlap
* embedding model
* top-k
* score threshold
* hybrid retrieval
* reranking

to be based on measured retrieval performance.

---

## Next Step

The next major stage introduces local language generation.

The architecture will then become:

```text
Question
   ↓
Semantic Retrieval
   ↓
Retrieved Evidence
   ↓
Context Construction
   ↓
Local LLM
   ↓
Generated Answer
```

This will be the transition from standalone retrieval into a complete Retrieval-Augmented Generation pipeline.

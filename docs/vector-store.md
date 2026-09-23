# Vector Storage

## Purpose

The vector-store layer provides persistent storage for document embeddings and the metadata required to reconstruct their original source.

Before this stage, embeddings existed only as Python objects:

```text
DocumentChunk
     ↓
Embedding Model
     ↓
ChunkEmbedding
     ↓
Python memory
```

Once the process exited, those embeddings disappeared.

The vector-store layer changes the pipeline to:

```text
DocumentChunk
     ↓
Embedding Model
     ↓
ChunkEmbedding
     ↓
Qdrant
     ↓
Persistent vector storage
```

This allows indexed research documents to remain searchable across application restarts.

---

## Why Qdrant

Qdrant is used as the baseline vector database because it provides:

* persistent vector storage
* vector similarity search
* metadata payloads
* payload filtering
* configurable distance metrics
* collection-level vector configuration
* deterministic point updates
* local development mode
* a migration path to a standalone server deployment

The project currently uses Qdrant's local persistent mode so vector storage can be developed without introducing Docker or an external service dependency.

Later, the same application layer can be adapted to communicate with a standalone Qdrant service.

---

## Storage Model

Qdrant stores vectors as points.

Conceptually:

```text
Point
│
├── id
├── vector
└── payload
```

For this project, one point represents one document chunk.

```text
Qdrant Point
│
├── id
│   └── deterministic UUID
│
├── vector
│   └── 384-dimensional BGE embedding
│
└── payload
    ├── chunk_id
    ├── document_id
    ├── file_name
    ├── page_number
    ├── chunk_index
    ├── page_chunk_index
    ├── text
    ├── token_count
    ├── char_count
    ├── token_start
    ├── token_end
    └── embedding_model
```

The vector represents semantic meaning.

The payload preserves provenance and the original retrieval text.

---

## Collection Design

The current collection is configured for the baseline embedding model:

```text
BAAI/bge-small-en-v1.5
```

which produces:

```text
384-dimensional vectors
```

The collection uses cosine distance.

Conceptually:

```text
Collection
├── vector size: 384
├── distance: cosine
└── points:
    ├── chunk A
    ├── chunk B
    ├── chunk C
    └── ...
```

The collection name includes the embedding-model family to reduce the risk of accidentally mixing incompatible vector spaces.

Example:

```text
research_chunks_bge_small_en_v1_5
```

---

## Why Embedding Models Must Not Be Mixed

Two embedding models may both produce 384-dimensional vectors while still representing entirely different semantic spaces.

For example:

```text
Model A
→ 384 dimensions

Model B
→ 384 dimensions
```

does not mean the resulting coordinates are compatible.

Therefore the vector-store layer validates both:

```text
embedding dimension
```

and:

```text
embedding model identity
```

before accepting a batch.

This prevents semantically incompatible vectors from being stored in the same collection.

---

## Distance Metric

The embedding layer produces normalised vectors:

```text
||v|| ≈ 1
```

The vector store therefore uses cosine distance to remain consistent with the retrieval geometry established during the embedding phase.

For normalised vectors, cosine similarity is closely related to the dot product used during the earlier in-memory retrieval smoke test.

---

## Vector Store Abstraction

Application code does not depend directly on Qdrant operations.

Instead, the project defines a vector-store interface with operations such as:

```text
ensure_collection()
replace_document()
delete_document()
count_document()
count()
close()
```

Conceptually:

```text
VectorStore
     │
     └── QdrantVectorStore
```

This keeps storage-specific code isolated from ingestion, embeddings, retrieval, and later API layers.

A different persistence implementation could therefore be introduced without rewriting the full RAG pipeline.

---

## Point Identity

Document chunks already have deterministic SHA-256-based chunk identifiers.

However, Qdrant point identifiers use supported point-ID formats such as UUIDs or integer IDs.

The project therefore converts each deterministic chunk identifier into a deterministic UUID.

```text
chunk_id
   ↓
UUID5
   ↓
Qdrant point ID
```

The original `chunk_id` is still retained in the payload.

This provides two useful identities:

```text
Qdrant point ID
    → database-compatible identifier

chunk_id
    → application-level chunk identity
```

Because the UUID conversion is deterministic, the same chunk produces the same point identifier across runs.

---

## Provenance Preservation

The storage layer preserves the provenance chain established during ingestion and chunking.

```text
Vector
  ↓
Qdrant Point
  ↓
Payload
  ↓
Chunk
  ↓
Page
  ↓
Document
```

A retrieved point can therefore later reconstruct:

* source document
* filename
* page number
* chunk identity
* chunk position
* original text

This forms the basis of the future citation system.

---

## Why Store Chunk Text in Qdrant

The original chunk text is stored as part of the Qdrant payload.

Without this, retrieval would return a vector identifier and require another storage system to recover the corresponding evidence.

Storing the text directly allows:

```text
nearest vector
      ↓
Qdrant point
      ↓
payload["text"]
      ↓
retrieved evidence
```

This simplifies the early architecture while maintaining traceability.

For very large-scale systems, separating vector storage and document storage could later be reconsidered.

---

## Document-Level Filtering

Every stored point includes:

```text
document_id
```

This allows all chunks belonging to the same source document to be selected using metadata filters.

Conceptually:

```text
Collection

Point A → document_id = X
Point B → document_id = X
Point C → document_id = Y
Point D → document_id = X
Point E → document_id = Y
```

A document filter can logically isolate:

```text
document_id = X
```

without requiring separate collections for every document.

This will later support:

* document deletion
* document-specific retrieval
* re-indexing
* metadata-constrained search

---

## Re-Indexing Strategy

The current application uses document replacement rather than blind repeated upserts.

Suppose the original indexing produces:

```text
Document A
├── Chunk 0
├── Chunk 1
├── Chunk 2
├── Chunk 3
└── Chunk 4
```

A later chunking configuration may produce only:

```text
Document A
├── Chunk 0
├── Chunk 1
└── Chunk 2
```

Simply inserting the new chunks could leave stale chunks 3 and 4 behind.

The current strategy is therefore:

```text
replace_document()

existing document vectors
        ↓
delete by document_id
        ↓
insert current vector set
```

This ensures the stored representation matches the current indexing configuration.

---

## Idempotent Indexing

Re-indexing the same document should not continuously increase the number of stored vectors.

For example:

```text
first indexing
→ 18 chunks

second indexing
→ 18 chunks

stored total
→ 18
```

not:

```text
36
```

Document-level replacement and deterministic identifiers make indexing predictable and repeatable.

---

## Current Transaction Limitation

The initial document replacement operation performs:

```text
delete existing document
        ↓
insert current document
```

as two separate operations.

This means a failure between deletion and insertion could temporarily leave the document absent from the store.

For the current local development environment, this is an acceptable trade-off.

A more production-oriented design could later use:

* document versions
* staging collections
* active-version metadata
* atomic switching strategies

This limitation is documented rather than hidden.

---

## Local Persistence

Development storage currently lives under:

```text
data/vector_store/qdrant/
```

The store survives application restarts.

Conceptually:

```text
Process A
   ↓
write embeddings
   ↓
close application
   ↓
disk
   ↓
Process B
   ↓
open same Qdrant path
   ↓
embeddings still available
```

This is the main capability introduced during the vector-storage phase.

---

## Test Strategy

Tests use Qdrant's in-memory mode instead of the persistent development database.

```text
Application development
→ persistent local Qdrant

Automated tests
→ in-memory Qdrant
```

This keeps tests:

* isolated
* fast
* reproducible
* independent of existing local data
* free from filesystem cleanup requirements

A separate persistence test verifies that a filesystem-backed store survives closing and reopening.

---

## Privacy Considerations

The local Qdrant database contains more than numerical vectors.

It also stores payload data such as:

```text
source text
filename
page number
document identity
```

Therefore the vector-store directory may contain sensitive or copyrighted research content.

The directory is intentionally excluded from Git.

```text
data/vector_store/
```

should be treated as document-derived private data, not merely as disposable application cache.

---

## Local Mode vs Server Mode

The current architecture uses Qdrant local mode for development convenience.

```text
Python Application
       ↓
Local Qdrant
```

A later production-style setup may use:

```text
Python Application
       ↓
HTTP / gRPC
       ↓
Qdrant Server
```

Potential benefits of server mode include:

* independent process lifecycle
* larger datasets
* production indexing
* operational monitoring
* networked access
* scalable deployment

Because the rest of the project communicates through the `VectorStore` abstraction, this migration should not require changes to higher-level RAG logic.

---

## Current Limitations

The current vector-store layer does not yet implement:

* semantic search
* similarity thresholds
* top-k retrieval
* metadata-constrained search APIs
* named vectors
* sparse vectors
* hybrid retrieval
* payload indexing optimisation
* collection migrations
* transactional document replacement
* remote Qdrant deployment

These belong to later retrieval and infrastructure phases.

---

## Next Step

The next stage is semantic retrieval.

The system will transform:

```text
User Query
    ↓
Query Embedding
    ↓
Qdrant Search
    ↓
Top-K Vector Matches
    ↓
Typed Retrieval Results
```

The retrieval layer will reconstruct source information from stored payloads and provide the evidence required by the later RAG generation pipeline.

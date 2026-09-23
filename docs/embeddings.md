Embeddings
Purpose

Embeddings transform text into dense numerical vectors representing learned semantic relationships.

The embedding stage converts:

DocumentChunk
      ↓
Embedding Model
      ↓
ChunkEmbedding

A query can later be embedded into the same vector space and compared against document vectors.

Baseline Model

The current baseline model is:

BAAI/bge-small-en-v1.5

The system currently uses 384-dimensional normalised vectors.

The model is accessed locally through Sentence Transformers.

Semantic Representation

An embedding does not assign individual dimensions simple human-readable meanings.

Instead, semantically related passages should occupy nearby areas of the learned vector space.

Conceptually:

"Adam was used for optimisation"
            ●

"Training used the Adam optimizer"
          ●


"Plants convert sunlight into energy"
                                  ●

Dense retrieval later searches for document vectors close to the query vector.

Query vs Document Embeddings

The embedding layer distinguishes between:

document passages

and:

retrieval queries

The provider therefore exposes separate methods:

embed_documents()
embed_query()

This leaves room for models that use different query instructions or prompting strategies for retrieval.

Embedding Provider Interface

Application code does not depend directly on Sentence Transformers.

Conceptually:

EmbeddingProvider
        │
        └── SentenceTransformerEmbedder

Future providers could be introduced without rewriting the rest of the RAG pipeline.

Model-Aware Chunking

The BGE baseline has a finite sequence length.

Phase 3 therefore changed the chunker from generic tiktoken counting to the embedding model's tokenizer.

Current working configuration:

384 content tokens
≈ 386 tokens including model special tokens

This remains comfortably below the model's inference limit.

The earlier tokenizer warning caused by pages longer than the model limit was not an inference failure: the entire page was being tokenised only so it could subsequently be split into smaller windows.

The tokenizer wrapper explicitly avoids truncating source pages during chunk construction.

Normalisation

Embeddings are generated with vector normalisation enabled.

For each vector:

||v|| ≈ 1

For normalised vectors, cosine similarity can be calculated efficiently using a dot product:

similarity ≈ query_vector · document_vector

This will be useful during dense retrieval.

Batching

Document chunks are embedded in batches rather than one at a time.

chunks
 ↓
batch
 ↓
neural-network inference
 ↓
vectors

Batching reduces repeated model overhead and allows better use of available hardware.

The current batch size is a baseline and can later be benchmarked.

Provenance Preservation

A ChunkEmbedding contains both:

embedding vector
+
original DocumentChunk

Therefore:

retrieved vector
      ↓
DocumentChunk
      ↓
page
      ↓
source document

remains available.

The vector itself is never treated as sufficient metadata.

Testing Strategy

Unit tests do not load the real embedding model.

Instead they use a fake implementation of the EmbeddingProvider interface.

This keeps normal test runs:

deterministic
fast
offline-friendly
independent of model downloads

Real-model behaviour is validated separately through smoke tests.

Phase 3 Smoke Tests

The model was checked for:

expected vector dimensionality
model sequence length
chunk compatibility
vector normalisation
one-to-one chunk/vector mapping
query embedding generation
semantic similarity behaviour

A simple in-memory retrieval experiment also demonstrated that query vectors can already rank document chunks without a vector database.

Current Limitations

Embedding quality has not yet been formally evaluated.

The current model should therefore be considered a baseline.

Future comparisons may include:

alternative BGE models
Nomic embeddings
E5-family models
GTE-family models

Evaluation should consider:

retrieval quality
latency
memory consumption
embedding dimensionality
indexing/storage cost

Model selection should ultimately be based on measured retrieval performance rather than reputation alone.
Document Chunking
Purpose

Embedding an entire research paper into a single vector would make retrieval too coarse.

Chunking converts document pages into smaller retrieval units:

ParsedDocument
      ↓
DocumentPage[]
      ↓
DocumentChunk[]

The chunker must balance two competing goals:

smaller chunks
    → precise retrieval
    → less context

larger chunks
    → more context
    → less precise retrieval

The correct configuration should eventually be determined experimentally.

Baseline Strategy

The current baseline uses overlapping token windows.

Current configuration:

chunk size: 384 model tokens
overlap:    64 model tokens

This is a baseline, not a claim of optimality.

Model-Aware Tokenisation

The initial implementation used tiktoken.

Phase 3 exposed an important issue:

tiktoken token count
        ≠
embedding-model token count

Since the embedding model ultimately consumes the chunks, the production chunking baseline now uses the embedding model's Hugging Face tokenizer.

This prevents silent assumptions about sequence length.

Tokenizer Abstraction

The chunker depends on a Tokenizer interface rather than a concrete implementation.

Conceptually:

Tokenizer
├── TiktokenTokenizer
└── HuggingFaceTokenizer

The chunking algorithm therefore does not need to know which tokenizer produced the token IDs.

Sliding Window

For:

chunk_size = 384
overlap = 64

the step size is:

384 - 64 = 320

A long page becomes approximately:

Chunk 0: tokens   0–384
Chunk 1: tokens 320–704
Chunk 2: tokens 640–...

Overlap helps prevent facts near boundaries from being split into completely independent pieces.

Page-Local Chunks

Chunks currently never span page boundaries.

Page 4
├── Chunk
├── Chunk
└── Chunk

Page 5
├── Chunk
└── Chunk

This simplifies citation provenance.

The trade-off is that a paragraph spanning two PDF pages is also separated by the chunker.

Cross-page or section-aware approaches may be investigated later.

Chunk Provenance

Each chunk retains:

chunk_id
document_id
filename
page number
global chunk index
page-local chunk index
source text
token count
character count
token start
token end

This maintains:

chunk
 ↓
page
 ↓
document

throughout later retrieval.

Deterministic Identifiers

Chunk IDs are deterministic rather than random UUIDs.

They are derived from:

document identity
+
page number
+
page-local chunk position

This improves reproducibility and will later help with:

vector-store updates
duplicate handling
evaluation datasets
debugging
citation tracing
Current Weaknesses

The baseline chunker does not understand:

sentences
paragraphs
headings
sections
equations
semantic topic transitions

A window can therefore begin or end at an awkward semantic position.

That is intentional: this provides a measurable baseline.

Future Strategies

Potential alternatives include:

sentence-aware chunking
recursive chunking
section-aware chunking
semantic chunking
parent-child retrieval
cross-page contextual chunks

These strategies should eventually be compared using retrieval metrics rather than subjective inspection alone.
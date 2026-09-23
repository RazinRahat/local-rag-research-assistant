Document Ingestion
Purpose

The ingestion layer converts external research documents into a stable internal representation that can be consumed by later RAG components.

The rest of the application should not depend directly on PyMuPDF objects.

External PDF
     ↓
Ingestion Boundary
     ↓
ParsedDocument

This allows the parser to evolve without forcing changes to chunking, retrieval, or generation.

Canonical Document Representation

The ingestion layer currently produces:

ParsedDocument
│
├── DocumentMetadata
│
└── DocumentPage[]

DocumentMetadata stores information including:

document identity
filename
file size
SHA-256 fingerprint
page count
number of pages containing extractable text
PDF title
author
subject
keywords
creator/producer
creation and modification metadata

Each DocumentPage contains:

human-readable page number
extracted text
character count
word count
text-presence flag
page dimensions
rotation
Page-Level Provenance

The ingestion pipeline intentionally does not flatten an entire PDF into one string.

Instead:

PDF
├── Page 1
├── Page 2
├── Page 3
└── ...

is retained.

This is important because later retrieved chunks should be able to produce traceable citations such as:

paper.pdf, p. 8

If pages were discarded during ingestion, citation reconstruction later would be much less reliable.

Page Numbering

PyMuPDF uses zero-based page indexing internally.

The application's document model converts these indexes to human-readable numbering at the ingestion boundary:

PyMuPDF index 0 → page 1
PyMuPDF index 1 → page 2

Downstream components therefore work entirely with human-facing page numbers.

Document Identity

Each document receives a SHA-256 fingerprint calculated from its file contents.

This means two files containing identical bytes will receive the same identity even if their filenames differ.

paper.pdf
paper-copy.pdf

        ↓ SHA-256

same document_id

This will later support duplicate detection and vector-store updates.

The file is hashed incrementally instead of loading the entire PDF into memory.

Text Extraction

PyMuPDF currently acts as the baseline PDF parser.

Text is extracted page by page using reading-order sorting.

PDFs do not inherently contain text in the same semantic order humans see on the page. Academic papers containing multiple columns, equations, figures, headers, or unusual layouts can therefore produce imperfect extraction.

This baseline is intentionally retained so more advanced parsing approaches can later be compared against it.

Possible future alternatives include:

block-aware PyMuPDF extraction
Markdown-oriented extraction
PyMuPDF4LLM
Docling
Unstructured
Text Normalisation

The normalisation stage currently performs conservative transformations:

Unicode NFC normalisation
null-character removal
line-ending normalisation
trailing whitespace removal
excessive blank-line reduction

The pipeline deliberately avoids aggressive transformations such as automatic dehyphenation.

For example:

architec-
ture

could potentially be reconstructed as:

architecture

but blindly applying this rule could corrupt legitimate hyphenated technical terms.

Aggressive normalisation should therefore be introduced only when its failure modes can be measured.

Empty Pages

Pages containing no extractable text are preserved.

For example:

Page 1 → text
Page 2 → text
Page 3 → no text
Page 4 → text

Page 3 remains part of the document representation.

Dropping it would cause downstream page numbering to drift and produce incorrect citations.

Scanned PDFs

A scanned PDF may contain pages without machine-readable text.

The current pipeline records this using:

page_count
text_page_count

A document with:

page_count = 20
text_page_count = 0

likely requires OCR.

OCR is intentionally outside the current ingestion baseline.

Error Handling

The ingestion domain defines explicit errors for cases such as:

unsupported formats
encrypted PDFs
empty documents
malformed documents

This will allow the API layer to translate domain failures into meaningful HTTP responses later.

Current Limitations

The baseline parser does not yet handle:

OCR
table reconstruction
image understanding
equation reconstruction
semantic section detection
sophisticated multi-column reading order
bibliography parsing

These limitations are intentionally documented rather than hidden.
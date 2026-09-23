from hashlib import sha256

from research_assistant.chunking.models import (
    ChunkingConfig,
    DocumentChunk,
)
from research_assistant.chunking.tokenizer import Tokenizer
from research_assistant.ingestion.models import ParsedDocument


def _build_chunk_id(
    document_id: str,
    page_number: int,
    page_chunk_index: int,
) -> str:
    """Generate a deterministic identifier for a document chunk."""

    value = f"{document_id}:{page_number}:{page_chunk_index}"

    return sha256(value.encode("utf-8")).hexdigest()


def chunk_document(
    document: ParsedDocument,
    tokenizer: Tokenizer,
    config: ChunkingConfig | None = None,
) -> tuple[DocumentChunk, ...]:
    """Split document pages into overlapping token windows."""

    if config is None:
        config = ChunkingConfig()

    chunks: list[DocumentChunk] = []

    chunk_index = 0

    step_size = config.chunk_size_tokens - config.chunk_overlap_tokens

    for page in document.pages:
        if not page.has_text:
            continue

        page_tokens = tokenizer.encode(page.text)

        if not page_tokens:
            continue

        token_start = 0
        page_chunk_index = 0

        while token_start < len(page_tokens):
            token_end = min(
                token_start + config.chunk_size_tokens,
                len(page_tokens),
            )

            token_slice = page_tokens[token_start:token_end]

            chunk_text = tokenizer.decode(token_slice).strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=_build_chunk_id(
                            document.metadata.document_id,
                            page.page_number,
                            page_chunk_index,
                        ),
                        document_id=(document.metadata.document_id),
                        file_name=(document.metadata.file_name),
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        page_chunk_index=page_chunk_index,
                        text=chunk_text,
                        token_count=tokenizer.count(chunk_text),
                        char_count=len(chunk_text),
                        token_start=token_start,
                        token_end=token_end,
                    )
                )

                chunk_index += 1

            if token_end == len(page_tokens):
                break

            token_start += step_size
            page_chunk_index += 1

    return tuple(chunks)

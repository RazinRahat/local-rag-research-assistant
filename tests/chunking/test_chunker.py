import pytest

from research_assistant.chunking.chunker import chunk_document
from research_assistant.chunking.models import ChunkingConfig
from research_assistant.chunking.tokenizer import TiktokenTokenizer
from research_assistant.ingestion.models import (
    DocumentMetadata,
    DocumentPage,
    ParsedDocument,
)


def _make_page(
    page_number: int,
    text: str,
) -> DocumentPage:
    return DocumentPage(
        page_number=page_number,
        text=text,
        char_count=len(text),
        word_count=len(text.split()),
        has_text=bool(text),
        width=612.0,
        height=792.0,
        rotation=0,
    )


def _make_document(
    *pages: DocumentPage,
) -> ParsedDocument:
    text_page_count = sum(1 for page in pages if page.has_text)

    metadata = DocumentMetadata(
        document_id="a" * 64,
        file_name="paper.pdf",
        file_size_bytes=1024,
        sha256="a" * 64,
        page_count=len(pages),
        text_page_count=text_page_count,
    )

    return ParsedDocument(
        metadata=metadata,
        pages=tuple(pages),
    )


def test_short_page_produces_single_chunk() -> None:
    document = _make_document(
        _make_page(
            1,
            "Retrieval augmented generation combines "
            "retrieval with language generation.",
        )
    )

    tokenizer = TiktokenTokenizer()

    chunks = chunk_document(
        document=document,
        tokenizer=tokenizer,
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.page_number == 1
    assert chunk.chunk_index == 0
    assert chunk.page_chunk_index == 0

    assert chunk.document_id == "a" * 64
    assert chunk.file_name == "paper.pdf"

    assert chunk.token_count > 0
    assert chunk.text


def test_long_page_produces_multiple_chunks() -> None:
    text = " ".join(f"research-token-{index}" for index in range(1000))

    document = _make_document(_make_page(1, text))

    tokenizer = TiktokenTokenizer()

    config = ChunkingConfig(
        chunk_size_tokens=100,
        chunk_overlap_tokens=20,
    )

    chunks = chunk_document(
        document=document,
        tokenizer=tokenizer,
        config=config,
    )

    assert len(chunks) > 1

    assert all(chunk.token_count <= config.chunk_size_tokens for chunk in chunks)


def test_chunks_use_expected_overlap() -> None:
    text = " ".join(f"token-{index}" for index in range(500))

    document = _make_document(_make_page(1, text))

    tokenizer = TiktokenTokenizer()

    config = ChunkingConfig(
        chunk_size_tokens=100,
        chunk_overlap_tokens=25,
    )

    chunks = chunk_document(
        document=document,
        tokenizer=tokenizer,
        config=config,
    )

    assert len(chunks) > 1

    for previous, current in zip(
        chunks,
        chunks[1:],
        strict=False,
    ):
        assert current.token_start == previous.token_end - config.chunk_overlap_tokens


def test_chunks_preserve_page_provenance() -> None:
    document = _make_document(
        _make_page(
            1,
            "Introduction to retrieval augmented generation.",
        ),
        _make_page(
            2,
            "Evaluation measures retrieval effectiveness.",
        ),
    )

    tokenizer = TiktokenTokenizer()

    chunks = chunk_document(
        document=document,
        tokenizer=tokenizer,
    )

    assert len(chunks) == 2

    assert chunks[0].page_number == 1
    assert chunks[1].page_number == 2


def test_empty_pages_do_not_create_chunks() -> None:
    document = _make_document(
        _make_page(1, ""),
        _make_page(
            2,
            "This page contains usable text.",
        ),
    )

    tokenizer = TiktokenTokenizer()

    chunks = chunk_document(
        document=document,
        tokenizer=tokenizer,
    )

    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_chunk_ids_are_deterministic() -> None:
    document = _make_document(
        _make_page(
            1,
            "Deterministic identifiers improve reproducibility.",
        )
    )

    tokenizer = TiktokenTokenizer()

    first_run = chunk_document(
        document=document,
        tokenizer=tokenizer,
    )

    second_run = chunk_document(
        document=document,
        tokenizer=tokenizer,
    )

    assert first_run[0].chunk_id == second_run[0].chunk_id


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(
            chunk_size_tokens=100,
            chunk_overlap_tokens=100,
        )

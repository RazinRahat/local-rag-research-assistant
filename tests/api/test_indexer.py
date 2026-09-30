"""Tests for the local PDF indexing pipeline."""

from collections.abc import Sequence
from io import BytesIO

import pymupdf
import pytest

from research_assistant.api.exceptions import (
    InvalidUploadError,
    UnsupportedMediaError,
    UploadTooLargeError,
)
from research_assistant.api.indexer import LocalPDFIndexer
from research_assistant.chunking.models import ChunkingConfig
from research_assistant.embeddings.embedder import EmbeddingVector
from research_assistant.embeddings.models import ChunkEmbedding
from research_assistant.vector_store.models import (
    VectorSearchResult,
)


class FakeTokenizer:
    def encode(
        self,
        text: str,
    ) -> list[int]:
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids: Sequence[int],
    ) -> str:
        return "".join(chr(token) for token in token_ids)

    def count(
        self,
        text: str,
    ) -> int:
        return len(self.encode(text))


class FakeEmbedder:
    @property
    def model_name(self) -> str:
        return "test-model"

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[EmbeddingVector, ...]:
        return tuple((1.0, 0.0, 0.0) for _ in texts)

    def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        return (
            1.0,
            0.0,
            0.0,
        )


class FakeVectorStore:
    def __init__(self) -> None:
        self.document_id: str | None = None

        self.embeddings: tuple[
            ChunkEmbedding,
            ...,
        ] = ()

    def ensure_collection(self) -> None:
        pass

    def replace_document(
        self,
        document_id: str,
        embeddings: Sequence[ChunkEmbedding],
    ) -> int:
        self.document_id = document_id
        self.embeddings = tuple(embeddings)

        return len(self.embeddings)

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        pass

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        score_threshold: float | None = None,
        document_id: str | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        return ()

    def count_document(
        self,
        document_id: str,
    ) -> int:
        if document_id == self.document_id:
            return len(self.embeddings)

        return 0

    def count(self) -> int:
        return len(self.embeddings)

    def close(self) -> None:
        pass


def make_pdf(
    text: str | None = None,
) -> bytes:
    document = pymupdf.open()

    page = document.new_page()

    if text is not None:
        page.insert_text(
            (72, 72),
            text,
        )

    data = bytes(document.tobytes())

    document.close()

    return data


def make_indexer(
    *,
    max_upload_bytes: int = 1024 * 1024,
) -> tuple[
    LocalPDFIndexer,
    FakeVectorStore,
]:
    store = FakeVectorStore()

    indexer = LocalPDFIndexer(
        tokenizer=FakeTokenizer(),
        embedder=FakeEmbedder(),
        store=store,
        chunking_config=ChunkingConfig(
            chunk_size_tokens=64,
            chunk_overlap_tokens=8,
        ),
        max_upload_bytes=max_upload_bytes,
    )

    return indexer, store


def test_indexes_pdf_into_vector_store() -> None:
    indexer, store = make_indexer()

    result = indexer.index(
        "paper.pdf",
        BytesIO(make_pdf("Transformers use attention mechanisms.")),
    )

    assert result.file_name == "paper.pdf"
    assert result.chunk_count >= 1

    assert result.document_id == store.document_id

    assert len(store.embeddings) == (result.chunk_count)

    assert all(
        embedding.chunk.document_id == result.document_id
        for embedding in store.embeddings
    )


def test_rejects_non_pdf_filename() -> None:
    indexer, store = make_indexer()

    with pytest.raises(UnsupportedMediaError):
        indexer.index(
            "notes.txt",
            BytesIO(b"example"),
        )

    assert store.embeddings == ()


def test_rejects_pdf_without_indexable_text() -> None:
    indexer, store = make_indexer()

    with pytest.raises(InvalidUploadError):
        indexer.index(
            "blank.pdf",
            BytesIO(make_pdf()),
        )

    assert store.embeddings == ()


def test_rejects_invalid_pdf() -> None:
    indexer, store = make_indexer()

    with pytest.raises(InvalidUploadError):
        indexer.index(
            "broken.pdf",
            BytesIO(b"not actually a pdf"),
        )

    assert store.embeddings == ()


def test_enforces_upload_size_limit() -> None:
    pdf = make_pdf("A small research document.")

    indexer, store = make_indexer(
        max_upload_bytes=len(pdf) - 1,
    )

    with pytest.raises(UploadTooLargeError):
        indexer.index(
            "paper.pdf",
            BytesIO(pdf),
        )

    assert store.embeddings == ()

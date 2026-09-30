"""Tests for indexed-document enumeration."""

from pathlib import Path

from research_assistant.chunking.models import DocumentChunk
from research_assistant.embeddings.models import ChunkEmbedding
from research_assistant.vector_store.models import VectorStoreConfig
from research_assistant.vector_store.qdrant_store import (
    QdrantVectorStore,
)


def make_embedding(
    *,
    document_id: str,
    file_name: str,
    chunk_id: str,
    chunk_index: int,
) -> ChunkEmbedding:
    text = f"Chunk {chunk_index}"

    chunk = DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        file_name=file_name,
        page_number=1,
        chunk_index=chunk_index,
        page_chunk_index=chunk_index,
        text=text,
        token_count=2,
        char_count=len(text),
        token_start=chunk_index * 2,
        token_end=(chunk_index + 1) * 2,
    )

    return ChunkEmbedding(
        chunk=chunk,
        model_name="test-model",
        dimension=3,
        vector=(
            1.0,
            0.0,
            0.0,
        ),
    )


def make_store() -> QdrantVectorStore:
    return QdrantVectorStore(
        VectorStoreConfig(
            path=Path(":memory:"),
            collection_name="test_documents",
            embedding_model="test-model",
            vector_size=3,
        )
    )


def test_list_documents_groups_chunks() -> None:
    store = make_store()

    store.ensure_collection()

    document_a = (
        make_embedding(
            document_id="a" * 64,
            file_name="paper-a.pdf",
            chunk_id="chunk-a-1",
            chunk_index=0,
        ),
        make_embedding(
            document_id="a" * 64,
            file_name="paper-a.pdf",
            chunk_id="chunk-a-2",
            chunk_index=1,
        ),
    )

    document_b = (
        make_embedding(
            document_id="b" * 64,
            file_name="paper-b.pdf",
            chunk_id="chunk-b-1",
            chunk_index=0,
        ),
    )

    store.replace_document(
        "a" * 64,
        document_a,
    )

    store.replace_document(
        "b" * 64,
        document_b,
    )

    documents = store.list_documents()

    assert len(documents) == 2

    assert documents[0].document_id == "a" * 64
    assert documents[0].file_name == "paper-a.pdf"
    assert documents[0].chunk_count == 2

    assert documents[1].document_id == "b" * 64
    assert documents[1].file_name == "paper-b.pdf"
    assert documents[1].chunk_count == 1

    store.close()


def test_list_documents_returns_empty_tuple() -> None:
    store = make_store()

    store.ensure_collection()

    assert store.list_documents() == ()

    store.close()


def test_document_replacement_updates_chunk_count() -> None:
    store = make_store()

    store.ensure_collection()

    document_id = "a" * 64

    original = (
        make_embedding(
            document_id=document_id,
            file_name="paper.pdf",
            chunk_id="chunk-1",
            chunk_index=0,
        ),
        make_embedding(
            document_id=document_id,
            file_name="paper.pdf",
            chunk_id="chunk-2",
            chunk_index=1,
        ),
    )

    replacement = (
        make_embedding(
            document_id=document_id,
            file_name="paper.pdf",
            chunk_id="chunk-3",
            chunk_index=0,
        ),
    )

    store.replace_document(
        document_id,
        original,
    )

    store.replace_document(
        document_id,
        replacement,
    )

    documents = store.list_documents()

    assert len(documents) == 1
    assert documents[0].document_id == document_id
    assert documents[0].file_name == "paper.pdf"
    assert documents[0].chunk_count == 1

    store.close()

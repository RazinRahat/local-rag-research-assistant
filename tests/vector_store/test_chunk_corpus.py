from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from qdrant_client import QdrantClient, models

from research_assistant.vector_store.models import (
    VectorStoreConfig,
)
from research_assistant.vector_store.qdrant_store import (
    QdrantVectorStore,
)

DOCUMENT_A = "a" * 64
DOCUMENT_B = "b" * 64

COLLECTION_NAME = "test_chunk_corpus"
EMBEDDING_MODEL = "test-embedding-model"


@pytest.fixture
def store_and_client() -> Iterator[
    tuple[
        QdrantVectorStore,
        QdrantClient,
    ]
]:
    client = QdrantClient(location=":memory:")

    store = QdrantVectorStore(
        config=VectorStoreConfig(
            collection_name=(COLLECTION_NAME),
            embedding_model=(EMBEDDING_MODEL),
            vector_size=3,
        ),
        client=client,
    )

    store.ensure_collection()

    try:
        yield store, client
    finally:
        store.close()


def make_payload(
    *,
    chunk_id: str,
    document_id: str,
    file_name: str,
    text: str,
    chunk_index: int,
    page_number: int,
) -> dict[str, Any]:
    token_count = max(
        1,
        len(text.split()),
    )

    return {
        "chunk_id": chunk_id,
        "document_id": document_id,
        "file_name": file_name,
        "page_number": page_number,
        "chunk_index": chunk_index,
        "page_chunk_index": 0,
        "text": text,
        "token_count": token_count,
        "char_count": len(text),
        "token_start": 0,
        "token_end": token_count,
        "embedding_model": (EMBEDDING_MODEL),
    }


def insert_chunks(
    client: QdrantClient,
) -> None:
    client.upsert(
        collection_name=(COLLECTION_NAME),
        points=[
            models.PointStruct(
                id=1,
                vector=[
                    1.0,
                    0.0,
                    0.0,
                ],
                payload=make_payload(
                    chunk_id="chunk-a-1",
                    document_id=DOCUMENT_A,
                    file_name="a.pdf",
                    text=("Attention replaces recurrence."),
                    chunk_index=0,
                    page_number=1,
                ),
            ),
            models.PointStruct(
                id=2,
                vector=[
                    0.0,
                    1.0,
                    0.0,
                ],
                payload=make_payload(
                    chunk_id="chunk-a-2",
                    document_id=DOCUMENT_A,
                    file_name="a.pdf",
                    text=("The optimizer was Adam."),
                    chunk_index=1,
                    page_number=2,
                ),
            ),
            models.PointStruct(
                id=3,
                vector=[
                    0.0,
                    0.0,
                    1.0,
                ],
                payload=make_payload(
                    chunk_id="chunk-b-1",
                    document_id=DOCUMENT_B,
                    file_name="b.pdf",
                    text=("Convolution is used for image features."),
                    chunk_index=0,
                    page_number=3,
                ),
            ),
        ],
        wait=True,
    )


def test_list_chunks_returns_all_persisted_chunks(
    store_and_client: tuple[
        QdrantVectorStore,
        QdrantClient,
    ],
) -> None:
    store, client = store_and_client

    insert_chunks(client)

    chunks = store.list_chunks()

    assert len(chunks) == 3

    assert tuple(chunk.chunk_id for chunk in chunks) == (
        "chunk-a-1",
        "chunk-a-2",
        "chunk-b-1",
    )

    assert chunks[0].text == "Attention replaces recurrence."

    assert chunks[1].page_number == 2


def test_list_chunks_reconstructs_provenance(
    store_and_client: tuple[
        QdrantVectorStore,
        QdrantClient,
    ],
) -> None:
    store, client = store_and_client

    insert_chunks(client)

    chunks = store.list_chunks(DOCUMENT_A)

    chunk = chunks[1]

    assert chunk.chunk_id == "chunk-a-2"

    assert chunk.document_id == DOCUMENT_A

    assert chunk.file_name == "a.pdf"

    assert chunk.page_number == 2

    assert chunk.chunk_index == 1

    assert chunk.text == "The optimizer was Adam."


def test_list_chunks_respects_document_scope(
    store_and_client: tuple[
        QdrantVectorStore,
        QdrantClient,
    ],
) -> None:
    store, client = store_and_client

    insert_chunks(client)

    chunks = store.list_chunks(DOCUMENT_B)

    assert len(chunks) == 1

    assert chunks[0].document_id == DOCUMENT_B

    assert chunks[0].chunk_id == "chunk-b-1"


def test_list_chunks_returns_empty_for_missing_document(
    store_and_client: tuple[
        QdrantVectorStore,
        QdrantClient,
    ],
) -> None:
    store, client = store_and_client

    insert_chunks(client)

    chunks = store.list_chunks("c" * 64)

    assert chunks == ()


def test_list_chunks_returns_empty_when_collection_does_not_exist() -> None:
    client = QdrantClient(location=":memory:")

    store = QdrantVectorStore(
        config=VectorStoreConfig(
            collection_name=("missing_collection"),
            embedding_model=(EMBEDDING_MODEL),
            vector_size=3,
        ),
        client=client,
    )

    try:
        assert store.list_chunks() == ()
    finally:
        store.close()


def test_qdrant_store_can_act_as_bm25_chunk_corpus(
    store_and_client: tuple[
        QdrantVectorStore,
        QdrantClient,
    ],
) -> None:
    from research_assistant.retrieval.lexical import (
        BM25Retriever,
    )

    store, client = store_and_client

    insert_chunks(client)

    retriever = BM25Retriever(store)

    results = retriever.retrieve("Adam optimizer")

    assert results

    assert results[0].chunk.chunk_id == "chunk-a-2"

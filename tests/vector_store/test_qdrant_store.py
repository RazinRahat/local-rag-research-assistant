from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from research_assistant.chunking.models import (
    DocumentChunk,
)
from research_assistant.embeddings.models import (
    ChunkEmbedding,
)
from research_assistant.vector_store.exceptions import (
    EmbeddingBatchError,
)
from research_assistant.vector_store.models import (
    VectorStoreConfig,
)
from research_assistant.vector_store.qdrant_store import (
    QdrantVectorStore,
)


def _make_embedding(
    document_id: str,
    chunk_index: int,
    model_name: str = "test-model",
    dimension: int = 3,
) -> ChunkEmbedding:
    text = f"Research chunk {chunk_index}"

    chunk = DocumentChunk(
        chunk_id=(f"{document_id}-chunk-{chunk_index}"),
        document_id=document_id,
        file_name="paper.pdf",
        page_number=1,
        chunk_index=chunk_index,
        page_chunk_index=chunk_index,
        text=text,
        token_count=4,
        char_count=len(text),
        token_start=chunk_index * 4,
        token_end=(chunk_index + 1) * 4,
    )

    vector = tuple(1.0 if index == 0 else 0.0 for index in range(dimension))

    return ChunkEmbedding(
        chunk=chunk,
        model_name=model_name,
        vector=vector,
        dimension=dimension,
    )


def _make_store(
    client: QdrantClient,
) -> QdrantVectorStore:
    return QdrantVectorStore(
        config=VectorStoreConfig(
            collection_name="test_collection",
            embedding_model="test-model",
            vector_size=3,
        ),
        client=client,
    )


def test_ensure_collection_creates_collection() -> None:
    client = QdrantClient(":memory:")
    store = _make_store(client)

    try:
        store.ensure_collection()

        assert client.collection_exists("test_collection")
    finally:
        store.close()


def test_replace_document_stores_embeddings() -> None:
    client = QdrantClient(":memory:")
    store = _make_store(client)

    document_id = "document-a"

    embeddings = (
        _make_embedding(document_id, 0),
        _make_embedding(document_id, 1),
    )

    try:
        stored_count = store.replace_document(
            document_id=document_id,
            embeddings=embeddings,
        )

        assert stored_count == 2
        assert store.count() == 2

        assert store.count_document(document_id) == 2
    finally:
        store.close()


def test_replace_document_removes_old_chunks() -> None:
    client = QdrantClient(":memory:")
    store = _make_store(client)

    document_id = "document-a"

    first_version = (
        _make_embedding(document_id, 0),
        _make_embedding(document_id, 1),
        _make_embedding(document_id, 2),
    )

    second_version = (
        _make_embedding(document_id, 0),
        _make_embedding(document_id, 1),
    )

    try:
        store.replace_document(
            document_id=document_id,
            embeddings=first_version,
        )

        assert store.count_document(document_id) == 3

        store.replace_document(
            document_id=document_id,
            embeddings=second_version,
        )

        assert store.count_document(document_id) == 2
    finally:
        store.close()


def test_delete_document_preserves_other_documents() -> None:
    client = QdrantClient(":memory:")
    store = _make_store(client)

    first_id = "document-a"
    second_id = "document-b"

    try:
        store.replace_document(
            document_id=first_id,
            embeddings=(_make_embedding(first_id, 0),),
        )

        store.replace_document(
            document_id=second_id,
            embeddings=(_make_embedding(second_id, 0),),
        )

        assert store.count() == 2

        store.delete_document(first_id)

        assert store.count_document(first_id) == 0

        assert store.count_document(second_id) == 1

        assert store.count() == 1
    finally:
        store.close()


def test_rejects_wrong_embedding_model() -> None:
    client = QdrantClient(":memory:")
    store = _make_store(client)

    document_id = "document-a"

    embedding = _make_embedding(
        document_id=document_id,
        chunk_index=0,
        model_name="different-model",
    )

    try:
        with pytest.raises(EmbeddingBatchError):
            store.replace_document(
                document_id=document_id,
                embeddings=(embedding,),
            )
    finally:
        store.close()


def test_local_store_persists_after_reopening(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qdrant"

    config = VectorStoreConfig(
        path=path,
        collection_name="test_collection",
        embedding_model="test-model",
        vector_size=3,
    )

    document_id = "document-a"

    first_store = QdrantVectorStore(config=config)

    first_store.replace_document(
        document_id=document_id,
        embeddings=(
            _make_embedding(document_id, 0),
            _make_embedding(document_id, 1),
        ),
    )

    assert first_store.count() == 2

    first_store.close()

    second_store = QdrantVectorStore(config=config)

    try:
        assert second_store.count() == 2

        assert second_store.count_document(document_id) == 2
    finally:
        second_store.close()


def _make_embedding_with_vector(
    document_id: str,
    chunk_index: int,
    vector: tuple[float, ...],
) -> ChunkEmbedding:
    text = f"Research chunk {chunk_index}"

    chunk = DocumentChunk(
        chunk_id=(f"{document_id}-chunk-{chunk_index}"),
        document_id=document_id,
        file_name="paper.pdf",
        page_number=1,
        chunk_index=chunk_index,
        page_chunk_index=chunk_index,
        text=text,
        token_count=4,
        char_count=len(text),
        token_start=chunk_index * 4,
        token_end=(chunk_index + 1) * 4,
    )

    return ChunkEmbedding(
        chunk=chunk,
        model_name="test-model",
        vector=vector,
        dimension=len(vector),
    )


def test_search_returns_nearest_vectors() -> None:
    client = QdrantClient(":memory:")
    store = _make_store(client)

    document_id = "document-a"

    embeddings = (
        _make_embedding_with_vector(
            document_id,
            0,
            (1.0, 0.0, 0.0),
        ),
        _make_embedding_with_vector(
            document_id,
            1,
            (0.0, 1.0, 0.0),
        ),
        _make_embedding_with_vector(
            document_id,
            2,
            (0.0, 0.0, 1.0),
        ),
    )

    try:
        store.replace_document(
            document_id=document_id,
            embeddings=embeddings,
        )

        results = store.search(
            query_vector=(1.0, 0.0, 0.0),
            top_k=2,
        )

        assert len(results) == 2

        assert results[0].chunk.chunk_index == 0

        assert results[0].score >= results[1].score
    finally:
        store.close()


def test_search_can_filter_by_document() -> None:
    client = QdrantClient(":memory:")
    store = _make_store(client)

    first_id = "document-a"
    second_id = "document-b"

    try:
        store.replace_document(
            document_id=first_id,
            embeddings=(
                _make_embedding_with_vector(
                    first_id,
                    0,
                    (1.0, 0.0, 0.0),
                ),
            ),
        )

        store.replace_document(
            document_id=second_id,
            embeddings=(
                _make_embedding_with_vector(
                    second_id,
                    0,
                    (0.99, 0.01, 0.0),
                ),
            ),
        )

        results = store.search(
            query_vector=(1.0, 0.0, 0.0),
            top_k=5,
            document_id=second_id,
        )

        assert len(results) == 1

        assert results[0].chunk.document_id == second_id
    finally:
        store.close()


def test_search_rejects_wrong_query_dimension() -> None:
    client = QdrantClient(":memory:")
    store = _make_store(client)

    try:
        with pytest.raises(ValueError):
            store.search(
                query_vector=(1.0, 0.0),
                top_k=5,
            )
    finally:
        store.close()

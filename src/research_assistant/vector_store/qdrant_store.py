from collections.abc import Sequence
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from research_assistant.chunking.models import (
    DocumentChunk,
)
from research_assistant.embeddings.models import (
    ChunkEmbedding,
)
from research_assistant.vector_store.exceptions import (
    CollectionConfigurationError,
    EmbeddingBatchError,
)
from research_assistant.vector_store.models import (
    VectorSearchResult,
    VectorStoreConfig,
)


def _required_str(
    payload: dict[str, Any],
    key: str,
) -> str:
    value = payload.get(key)

    if not isinstance(value, str):
        raise ValueError(f"Expected payload field '{key}' to contain a string")

    return value


def _required_int(
    payload: dict[str, Any],
    key: str,
) -> int:
    value = payload.get(key)

    if not isinstance(value, int):
        raise ValueError(f"Expected payload field '{key}' to contain an integer")

    return value


def _payload_to_chunk(
    payload: dict[str, Any],
) -> DocumentChunk:
    """Reconstruct a document chunk from persisted payload."""

    text = _required_str(
        payload,
        "text",
    )

    return DocumentChunk(
        chunk_id=_required_str(
            payload,
            "chunk_id",
        ),
        document_id=_required_str(
            payload,
            "document_id",
        ),
        file_name=_required_str(
            payload,
            "file_name",
        ),
        page_number=_required_int(
            payload,
            "page_number",
        ),
        chunk_index=_required_int(
            payload,
            "chunk_index",
        ),
        page_chunk_index=_required_int(
            payload,
            "page_chunk_index",
        ),
        text=text,
        token_count=_required_int(
            payload,
            "token_count",
        ),
        char_count=_required_int(
            payload,
            "char_count",
        ),
        token_start=_required_int(
            payload,
            "token_start",
        ),
        token_end=_required_int(
            payload,
            "token_end",
        ),
    )


def _point_id(chunk_id: str) -> str:
    """Create a deterministic Qdrant-compatible UUID."""

    return str(
        uuid5(
            NAMESPACE_URL,
            f"local-rag:{chunk_id}",
        )
    )


def _document_filter(
    document_id: str,
) -> models.Filter:
    """Build a filter matching one source document."""

    return models.Filter(
        must=[
            models.FieldCondition(
                key="document_id",
                match=models.MatchValue(value=document_id),
            )
        ]
    )


class QdrantVectorStore:
    """Qdrant-backed storage for document embeddings."""

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self._config = config or VectorStoreConfig()

        self._client = (
            client if client is not None else QdrantClient(path=str(self._config.path))
        )

    def ensure_collection(self) -> None:
        """Create or validate the embedding collection."""

        if not self._client.collection_exists(self._config.collection_name):
            self._client.create_collection(
                collection_name=(self._config.collection_name),
                vectors_config=models.VectorParams(
                    size=self._config.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

            return

        collection = self._client.get_collection(self._config.collection_name)

        vectors_config = collection.config.params.vectors

        if not isinstance(
            vectors_config,
            models.VectorParams,
        ):
            raise CollectionConfigurationError(
                "Expected a single unnamed dense vector configuration"
            )

        if vectors_config.size != self._config.vector_size:
            raise CollectionConfigurationError(
                "Existing collection vector size "
                f"{vectors_config.size} does not match "
                f"expected size "
                f"{self._config.vector_size}"
            )

        if vectors_config.distance != models.Distance.COSINE:
            raise CollectionConfigurationError(
                "Existing collection does not use cosine distance"
            )

    def _validate_embeddings(
        self,
        document_id: str,
        embeddings: Sequence[ChunkEmbedding],
    ) -> None:
        for embedding in embeddings:
            if embedding.chunk.document_id != document_id:
                raise EmbeddingBatchError("Embedding belongs to a different document")

            if embedding.model_name != self._config.embedding_model:
                raise EmbeddingBatchError(
                    "Embedding model does not match the vector-store configuration"
                )

            if embedding.dimension != self._config.vector_size:
                raise EmbeddingBatchError(
                    "Embedding dimension does not match the vector-store configuration"
                )

    def _build_point(
        self,
        embedding: ChunkEmbedding,
    ) -> models.PointStruct:
        chunk = embedding.chunk

        payload: dict[str, Any] = {
            "chunk_id": chunk.chunk_id,
            "document_id": chunk.document_id,
            "file_name": chunk.file_name,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "page_chunk_index": (chunk.page_chunk_index),
            "text": chunk.text,
            "token_count": chunk.token_count,
            "char_count": chunk.char_count,
            "token_start": chunk.token_start,
            "token_end": chunk.token_end,
            "embedding_model": (embedding.model_name),
        }

        return models.PointStruct(
            id=_point_id(chunk.chunk_id),
            vector=list(embedding.vector),
            payload=payload,
        )

    def replace_document(
        self,
        document_id: str,
        embeddings: Sequence[ChunkEmbedding],
    ) -> int:
        """Replace all vectors associated with a document."""

        self.ensure_collection()

        self._validate_embeddings(
            document_id=document_id,
            embeddings=embeddings,
        )

        self.delete_document(document_id)

        if not embeddings:
            return 0

        points = [self._build_point(embedding) for embedding in embeddings]

        self._client.upsert(
            collection_name=(self._config.collection_name),
            points=points,
            wait=True,
        )

        return len(points)

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        """Delete every chunk belonging to a document."""

        if not self._client.collection_exists(self._config.collection_name):
            return

        self._client.delete(
            collection_name=(self._config.collection_name),
            points_selector=models.FilterSelector(filter=_document_filter(document_id)),
            wait=True,
        )

    def count_document(
        self,
        document_id: str,
    ) -> int:
        """Count vectors belonging to one document."""

        if not self._client.collection_exists(self._config.collection_name):
            return 0

        result = self._client.count(
            collection_name=(self._config.collection_name),
            count_filter=_document_filter(document_id),
            exact=True,
        )

        return result.count

    def count(self) -> int:
        """Count all vectors in the collection."""

        if not self._client.collection_exists(self._config.collection_name):
            return 0

        result = self._client.count(
            collection_name=(self._config.collection_name),
            exact=True,
        )

        return result.count

    def close(self) -> None:
        """Close the Qdrant client."""

        self._client.close()

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        score_threshold: float | None = None,
        document_id: str | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Search for chunks nearest to the query vector."""

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if len(query_vector) != self._config.vector_size:
            raise ValueError(
                "Query vector dimension does not match the vector-store configuration"
            )

        if not self._client.collection_exists(self._config.collection_name):
            return ()

        query_filter = None

        if document_id is not None:
            query_filter = _document_filter(document_id)

        response = self._client.query_points(
            collection_name=(self._config.collection_name),
            query=list(query_vector),
            query_filter=query_filter,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )

        results: list[VectorSearchResult] = []

        for point in response.points:
            payload = point.payload

            if payload is None:
                raise ValueError("Vector search result is missing payload")

            payload_dict = dict(payload)

            chunk = _payload_to_chunk(payload_dict)

            embedding_model = _required_str(
                payload_dict,
                "embedding_model",
            )

            results.append(
                VectorSearchResult(
                    point_id=str(point.id),
                    score=float(point.score),
                    chunk=chunk,
                    embedding_model=embedding_model,
                )
            )

        return tuple(results)

from collections.abc import Sequence
from typing import Protocol

from research_assistant.embeddings.models import (
    ChunkEmbedding,
)
from research_assistant.vector_store.models import (
    VectorSearchResult,
)


class VectorStore(Protocol):
    """Persistence interface for document embeddings."""

    def ensure_collection(self) -> None:
        """Ensure compatible storage exists."""
        ...

    def replace_document(
        self,
        document_id: str,
        embeddings: Sequence[ChunkEmbedding],
    ) -> int:
        """Replace all stored chunks for one document."""
        ...

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        """Delete all chunks belonging to a document."""
        ...

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        score_threshold: float | None = None,
        document_id: str | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        """Find chunks nearest to a query vector."""
        ...

    def count_document(
        self,
        document_id: str,
    ) -> int:
        """Count stored chunks for a document."""
        ...

    def count(self) -> int:
        """Count all stored chunks."""
        ...

    def close(self) -> None:
        """Release vector-store resources."""
        ...

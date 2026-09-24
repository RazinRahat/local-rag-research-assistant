from research_assistant.embeddings.embedder import (
    EmbeddingProvider,
)
from research_assistant.retrieval.models import (
    RetrievalConfig,
    RetrievalResult,
)
from research_assistant.vector_store.base import (
    VectorStore,
)


class SemanticRetriever:
    """Retrieve semantically relevant document chunks."""

    def __init__(
        self,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Retrieve ranked evidence for a natural-language query."""

        clean_query = query.strip()

        if not clean_query:
            raise ValueError("Retrieval query cannot be empty")

        retrieval_config = config or RetrievalConfig()

        query_vector = self._embedder.embed_query(clean_query)

        matches = self._vector_store.search(
            query_vector=query_vector,
            top_k=retrieval_config.top_k,
            score_threshold=(retrieval_config.score_threshold),
            document_id=(retrieval_config.document_id),
        )

        return tuple(
            RetrievalResult(
                rank=rank,
                score=match.score,
                chunk=match.chunk,
            )
            for rank, match in enumerate(
                matches,
                start=1,
            )
        )

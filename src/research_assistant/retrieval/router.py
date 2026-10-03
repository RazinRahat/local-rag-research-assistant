from research_assistant.retrieval.base import (
    Retriever,
)
from research_assistant.retrieval.models import (
    RetrievalConfig,
    RetrievalMode,
    RetrievalResult,
)


class RetrievalRouter:
    """Route retrieval requests to a configured strategy."""

    def __init__(
        self,
        *,
        dense_retriever: Retriever,
        lexical_retriever: Retriever,
        hybrid_retriever: Retriever,
        hybrid_reranked_retriever: Retriever,
    ) -> None:
        self._retrievers: dict[
            RetrievalMode,
            Retriever,
        ] = {
            RetrievalMode.DENSE: (dense_retriever),
            RetrievalMode.LEXICAL: (lexical_retriever),
            RetrievalMode.HYBRID: (hybrid_retriever),
            RetrievalMode.HYBRID_RERANKED: (hybrid_reranked_retriever),
        }

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Retrieve using the configured strategy."""

        retrieval_config = config or RetrievalConfig()

        retriever = self._retrievers[retrieval_config.mode]

        return retriever.retrieve(
            query,
            retrieval_config,
        )

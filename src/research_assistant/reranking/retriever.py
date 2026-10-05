from research_assistant.reranking.base import (
    Reranker,
)
from research_assistant.reranking.models import (
    RerankingConfig,
)
from research_assistant.retrieval.base import (
    Retriever,
)
from research_assistant.retrieval.models import (
    RetrievalConfig,
    RetrievalResult,
)


class RerankingRetriever:
    """Retrieve a candidate pool and rerank its strongest subset."""

    def __init__(
        self,
        *,
        retriever: Retriever,
        reranker: Reranker,
        config: RerankingConfig | None = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._config = config or RerankingConfig()

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> tuple[
        RetrievalResult,
        ...,
    ]:
        """Retrieve candidates and apply second-stage reranking."""

        clean_query = query.strip()

        if not clean_query:
            raise ValueError("Retrieval query cannot be empty")

        retrieval_config = config or RetrievalConfig()

        candidate_top_k = max(
            retrieval_config.top_k,
            self._config.candidate_pool_size,
        )

        candidate_config = retrieval_config.model_copy(
            update={
                "top_k": (candidate_top_k),
            }
        )

        candidates = self._retriever.retrieve(
            clean_query,
            candidate_config,
        )

        if not candidates:
            return ()

        rerank_candidate_count = max(
            retrieval_config.top_k,
            self._config.rerank_pool_size,
        )

        rerank_candidates = candidates[:rerank_candidate_count]

        return self._reranker.rerank(
            clean_query,
            rerank_candidates,
            top_k=(retrieval_config.top_k),
        )

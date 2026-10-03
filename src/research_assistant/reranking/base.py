from collections.abc import Sequence
from typing import Protocol

from research_assistant.retrieval.models import (
    RetrievalResult,
)


class Reranker(Protocol):
    """Interface for second-stage evidence reranking."""

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> tuple[RetrievalResult, ...]:
        """Reorder retrieval candidates by query relevance."""
        ...

from typing import Protocol

from research_assistant.retrieval.models import (
    RetrievalConfig,
    RetrievalResult,
)


class Retriever(Protocol):
    """Interface for evidence retrieval."""

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Return ranked evidence for a query."""
        ...

from pydantic import BaseModel, ConfigDict, Field

from research_assistant.chunking.models import (
    DocumentChunk,
)


class RetrievalConfig(BaseModel):
    """Configuration for semantic retrieval."""

    model_config = ConfigDict(frozen=True)

    top_k: int = Field(
        default=5,
        ge=1,
    )

    score_threshold: float | None = None

    document_id: str | None = None


class RetrievalResult(BaseModel):
    """Ranked evidence returned by semantic retrieval."""

    model_config = ConfigDict(frozen=True)

    rank: int = Field(
        ge=1,
    )

    score: float

    chunk: DocumentChunk

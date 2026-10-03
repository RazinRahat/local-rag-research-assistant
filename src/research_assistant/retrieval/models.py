from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)

from research_assistant.chunking.models import (
    DocumentChunk,
)


class RetrievalMode(StrEnum):
    """Available retrieval strategies."""

    DENSE = "dense"
    LEXICAL = "lexical"
    HYBRID = "hybrid"
    HYBRID_RERANKED = "hybrid_reranked"


class RetrievalConfig(BaseModel):
    """Configuration for evidence retrieval."""

    model_config = ConfigDict(
        frozen=True,
    )

    top_k: int = Field(
        default=5,
        ge=1,
    )

    score_threshold: float | None = None

    document_id: str | None = None

    mode: RetrievalMode = RetrievalMode.DENSE


class RetrievalResult(BaseModel):
    """Ranked evidence returned by retrieval."""

    model_config = ConfigDict(
        frozen=True,
    )

    rank: int = Field(
        ge=1,
    )

    score: float

    chunk: DocumentChunk

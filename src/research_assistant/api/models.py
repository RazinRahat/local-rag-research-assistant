"""HTTP request and response models for the research API."""

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from research_assistant.retrieval.models import (
    RetrievalConfig,
    RetrievalResult,
)


class RetrievalOptions(BaseModel):
    """Retrieval settings shared by search and RAG endpoints."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    score_threshold: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )

    document_id: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )

    def to_retrieval_config(self) -> RetrievalConfig:
        """Convert HTTP options into the existing domain model."""

        return RetrievalConfig(
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            document_id=self.document_id,
        )


class SearchRequest(RetrievalOptions):
    """Request body for semantic retrieval."""

    query: str = Field(
        min_length=1,
        max_length=4000,
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        clean_value = value.strip()

        if not clean_value:
            raise ValueError("Search query cannot be blank")

        return clean_value


class QueryRequest(RetrievalOptions):
    """Request body for cited RAG question answering."""

    question: str = Field(
        min_length=1,
        max_length=4000,
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        clean_value = value.strip()

        if not clean_value:
            raise ValueError("Question cannot be blank")

        return clean_value


class SearchResponse(BaseModel):
    """Response returned by the semantic-search endpoint."""

    model_config = ConfigDict(frozen=True)

    query: str

    results: tuple[RetrievalResult, ...]

from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from research_assistant.rag.models import (
    EvidenceBlock,
    RAGResponse,
)


class CitationReference(BaseModel):
    """One inline citation occurrence in generated text."""

    model_config = ConfigDict(frozen=True)

    source_id: str

    start_index: int = Field(
        ge=0,
    )

    end_index: int = Field(
        gt=0,
    )

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.end_index <= self.start_index:
            raise ValueError("Citation end_index must be greater than start_index")

        return self


class CitationSource(BaseModel):
    """Validated evidence referenced by the generated answer."""

    model_config = ConfigDict(frozen=True)

    source_id: str
    evidence: EvidenceBlock


class CitationValidationResult(BaseModel):
    """Validated citation references and unique sources."""

    model_config = ConfigDict(frozen=True)

    references: tuple[CitationReference, ...]
    sources: tuple[CitationSource, ...]


class CitedRAGResponse(BaseModel):
    """RAG response with validated citation metadata."""

    model_config = ConfigDict(frozen=True)

    rag: RAGResponse

    citations: CitationValidationResult

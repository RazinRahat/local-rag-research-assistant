from pydantic import BaseModel, ConfigDict, Field

from research_assistant.chunking.models import DocumentChunk
from research_assistant.generation.models import (
    ChatMessage,
    GenerationResult,
)


class ContextConfig(BaseModel):
    """Configuration for RAG context construction."""

    model_config = ConfigDict(frozen=True)

    safety_margin_tokens: int = Field(
        default=512,
        ge=0,
    )

    overlap_dedupe_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
    )


class EvidenceBlock(BaseModel):
    """Evidence selected for the LLM prompt."""

    model_config = ConfigDict(frozen=True)

    source_id: str

    retrieval_rank: int = Field(
        ge=1,
    )

    score: float

    chunk: DocumentChunk


class ContextBuildResult(BaseModel):
    """Result of bounded evidence-context construction."""

    model_config = ConfigDict(frozen=True)

    messages: tuple[ChatMessage, ...]

    evidence: tuple[EvidenceBlock, ...]

    estimated_prompt_tokens: int = Field(
        ge=0,
    )

    prompt_budget_tokens: int = Field(
        ge=0,
    )

    skipped_redundant: int = Field(
        default=0,
        ge=0,
    )

    skipped_for_budget: int = Field(
        default=0,
        ge=0,
    )

    truncated: bool = False


class RAGResponse(BaseModel):
    """Complete response from the RAG pipeline."""

    model_config = ConfigDict(frozen=True)

    question: str
    answer: str

    evidence: tuple[EvidenceBlock, ...]

    retrieved_count: int = Field(
        ge=0,
    )

    used_evidence_count: int = Field(
        ge=0,
    )

    estimated_prompt_tokens: int = Field(
        default=0,
        ge=0,
    )

    actual_prompt_tokens: int = Field(
        default=0,
        ge=0,
    )

    context_truncated: bool = False

    insufficient_evidence: bool = False

    generation: GenerationResult | None = None

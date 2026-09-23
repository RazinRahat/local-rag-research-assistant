from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChunkingConfig(BaseModel):
    """Configuration for token-based document chunking."""

    model_config = ConfigDict(frozen=True)

    chunk_size_tokens: int = Field(default=384, ge=1)
    chunk_overlap_tokens: int = Field(default=64, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError(
                "chunk_overlap_tokens must be smaller than chunk_size_tokens"
            )

        return self


class DocumentChunk(BaseModel):
    """A retrieval unit derived from a document page."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str

    document_id: str
    file_name: str

    page_number: int = Field(ge=1)

    chunk_index: int = Field(ge=0)
    page_chunk_index: int = Field(ge=0)

    text: str

    token_count: int = Field(ge=1)
    char_count: int = Field(ge=1)

    token_start: int = Field(ge=0)
    token_end: int = Field(gt=0)

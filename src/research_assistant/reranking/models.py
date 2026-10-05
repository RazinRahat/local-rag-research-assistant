from typing import Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class CrossEncoderConfig(BaseModel):
    """Configuration for local cross-encoder reranking."""

    model_config = ConfigDict(
        frozen=True,
    )

    model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"

    batch_size: int = Field(
        default=8,
        ge=1,
    )

    max_length: int = Field(
        default=512,
        ge=8,
    )

    device: str | None = None


class RerankingConfig(BaseModel):
    """Configuration for second-stage reranking."""

    model_config = ConfigDict(
        frozen=True,
    )

    candidate_pool_size: int = Field(
        default=20,
        ge=1,
    )

    rerank_pool_size: int = Field(
        default=10,
        ge=1,
    )

    @model_validator(mode="after")
    def validate_pool_sizes(
        self,
    ) -> Self:
        if self.rerank_pool_size > self.candidate_pool_size:
            raise ValueError("rerank_pool_size cannot exceed candidate_pool_size")

        return self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
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

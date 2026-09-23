from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from research_assistant.chunking.models import DocumentChunk


class EmbeddingConfig(BaseModel):
    """Configuration for local embedding generation."""

    model_config = ConfigDict(frozen=True)

    model_name: str = "BAAI/bge-small-en-v1.5"

    batch_size: int = Field(
        default=32,
        ge=1,
    )

    normalize_embeddings: bool = True

    query_instruction: str = "Represent this sentence for searching relevant passages: "


class ChunkEmbedding(BaseModel):
    """Embedding associated with its source document chunk."""

    model_config = ConfigDict(frozen=True)

    chunk: DocumentChunk

    model_name: str

    vector: tuple[float, ...]

    dimension: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_dimension(self) -> Self:
        if len(self.vector) != self.dimension:
            raise ValueError(
                "Embedding vector length does not match declared dimension"
            )

        return self

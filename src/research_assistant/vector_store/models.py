from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from research_assistant.chunking.models import DocumentChunk


class VectorStoreConfig(BaseModel):
    """Configuration for persistent vector storage."""

    model_config = ConfigDict(frozen=True)

    path: Path = Path("data/vector_store/qdrant")

    collection_name: str = "research_chunks_bge_small_en_v1_5"

    embedding_model: str = "BAAI/bge-small-en-v1.5"

    vector_size: int = Field(
        default=384,
        ge=1,
    )


class VectorSearchResult(BaseModel):
    """Vector-store search result independent of database implementation."""

    model_config = ConfigDict(frozen=True)

    point_id: str

    score: float

    chunk: DocumentChunk

    embedding_model: str

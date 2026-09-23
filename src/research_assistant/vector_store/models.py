from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


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

import pytest
from pydantic import ValidationError

from research_assistant.chunking.models import DocumentChunk
from research_assistant.embeddings.models import (
    ChunkEmbedding,
)


def _make_chunk() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="chunk-0",
        document_id="a" * 64,
        file_name="paper.pdf",
        page_number=1,
        chunk_index=0,
        page_chunk_index=0,
        text="Embedding test.",
        token_count=3,
        char_count=15,
        token_start=0,
        token_end=3,
    )


def test_embedding_dimension_matches_vector() -> None:
    embedding = ChunkEmbedding(
        chunk=_make_chunk(),
        model_name="test-model",
        vector=(0.1, 0.2, 0.3),
        dimension=3,
    )

    assert embedding.dimension == 3


def test_embedding_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValidationError):
        ChunkEmbedding(
            chunk=_make_chunk(),
            model_name="test-model",
            vector=(0.1, 0.2),
            dimension=3,
        )

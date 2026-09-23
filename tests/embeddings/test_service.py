from collections.abc import Sequence

from research_assistant.chunking.models import DocumentChunk
from research_assistant.embeddings.embedder import (
    EmbeddingVector,
)
from research_assistant.embeddings.service import (
    embed_chunks,
)


class FakeEmbedder:
    @property
    def model_name(self) -> str:
        return "fake-embedding-model"

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[EmbeddingVector, ...]:
        return tuple((1.0, 0.0, 0.0) for _ in texts)

    def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        return (1.0, 0.0, 0.0)


def _make_chunk(
    chunk_index: int,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"chunk-{chunk_index}",
        document_id="a" * 64,
        file_name="paper.pdf",
        page_number=1,
        chunk_index=chunk_index,
        page_chunk_index=chunk_index,
        text=text,
        token_count=10,
        char_count=len(text),
        token_start=chunk_index * 10,
        token_end=(chunk_index + 1) * 10,
    )


def test_embed_chunks_preserves_provenance() -> None:
    chunks = (
        _make_chunk(
            0,
            "Retrieval augmented generation.",
        ),
        _make_chunk(
            1,
            "Dense retrieval uses embeddings.",
        ),
    )

    result = embed_chunks(
        chunks=chunks,
        embedder=FakeEmbedder(),
    )

    assert len(result) == 2

    assert result[0].chunk == chunks[0]
    assert result[1].chunk == chunks[1]

    assert result[0].model_name == ("fake-embedding-model")

    assert result[0].dimension == 3

    assert result[0].vector == (
        1.0,
        0.0,
        0.0,
    )

from collections.abc import Sequence

from research_assistant.chunking.models import DocumentChunk
from research_assistant.embeddings.embedder import (
    EmbeddingProvider,
)
from research_assistant.embeddings.models import (
    ChunkEmbedding,
)


def embed_chunks(
    chunks: Sequence[DocumentChunk],
    embedder: EmbeddingProvider,
) -> tuple[ChunkEmbedding, ...]:
    """Generate embeddings while preserving chunk provenance."""

    if not chunks:
        return ()

    texts = [chunk.text for chunk in chunks]

    vectors = embedder.embed_documents(texts)

    if len(vectors) != len(chunks):
        raise ValueError("Embedding count does not match chunk count")

    return tuple(
        ChunkEmbedding(
            chunk=chunk,
            model_name=embedder.model_name,
            vector=vector,
            dimension=embedder.dimension,
        )
        for chunk, vector in zip(
            chunks,
            vectors,
            strict=True,
        )
    )

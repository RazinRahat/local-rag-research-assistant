from collections.abc import Sequence

import pytest

from research_assistant.chunking.models import (
    DocumentChunk,
)
from research_assistant.embeddings.embedder import (
    EmbeddingVector,
)
from research_assistant.embeddings.models import (
    ChunkEmbedding,
)
from research_assistant.retrieval.models import (
    RetrievalConfig,
)
from research_assistant.retrieval.retriever import (
    SemanticRetriever,
)
from research_assistant.vector_store.models import (
    VectorSearchResult,
)


class FakeEmbedder:
    @property
    def model_name(self) -> str:
        return "fake-model"

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
) -> DocumentChunk:
    text = f"Chunk {chunk_index}"

    return DocumentChunk(
        chunk_id=f"chunk-{chunk_index}",
        document_id="document-a",
        file_name="paper.pdf",
        page_number=chunk_index + 1,
        chunk_index=chunk_index,
        page_chunk_index=0,
        text=text,
        token_count=2,
        char_count=len(text),
        token_start=0,
        token_end=2,
    )


class FakeVectorStore:
    def ensure_collection(self) -> None:
        return None

    def replace_document(
        self,
        document_id: str,
        embeddings: Sequence[ChunkEmbedding],
    ) -> int:
        return len(embeddings)

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        return None

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int,
        score_threshold: float | None = None,
        document_id: str | None = None,
    ) -> tuple[VectorSearchResult, ...]:
        results = (
            VectorSearchResult(
                point_id="point-1",
                score=0.91,
                chunk=_make_chunk(0),
                embedding_model="fake-model",
            ),
            VectorSearchResult(
                point_id="point-2",
                score=0.82,
                chunk=_make_chunk(1),
                embedding_model="fake-model",
            ),
        )

        return results[:top_k]

    def count_document(
        self,
        document_id: str,
    ) -> int:
        return 2

    def count(self) -> int:
        return 2

    def close(self) -> None:
        return None


def test_retriever_returns_ranked_results() -> None:
    retriever = SemanticRetriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    results = retriever.retrieve("How does attention work?")

    assert len(results) == 2

    assert results[0].rank == 1
    assert results[1].rank == 2

    assert results[0].score == 0.91
    assert results[1].score == 0.82

    assert results[0].chunk.page_number == 1


def test_retriever_respects_top_k() -> None:
    retriever = SemanticRetriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    results = retriever.retrieve(
        "How does attention work?",
        config=RetrievalConfig(
            top_k=1,
        ),
    )

    assert len(results) == 1


def test_retriever_rejects_empty_query() -> None:
    retriever = SemanticRetriever(
        embedder=FakeEmbedder(),
        vector_store=FakeVectorStore(),
    )

    with pytest.raises(ValueError):
        retriever.retrieve("   ")

from __future__ import annotations

from collections.abc import Sequence

import pytest

from research_assistant.chunking.models import (
    DocumentChunk,
)
from research_assistant.retrieval.lexical import (
    BM25Config,
    BM25Retriever,
    tokenize_lexically,
)
from research_assistant.retrieval.models import (
    RetrievalConfig,
)

DOCUMENT_A = "a" * 64
DOCUMENT_B = "b" * 64


def make_chunk(
    *,
    chunk_id: str,
    document_id: str,
    text: str,
    chunk_index: int,
    page_number: int = 1,
) -> DocumentChunk:
    token_count = max(
        1,
        len(text.split()),
    )

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        file_name=f"{document_id[0]}.pdf",
        page_number=page_number,
        chunk_index=chunk_index,
        page_chunk_index=chunk_index,
        text=text,
        token_count=token_count,
        char_count=len(text),
        token_start=0,
        token_end=token_count,
    )


class FakeChunkCorpus:
    def __init__(
        self,
        chunks: Sequence[DocumentChunk],
    ) -> None:
        self._chunks = tuple(chunks)

    def list_chunks(
        self,
        document_id: str | None = None,
    ) -> tuple[DocumentChunk, ...]:
        if document_id is None:
            return self._chunks

        return tuple(
            chunk for chunk in self._chunks if chunk.document_id == document_id
        )


def test_tokenize_lexically_normalises_case() -> None:
    assert tokenize_lexically("Transformer ATTENTION attention") == (
        "transformer",
        "attention",
        "attention",
    )


def test_tokenizer_preserves_research_terms() -> None:
    assert tokenize_lexically("BERT-base F1-score model's output") == (
        "bert-base",
        "f1-score",
        "model's",
        "output",
    )


def test_tokenizer_ignores_surrounding_punctuation() -> None:
    assert tokenize_lexically("(Attention), transformer; Adam.") == (
        "attention",
        "transformer",
        "adam",
    )


def test_bm25_prefers_exact_lexical_overlap() -> None:
    exact = make_chunk(
        chunk_id="exact",
        document_id=DOCUMENT_A,
        chunk_index=0,
        text=("The Adam optimizer was used for model training."),
    )

    unrelated = make_chunk(
        chunk_id="unrelated",
        document_id=DOCUMENT_A,
        chunk_index=1,
        text=("The dataset contains images of domestic animals."),
    )

    retriever = BM25Retriever(
        FakeChunkCorpus(
            (
                unrelated,
                exact,
            )
        )
    )

    results = retriever.retrieve("Adam optimizer")

    assert results

    assert results[0].chunk.chunk_id == "exact"

    assert results[0].rank == 1
    assert results[0].score > 0


def test_bm25_returns_only_matching_chunks() -> None:
    chunks = (
        make_chunk(
            chunk_id="one",
            document_id=DOCUMENT_A,
            chunk_index=0,
            text=("Attention mechanism."),
        ),
        make_chunk(
            chunk_id="two",
            document_id=DOCUMENT_A,
            chunk_index=1,
            text=("Convolutional network."),
        ),
    )

    retriever = BM25Retriever(FakeChunkCorpus(chunks))

    results = retriever.retrieve("attention")

    assert tuple(result.chunk.chunk_id for result in results) == ("one",)


def test_bm25_respects_top_k() -> None:
    chunks = tuple(
        make_chunk(
            chunk_id=f"chunk-{index}",
            document_id=DOCUMENT_A,
            chunk_index=index,
            text=(f"attention model example {index}"),
        )
        for index in range(5)
    )

    retriever = BM25Retriever(FakeChunkCorpus(chunks))

    results = retriever.retrieve(
        "attention",
        RetrievalConfig(
            top_k=2,
        ),
    )

    assert len(results) == 2

    assert tuple(result.rank for result in results) == (
        1,
        2,
    )


def test_bm25_respects_document_scope() -> None:
    chunk_a = make_chunk(
        chunk_id="a-chunk",
        document_id=DOCUMENT_A,
        chunk_index=0,
        text=("attention transformer"),
    )

    chunk_b = make_chunk(
        chunk_id="b-chunk",
        document_id=DOCUMENT_B,
        chunk_index=0,
        text=("attention attention transformer"),
    )

    retriever = BM25Retriever(
        FakeChunkCorpus(
            (
                chunk_a,
                chunk_b,
            )
        )
    )

    results = retriever.retrieve(
        "attention transformer",
        RetrievalConfig(
            document_id=DOCUMENT_A,
        ),
    )

    assert len(results) == 1

    assert results[0].chunk.document_id == DOCUMENT_A

    assert results[0].chunk.chunk_id == "a-chunk"


def test_bm25_returns_empty_when_corpus_is_empty() -> None:
    retriever = BM25Retriever(FakeChunkCorpus(()))

    results = retriever.retrieve("attention")

    assert results == ()


def test_bm25_rejects_blank_query() -> None:
    retriever = BM25Retriever(FakeChunkCorpus(()))

    with pytest.raises(
        ValueError,
        match=("Retrieval query cannot be empty"),
    ):
        retriever.retrieve("   ")


def test_bm25_returns_empty_without_matching_terms() -> None:
    chunk = make_chunk(
        chunk_id="chunk",
        document_id=DOCUMENT_A,
        chunk_index=0,
        text=("attention transformer"),
    )

    retriever = BM25Retriever(FakeChunkCorpus((chunk,)))

    assert retriever.retrieve("photosynthesis") == ()


def test_bm25_rejects_dense_score_threshold() -> None:
    chunk = make_chunk(
        chunk_id="chunk",
        document_id=DOCUMENT_A,
        chunk_index=0,
        text=("attention transformer"),
    )

    retriever = BM25Retriever(FakeChunkCorpus((chunk,)))

    with pytest.raises(
        ValueError,
        match=("score_threshold is not supported"),
    ):
        retriever.retrieve(
            "attention",
            RetrievalConfig(
                score_threshold=0.5,
            ),
        )


def test_bm25_configuration_rejects_non_positive_k1() -> None:
    with pytest.raises(
        ValueError,
        match=("k1 must be greater than 0"),
    ):
        BM25Config(
            k1=0,
        )


@pytest.mark.parametrize(
    "b",
    (
        -0.1,
        1.1,
    ),
)
def test_bm25_configuration_rejects_invalid_b(
    b: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=("b must be between 0 and 1"),
    ):
        BM25Config(
            b=b,
        )


def test_bm25_configuration_accepts_boundary_b_values() -> None:
    assert BM25Config(b=0.0).b == 0.0

    assert BM25Config(b=1.0).b == 1.0


def test_bm25_returns_results_as_tuple() -> None:
    chunk = make_chunk(
        chunk_id="chunk",
        document_id=DOCUMENT_A,
        chunk_index=0,
        text=("attention transformer"),
    )

    retriever = BM25Retriever(FakeChunkCorpus((chunk,)))

    results = retriever.retrieve("attention")

    assert isinstance(
        results,
        tuple,
    )


def test_bm25_ranks_results_sequentially() -> None:
    chunks = (
        make_chunk(
            chunk_id="strong",
            document_id=DOCUMENT_A,
            chunk_index=0,
            text=("attention attention attention transformer"),
        ),
        make_chunk(
            chunk_id="weaker",
            document_id=DOCUMENT_A,
            chunk_index=1,
            text=("attention transformer"),
        ),
    )

    retriever = BM25Retriever(FakeChunkCorpus(chunks))

    results = retriever.retrieve("attention")

    assert tuple(result.rank for result in results) == tuple(
        range(
            1,
            len(results) + 1,
        )
    )


def test_bm25_uses_chunk_id_as_deterministic_tie_breaker() -> None:
    chunk_b = make_chunk(
        chunk_id="b-chunk",
        document_id=DOCUMENT_A,
        chunk_index=1,
        text="attention",
    )

    chunk_a = make_chunk(
        chunk_id="a-chunk",
        document_id=DOCUMENT_A,
        chunk_index=0,
        text="attention",
    )

    retriever = BM25Retriever(
        FakeChunkCorpus(
            (
                chunk_b,
                chunk_a,
            )
        )
    )

    results = retriever.retrieve("attention")

    assert tuple(result.chunk.chunk_id for result in results) == (
        "a-chunk",
        "b-chunk",
    )

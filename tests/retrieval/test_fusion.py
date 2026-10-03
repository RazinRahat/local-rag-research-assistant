from __future__ import annotations

import pytest

from research_assistant.chunking.models import (
    DocumentChunk,
)
from research_assistant.retrieval.fusion import (
    RRFConfig,
    reciprocal_rank_fusion,
)
from research_assistant.retrieval.models import (
    RetrievalResult,
)

DOCUMENT_ID = "a" * 64


def make_chunk(
    *,
    chunk_id: str,
    text: str | None = None,
    chunk_index: int = 0,
) -> DocumentChunk:
    chunk_text = text or f"Text for {chunk_id}"

    token_count = max(
        1,
        len(chunk_text.split()),
    )

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=DOCUMENT_ID,
        file_name="paper.pdf",
        page_number=1,
        chunk_index=chunk_index,
        page_chunk_index=chunk_index,
        text=chunk_text,
        token_count=token_count,
        char_count=len(chunk_text),
        token_start=0,
        token_end=token_count,
    )


def make_result(
    *,
    rank: int,
    chunk: DocumentChunk,
    score: float = 1.0,
) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=score,
        chunk=chunk,
    )


def test_rrf_configuration_uses_standard_baseline() -> None:
    config = RRFConfig()

    assert config.rank_constant == 60


def test_rrf_rejects_invalid_rank_constant() -> None:
    with pytest.raises(
        ValueError,
        match=("rank_constant must be at least 1"),
    ):
        RRFConfig(rank_constant=0)


def test_rrf_rejects_invalid_top_k() -> None:
    with pytest.raises(
        ValueError,
        match=("top_k must be at least 1"),
    ):
        reciprocal_rank_fusion(
            (),
            top_k=0,
        )


def test_rrf_returns_empty_without_rankings() -> None:
    results = reciprocal_rank_fusion(
        (),
        top_k=5,
    )

    assert results == ()


def test_rrf_preserves_single_ranking_order() -> None:
    chunk_a = make_chunk(
        chunk_id="a",
        chunk_index=0,
    )

    chunk_b = make_chunk(
        chunk_id="b",
        chunk_index=1,
    )

    ranking = (
        make_result(
            rank=1,
            chunk=chunk_a,
            score=0.91,
        ),
        make_result(
            rank=2,
            chunk=chunk_b,
            score=0.82,
        ),
    )

    results = reciprocal_rank_fusion(
        (ranking,),
        top_k=5,
    )

    assert tuple(result.chunk.chunk_id for result in results) == (
        "a",
        "b",
    )

    assert tuple(result.rank for result in results) == (
        1,
        2,
    )


def test_rrf_rewards_consensus_between_rankings() -> None:
    chunk_a = make_chunk(
        chunk_id="a",
        chunk_index=0,
    )

    chunk_b = make_chunk(
        chunk_id="b",
        chunk_index=1,
    )

    chunk_c = make_chunk(
        chunk_id="c",
        chunk_index=2,
    )

    dense = (
        make_result(
            rank=1,
            chunk=chunk_a,
            score=0.93,
        ),
        make_result(
            rank=2,
            chunk=chunk_b,
            score=0.84,
        ),
    )

    lexical = (
        make_result(
            rank=1,
            chunk=chunk_b,
            score=6.4,
        ),
        make_result(
            rank=2,
            chunk=chunk_c,
            score=4.1,
        ),
    )

    results = reciprocal_rank_fusion(
        (
            dense,
            lexical,
        ),
        top_k=3,
    )

    assert results[0].chunk.chunk_id == "b"

    expected_score = 1 / 62 + 1 / 61

    assert results[0].score == pytest.approx(expected_score)


def test_rrf_does_not_compare_original_scores() -> None:
    chunk_a = make_chunk(
        chunk_id="a",
        chunk_index=0,
    )

    chunk_b = make_chunk(
        chunk_id="b",
        chunk_index=1,
    )

    dense = (
        make_result(
            rank=1,
            chunk=chunk_a,
            score=0.01,
        ),
        make_result(
            rank=2,
            chunk=chunk_b,
            score=9999.0,
        ),
    )

    results = reciprocal_rank_fusion(
        (dense,),
        top_k=2,
    )

    assert results[0].chunk.chunk_id == "a"

    assert results[1].chunk.chunk_id == "b"


def test_rrf_includes_chunks_seen_by_only_one_retriever() -> None:
    chunk_a = make_chunk(
        chunk_id="a",
        chunk_index=0,
    )

    chunk_b = make_chunk(
        chunk_id="b",
        chunk_index=1,
    )

    dense = (
        make_result(
            rank=1,
            chunk=chunk_a,
        ),
    )

    lexical = (
        make_result(
            rank=1,
            chunk=chunk_b,
        ),
    )

    results = reciprocal_rank_fusion(
        (
            dense,
            lexical,
        ),
        top_k=5,
    )

    assert {result.chunk.chunk_id for result in results} == {
        "a",
        "b",
    }


def test_rrf_respects_top_k() -> None:
    chunks = tuple(
        make_chunk(
            chunk_id=f"chunk-{index}",
            chunk_index=index,
        )
        for index in range(5)
    )

    ranking = tuple(
        make_result(
            rank=index + 1,
            chunk=chunk,
        )
        for index, chunk in enumerate(chunks)
    )

    results = reciprocal_rank_fusion(
        (ranking,),
        top_k=2,
    )

    assert len(results) == 2

    assert tuple(result.rank for result in results) == (
        1,
        2,
    )


def test_rrf_uses_deterministic_chunk_id_tie_breaker() -> None:
    chunk_b = make_chunk(
        chunk_id="b",
        chunk_index=1,
    )

    chunk_a = make_chunk(
        chunk_id="a",
        chunk_index=0,
    )

    dense = (
        make_result(
            rank=1,
            chunk=chunk_b,
        ),
    )

    lexical = (
        make_result(
            rank=1,
            chunk=chunk_a,
        ),
    )

    results = reciprocal_rank_fusion(
        (
            dense,
            lexical,
        ),
        top_k=2,
    )

    assert tuple(result.chunk.chunk_id for result in results) == (
        "a",
        "b",
    )


def test_rrf_rejects_non_sequential_input_ranks() -> None:
    chunk = make_chunk(chunk_id="chunk")

    ranking = (
        make_result(
            rank=2,
            chunk=chunk,
        ),
    )

    with pytest.raises(
        ValueError,
        match=("sequential ranks starting at 1"),
    ):
        reciprocal_rank_fusion(
            (ranking,),
            top_k=5,
        )


def test_rrf_rejects_duplicate_chunk_within_ranking() -> None:
    chunk = make_chunk(chunk_id="chunk")

    ranking = (
        make_result(
            rank=1,
            chunk=chunk,
        ),
        make_result(
            rank=2,
            chunk=chunk,
        ),
    )

    with pytest.raises(
        ValueError,
        match=("duplicate chunk_id"),
    ):
        reciprocal_rank_fusion(
            (ranking,),
            top_k=5,
        )


def test_rrf_rejects_conflicting_chunk_provenance() -> None:
    first_chunk = make_chunk(
        chunk_id="shared",
        text="Original chunk text.",
    )

    conflicting_chunk = make_chunk(
        chunk_id="shared",
        text="Different chunk text.",
    )

    dense = (
        make_result(
            rank=1,
            chunk=first_chunk,
        ),
    )

    lexical = (
        make_result(
            rank=1,
            chunk=conflicting_chunk,
        ),
    )

    with pytest.raises(
        ValueError,
        match=("Conflicting chunk provenance"),
    ):
        reciprocal_rank_fusion(
            (
                dense,
                lexical,
            ),
            top_k=5,
        )


def test_rrf_returns_sequential_final_ranks() -> None:
    chunk_a = make_chunk(
        chunk_id="a",
        chunk_index=0,
    )

    chunk_b = make_chunk(
        chunk_id="b",
        chunk_index=1,
    )

    chunk_c = make_chunk(
        chunk_id="c",
        chunk_index=2,
    )

    dense = (
        make_result(
            rank=1,
            chunk=chunk_a,
        ),
        make_result(
            rank=2,
            chunk=chunk_b,
        ),
        make_result(
            rank=3,
            chunk=chunk_c,
        ),
    )

    results = reciprocal_rank_fusion(
        (dense,),
        top_k=3,
    )

    assert tuple(result.rank for result in results) == (
        1,
        2,
        3,
    )

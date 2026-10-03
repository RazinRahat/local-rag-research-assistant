from __future__ import annotations

import pytest

from research_assistant.chunking.models import (
    DocumentChunk,
)
from research_assistant.retrieval.base import (
    Retriever,
)
from research_assistant.retrieval.fusion import (
    RRFConfig,
)
from research_assistant.retrieval.hybrid import (
    HybridConfig,
    HybridRetriever,
)
from research_assistant.retrieval.models import (
    RetrievalConfig,
    RetrievalResult,
)

DOCUMENT_A = "a" * 64
DOCUMENT_B = "b" * 64


def make_chunk(
    *,
    chunk_id: str,
    chunk_index: int,
    document_id: str = DOCUMENT_A,
    text: str | None = None,
) -> DocumentChunk:
    chunk_text = text or f"Evidence for {chunk_id}"

    token_count = max(
        1,
        len(chunk_text.split()),
    )

    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        file_name=(f"{document_id[0]}.pdf"),
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
    score: float,
) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=score,
        chunk=chunk,
    )


class FakeRetriever:
    def __init__(
        self,
        results: tuple[
            RetrievalResult,
            ...,
        ],
    ) -> None:
        self._results = results

        self.calls: list[
            tuple[
                str,
                RetrievalConfig | None,
            ]
        ] = []

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> tuple[
        RetrievalResult,
        ...,
    ]:
        self.calls.append(
            (
                query,
                config,
            )
        )

        if config is None:
            return self._results

        return self._results[: config.top_k]


def test_hybrid_configuration_defaults() -> None:
    config = HybridConfig()

    assert config.candidate_multiplier == 2

    assert config.rrf_config.rank_constant == 60


def test_hybrid_configuration_rejects_invalid_candidate_multiplier() -> None:
    with pytest.raises(
        ValueError,
        match=("candidate_multiplier must be at least 1"),
    ):
        HybridConfig(candidate_multiplier=0)


def test_hybrid_satisfies_retriever_protocol() -> None:
    dense = FakeRetriever(())
    lexical = FakeRetriever(())

    retriever: Retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

    assert retriever.retrieve("attention") == ()


def test_hybrid_rejects_blank_query() -> None:
    dense = FakeRetriever(())
    lexical = FakeRetriever(())

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

    with pytest.raises(
        ValueError,
        match=("Retrieval query cannot be empty"),
    ):
        retriever.retrieve("   ")

    assert dense.calls == []
    assert lexical.calls == []


def test_hybrid_rejects_score_threshold() -> None:
    dense = FakeRetriever(())
    lexical = FakeRetriever(())

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

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

    assert dense.calls == []
    assert lexical.calls == []


def test_hybrid_overfetches_candidates_from_both_retrievers() -> None:
    dense = FakeRetriever(())
    lexical = FakeRetriever(())

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
        config=HybridConfig(candidate_multiplier=3),
    )

    retriever.retrieve(
        "attention",
        RetrievalConfig(
            top_k=4,
            document_id=DOCUMENT_A,
        ),
    )

    assert len(dense.calls) == 1

    assert len(lexical.calls) == 1

    dense_query, dense_config = dense.calls[0]

    lexical_query, lexical_config = lexical.calls[0]

    assert dense_query == "attention"

    assert lexical_query == "attention"

    assert dense_config is not None

    assert lexical_config is not None

    assert dense_config.top_k == 12

    assert lexical_config.top_k == 12

    assert dense_config.document_id == DOCUMENT_A

    assert lexical_config.document_id == DOCUMENT_A

    assert dense_config.score_threshold is None

    assert lexical_config.score_threshold is None


def test_hybrid_rewards_dense_and_lexical_consensus() -> None:
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

    dense = FakeRetriever(
        (
            make_result(
                rank=1,
                chunk=chunk_a,
                score=0.95,
            ),
            make_result(
                rank=2,
                chunk=chunk_b,
                score=0.87,
            ),
        )
    )

    lexical = FakeRetriever(
        (
            make_result(
                rank=1,
                chunk=chunk_b,
                score=7.2,
            ),
            make_result(
                rank=2,
                chunk=chunk_c,
                score=5.9,
            ),
        )
    )

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

    results = retriever.retrieve(
        "attention transformer",
        RetrievalConfig(
            top_k=3,
        ),
    )

    assert results[0].chunk.chunk_id == "b"

    assert tuple(result.chunk.chunk_id for result in results) == (
        "b",
        "a",
        "c",
    )


def test_hybrid_returns_only_final_top_k() -> None:
    chunks = tuple(
        make_chunk(
            chunk_id=f"chunk-{index}",
            chunk_index=index,
        )
        for index in range(6)
    )

    dense = FakeRetriever(
        tuple(
            make_result(
                rank=index + 1,
                chunk=chunk,
                score=(1.0 - index * 0.1),
            )
            for index, chunk in enumerate(chunks)
        )
    )

    lexical = FakeRetriever(())

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

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


def test_hybrid_works_when_dense_returns_no_results() -> None:
    lexical_chunk = make_chunk(
        chunk_id="lexical",
        chunk_index=0,
    )

    dense = FakeRetriever(())

    lexical = FakeRetriever(
        (
            make_result(
                rank=1,
                chunk=lexical_chunk,
                score=4.2,
            ),
        )
    )

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

    results = retriever.retrieve("exact terminology")

    assert len(results) == 1

    assert results[0].chunk.chunk_id == "lexical"


def test_hybrid_works_when_lexical_returns_no_results() -> None:
    dense_chunk = make_chunk(
        chunk_id="dense",
        chunk_index=0,
    )

    dense = FakeRetriever(
        (
            make_result(
                rank=1,
                chunk=dense_chunk,
                score=0.91,
            ),
        )
    )

    lexical = FakeRetriever(())

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

    results = retriever.retrieve("semantic paraphrase")

    assert len(results) == 1

    assert results[0].chunk.chunk_id == "dense"


def test_hybrid_returns_empty_when_both_retrievers_are_empty() -> None:
    retriever = HybridRetriever(
        dense_retriever=(FakeRetriever(())),
        lexical_retriever=(FakeRetriever(())),
    )

    assert retriever.retrieve("attention") == ()


def test_hybrid_preserves_document_scope() -> None:
    dense = FakeRetriever(())
    lexical = FakeRetriever(())

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

    retriever.retrieve(
        "attention",
        RetrievalConfig(
            top_k=5,
            document_id=DOCUMENT_B,
        ),
    )

    dense_config = dense.calls[0][1]

    lexical_config = lexical.calls[0][1]

    assert dense_config is not None

    assert lexical_config is not None

    assert dense_config.document_id == DOCUMENT_B

    assert lexical_config.document_id == DOCUMENT_B


def test_hybrid_uses_configured_rrf_constant() -> None:
    shared = make_chunk(
        chunk_id="shared",
        chunk_index=0,
    )

    dense = FakeRetriever(
        (
            make_result(
                rank=1,
                chunk=shared,
                score=0.9,
            ),
        )
    )

    lexical = FakeRetriever(
        (
            make_result(
                rank=1,
                chunk=shared,
                score=10.0,
            ),
        )
    )

    retriever = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
        config=HybridConfig(rrf_config=RRFConfig(rank_constant=10)),
    )

    results = retriever.retrieve("attention")

    assert results[0].score == pytest.approx(2 / 11)

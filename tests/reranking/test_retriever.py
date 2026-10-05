from __future__ import annotations

from collections.abc import Sequence

import pytest

from research_assistant.chunking.models import (
    DocumentChunk,
)
from research_assistant.reranking.models import (
    RerankingConfig,
)
from research_assistant.reranking.retriever import (
    RerankingRetriever,
)
from research_assistant.retrieval.base import (
    Retriever,
)
from research_assistant.retrieval.models import (
    RetrievalConfig,
    RetrievalResult,
)

DOCUMENT_ID = "a" * 64


def make_chunk(
    chunk_id: str,
    text: str,
    *,
    chunk_index: int,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=DOCUMENT_ID,
        file_name="paper.pdf",
        page_number=1,
        chunk_index=chunk_index,
        page_chunk_index=chunk_index,
        text=text,
        token_count=10,
        char_count=len(text),
        token_start=0,
        token_end=10,
    )


def make_result(
    rank: int,
    chunk_id: str,
    text: str,
    *,
    score: float = 0.0,
) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=score,
        chunk=make_chunk(
            chunk_id,
            text,
            chunk_index=rank - 1,
        ),
    )


class FakeRetriever:
    def __init__(
        self,
        results: Sequence[RetrievalResult],
    ) -> None:
        self._results = tuple(results)

        self.calls: list[
            tuple[
                str,
                RetrievalConfig,
            ]
        ] = []

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> tuple[RetrievalResult, ...]:
        retrieval_config = config or RetrievalConfig()

        self.calls.append(
            (
                query,
                retrieval_config,
            )
        )

        return self._results[: retrieval_config.top_k]


class FakeReranker:
    def __init__(self) -> None:
        self.calls: list[
            tuple[
                str,
                tuple[
                    RetrievalResult,
                    ...,
                ],
                int,
            ]
        ] = []

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> tuple[
        RetrievalResult,
        ...,
    ]:
        candidate_tuple = tuple(candidates)

        self.calls.append(
            (
                query,
                candidate_tuple,
                top_k,
            )
        )

        reordered = tuple(reversed(candidate_tuple))[:top_k]

        return tuple(
            RetrievalResult(
                rank=rank,
                score=float(100 - rank),
                chunk=candidate.chunk,
            )
            for rank, candidate in enumerate(
                reordered,
                start=1,
            )
        )


def make_candidates(
    count: int,
) -> tuple[
    RetrievalResult,
    ...,
]:
    return tuple(
        make_result(
            rank,
            f"chunk-{rank}",
            f"passage {rank}",
            score=float(count - rank),
        )
        for rank in range(
            1,
            count + 1,
        )
    )


def test_default_reranking_config() -> None:
    config = RerankingConfig()

    assert config.candidate_pool_size == 20

    assert config.rerank_pool_size == 10


def test_reranking_config_rejects_invalid_pool_size() -> None:
    with pytest.raises(ValueError):
        RerankingConfig(candidate_pool_size=0)


def test_reranking_retriever_satisfies_retriever_protocol() -> None:
    base = FakeRetriever(make_candidates(5))

    reranker = FakeReranker()

    retriever: Retriever = RerankingRetriever(
        retriever=base,
        reranker=reranker,
    )

    results = retriever.retrieve(
        "attention",
        RetrievalConfig(
            top_k=2,
        ),
    )

    assert len(results) == 2


def test_blank_query_is_rejected_before_retrieval() -> None:
    base = FakeRetriever(make_candidates(5))

    reranker = FakeReranker()

    retriever = RerankingRetriever(
        retriever=base,
        reranker=reranker,
    )

    with pytest.raises(
        ValueError,
        match=("Retrieval query cannot be empty"),
    ):
        retriever.retrieve("   ")

    assert base.calls == []
    assert reranker.calls == []


def test_default_candidate_pool_is_twenty() -> None:
    base = FakeRetriever(make_candidates(30))

    reranker = FakeReranker()

    retriever = RerankingRetriever(
        retriever=base,
        reranker=reranker,
    )

    retriever.retrieve(
        "attention",
        RetrievalConfig(
            top_k=5,
        ),
    )

    assert len(base.calls) == 1

    _, config = base.calls[0]

    assert config.top_k == 20


def test_custom_candidate_pool_is_forwarded() -> None:
    base = FakeRetriever(make_candidates(30))

    reranker = FakeReranker()

    retriever = RerankingRetriever(
        retriever=base,
        reranker=reranker,
        config=RerankingConfig(
            candidate_pool_size=15,
        ),
    )

    retriever.retrieve(
        "attention",
        RetrievalConfig(
            top_k=5,
        ),
    )

    _, config = base.calls[0]

    assert config.top_k == 15


def test_candidate_pool_never_drops_below_final_top_k() -> None:
    base = FakeRetriever(make_candidates(30))

    reranker = FakeReranker()

    retriever = RerankingRetriever(
        retriever=base,
        reranker=reranker,
        config=RerankingConfig(
            candidate_pool_size=10,
        ),
    )

    retriever.retrieve(
        "attention",
        RetrievalConfig(
            top_k=25,
        ),
    )

    _, config = base.calls[0]

    assert config.top_k == 25


def test_document_scope_is_preserved() -> None:
    base = FakeRetriever(make_candidates(20))

    reranker = FakeReranker()

    retriever = RerankingRetriever(
        retriever=base,
        reranker=reranker,
    )

    retriever.retrieve(
        "attention",
        RetrievalConfig(
            top_k=5,
            document_id=DOCUMENT_ID,
        ),
    )

    _, config = base.calls[0]

    assert config.document_id == DOCUMENT_ID


def test_score_threshold_is_preserved_for_wrapped_retriever() -> None:
    base = FakeRetriever(make_candidates(20))

    reranker = FakeReranker()

    retriever = RerankingRetriever(
        retriever=base,
        reranker=reranker,
    )

    retriever.retrieve(
        "attention",
        RetrievalConfig(
            top_k=5,
            score_threshold=0.5,
        ),
    )

    _, config = base.calls[0]

    assert config.score_threshold == 0.5


def test_final_result_comes_from_reranker() -> None:
    base = FakeRetriever(make_candidates(20))

    reranker = FakeReranker()

    retriever = RerankingRetriever(
        retriever=base,
        reranker=reranker,
    )

    results = retriever.retrieve(
        "attention",
        RetrievalConfig(
            top_k=3,
        ),
    )

    assert [result.chunk.chunk_id for result in results] == [
        "chunk-10",
        "chunk-9",
        "chunk-8",
    ]

    assert [result.rank for result in results] == [
        1,
        2,
        3,
    ]


def test_empty_candidate_set_skips_reranker() -> None:
    base = FakeRetriever(())

    reranker = FakeReranker()

    retriever = RerankingRetriever(
        retriever=base,
        reranker=reranker,
    )

    results = retriever.retrieve(
        "attention",
        RetrievalConfig(
            top_k=5,
        ),
    )

    assert results == ()
    assert reranker.calls == []


def test_query_is_trimmed_before_retrieval_and_reranking() -> None:
    base = FakeRetriever(make_candidates(5))

    reranker = FakeReranker()

    retriever = RerankingRetriever(
        retriever=base,
        reranker=reranker,
    )

    retriever.retrieve(
        "  attention  ",
        RetrievalConfig(
            top_k=2,
        ),
    )

    base_query, _ = base.calls[0]

    (
        reranker_query,
        _,
        _,
    ) = reranker.calls[0]

    assert base_query == "attention"

    assert reranker_query == "attention"


def test_original_retrieval_config_is_not_mutated() -> None:
    base = FakeRetriever(make_candidates(20))

    reranker = FakeReranker()

    retriever = RerankingRetriever(
        retriever=base,
        reranker=reranker,
    )

    original_config = RetrievalConfig(
        top_k=5,
        document_id=(DOCUMENT_ID),
    )

    retriever.retrieve(
        "attention",
        original_config,
    )

    assert original_config.top_k == 5

    assert original_config.document_id == DOCUMENT_ID


def test_reranking_config_rejects_invalid_rerank_pool_size() -> None:
    with pytest.raises(ValueError):
        RerankingConfig(rerank_pool_size=0)


def test_rerank_pool_cannot_exceed_candidate_pool() -> None:
    with pytest.raises(
        ValueError,
        match=("rerank_pool_size cannot exceed candidate_pool_size"),
    ):
        RerankingConfig(
            candidate_pool_size=10,
            rerank_pool_size=11,
        )


def test_custom_rerank_pool_is_forwarded() -> None:
    candidates = make_candidates(20)

    base = FakeRetriever(candidates)

    reranker = FakeReranker()

    retriever = RerankingRetriever(
        retriever=base,
        reranker=reranker,
        config=RerankingConfig(
            candidate_pool_size=20,
            rerank_pool_size=6,
        ),
    )

    retriever.retrieve(
        "attention",
        RetrievalConfig(
            top_k=5,
        ),
    )

    assert len(reranker.calls) == 1

    (
        _,
        forwarded_candidates,
        top_k,
    ) = reranker.calls[0]

    assert forwarded_candidates == candidates[:6]

    assert top_k == 5


def test_candidates_are_forwarded_to_reranker() -> None:
    candidates = make_candidates(20)

    base = FakeRetriever(candidates)

    reranker = FakeReranker()

    retriever = RerankingRetriever(
        retriever=base,
        reranker=reranker,
    )

    retriever.retrieve(
        "attention",
        RetrievalConfig(
            top_k=5,
        ),
    )

    assert len(base.calls) == 1
    assert len(reranker.calls) == 1

    _, base_config = base.calls[0]

    assert base_config.top_k == 20

    (
        query,
        forwarded_candidates,
        top_k,
    ) = reranker.calls[0]

    assert query == "attention"

    assert forwarded_candidates == candidates[:10]

    assert top_k == 5


def test_rerank_pool_never_drops_below_final_top_k() -> None:
    candidates = make_candidates(30)

    base = FakeRetriever(candidates)

    reranker = FakeReranker()

    retriever = RerankingRetriever(
        retriever=base,
        reranker=reranker,
    )

    retriever.retrieve(
        "attention",
        RetrievalConfig(
            top_k=12,
        ),
    )

    (
        _,
        forwarded_candidates,
        top_k,
    ) = reranker.calls[0]

    assert len(forwarded_candidates) == 12

    assert top_k == 12

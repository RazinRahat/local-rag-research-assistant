from collections.abc import Sequence

import pytest

from research_assistant.chunking.models import (
    DocumentChunk,
)
from research_assistant.reranking.base import (
    Reranker,
)
from research_assistant.reranking.cross_encoder import (
    CrossEncoderReranker,
)
from research_assistant.reranking.models import (
    CrossEncoderConfig,
)
from research_assistant.retrieval.models import (
    RetrievalResult,
)

DOCUMENT_ID = "a" * 64


def make_chunk(
    chunk_id: str,
    text: str,
) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=chunk_id,
        document_id=DOCUMENT_ID,
        file_name="paper.pdf",
        page_number=1,
        chunk_index=0,
        page_chunk_index=0,
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
    score: float = 0.0,
) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=score,
        chunk=make_chunk(
            chunk_id,
            text,
        ),
    )


class FakePairScorer:
    def __init__(
        self,
        scores: Sequence[float],
    ) -> None:
        self._scores = tuple(scores)

        self.calls: list[
            tuple[
                str,
                tuple[str, ...],
            ]
        ] = []

    def score(
        self,
        query: str,
        passages: Sequence[str],
    ) -> tuple[float, ...]:
        self.calls.append(
            (
                query,
                tuple(passages),
            )
        )

        return self._scores


def test_default_cross_encoder_config() -> None:
    config = CrossEncoderConfig()

    assert config.model_name == ("cross-encoder/ms-marco-MiniLM-L6-v2")

    assert config.batch_size == 8
    assert config.max_length == 512
    assert config.device is None


def test_cross_encoder_config_rejects_invalid_batch_size() -> None:
    with pytest.raises(ValueError):
        CrossEncoderConfig(batch_size=0)


def test_cross_encoder_config_rejects_invalid_max_length() -> None:
    with pytest.raises(ValueError):
        CrossEncoderConfig(max_length=1)


def test_reranker_satisfies_protocol() -> None:
    scorer = FakePairScorer(
        [
            1.0,
        ]
    )

    reranker: Reranker = CrossEncoderReranker(scorer)

    results = reranker.rerank(
        "attention",
        (
            make_result(
                1,
                "a",
                "attention",
            ),
        ),
        top_k=1,
    )

    assert len(results) == 1


def test_blank_query_is_rejected() -> None:
    scorer = FakePairScorer([])

    reranker = CrossEncoderReranker(scorer)

    with pytest.raises(
        ValueError,
        match=("Reranking query cannot be empty"),
    ):
        reranker.rerank(
            "   ",
            (),
            top_k=1,
        )

    assert scorer.calls == []


def test_invalid_top_k_is_rejected() -> None:
    scorer = FakePairScorer([])

    reranker = CrossEncoderReranker(scorer)

    with pytest.raises(
        ValueError,
        match=("top_k must be at least 1"),
    ):
        reranker.rerank(
            "attention",
            (),
            top_k=0,
        )


def test_empty_candidates_return_empty_tuple() -> None:
    scorer = FakePairScorer([])

    reranker = CrossEncoderReranker(scorer)

    results = reranker.rerank(
        "attention",
        (),
        top_k=5,
    )

    assert results == ()
    assert scorer.calls == []


def test_query_and_passages_are_forwarded() -> None:
    scorer = FakePairScorer(
        [
            1.0,
            2.0,
        ]
    )

    reranker = CrossEncoderReranker(scorer)

    candidates = (
        make_result(
            1,
            "a",
            "first passage",
        ),
        make_result(
            2,
            "b",
            "second passage",
        ),
    )

    reranker.rerank(
        "  attention  ",
        candidates,
        top_k=2,
    )

    assert scorer.calls == [
        (
            "attention",
            (
                "first passage",
                "second passage",
            ),
        )
    ]


def test_candidates_are_sorted_by_reranker_score() -> None:
    scorer = FakePairScorer(
        [
            1.0,
            5.0,
            3.0,
        ]
    )

    reranker = CrossEncoderReranker(scorer)

    results = reranker.rerank(
        "attention",
        (
            make_result(
                1,
                "a",
                "first",
                score=0.9,
            ),
            make_result(
                2,
                "b",
                "second",
                score=0.8,
            ),
            make_result(
                3,
                "c",
                "third",
                score=0.7,
            ),
        ),
        top_k=3,
    )

    assert [result.chunk.chunk_id for result in results] == [
        "b",
        "c",
        "a",
    ]

    assert [result.score for result in results] == [
        5.0,
        3.0,
        1.0,
    ]


def test_original_retrieval_scores_do_not_affect_reranking() -> None:
    scorer = FakePairScorer(
        [
            0.1,
            9.0,
        ]
    )

    reranker = CrossEncoderReranker(scorer)

    results = reranker.rerank(
        "attention",
        (
            make_result(
                1,
                "a",
                "first",
                score=1000.0,
            ),
            make_result(
                2,
                "b",
                "second",
                score=-1000.0,
            ),
        ),
        top_k=2,
    )

    assert results[0].chunk.chunk_id == "b"


def test_top_k_limits_final_results() -> None:
    scorer = FakePairScorer(
        [
            1.0,
            3.0,
            2.0,
        ]
    )

    reranker = CrossEncoderReranker(scorer)

    results = reranker.rerank(
        "attention",
        (
            make_result(
                1,
                "a",
                "first",
            ),
            make_result(
                2,
                "b",
                "second",
            ),
            make_result(
                3,
                "c",
                "third",
            ),
        ),
        top_k=2,
    )

    assert len(results) == 2

    assert [result.chunk.chunk_id for result in results] == [
        "b",
        "c",
    ]


def test_final_ranks_are_sequential() -> None:
    scorer = FakePairScorer(
        [
            1.0,
            5.0,
            3.0,
        ]
    )

    reranker = CrossEncoderReranker(scorer)

    results = reranker.rerank(
        "attention",
        (
            make_result(
                1,
                "a",
                "first",
            ),
            make_result(
                2,
                "b",
                "second",
            ),
            make_result(
                3,
                "c",
                "third",
            ),
        ),
        top_k=3,
    )

    assert [result.rank for result in results] == [
        1,
        2,
        3,
    ]


def test_equal_scores_preserve_original_rank_order() -> None:
    scorer = FakePairScorer(
        [
            5.0,
            5.0,
            5.0,
        ]
    )

    reranker = CrossEncoderReranker(scorer)

    results = reranker.rerank(
        "attention",
        (
            make_result(
                1,
                "c",
                "first",
            ),
            make_result(
                2,
                "a",
                "second",
            ),
            make_result(
                3,
                "b",
                "third",
            ),
        ),
        top_k=3,
    )

    assert [result.chunk.chunk_id for result in results] == [
        "c",
        "a",
        "b",
    ]


def test_non_sequential_candidate_ranks_are_rejected() -> None:
    scorer = FakePairScorer(
        [
            1.0,
            2.0,
        ]
    )

    reranker = CrossEncoderReranker(scorer)

    candidates = (
        make_result(
            1,
            "a",
            "first",
        ),
        make_result(
            3,
            "b",
            "second",
        ),
    )

    with pytest.raises(
        ValueError,
        match=("sequential ranks"),
    ):
        reranker.rerank(
            "attention",
            candidates,
            top_k=2,
        )

    assert scorer.calls == []


def test_duplicate_chunks_are_rejected() -> None:
    scorer = FakePairScorer(
        [
            1.0,
            2.0,
        ]
    )

    reranker = CrossEncoderReranker(scorer)

    candidates = (
        make_result(
            1,
            "same",
            "first",
        ),
        make_result(
            2,
            "same",
            "first",
        ),
    )

    with pytest.raises(
        ValueError,
        match=("duplicate chunk_id"),
    ):
        reranker.rerank(
            "attention",
            candidates,
            top_k=2,
        )

    assert scorer.calls == []


def test_wrong_score_count_is_rejected() -> None:
    scorer = FakePairScorer(
        [
            1.0,
        ]
    )

    reranker = CrossEncoderReranker(scorer)

    candidates = (
        make_result(
            1,
            "a",
            "first",
        ),
        make_result(
            2,
            "b",
            "second",
        ),
    )

    with pytest.raises(
        ValueError,
        match=("unexpected number of scores"),
    ):
        reranker.rerank(
            "attention",
            candidates,
            top_k=2,
        )


@pytest.mark.parametrize(
    "invalid_score",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
)
def test_non_finite_scores_are_rejected(
    invalid_score: float,
) -> None:
    scorer = FakePairScorer(
        [
            invalid_score,
        ]
    )

    reranker = CrossEncoderReranker(scorer)

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        reranker.rerank(
            "attention",
            (
                make_result(
                    1,
                    "a",
                    "first",
                ),
            ),
            top_k=1,
        )

from __future__ import annotations

from collections.abc import Sequence

from research_assistant.chunking.models import (
    DocumentChunk,
)
from research_assistant.reranking.cross_encoder import (
    CrossEncoderReranker,
)
from research_assistant.reranking.retriever import (
    RerankingRetriever,
)
from research_assistant.retrieval.hybrid import (
    HybridRetriever,
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
        token_count=20,
        char_count=len(text),
        token_start=0,
        token_end=20,
    )


CHUNKS = {
    "a": make_chunk(
        "a",
        "Adam is cited in the references.",
        chunk_index=0,
    ),
    "b": make_chunk(
        "b",
        ("We used the Adam optimizer during model training."),
        chunk_index=1,
    ),
    "c": make_chunk(
        "c",
        ("The optimizer uses beta parameters during training."),
        chunk_index=2,
    ),
    "d": make_chunk(
        "d",
        ("The architecture uses multi-head attention."),
        chunk_index=3,
    ),
    "e": make_chunk(
        "e",
        ("Training used an adaptive optimization method."),
        chunk_index=4,
    ),
}


def result(
    *,
    rank: int,
    score: float,
    chunk_id: str,
) -> RetrievalResult:
    return RetrievalResult(
        rank=rank,
        score=score,
        chunk=CHUNKS[chunk_id],
    )


class FakeBranchRetriever:
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
    ) -> tuple[
        RetrievalResult,
        ...,
    ]:
        retrieval_config = config or RetrievalConfig()

        self.calls.append(
            (
                query,
                retrieval_config,
            )
        )

        return self._results[: retrieval_config.top_k]


class MappingPairScorer:
    def __init__(
        self,
        scores_by_text: dict[
            str,
            float,
        ],
    ) -> None:
        self._scores_by_text = scores_by_text

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
        passage_tuple = tuple(passages)

        self.calls.append(
            (
                query,
                passage_tuple,
            )
        )

        return tuple(self._scores_by_text[passage] for passage in passage_tuple)


def make_pipeline() -> tuple[
    RerankingRetriever,
    FakeBranchRetriever,
    FakeBranchRetriever,
    MappingPairScorer,
]:
    dense = FakeBranchRetriever(
        (
            result(
                rank=1,
                score=0.90,
                chunk_id="a",
            ),
            result(
                rank=2,
                score=0.85,
                chunk_id="b",
            ),
            result(
                rank=3,
                score=0.80,
                chunk_id="c",
            ),
            result(
                rank=4,
                score=0.70,
                chunk_id="d",
            ),
        )
    )

    lexical = FakeBranchRetriever(
        (
            result(
                rank=1,
                score=8.0,
                chunk_id="b",
            ),
            result(
                rank=2,
                score=7.0,
                chunk_id="e",
            ),
            result(
                rank=3,
                score=6.0,
                chunk_id="a",
            ),
            result(
                rank=4,
                score=4.0,
                chunk_id="d",
            ),
        )
    )

    hybrid = HybridRetriever(
        dense_retriever=dense,
        lexical_retriever=lexical,
    )

    scorer = MappingPairScorer(
        {
            CHUNKS["a"].text: 1.0,
            CHUNKS["b"].text: 5.0,
            CHUNKS["c"].text: 9.0,
            CHUNKS["d"].text: 0.5,
            CHUNKS["e"].text: 8.0,
        }
    )

    reranker = CrossEncoderReranker(scorer)

    pipeline = RerankingRetriever(
        retriever=hybrid,
        reranker=reranker,
    )

    return (
        pipeline,
        dense,
        lexical,
        scorer,
    )


def test_complete_hybrid_reranking_pipeline() -> None:
    (
        pipeline,
        _,
        _,
        scorer,
    ) = make_pipeline()

    results = pipeline.retrieve(
        "Adam optimizer",
        RetrievalConfig(
            top_k=3,
        ),
    )

    assert [item.chunk.chunk_id for item in results] == [
        "c",
        "e",
        "b",
    ]

    assert [item.rank for item in results] == [
        1,
        2,
        3,
    ]

    assert [item.score for item in results] == [
        9.0,
        8.0,
        5.0,
    ]

    assert len(scorer.calls) == 1


def test_reranking_can_promote_lower_hybrid_candidate() -> None:
    (
        pipeline,
        _,
        _,
        _,
    ) = make_pipeline()

    results = pipeline.retrieve(
        "Adam optimizer",
        RetrievalConfig(
            top_k=1,
        ),
    )

    assert results[0].chunk.chunk_id == "c"


def test_nested_candidate_expansion_reaches_branches() -> None:
    (
        pipeline,
        dense,
        lexical,
        _,
    ) = make_pipeline()

    pipeline.retrieve(
        "Adam optimizer",
        RetrievalConfig(
            top_k=5,
        ),
    )

    assert len(dense.calls) == 1

    assert len(lexical.calls) == 1

    _, dense_config = dense.calls[0]

    _, lexical_config = lexical.calls[0]

    # RerankingRetriever:
    # final top_k 5
    #     ↓
    # candidate pool 20
    #
    # HybridRetriever:
    # candidate_multiplier 2
    #     ↓
    # dense top_k 40
    # lexical top_k 40

    assert dense_config.top_k == 40

    assert lexical_config.top_k == 40


def test_document_scope_survives_complete_pipeline() -> None:
    (
        pipeline,
        dense,
        lexical,
        _,
    ) = make_pipeline()

    pipeline.retrieve(
        "Adam optimizer",
        RetrievalConfig(
            top_k=5,
            document_id=(DOCUMENT_ID),
        ),
    )

    _, dense_config = dense.calls[0]

    _, lexical_config = lexical.calls[0]

    assert dense_config.document_id == DOCUMENT_ID

    assert lexical_config.document_id == DOCUMENT_ID


def test_query_reaches_all_pipeline_stages() -> None:
    (
        pipeline,
        dense,
        lexical,
        scorer,
    ) = make_pipeline()

    pipeline.retrieve(
        "  Adam optimizer  ",
        RetrievalConfig(
            top_k=2,
        ),
    )

    dense_query, _ = dense.calls[0]

    lexical_query, _ = lexical.calls[0]

    scorer_query, _ = scorer.calls[0]

    assert dense_query == "Adam optimizer"

    assert lexical_query == "Adam optimizer"

    assert scorer_query == "Adam optimizer"

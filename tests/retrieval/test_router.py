from research_assistant.retrieval.base import (
    Retriever,
)
from research_assistant.retrieval.models import (
    RetrievalConfig,
    RetrievalMode,
    RetrievalResult,
)
from research_assistant.retrieval.router import (
    RetrievalRouter,
)

DOCUMENT_ID = "a" * 64


class FakeRetriever:
    def __init__(self) -> None:
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

        return ()


def make_router() -> tuple[
    RetrievalRouter,
    FakeRetriever,
    FakeRetriever,
    FakeRetriever,
    FakeRetriever,
]:
    dense = FakeRetriever()
    lexical = FakeRetriever()
    hybrid = FakeRetriever()
    hybrid_reranked = FakeRetriever()

    router = RetrievalRouter(
        dense_retriever=dense,
        lexical_retriever=lexical,
        hybrid_retriever=hybrid,
        hybrid_reranked_retriever=(hybrid_reranked),
    )

    return (
        router,
        dense,
        lexical,
        hybrid,
        hybrid_reranked,
    )


def test_router_satisfies_retriever_protocol() -> None:
    (
        router,
        _,
        _,
        _,
        _,
    ) = make_router()

    retriever: Retriever = router

    results = retriever.retrieve("attention")

    assert results == ()


def test_router_uses_dense_by_default() -> None:
    (
        router,
        dense,
        lexical,
        hybrid,
        hybrid_reranked,
    ) = make_router()

    router.retrieve("attention")

    assert len(dense.calls) == 1

    assert lexical.calls == []
    assert hybrid.calls == []
    assert hybrid_reranked.calls == []

    query, config = dense.calls[0]

    assert query == "attention"

    assert config.mode == RetrievalMode.DENSE


def test_router_selects_dense_mode() -> None:
    (
        router,
        dense,
        lexical,
        hybrid,
        hybrid_reranked,
    ) = make_router()

    config = RetrievalConfig(
        mode=RetrievalMode.DENSE,
    )

    router.retrieve(
        "attention",
        config,
    )

    assert dense.calls == [
        (
            "attention",
            config,
        )
    ]

    assert lexical.calls == []
    assert hybrid.calls == []
    assert hybrid_reranked.calls == []


def test_router_selects_lexical_mode() -> None:
    (
        router,
        dense,
        lexical,
        hybrid,
        hybrid_reranked,
    ) = make_router()

    config = RetrievalConfig(
        mode=RetrievalMode.LEXICAL,
    )

    router.retrieve(
        "attention",
        config,
    )

    assert dense.calls == []

    assert lexical.calls == [
        (
            "attention",
            config,
        )
    ]

    assert hybrid.calls == []
    assert hybrid_reranked.calls == []


def test_router_selects_hybrid_mode() -> None:
    (
        router,
        dense,
        lexical,
        hybrid,
        hybrid_reranked,
    ) = make_router()

    config = RetrievalConfig(
        mode=RetrievalMode.HYBRID,
    )

    router.retrieve(
        "attention",
        config,
    )

    assert dense.calls == []
    assert lexical.calls == []

    assert hybrid.calls == [
        (
            "attention",
            config,
        )
    ]

    assert hybrid_reranked.calls == []


def test_router_selects_hybrid_reranked_mode() -> None:
    (
        router,
        dense,
        lexical,
        hybrid,
        hybrid_reranked,
    ) = make_router()

    config = RetrievalConfig(
        mode=(RetrievalMode.HYBRID_RERANKED),
    )

    router.retrieve(
        "attention",
        config,
    )

    assert dense.calls == []
    assert lexical.calls == []
    assert hybrid.calls == []

    assert hybrid_reranked.calls == [
        (
            "attention",
            config,
        )
    ]


def test_router_preserves_retrieval_configuration() -> None:
    (
        router,
        dense,
        _,
        _,
        _,
    ) = make_router()

    config = RetrievalConfig(
        top_k=8,
        score_threshold=0.4,
        document_id=DOCUMENT_ID,
        mode=RetrievalMode.DENSE,
    )

    router.retrieve(
        "attention",
        config,
    )

    assert len(dense.calls) == 1

    query, received_config = dense.calls[0]

    assert query == "attention"

    assert received_config is config

    assert received_config.top_k == 8

    assert received_config.score_threshold == 0.4

    assert received_config.document_id == DOCUMENT_ID

    assert received_config.mode == RetrievalMode.DENSE

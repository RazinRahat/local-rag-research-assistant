from __future__ import annotations

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
                RetrievalConfig | None,
            ]
        ] = []

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> tuple[RetrievalResult, ...]:
        self.calls.append(
            (
                query,
                config,
            )
        )

        return ()


def make_router() -> tuple[
    RetrievalRouter,
    FakeRetriever,
    FakeRetriever,
    FakeRetriever,
]:
    dense = FakeRetriever()
    lexical = FakeRetriever()
    hybrid = FakeRetriever()

    router = RetrievalRouter(
        dense_retriever=dense,
        lexical_retriever=lexical,
        hybrid_retriever=hybrid,
    )

    return (
        router,
        dense,
        lexical,
        hybrid,
    )


def test_router_satisfies_retriever_protocol() -> None:
    router, _, _, _ = make_router()

    retriever: Retriever = router

    assert retriever.retrieve("attention") == ()


def test_router_uses_dense_by_default() -> None:
    (
        router,
        dense,
        lexical,
        hybrid,
    ) = make_router()

    router.retrieve("attention")

    assert len(dense.calls) == 1
    assert lexical.calls == []
    assert hybrid.calls == []

    _, config = dense.calls[0]

    assert config is not None

    assert config.mode == RetrievalMode.DENSE


def test_router_selects_dense_mode() -> None:
    (
        router,
        dense,
        lexical,
        hybrid,
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


def test_router_selects_lexical_mode() -> None:
    (
        router,
        dense,
        lexical,
        hybrid,
    ) = make_router()

    config = RetrievalConfig(
        mode=RetrievalMode.LEXICAL,
    )

    router.retrieve(
        "Adam optimizer",
        config,
    )

    assert dense.calls == []

    assert lexical.calls == [
        (
            "Adam optimizer",
            config,
        )
    ]

    assert hybrid.calls == []


def test_router_selects_hybrid_mode() -> None:
    (
        router,
        dense,
        lexical,
        hybrid,
    ) = make_router()

    config = RetrievalConfig(
        top_k=7,
        document_id=DOCUMENT_ID,
        mode=RetrievalMode.HYBRID,
    )

    router.retrieve(
        "self attention",
        config,
    )

    assert dense.calls == []
    assert lexical.calls == []

    assert hybrid.calls == [
        (
            "self attention",
            config,
        )
    ]


def test_router_preserves_retrieval_configuration() -> None:
    (
        router,
        dense,
        _,
        _,
    ) = make_router()

    config = RetrievalConfig(
        top_k=3,
        score_threshold=0.5,
        document_id=DOCUMENT_ID,
        mode=RetrievalMode.DENSE,
    )

    router.retrieve(
        "transformer",
        config,
    )

    _, forwarded = dense.calls[0]

    assert forwarded is config

    assert forwarded.top_k == 3

    assert forwarded.score_threshold == 0.5

    assert forwarded.document_id == DOCUMENT_ID

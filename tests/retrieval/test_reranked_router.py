from research_assistant.retrieval.models import (
    RetrievalConfig,
    RetrievalMode,
    RetrievalResult,
)
from research_assistant.retrieval.router import (
    RetrievalRouter,
)


class RecordingRetriever:
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
    RecordingRetriever,
    RecordingRetriever,
    RecordingRetriever,
    RecordingRetriever,
]:
    dense = RecordingRetriever()
    lexical = RecordingRetriever()
    hybrid = RecordingRetriever()
    reranked = RecordingRetriever()

    router = RetrievalRouter(
        dense_retriever=dense,
        lexical_retriever=lexical,
        hybrid_retriever=hybrid,
        hybrid_reranked_retriever=(reranked),
    )

    return (
        router,
        dense,
        lexical,
        hybrid,
        reranked,
    )


def test_hybrid_reranked_mode_routes_to_reranked_retriever() -> None:
    (
        router,
        dense,
        lexical,
        hybrid,
        reranked,
    ) = make_router()

    config = RetrievalConfig(
        top_k=5,
        mode=(RetrievalMode.HYBRID_RERANKED),
    )

    router.retrieve(
        "Adam optimizer",
        config,
    )

    assert dense.calls == []
    assert lexical.calls == []
    assert hybrid.calls == []

    assert reranked.calls == [
        (
            "Adam optimizer",
            config,
        )
    ]


def test_dense_default_is_unchanged() -> None:
    (
        router,
        dense,
        lexical,
        hybrid,
        reranked,
    ) = make_router()

    router.retrieve("attention")

    assert len(dense.calls) == 1

    assert lexical.calls == []
    assert hybrid.calls == []
    assert reranked.calls == []


def test_existing_hybrid_mode_is_unchanged() -> None:
    (
        router,
        dense,
        lexical,
        hybrid,
        reranked,
    ) = make_router()

    config = RetrievalConfig(
        mode=(RetrievalMode.HYBRID),
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

    assert reranked.calls == []

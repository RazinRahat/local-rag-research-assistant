import pytest
from pydantic import (
    ValidationError,
)

from research_assistant.api.models import (
    QueryRequest,
    SearchRequest,
)
from research_assistant.retrieval.models import (
    RetrievalMode,
)


def test_search_accepts_hybrid_reranked_mode() -> None:
    request = SearchRequest(
        query="Adam optimizer",
        mode=(RetrievalMode.HYBRID_RERANKED),
    )

    assert request.mode == RetrievalMode.HYBRID_RERANKED

    config = request.to_retrieval_config()

    assert config.mode == RetrievalMode.HYBRID_RERANKED


def test_query_accepts_hybrid_reranked_mode() -> None:
    request = QueryRequest(
        question=("Why does the Transformer avoid recurrence?"),
        mode=(RetrievalMode.HYBRID_RERANKED),
    )

    config = request.to_retrieval_config()

    assert config.mode == RetrievalMode.HYBRID_RERANKED


def test_hybrid_reranked_rejects_score_threshold() -> None:
    with pytest.raises(
        ValidationError,
        match=("score_threshold is only supported for dense retrieval"),
    ):
        SearchRequest(
            query="Adam optimizer",
            mode=(RetrievalMode.HYBRID_RERANKED),
            score_threshold=0.5,
        )


def test_dense_threshold_remains_valid() -> None:
    request = SearchRequest(
        query="Adam optimizer",
        mode=RetrievalMode.DENSE,
        score_threshold=0.5,
    )

    assert request.score_threshold == 0.5


def test_invalid_mode_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest.model_validate(
            {
                "query": ("Adam optimizer"),
                "mode": "reranked",
            }
        )

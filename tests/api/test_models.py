"""Tests for API request validation."""

import pytest
from pydantic import ValidationError

from research_assistant.api.models import (
    QueryRequest,
    SearchRequest,
)
from research_assistant.retrieval.models import (
    RetrievalConfig,
    RetrievalMode,
)


def test_search_request_defaults() -> None:
    request = SearchRequest(query="Explain self-attention.")

    assert request.top_k == 5

    assert request.score_threshold is None

    assert request.document_id is None

    assert request.mode == RetrievalMode.DENSE


def test_search_request_strips_whitespace() -> None:
    request = SearchRequest(query=("  What is positional encoding?  "))

    assert request.query == ("What is positional encoding?")


def test_search_request_converts_to_domain_config() -> None:
    document_id = "a" * 64

    request = SearchRequest(
        query=("Explain multi-head attention."),
        top_k=3,
        score_threshold=0.5,
        document_id=document_id,
        mode=RetrievalMode.DENSE,
    )

    config = request.to_retrieval_config()

    assert isinstance(
        config,
        RetrievalConfig,
    )

    assert config.top_k == 3

    assert config.score_threshold == 0.5

    assert config.document_id == document_id

    assert config.mode == RetrievalMode.DENSE


@pytest.mark.parametrize(
    "mode",
    [
        RetrievalMode.DENSE,
        RetrievalMode.LEXICAL,
        RetrievalMode.HYBRID,
    ],
)
def test_search_request_accepts_retrieval_modes(
    mode: RetrievalMode,
) -> None:
    request = SearchRequest(
        query="Attention",
        mode=mode,
    )

    assert request.mode == mode

    assert request.to_retrieval_config().mode == mode


def test_invalid_retrieval_mode_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(
            query="Attention",
            mode="unknown",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "mode",
    [
        RetrievalMode.LEXICAL,
        RetrievalMode.HYBRID,
    ],
)
def test_non_dense_mode_rejects_score_threshold(
    mode: RetrievalMode,
) -> None:
    with pytest.raises(
        ValidationError,
        match=("score_threshold is only supported for dense retrieval"),
    ):
        SearchRequest(
            query="Attention",
            mode=mode,
            score_threshold=0.5,
        )


def test_blank_query_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="   ")


@pytest.mark.parametrize(
    "invalid_top_k",
    [
        -1,
        0,
        21,
        100,
    ],
)
def test_invalid_top_k_is_rejected(
    invalid_top_k: int,
) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(
            query="Attention",
            top_k=invalid_top_k,
        )


def test_invalid_document_id_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(
            query="Attention",
            document_id=("invalid-id"),
        )


def test_query_request_strips_whitespace() -> None:
    request = QueryRequest(question=("  Why avoid recurrence?  "))

    assert request.question == ("Why avoid recurrence?")


def test_query_request_accepts_hybrid_mode() -> None:
    request = QueryRequest(
        question=("Why avoid recurrence?"),
        mode=RetrievalMode.HYBRID,
    )

    config = request.to_retrieval_config()

    assert config.mode == RetrievalMode.HYBRID


def test_query_request_accepts_lexical_mode() -> None:
    request = QueryRequest(
        question=("Which optimizer was used?"),
        mode=RetrievalMode.LEXICAL,
    )

    config = request.to_retrieval_config()

    assert config.mode == RetrievalMode.LEXICAL


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        QueryRequest.model_validate(
            {
                "question": ("Explain attention."),
                "unknown_option": True,
            }
        )

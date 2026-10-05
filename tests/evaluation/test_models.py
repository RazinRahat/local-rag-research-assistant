import pytest
from pydantic import (
    ValidationError,
)

from research_assistant.evaluation.models import (
    EvaluationQuery,
    RelevanceJudgment,
)


def make_judgment(
    chunk_id: str,
    relevance: int,
) -> RelevanceJudgment:
    return RelevanceJudgment(
        chunk_id=chunk_id,
        relevance=relevance,
    )


def test_relevance_accepts_zero_to_three() -> None:
    for relevance in range(4):
        judgment = make_judgment(
            f"chunk-{relevance}",
            relevance,
        )

        assert judgment.relevance == relevance


def test_relevance_rejects_negative_value() -> None:
    with pytest.raises(ValidationError):
        make_judgment(
            "chunk-a",
            -1,
        )


def test_relevance_rejects_value_above_three() -> None:
    with pytest.raises(ValidationError):
        make_judgment(
            "chunk-a",
            4,
        )


def test_query_requires_relevant_judgment() -> None:
    with pytest.raises(
        ValidationError,
        match=("at least one relevant judgment"),
    ):
        EvaluationQuery(
            id="q1",
            query="test",
            judgments=(
                make_judgment(
                    "chunk-a",
                    0,
                ),
            ),
        )


def test_query_rejects_duplicate_chunk_ids() -> None:
    with pytest.raises(
        ValidationError,
        match=("duplicate chunk IDs"),
    ):
        EvaluationQuery(
            id="q1",
            query="test",
            judgments=(
                make_judgment(
                    "chunk-a",
                    3,
                ),
                make_judgment(
                    "chunk-a",
                    1,
                ),
            ),
        )


def test_query_accepts_graded_judgments() -> None:
    query = EvaluationQuery(
        id="q1",
        query="test",
        judgments=(
            make_judgment(
                "chunk-a",
                3,
            ),
            make_judgment(
                "chunk-b",
                2,
            ),
            make_judgment(
                "chunk-c",
                0,
            ),
        ),
        tags=(
            "semantic",
            "document-scoped",
        ),
    )

    assert len(query.judgments) == 3

    assert query.tags == (
        "semantic",
        "document-scoped",
    )


def test_query_is_included_in_aggregate_by_default() -> None:
    query = EvaluationQuery(
        id="q1",
        query="test",
        judgments=(
            make_judgment(
                "chunk-a",
                3,
            ),
        ),
    )

    assert query.include_in_aggregate is True


def test_query_can_be_diagnostic_only() -> None:
    query = EvaluationQuery(
        id="q1",
        query="test",
        include_in_aggregate=False,
        judgments=(
            make_judgment(
                "chunk-a",
                3,
            ),
        ),
    )

    assert query.include_in_aggregate is False

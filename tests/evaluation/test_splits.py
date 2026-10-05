import pytest

from research_assistant.evaluation.models import (
    EvaluationQuery,
    RelevanceJudgment,
)
from research_assistant.evaluation.splits import (
    EvaluationSplit,
    query_split,
    select_queries,
)


def make_query(
    query_id: str,
) -> EvaluationQuery:
    return EvaluationQuery(
        id=query_id,
        query="example",
        judgments=(
            RelevanceJudgment(
                chunk_id=(f"{query_id}-chunk"),
                relevance=3,
            ),
        ),
    )


def test_detects_development_split() -> None:
    query = make_query("dev_example")

    assert query_split(query) == EvaluationSplit.DEVELOPMENT


def test_detects_holdout_split() -> None:
    query = make_query("holdout_example")

    assert query_split(query) == EvaluationSplit.HOLDOUT


def test_unknown_split_is_rejected() -> None:
    query = make_query("example")

    with pytest.raises(
        ValueError,
        match="does not declare",
    ):
        query_split(query)


def test_selects_development_queries() -> None:
    queries = (
        make_query("dev_one"),
        make_query("holdout_one"),
        make_query("dev_two"),
    )

    selected = select_queries(
        queries,
        split=(EvaluationSplit.DEVELOPMENT),
    )

    assert tuple(query.id for query in selected) == (
        "dev_one",
        "dev_two",
    )


def test_selects_holdout_queries() -> None:
    queries = (
        make_query("dev_one"),
        make_query("holdout_one"),
    )

    selected = select_queries(
        queries,
        split=(EvaluationSplit.HOLDOUT),
    )

    assert tuple(query.id for query in selected) == ("holdout_one",)

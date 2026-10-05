import math

import pytest

from research_assistant.evaluation.metrics import (
    evaluate_ranking,
)
from research_assistant.evaluation.models import (
    EvaluationQuery,
    RelevanceJudgment,
)


def make_query() -> EvaluationQuery:
    return EvaluationQuery(
        id="query-1",
        query="example query",
        judgments=(
            RelevanceJudgment(
                chunk_id="a",
                relevance=3,
            ),
            RelevanceJudgment(
                chunk_id="b",
                relevance=2,
            ),
            RelevanceJudgment(
                chunk_id="c",
                relevance=1,
            ),
        ),
    )


def test_perfect_binary_ranking() -> None:
    metrics = evaluate_ranking(
        (
            "a",
            "b",
        ),
        make_query(),
        k=2,
    )

    assert metrics.relevance_threshold == 2

    assert metrics.precision_at_k == 1.0

    assert metrics.recall_at_k == 1.0

    assert metrics.reciprocal_rank == 1.0

    assert metrics.ndcg_at_k == pytest.approx(1.0)


def test_partial_retrieval() -> None:
    metrics = evaluate_ranking(
        (
            "x",
            "b",
            "y",
        ),
        make_query(),
        k=3,
    )

    assert metrics.precision_at_k == pytest.approx(1 / 3)

    assert metrics.recall_at_k == pytest.approx(1 / 2)

    assert metrics.reciprocal_rank == 0.5

    assert 0.0 < metrics.ndcg_at_k < 1.0


def test_marginal_relevance_does_not_count_as_binary_relevant() -> None:
    metrics = evaluate_ranking(
        (
            "c",
            "a",
        ),
        make_query(),
        k=2,
    )

    assert metrics.relevant_retrieved == 1

    assert metrics.reciprocal_rank == 0.5

    assert metrics.ndcg_at_k > 0.0


def test_unjudged_chunks_are_zero_relevance() -> None:
    metrics = evaluate_ranking(
        (
            "x",
            "y",
            "z",
        ),
        make_query(),
        k=3,
    )

    assert metrics.precision_at_k == 0.0

    assert metrics.recall_at_k == 0.0

    assert metrics.reciprocal_rank == 0.0

    assert metrics.ndcg_at_k == 0.0


def test_mrr_uses_first_binary_relevant_result() -> None:
    metrics = evaluate_ranking(
        (
            "x",
            "c",
            "a",
        ),
        make_query(),
        k=3,
    )

    assert metrics.reciprocal_rank == pytest.approx(1 / 3)


def test_ndcg_uses_graded_relevance() -> None:
    good = evaluate_ranking(
        (
            "a",
            "b",
            "c",
        ),
        make_query(),
        k=3,
    )

    reversed_order = evaluate_ranking(
        (
            "c",
            "b",
            "a",
        ),
        make_query(),
        k=3,
    )

    assert good.ndcg_at_k > reversed_order.ndcg_at_k


def test_precision_uses_requested_k() -> None:
    metrics = evaluate_ranking(
        ("a",),
        make_query(),
        k=5,
    )

    assert metrics.retrieved_count == 1

    assert metrics.precision_at_k == 0.2


def test_custom_relevance_threshold() -> None:
    metrics = evaluate_ranking(
        (
            "b",
            "a",
        ),
        make_query(),
        k=2,
        relevance_threshold=3,
    )

    assert metrics.relevant_total == 1

    assert metrics.reciprocal_rank == 0.5


def test_invalid_relevance_threshold_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=("relevance_threshold"),
    ):
        evaluate_ranking(
            ("a",),
            make_query(),
            k=1,
            relevance_threshold=4,
        )


def test_duplicate_ranking_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="duplicate chunk IDs",
    ):
        evaluate_ranking(
            (
                "a",
                "a",
            ),
            make_query(),
            k=2,
        )


def test_invalid_k_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="k must be at least 1",
    ):
        evaluate_ranking(
            ("a",),
            make_query(),
            k=0,
        )


def test_metrics_are_finite() -> None:
    metrics = evaluate_ranking(
        (
            "x",
            "a",
        ),
        make_query(),
        k=2,
    )

    values = (
        metrics.precision_at_k,
        metrics.recall_at_k,
        metrics.reciprocal_rank,
        metrics.ndcg_at_k,
    )

    assert all(math.isfinite(value) for value in values)

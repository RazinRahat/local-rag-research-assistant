import pytest

from research_assistant.evaluation.aggregate import (
    aggregate_metrics,
)
from research_assistant.evaluation.models import (
    RetrievalMetrics,
)


def make_metrics(
    *,
    precision: float,
    recall: float,
    reciprocal_rank: float,
    ndcg: float,
    k: int = 5,
    relevance_threshold: int = 2,
) -> RetrievalMetrics:
    return RetrievalMetrics(
        k=k,
        relevance_threshold=(relevance_threshold),
        retrieved_count=k,
        relevant_total=2,
        relevant_retrieved=1,
        precision_at_k=precision,
        recall_at_k=recall,
        reciprocal_rank=(reciprocal_rank),
        ndcg_at_k=ndcg,
    )


def test_macro_averages_metrics() -> None:
    aggregate = aggregate_metrics(
        (
            make_metrics(
                precision=0.2,
                recall=0.5,
                reciprocal_rank=1.0,
                ndcg=0.8,
            ),
            make_metrics(
                precision=0.4,
                recall=1.0,
                reciprocal_rank=0.5,
                ndcg=0.6,
            ),
        )
    )

    assert aggregate.query_count == 2

    assert aggregate.mean_precision_at_k == pytest.approx(0.3)

    assert aggregate.mean_recall_at_k == pytest.approx(0.75)

    assert aggregate.mean_reciprocal_rank == pytest.approx(0.75)

    assert aggregate.mean_ndcg_at_k == pytest.approx(0.7)


def test_empty_metric_set_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="empty metric set",
    ):
        aggregate_metrics(())


def test_different_k_values_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="different k values",
    ):
        aggregate_metrics(
            (
                make_metrics(
                    precision=0.2,
                    recall=0.5,
                    reciprocal_rank=1.0,
                    ndcg=0.8,
                    k=5,
                ),
                make_metrics(
                    precision=0.2,
                    recall=0.5,
                    reciprocal_rank=1.0,
                    ndcg=0.8,
                    k=10,
                ),
            )
        )


def test_different_thresholds_are_rejected() -> None:
    with pytest.raises(
        ValueError,
        match=("different relevance thresholds"),
    ):
        aggregate_metrics(
            (
                make_metrics(
                    precision=0.2,
                    recall=0.5,
                    reciprocal_rank=1.0,
                    ndcg=0.8,
                    relevance_threshold=2,
                ),
                make_metrics(
                    precision=0.2,
                    recall=0.5,
                    reciprocal_rank=1.0,
                    ndcg=0.8,
                    relevance_threshold=3,
                ),
            )
        )

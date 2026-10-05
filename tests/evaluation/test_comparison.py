import pytest

from research_assistant.evaluation.comparison import (
    compare_metrics,
)
from research_assistant.evaluation.models import (
    ComparisonOutcome,
    RetrievalMetrics,
)


def make_metrics(
    *,
    precision: float,
    recall: float,
    reciprocal_rank: float,
    ndcg: float,
    k: int = 5,
) -> RetrievalMetrics:
    return RetrievalMetrics(
        k=k,
        relevance_threshold=2,
        retrieved_count=k,
        relevant_total=5,
        relevant_retrieved=round(recall * 5),
        precision_at_k=precision,
        recall_at_k=recall,
        reciprocal_rank=(reciprocal_rank),
        ndcg_at_k=ndcg,
    )


def test_improvement() -> None:
    baseline = make_metrics(
        precision=0.4,
        recall=0.4,
        reciprocal_rank=0.5,
        ndcg=0.5,
    )

    candidate = make_metrics(
        precision=0.6,
        recall=0.6,
        reciprocal_rank=1.0,
        ndcg=0.8,
    )

    comparison = compare_metrics(
        baseline_mode="hybrid",
        candidate_mode=("hybrid_reranked"),
        baseline=baseline,
        candidate=candidate,
    )

    assert comparison.outcome == ComparisonOutcome.IMPROVEMENT

    assert comparison.delta.ndcg_at_k == pytest.approx(0.3)


def test_degradation() -> None:
    baseline = make_metrics(
        precision=0.8,
        recall=0.8,
        reciprocal_rank=1.0,
        ndcg=0.9,
    )

    candidate = make_metrics(
        precision=0.4,
        recall=0.4,
        reciprocal_rank=0.5,
        ndcg=0.6,
    )

    comparison = compare_metrics(
        baseline_mode="hybrid",
        candidate_mode=("hybrid_reranked"),
        baseline=baseline,
        candidate=candidate,
    )

    assert comparison.outcome == ComparisonOutcome.DEGRADATION


def test_no_change() -> None:
    metrics = make_metrics(
        precision=0.6,
        recall=0.6,
        reciprocal_rank=1.0,
        ndcg=0.8,
    )

    comparison = compare_metrics(
        baseline_mode="hybrid",
        candidate_mode=("hybrid_reranked"),
        baseline=metrics,
        candidate=metrics,
    )

    assert comparison.outcome == ComparisonOutcome.NO_CHANGE


def test_mixed_change() -> None:
    baseline = make_metrics(
        precision=0.4,
        recall=0.4,
        reciprocal_rank=1.0,
        ndcg=0.7,
    )

    candidate = make_metrics(
        precision=0.6,
        recall=0.6,
        reciprocal_rank=0.5,
        ndcg=0.8,
    )

    comparison = compare_metrics(
        baseline_mode="hybrid",
        candidate_mode=("hybrid_reranked"),
        baseline=baseline,
        candidate=candidate,
    )

    assert comparison.outcome == ComparisonOutcome.MIXED


def test_different_k_values_are_rejected() -> None:
    baseline = make_metrics(
        precision=0.4,
        recall=0.4,
        reciprocal_rank=1.0,
        ndcg=0.7,
        k=5,
    )

    candidate = make_metrics(
        precision=0.4,
        recall=0.4,
        reciprocal_rank=1.0,
        ndcg=0.7,
        k=10,
    )

    with pytest.raises(
        ValueError,
        match="different k values",
    ):
        compare_metrics(
            baseline_mode="hybrid",
            candidate_mode=("hybrid_reranked"),
            baseline=baseline,
            candidate=candidate,
        )


def test_negative_epsilon_is_rejected() -> None:
    metrics = make_metrics(
        precision=0.4,
        recall=0.4,
        reciprocal_rank=1.0,
        ndcg=0.7,
    )

    with pytest.raises(
        ValueError,
        match=("epsilon cannot be negative"),
    ):
        compare_metrics(
            baseline_mode="hybrid",
            candidate_mode=("hybrid_reranked"),
            baseline=metrics,
            candidate=metrics,
            epsilon=-0.1,
        )


def test_different_relevance_thresholds_are_rejected() -> None:
    baseline = RetrievalMetrics(
        k=5,
        relevance_threshold=2,
        retrieved_count=5,
        relevant_total=2,
        relevant_retrieved=1,
        precision_at_k=0.2,
        recall_at_k=0.5,
        reciprocal_rank=1.0,
        ndcg_at_k=0.7,
    )

    candidate = RetrievalMetrics(
        k=5,
        relevance_threshold=3,
        retrieved_count=5,
        relevant_total=1,
        relevant_retrieved=1,
        precision_at_k=0.2,
        recall_at_k=1.0,
        reciprocal_rank=1.0,
        ndcg_at_k=0.7,
    )

    with pytest.raises(
        ValueError,
        match=("different relevance thresholds"),
    ):
        compare_metrics(
            baseline_mode="hybrid",
            candidate_mode=("hybrid_reranked"),
            baseline=baseline,
            candidate=candidate,
        )

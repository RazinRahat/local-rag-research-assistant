from research_assistant.evaluation.models import (
    ComparisonOutcome,
    MetricDelta,
    RetrievalComparison,
    RetrievalMetrics,
)

DEFAULT_EPSILON = 1e-9


def compare_metrics(
    *,
    baseline_mode: str,
    candidate_mode: str,
    baseline: RetrievalMetrics,
    candidate: RetrievalMetrics,
    epsilon: float = DEFAULT_EPSILON,
) -> RetrievalComparison:
    """Compare candidate retrieval metrics against a baseline."""

    if baseline.k != candidate.k:
        raise ValueError("Cannot compare metrics with different k values")

    if baseline.relevance_threshold != candidate.relevance_threshold:
        raise ValueError("Cannot compare metrics with different relevance thresholds")

    if epsilon < 0.0:
        raise ValueError("epsilon cannot be negative")

    delta = MetricDelta(
        precision_at_k=(candidate.precision_at_k - baseline.precision_at_k),
        recall_at_k=(candidate.recall_at_k - baseline.recall_at_k),
        reciprocal_rank=(candidate.reciprocal_rank - baseline.reciprocal_rank),
        ndcg_at_k=(candidate.ndcg_at_k - baseline.ndcg_at_k),
    )

    values = (
        delta.precision_at_k,
        delta.recall_at_k,
        delta.reciprocal_rank,
        delta.ndcg_at_k,
    )

    has_positive = any(value > epsilon for value in values)

    has_negative = any(value < -epsilon for value in values)

    if has_positive and has_negative:
        outcome = ComparisonOutcome.MIXED

    elif has_positive:
        outcome = ComparisonOutcome.IMPROVEMENT

    elif has_negative:
        outcome = ComparisonOutcome.DEGRADATION

    else:
        outcome = ComparisonOutcome.NO_CHANGE

    return RetrievalComparison(
        baseline_mode=(baseline_mode),
        candidate_mode=(candidate_mode),
        baseline=baseline,
        candidate=candidate,
        delta=delta,
        outcome=outcome,
    )

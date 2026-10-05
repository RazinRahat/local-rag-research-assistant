from collections.abc import (
    Sequence,
)
from statistics import (
    fmean,
)

from research_assistant.evaluation.models import (
    AggregateRetrievalMetrics,
    RetrievalMetrics,
)


def aggregate_metrics(
    metrics: Sequence[RetrievalMetrics],
) -> AggregateRetrievalMetrics:
    """Macro-average retrieval metrics across queries."""

    if not metrics:
        raise ValueError("Cannot aggregate an empty metric set")

    k = metrics[0].k

    relevance_threshold = metrics[0].relevance_threshold

    for item in metrics[1:]:
        if item.k != k:
            raise ValueError("Cannot aggregate metrics with different k values")

        if item.relevance_threshold != relevance_threshold:
            raise ValueError(
                "Cannot aggregate metrics with different relevance thresholds"
            )

    return AggregateRetrievalMetrics(
        query_count=len(metrics),
        k=k,
        relevance_threshold=(relevance_threshold),
        mean_precision_at_k=fmean(item.precision_at_k for item in metrics),
        mean_recall_at_k=fmean(item.recall_at_k for item in metrics),
        mean_reciprocal_rank=fmean(item.reciprocal_rank for item in metrics),
        mean_ndcg_at_k=fmean(item.ndcg_at_k for item in metrics),
    )

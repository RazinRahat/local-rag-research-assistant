from collections.abc import (
    Sequence,
)
from math import (
    log2,
)

from research_assistant.evaluation.models import (
    EvaluationQuery,
    RetrievalMetrics,
)


def _dcg(
    relevance_values: Sequence[int],
) -> float:
    total = 0.0

    for rank, relevance in enumerate(
        relevance_values,
        start=1,
    ):
        gain = float((2**relevance) - 1)

        discount = log2(rank + 1)

        total += gain / discount

    return total


def evaluate_ranking(
    retrieved_chunk_ids: Sequence[str],
    evaluation_query: EvaluationQuery,
    *,
    k: int,
    relevance_threshold: int = 2,
) -> RetrievalMetrics:
    """Evaluate a ranked chunk list against graded judgments."""

    if k < 1:
        raise ValueError("k must be at least 1")

    if relevance_threshold not in {
        1,
        2,
        3,
    }:
        raise ValueError("relevance_threshold must be between 1 and 3")

    top_ids = tuple(retrieved_chunk_ids[:k])

    if len(top_ids) != len(set(top_ids)):
        raise ValueError("Retrieved ranking contains duplicate chunk IDs")

    relevance_by_chunk = {
        judgment.chunk_id: (judgment.relevance)
        for judgment in evaluation_query.judgments
    }

    relevant_total = sum(
        1
        for relevance in relevance_by_chunk.values()
        if (relevance >= relevance_threshold)
    )

    if relevant_total == 0:
        raise ValueError(
            "Evaluation query has no judgments meeting the binary relevance threshold"
        )

    retrieved_relevances = tuple(
        relevance_by_chunk.get(
            chunk_id,
            0,
        )
        for chunk_id in top_ids
    )

    relevant_retrieved = sum(
        1 for relevance in retrieved_relevances if (relevance >= relevance_threshold)
    )

    precision_at_k = relevant_retrieved / k

    recall_at_k = relevant_retrieved / relevant_total

    reciprocal_rank = 0.0

    for rank, relevance in enumerate(
        retrieved_relevances,
        start=1,
    ):
        if relevance >= relevance_threshold:
            reciprocal_rank = 1.0 / rank
            break

    dcg = _dcg(retrieved_relevances)

    ideal_relevances = sorted(
        (relevance for relevance in relevance_by_chunk.values() if relevance > 0),
        reverse=True,
    )[:k]

    ideal_dcg = _dcg(ideal_relevances)

    ndcg_at_k = dcg / ideal_dcg if ideal_dcg > 0.0 else 0.0

    return RetrievalMetrics(
        k=k,
        relevance_threshold=(relevance_threshold),
        retrieved_count=len(top_ids),
        relevant_total=(relevant_total),
        relevant_retrieved=(relevant_retrieved),
        precision_at_k=(precision_at_k),
        recall_at_k=(recall_at_k),
        reciprocal_rank=(reciprocal_rank),
        ndcg_at_k=(ndcg_at_k),
    )

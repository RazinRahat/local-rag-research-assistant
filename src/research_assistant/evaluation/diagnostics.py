from research_assistant.evaluation.models import (
    ComparisonOutcome,
)


def classify_top1_change(
    baseline_relevance: int,
    candidate_relevance: int,
) -> ComparisonOutcome:
    """Classify change in top-ranked evidence quality."""

    if candidate_relevance > baseline_relevance:
        return ComparisonOutcome.IMPROVEMENT

    if candidate_relevance < baseline_relevance:
        return ComparisonOutcome.DEGRADATION

    return ComparisonOutcome.NO_CHANGE


def combine_outcomes(
    metric_outcome: ComparisonOutcome,
    top1_outcome: ComparisonOutcome,
) -> ComparisonOutcome:
    """Combine metric and top-1 evidence outcomes."""

    if (
        metric_outcome == ComparisonOutcome.MIXED
        or top1_outcome == ComparisonOutcome.MIXED
    ):
        return ComparisonOutcome.MIXED

    substantive = {
        outcome
        for outcome in (
            metric_outcome,
            top1_outcome,
        )
        if (outcome != ComparisonOutcome.NO_CHANGE)
    }

    if not substantive:
        return ComparisonOutcome.NO_CHANGE

    if len(substantive) == 1:
        return substantive.pop()

    return ComparisonOutcome.MIXED

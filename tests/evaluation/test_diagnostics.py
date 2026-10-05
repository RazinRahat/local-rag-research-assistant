from research_assistant.evaluation.diagnostics import (
    classify_top1_change,
    combine_outcomes,
)
from research_assistant.evaluation.models import (
    ComparisonOutcome,
)


def test_top1_improvement() -> None:
    assert (
        classify_top1_change(
            2,
            3,
        )
        == ComparisonOutcome.IMPROVEMENT
    )


def test_top1_degradation() -> None:
    assert (
        classify_top1_change(
            3,
            2,
        )
        == ComparisonOutcome.DEGRADATION
    )


def test_top1_no_change() -> None:
    assert (
        classify_top1_change(
            3,
            3,
        )
        == ComparisonOutcome.NO_CHANGE
    )


def test_metric_improvement_with_top1_degradation_is_mixed() -> None:
    outcome = combine_outcomes(
        ComparisonOutcome.IMPROVEMENT,
        ComparisonOutcome.DEGRADATION,
    )

    assert outcome == ComparisonOutcome.MIXED


def test_metric_improvement_with_stable_top1_is_improvement() -> None:
    outcome = combine_outcomes(
        ComparisonOutcome.IMPROVEMENT,
        ComparisonOutcome.NO_CHANGE,
    )

    assert outcome == ComparisonOutcome.IMPROVEMENT


def test_matching_degradations_remain_degradation() -> None:
    outcome = combine_outcomes(
        ComparisonOutcome.DEGRADATION,
        ComparisonOutcome.DEGRADATION,
    )

    assert outcome == ComparisonOutcome.DEGRADATION


def test_all_no_change_remains_no_change() -> None:
    outcome = combine_outcomes(
        ComparisonOutcome.NO_CHANGE,
        ComparisonOutcome.NO_CHANGE,
    )

    assert outcome == ComparisonOutcome.NO_CHANGE

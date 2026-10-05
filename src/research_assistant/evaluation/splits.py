from enum import StrEnum

from research_assistant.evaluation.models import (
    EvaluationQuery,
)


class EvaluationSplit(StrEnum):
    """Supported benchmark partitions."""

    DEVELOPMENT = "dev"
    HOLDOUT = "holdout"


def query_split(
    query: EvaluationQuery,
) -> EvaluationSplit:
    """Infer a benchmark split from its frozen query ID."""

    if query.id.startswith("dev_"):
        return EvaluationSplit.DEVELOPMENT

    if query.id.startswith("holdout_"):
        return EvaluationSplit.HOLDOUT

    raise ValueError(
        f"Evaluation query ID does not declare a supported split: {query.id}"
    )


def select_queries(
    queries: tuple[
        EvaluationQuery,
        ...,
    ],
    *,
    split: EvaluationSplit,
) -> tuple[
    EvaluationQuery,
    ...,
]:
    """Return only queries belonging to one benchmark split."""

    selected = tuple(query for query in queries if query_split(query) == split)

    if not selected:
        raise ValueError(f"No evaluation queries found for split: {split}")

    return selected

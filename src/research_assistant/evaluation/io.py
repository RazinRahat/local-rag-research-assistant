import json
from pathlib import Path

from research_assistant.evaluation.models import (
    EvaluationQuery,
)


def load_evaluation_queries(
    path: Path,
) -> tuple[EvaluationQuery, ...]:
    """Load and validate canonical retrieval qrels."""

    raw: object = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(
        raw,
        dict,
    ):
        raise ValueError("Evaluation file must contain a JSON object")

    raw_queries = raw.get("queries")

    if not isinstance(
        raw_queries,
        list,
    ):
        raise ValueError("Evaluation file must contain a queries list")

    if not raw_queries:
        raise ValueError("Evaluation query set cannot be empty")

    queries = tuple(
        EvaluationQuery.model_validate(raw_query) for raw_query in raw_queries
    )

    query_ids = tuple(query.id for query in queries)

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("Evaluation query set contains duplicate IDs")

    return queries

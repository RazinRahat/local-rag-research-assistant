from __future__ import annotations

import argparse
import csv
import json
from collections import (
    OrderedDict,
)
from datetime import (
    UTC,
    datetime,
)
from pathlib import Path
from typing import Any

from research_assistant.evaluation.models import (
    EvaluationQuery,
    RelevanceJudgment,
)

DEFAULT_INPUT = Path("experiments/relevance_review.csv")

DEFAULT_OUTPUT = Path("experiments/relevance_judgments.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a completed relevance-review worksheet and build canonical qrels."
        )
    )

    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def parse_bool(
    value: str,
    *,
    field_name: str,
    row_number: int,
) -> bool:
    clean = value.strip().casefold()

    if clean == "true":
        return True

    if clean == "false":
        return False

    raise ValueError(f"Row {row_number}: {field_name} must be 'true' or 'false'")


def parse_relevance(
    value: str,
    *,
    row_number: int,
) -> int:
    clean = value.strip()

    if not clean:
        raise ValueError(f"Row {row_number}: relevance has not been assigned")

    try:
        relevance = int(clean)
    except ValueError as error:
        raise ValueError(
            f"Row {row_number}: relevance must be an integer from 0 to 3"
        ) from error

    if relevance not in {
        0,
        1,
        2,
        3,
    }:
        raise ValueError(f"Row {row_number}: relevance must be between 0 and 3")

    return relevance


def normalise_document_id(
    value: str,
) -> str | None:
    clean = value.strip()

    return clean if clean else None


def main() -> None:
    args = parse_args()

    grouped: OrderedDict[
        str,
        dict[str, Any],
    ] = OrderedDict()

    with args.csv.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as input_file:
        reader = csv.DictReader(input_file)

        required_columns = {
            "query_id",
            "query",
            "include_in_aggregate",
            "document_id",
            "chunk_id",
            "relevance",
            "rationale",
        }

        fieldnames = set(reader.fieldnames or ())

        missing = required_columns - fieldnames

        if missing:
            raise ValueError(
                "Worksheet is missing required columns: " + ", ".join(sorted(missing))
            )

        for row_number, row in enumerate(
            reader,
            start=2,
        ):
            query_id = row["query_id"].strip()

            query = row["query"].strip()

            chunk_id = row["chunk_id"].strip()

            if not query_id:
                raise ValueError(f"Row {row_number}: query_id is empty")

            if not query:
                raise ValueError(f"Row {row_number}: query is empty")

            if not chunk_id:
                raise ValueError(f"Row {row_number}: chunk_id is empty")

            include_in_aggregate = parse_bool(
                row["include_in_aggregate"],
                field_name=("include_in_aggregate"),
                row_number=(row_number),
            )

            document_id = normalise_document_id(row["document_id"])

            relevance = parse_relevance(
                row["relevance"],
                row_number=(row_number),
            )

            rationale = row["rationale"].strip()

            if relevance > 0 and not rationale:
                raise ValueError(
                    f"Row {row_number}: a rationale is required for relevant judgments"
                )

            group = grouped.get(query_id)

            if group is None:
                group = {
                    "id": query_id,
                    "query": query,
                    "document_id": (document_id),
                    "include_in_aggregate": (include_in_aggregate),
                    "judgments": [],
                    "seen_chunk_ids": set(),
                }

                grouped[query_id] = group

            if group["query"] != query:
                raise ValueError(
                    f"Row {row_number}: "
                    f"query text for "
                    f"{query_id} changed "
                    "within the worksheet"
                )

            if group["document_id"] != document_id:
                raise ValueError(
                    f"Row {row_number}: document scope changed for {query_id}"
                )

            if group["include_in_aggregate"] != include_in_aggregate:
                raise ValueError(
                    f"Row {row_number}: "
                    "include_in_aggregate "
                    "changed within query "
                    f"{query_id}"
                )

            seen_chunk_ids: set[str] = group["seen_chunk_ids"]

            if chunk_id in seen_chunk_ids:
                raise ValueError(
                    f"Row {row_number}: duplicate chunk_id {chunk_id} for {query_id}"
                )

            seen_chunk_ids.add(chunk_id)

            judgments: list[RelevanceJudgment] = group["judgments"]

            judgments.append(
                RelevanceJudgment(
                    chunk_id=chunk_id,
                    relevance=relevance,
                    rationale=(rationale or None),
                )
            )

    if not grouped:
        raise ValueError("Worksheet contains no judgments")

    queries: list[EvaluationQuery] = []

    for group in grouped.values():
        query = EvaluationQuery(
            id=group["id"],
            query=group["query"],
            document_id=(group["document_id"]),
            include_in_aggregate=(group["include_in_aggregate"]),
            judgments=tuple(group["judgments"]),
        )

        queries.append(query)

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "source_csv": str(args.csv),
        "relevance_scale": {
            "0": ("Not relevant"),
            "1": ("Marginally relevant or topically related"),
            "2": ("Relevant supporting evidence"),
            "3": ("Directly answers or strongly supports the query"),
        },
        "queries": [query.model_dump(mode="json") for query in queries],
    }

    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    aggregate_queries = sum(1 for query in queries if query.include_in_aggregate)

    diagnostic_queries = len(queries) - aggregate_queries

    judgment_count = sum(len(query.judgments) for query in queries)

    relevant_count = sum(
        1 for query in queries for judgment in query.judgments if judgment.relevance > 0
    )

    print(
        "Output:",
        args.output,
    )

    print(
        "Queries:",
        len(queries),
    )

    print(
        "Primary aggregate queries:",
        aggregate_queries,
    )

    print(
        "Diagnostic-only queries:",
        diagnostic_queries,
    )

    print(
        "Judgments:",
        judgment_count,
    )

    print(
        "Relevant judgments:",
        relevant_count,
    )


if __name__ == "__main__":
    main()

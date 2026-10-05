from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

DEFAULT_RESULTS_DIR = Path("experiments/results")

DEFAULT_OUTPUT = Path("experiments/relevance_review.csv")


FIELDNAMES = (
    "query_id",
    "query",
    "include_in_aggregate",
    "document_id",
    "candidate_index",
    "chunk_id",
    "file_name",
    "page_number",
    "chunk_index",
    "text",
    "relevance",
    "rationale",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the blind Phase 13 relevance pool as a CSV review worksheet."
        )
    )

    parser.add_argument(
        "--review-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    return parser.parse_args()


def find_latest_review_file() -> Path:
    candidates = sorted(
        DEFAULT_RESULTS_DIR.glob("relevance_pool_review_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError(
            "No relevance pool review file was found under experiments/results/"
        )

    return candidates[0]


def load_review(
    path: Path,
) -> dict[str, Any]:
    raw: object = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(
        raw,
        dict,
    ):
        raise ValueError("Review file must contain a JSON object")

    return raw


def main() -> None:
    args = parse_args()

    review_path = args.review_file or find_latest_review_file()

    review = load_review(review_path)

    raw_queries = review.get("queries")

    if not isinstance(
        raw_queries,
        list,
    ):
        raise ValueError("Review file contains no valid queries list")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    row_count = 0

    with args.output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()

        for raw_query in raw_queries:
            if not isinstance(
                raw_query,
                dict,
            ):
                raise ValueError("Review query must be an object")

            query_id = raw_query.get("id")

            query = raw_query.get("query")

            document_id = raw_query.get("document_id")

            raw_candidates = raw_query.get("candidates")

            if not isinstance(
                query_id,
                str,
            ):
                raise ValueError("Review query has no valid id")

            if not isinstance(
                query,
                str,
            ):
                raise ValueError(f"{query_id} has no valid query text")

            if not isinstance(
                raw_candidates,
                list,
            ):
                raise ValueError(f"{query_id} has no candidate list")

            include_in_aggregate = query_id != "dataset"

            for candidate in raw_candidates:
                if not isinstance(
                    candidate,
                    dict,
                ):
                    raise ValueError(f"{query_id} contains an invalid candidate")

                writer.writerow(
                    {
                        "query_id": (query_id),
                        "query": (query),
                        "include_in_aggregate": (str(include_in_aggregate).lower()),
                        "document_id": (document_id or ""),
                        "candidate_index": (candidate.get("candidate_index")),
                        "chunk_id": (candidate.get("chunk_id")),
                        "file_name": (candidate.get("file_name")),
                        "page_number": (candidate.get("page_number")),
                        "chunk_index": (candidate.get("chunk_index")),
                        "text": (candidate.get("text")),
                        "relevance": "",
                        "rationale": "",
                    }
                )

                row_count += 1

    print(
        "Source:",
        review_path,
    )

    print(
        "Worksheet:",
        args.output,
    )

    print(
        "Candidates:",
        row_count,
    )

    print()
    print(
        "The dataset query is marked "
        "include_in_aggregate=false "
        "because its whole-corpus wording "
        "is ambiguous."
    )


if __name__ == "__main__":
    main()

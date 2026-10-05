import json
from pathlib import Path

import pytest

from research_assistant.evaluation.io import (
    load_evaluation_queries,
)


def write_json(
    path: Path,
    payload: object,
) -> None:
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def make_query(
    query_id: str,
) -> dict[str, object]:
    return {
        "id": query_id,
        "query": "example query",
        "document_id": None,
        "include_in_aggregate": True,
        "judgments": [
            {
                "chunk_id": "chunk-a",
                "relevance": 3,
                "rationale": ("Direct evidence"),
            }
        ],
        "tags": [],
        "notes": None,
    }


def test_loads_valid_queries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qrels.json"

    write_json(
        path,
        {"queries": [make_query("q1")]},
    )

    queries = load_evaluation_queries(path)

    assert len(queries) == 1

    assert queries[0].id == "q1"


def test_preserves_aggregate_flag(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qrels.json"

    query = make_query("dataset")

    query["include_in_aggregate"] = False

    write_json(
        path,
        {"queries": [query]},
    )

    queries = load_evaluation_queries(path)

    assert queries[0].include_in_aggregate is False


def test_rejects_empty_query_set(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qrels.json"

    write_json(
        path,
        {"queries": []},
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        load_evaluation_queries(path)


def test_rejects_duplicate_query_ids(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qrels.json"

    write_json(
        path,
        {
            "queries": [
                make_query("q1"),
                make_query("q1"),
            ]
        },
    )

    with pytest.raises(
        ValueError,
        match="duplicate IDs",
    ):
        load_evaluation_queries(path)


def test_rejects_missing_queries_list(
    tmp_path: Path,
) -> None:
    path = tmp_path / "qrels.json"

    write_json(
        path,
        {},
    )

    with pytest.raises(
        ValueError,
        match="queries list",
    ):
        load_evaluation_queries(path)

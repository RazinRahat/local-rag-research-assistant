from __future__ import annotations

import argparse
import json
from datetime import (
    UTC,
    datetime,
)
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"

DEFAULT_QUERY_FILE = Path("experiments/retrieval_queries.json")

DEFAULT_OUTPUT_DIR = Path("experiments/results")

MODES = (
    "dense",
    "lexical",
    "hybrid",
    "hybrid_reranked",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a pooled candidate set for human retrieval relevance judgments."
        )
    )

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
    )

    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_QUERY_FILE,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--pool-k",
        type=int,
        default=20,
    )

    args = parser.parse_args()

    if not 1 <= args.pool_k <= 20:
        parser.error("--pool-k must be between 1 and 20")

    return args


def load_queries(
    path: Path,
) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(raw, dict):
        candidate_queries = raw.get("queries")
    else:
        candidate_queries = raw

    if not isinstance(
        candidate_queries,
        list,
    ):
        raise ValueError(
            "Query file must contain a list or an object with a 'queries' list."
        )

    queries: list[dict[str, Any]] = []

    for index, item in enumerate(
        candidate_queries,
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(f"Query #{index} must be an object")

        query_id = item.get("id")
        query_text = item.get("query")

        if (
            not isinstance(
                query_id,
                str,
            )
            or not query_id.strip()
        ):
            raise ValueError(f"Query #{index} has no valid id")

        if (
            not isinstance(
                query_text,
                str,
            )
            or not query_text.strip()
        ):
            raise ValueError(f"Query #{index} has no valid query")

        document_id = item.get("document_id")

        if document_id is not None and not isinstance(
            document_id,
            str,
        ):
            raise ValueError(f"Query #{index} has invalid document_id")

        queries.append(
            {
                "id": (query_id.strip()),
                "query": (query_text.strip()),
                "document_id": (document_id),
            }
        )

    if not queries:
        raise ValueError("Query set cannot be empty")

    return queries


def search(
    client: httpx.Client,
    *,
    base_url: str,
    query: str,
    mode: str,
    top_k: int,
    document_id: str | None,
) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {
        "query": query,
        "top_k": top_k,
        "mode": mode,
    }

    if document_id is not None:
        payload["document_id"] = document_id

    response = client.post(
        (f"{base_url.rstrip('/')}/search"),
        json=payload,
    )

    response.raise_for_status()

    body = response.json()

    results = body.get("results")

    if not isinstance(
        results,
        list,
    ):
        raise ValueError("Search response contains no valid results list")

    return results


def extract_candidate(
    result: dict[str, Any],
) -> dict[str, Any]:
    chunk = result.get("chunk")

    if not isinstance(
        chunk,
        dict,
    ):
        raise ValueError("Search result is missing chunk data")

    chunk_id = chunk.get("chunk_id")

    if not isinstance(
        chunk_id,
        str,
    ):
        raise ValueError("Chunk has no valid chunk_id")

    return {
        "chunk_id": chunk_id,
        "document_id": (chunk.get("document_id")),
        "file_name": (chunk.get("file_name")),
        "page_number": (chunk.get("page_number")),
        "chunk_index": (chunk.get("chunk_index")),
        "page_chunk_index": (chunk.get("page_chunk_index")),
        "token_count": (chunk.get("token_count")),
        "text": chunk.get("text"),
    }


def validate_same_candidate(
    existing: dict[str, Any],
    observed: dict[str, Any],
) -> None:
    fields = (
        "document_id",
        "file_name",
        "page_number",
        "chunk_index",
        "page_chunk_index",
        "token_count",
        "text",
    )

    for field in fields:
        if existing.get(field) != observed.get(field):
            raise ValueError(
                f"Conflicting provenance for chunk {existing['chunk_id']}: {field}"
            )


def build_query_pool(
    client: httpx.Client,
    *,
    base_url: str,
    query_item: dict[str, Any],
    pool_k: int,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    query_id = query_item["id"]
    query = query_item["query"]

    document_id = query_item.get("document_id")

    candidates: dict[
        str,
        dict[str, Any],
    ] = {}

    trace_candidates: dict[
        str,
        dict[str, Any],
    ] = {}

    systems: dict[
        str,
        list[str],
    ] = {}

    for mode in MODES:
        results = search(
            client,
            base_url=base_url,
            query=query,
            mode=mode,
            top_k=pool_k,
            document_id=document_id,
        )

        system_ids: list[str] = []

        for result in results:
            candidate = extract_candidate(result)

            chunk_id = candidate["chunk_id"]

            existing = candidates.get(chunk_id)

            if existing is None:
                candidates[chunk_id] = candidate
            else:
                validate_same_candidate(
                    existing,
                    candidate,
                )

            system_ids.append(chunk_id)

            trace = trace_candidates.setdefault(
                chunk_id,
                {
                    "chunk_id": (chunk_id),
                    "retrieved_by": {},
                },
            )

            trace["retrieved_by"][mode] = {
                "rank": result.get("rank"),
                "score": result.get("score"),
            }

        systems[mode] = system_ids

    ordered_candidates = sorted(
        candidates.values(),
        key=lambda item: item["chunk_id"],
    )

    review_candidates = []

    for index, candidate in enumerate(
        ordered_candidates,
        start=1,
    ):
        review_candidates.append(
            {
                "candidate_index": (index),
                **candidate,
                "relevance": None,
                "rationale": None,
            }
        )

    review_query = {
        "id": query_id,
        "query": query,
        "document_id": document_id,
        "candidate_count": len(review_candidates),
        "candidates": (review_candidates),
    }

    trace_query = {
        "id": query_id,
        "query": query,
        "document_id": document_id,
        "systems": systems,
        "candidates": (
            sorted(
                trace_candidates.values(),
                key=lambda item: item["chunk_id"],
            )
        ),
    }

    return (
        review_query,
        trace_query,
    )


def main() -> None:
    args = parse_args()

    queries = load_queries(args.queries)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    review_queries = []
    trace_queries = []

    with httpx.Client(timeout=120.0) as client:
        for query_item in queries:
            (
                review_query,
                trace_query,
            ) = build_query_pool(
                client,
                base_url=args.base_url,
                query_item=query_item,
                pool_k=args.pool_k,
            )

            review_queries.append(review_query)

            trace_queries.append(trace_query)

            print(
                f"{query_item['id']}: "
                f"{review_query['candidate_count']} "
                "unique candidates"
            )

    review_report = {
        "schema_version": 1,
        "purpose": ("Human relevance judgments for Phase 13 retrieval evaluation"),
        "created_at": datetime.now(UTC).isoformat(),
        "query_file": str(args.queries),
        "pool_depth_per_mode": (args.pool_k),
        "modes_used_for_pooling": (list(MODES)),
        "relevance_scale": {
            "0": ("Not relevant to answering the query"),
            "1": ("Marginally relevant or topically related"),
            "2": ("Relevant and useful supporting evidence"),
            "3": ("Directly answers or strongly supports the query"),
        },
        "judging_instructions": [
            ("Judge the passage against the query itself."),
            (
                "Do not infer relevance from "
                "which retrieval system may "
                "have produced the passage."
            ),
            (
                "Use relevance 3 only for "
                "directly answering or "
                "strongly supporting evidence."
            ),
            (
                "Use relevance 1 for passages "
                "that merely mention the topic "
                "without answering the likely "
                "information need."
            ),
            (
                "Write a short rationale for "
                "every relevance value above 0 "
                "and for any difficult 0."
            ),
            (
                "For ambiguous whole-corpus "
                "queries, judge the literal "
                "query first and document the "
                "ambiguity in the rationale."
            ),
        ],
        "queries": review_queries,
    }

    trace_report = {
        "schema_version": 1,
        "experiment": ("phase13_relevance_pool"),
        "created_at": datetime.now(UTC).isoformat(),
        "query_file": str(args.queries),
        "pool_depth_per_mode": (args.pool_k),
        "modes": list(MODES),
        "warning": (
            "This trace contains retrieval "
            "system ranks and scores. Do not "
            "use it while assigning relevance "
            "judgments because it can bias "
            "human labels."
        ),
        "queries": trace_queries,
    }

    review_path = args.output_dir / (f"relevance_pool_review_{timestamp}.json")

    trace_path = args.output_dir / (f"relevance_pool_trace_{timestamp}.json")

    review_path.write_text(
        json.dumps(
            review_report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    trace_path.write_text(
        json.dumps(
            trace_report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Review file:",
        review_path,
    )

    print(
        "Trace file:",
        trace_path,
    )

    print()
    print("IMPORTANT: judge relevance using the review file only.")


if __name__ == "__main__":
    main()

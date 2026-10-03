from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

RETRIEVAL_MODES = (
    "dense",
    "lexical",
    "hybrid",
)


@dataclass(frozen=True)
class QueryCase:
    id: str
    query: str


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    score: float
    chunk_id: str
    document_id: str
    file_name: str
    page_number: int
    chunk_index: int
    page_chunk_index: int
    token_count: int
    text: str


@dataclass(frozen=True)
class ModeResult:
    mode: str
    elapsed_ms: float
    results: tuple[RetrievedChunk, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare dense, lexical, and hybrid retrieval against the same query set."
        )
    )

    parser.add_argument(
        "--query-file",
        type=Path,
        default=Path("experiments/retrieval_queries.json"),
        help=("JSON file containing retrieval comparison queries."),
    )

    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Research API base URL.",
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of final results per mode.",
    )

    parser.add_argument(
        "--document-id",
        default=None,
        help=("Optional SHA-256 document ID used to scope every comparison query."),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results"),
        help="Directory for experiment reports.",
    )

    return parser.parse_args()


def load_queries(
    path: Path,
) -> tuple[QueryCase, ...]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError("Query file must contain a JSON list.")

    queries: list[QueryCase] = []

    for index, item in enumerate(
        data,
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(f"Query entry {index} must be an object.")

        query_id = item.get("id")

        query = item.get("query")

        if (
            not isinstance(
                query_id,
                str,
            )
            or not query_id.strip()
        ):
            raise ValueError(f"Query entry {index} has an invalid id.")

        if (
            not isinstance(
                query,
                str,
            )
            or not query.strip()
        ):
            raise ValueError(f"Query entry {index} has an invalid query.")

        queries.append(
            QueryCase(
                id=query_id.strip(),
                query=query.strip(),
            )
        )

    if not queries:
        raise ValueError("Query file cannot be empty.")

    return tuple(queries)


def parse_search_results(
    payload: dict[str, Any],
) -> tuple[RetrievedChunk, ...]:
    raw_results = payload.get("results")

    if not isinstance(
        raw_results,
        list,
    ):
        raise ValueError("Search response is missing results.")

    parsed: list[RetrievedChunk] = []

    for item in raw_results:
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError("Search result must be an object.")

        chunk = item.get("chunk")

        if not isinstance(
            chunk,
            dict,
        ):
            raise ValueError("Search result is missing chunk data.")

        parsed.append(
            RetrievedChunk(
                rank=int(item["rank"]),
                score=float(item["score"]),
                chunk_id=str(chunk["chunk_id"]),
                document_id=str(chunk["document_id"]),
                file_name=str(chunk["file_name"]),
                page_number=int(chunk["page_number"]),
                chunk_index=int(chunk["chunk_index"]),
                page_chunk_index=int(chunk["page_chunk_index"]),
                token_count=int(chunk["token_count"]),
                text=str(chunk["text"]),
            )
        )

    return tuple(parsed)


def run_mode(
    client: httpx.Client,
    *,
    query: str,
    mode: str,
    top_k: int,
    document_id: str | None,
) -> ModeResult:
    request_body: dict[
        str,
        Any,
    ] = {
        "query": query,
        "top_k": top_k,
        "mode": mode,
    }

    if document_id:
        request_body["document_id"] = document_id

    started = time.perf_counter()

    response = client.post(
        "/search",
        json=request_body,
    )

    elapsed_ms = (time.perf_counter() - started) * 1000

    response.raise_for_status()

    payload = response.json()

    if not isinstance(
        payload,
        dict,
    ):
        raise ValueError("Search response must be an object.")

    return ModeResult(
        mode=mode,
        elapsed_ms=elapsed_ms,
        results=(parse_search_results(payload)),
    )


def chunk_ids(
    result: ModeResult,
) -> tuple[str, ...]:
    return tuple(item.chunk_id for item in result.results)


def pairwise_overlap(
    first: ModeResult,
    second: ModeResult,
) -> dict[str, Any]:
    first_ids = set(chunk_ids(first))

    second_ids = set(chunk_ids(second))

    shared = first_ids & second_ids

    union = first_ids | second_ids

    jaccard = len(shared) / len(union) if union else 1.0

    return {
        "first_mode": first.mode,
        "second_mode": second.mode,
        "shared_chunk_count": len(shared),
        "jaccard": jaccard,
        "shared_chunk_ids": sorted(shared),
        "first_only": sorted(first_ids - second_ids),
        "second_only": sorted(second_ids - first_ids),
    }


def build_comparison(
    mode_results: tuple[
        ModeResult,
        ...,
    ],
) -> dict[str, Any]:
    by_mode = {result.mode: result for result in mode_results}

    dense = by_mode["dense"]

    lexical = by_mode["lexical"]

    hybrid = by_mode["hybrid"]

    return {
        "dense_vs_lexical": pairwise_overlap(
            dense,
            lexical,
        ),
        "dense_vs_hybrid": pairwise_overlap(
            dense,
            hybrid,
        ),
        "lexical_vs_hybrid": pairwise_overlap(
            lexical,
            hybrid,
        ),
    }


def print_query_summary(
    query_case: QueryCase,
    mode_results: tuple[
        ModeResult,
        ...,
    ],
) -> None:
    print()
    print("=" * 80)
    print(f"{query_case.id}: {query_case.query}")
    print("=" * 80)

    for mode_result in mode_results:
        print()
        print(mode_result.mode.upper())

        print(f"elapsed: {mode_result.elapsed_ms:.1f} ms")

        if not (mode_result.results):
            print("  no results")

            continue

        for result in mode_result.results:
            preview = " ".join(result.text.split())

            if len(preview) > 120:
                preview = preview[:117] + "..."

            print(
                f"  #{result.rank} "
                f"score={result.score:.6f} "
                f"{result.file_name} "
                f"p.{result.page_number}"
            )

            print(f"      {result.chunk_id}")

            print(f"      {preview}")


def main() -> None:
    args = parse_args()

    if args.top_k < 1:
        raise ValueError("top_k must be at least 1.")

    queries = load_queries(args.query_file)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    run_started_at = datetime.now(UTC)

    report_queries: list[dict[str, Any]] = []

    with httpx.Client(
        base_url=(args.base_url.rstrip("/")),
        timeout=120.0,
    ) as client:
        health = client.get("/health")

        health.raise_for_status()

        for query_case in queries:
            mode_results = tuple(
                run_mode(
                    client,
                    query=(query_case.query),
                    mode=mode,
                    top_k=(args.top_k),
                    document_id=(args.document_id),
                )
                for mode in RETRIEVAL_MODES
            )

            print_query_summary(
                query_case,
                mode_results,
            )

            report_queries.append(
                {
                    "id": query_case.id,
                    "query": query_case.query,
                    "modes": {
                        result.mode: {
                            "elapsed_ms": result.elapsed_ms,
                            "results": [asdict(item) for item in result.results],
                        }
                        for result in mode_results
                    },
                    "overlap": build_comparison(mode_results),
                }
            )

    finished_at = datetime.now(UTC)

    timestamp = run_started_at.strftime("%Y%m%dT%H%M%SZ")

    output_path = args.output_dir / (f"retrieval_comparison_{timestamp}.json")

    report = {
        "experiment": "phase11_retrieval_comparison",
        "started_at": run_started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "base_url": args.base_url,
        "top_k": args.top_k,
        "document_id": args.document_id,
        "modes": list(RETRIEVAL_MODES),
        "score_warning": (
            "Scores from dense, lexical, "
            "and hybrid retrieval use "
            "different scales and must not "
            "be compared numerically."
        ),
        "queries": report_queries,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
            ensure_ascii=False,
        )

        handle.write("\n")

    print()
    print("Report written to:")

    print(output_path)


if __name__ == "__main__":
    main()

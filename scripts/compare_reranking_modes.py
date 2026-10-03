from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_QUERY_FILE = Path("experiments/retrieval_queries.json")
DEFAULT_OUTPUT_DIR = Path("experiments/results")

MODES = (
    "hybrid",
    "hybrid_reranked",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare Phase 11 hybrid retrieval "
            "against Phase 12 cross-encoder reranking."
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
        "--top-k",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--document-id",
        default=None,
    )

    parser.add_argument(
        "--no-warmup",
        action="store_true",
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    if args.repeats < 1:
        parser.error("--repeats must be at least 1")

    return args


def load_queries(
    path: Path,
) -> list[dict[str, str]]:
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
            "Query file must contain either "
            "a list of queries or an object "
            "with a 'queries' list."
        )

    queries: list[dict[str, str]] = []

    for index, item in enumerate(
        candidate_queries,
        start=1,
    ):
        if not isinstance(
            item,
            dict,
        ):
            raise ValueError(f"Query #{index} must be an object.")

        query_id = item.get("id")
        query_text = item.get("query")

        if (
            not isinstance(
                query_id,
                str,
            )
            or not query_id.strip()
        ):
            raise ValueError(f"Query #{index} has no valid 'id'.")

        if (
            not isinstance(
                query_text,
                str,
            )
            or not query_text.strip()
        ):
            raise ValueError(f"Query #{index} has no valid 'query'.")

        queries.append(
            {
                "id": query_id.strip(),
                "query": (query_text.strip()),
            }
        )

    if not queries:
        raise ValueError("Query set cannot be empty.")

    return queries


def search(
    client: httpx.Client,
    *,
    base_url: str,
    query: str,
    mode: str,
    top_k: int,
    document_id: str | None,
) -> tuple[
    float,
    list[dict[str, Any]],
]:
    payload: dict[
        str,
        Any,
    ] = {
        "query": query,
        "top_k": top_k,
        "mode": mode,
    }

    if document_id is not None:
        payload["document_id"] = document_id

    started = time.perf_counter()

    response = client.post(
        f"{base_url.rstrip('/')}/search",
        json=payload,
    )

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    response.raise_for_status()

    body = response.json()

    raw_results = body.get("results")

    if not isinstance(
        raw_results,
        list,
    ):
        raise ValueError("Search response does not contain a valid results list.")

    results: list[dict[str, Any]] = []

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
            raise ValueError("Search result is missing chunk metadata.")

        results.append(
            {
                "rank": item.get("rank"),
                "score": item.get("score"),
                "chunk_id": chunk.get("chunk_id"),
                "document_id": (chunk.get("document_id")),
                "file_name": chunk.get("file_name"),
                "page_number": (chunk.get("page_number")),
                "chunk_index": (chunk.get("chunk_index")),
                "page_chunk_index": (chunk.get("page_chunk_index")),
                "token_count": (chunk.get("token_count")),
                "text": chunk.get("text"),
            }
        )

    return (
        elapsed_ms,
        results,
    )


def chunk_ids(
    results: list[dict[str, Any]],
) -> list[str]:
    ids: list[str] = []

    for result in results:
        chunk_id = result.get("chunk_id")

        if not isinstance(
            chunk_id,
            str,
        ):
            raise ValueError("Result has no valid chunk_id.")

        ids.append(chunk_id)

    return ids


def ensure_stable_results(
    runs: list[list[dict[str, Any]]],
    *,
    mode: str,
    query_id: str,
) -> None:
    if not runs:
        raise ValueError("No experiment runs were recorded.")

    expected = chunk_ids(runs[0])

    for run_number, run in enumerate(
        runs[1:],
        start=2,
    ):
        observed = chunk_ids(run)

        if observed != expected:
            raise RuntimeError(
                "Retrieval ranking changed "
                "between repeated runs for "
                f"query '{query_id}', "
                f"mode '{mode}', "
                f"run {run_number}."
            )


def build_overlap(
    hybrid_results: list[dict[str, Any]],
    reranked_results: list[dict[str, Any]],
) -> dict[str, Any]:
    hybrid_ids = chunk_ids(hybrid_results)

    reranked_ids = chunk_ids(reranked_results)

    hybrid_set = set(hybrid_ids)

    reranked_set = set(reranked_ids)

    shared = hybrid_set & reranked_set

    union = hybrid_set | reranked_set

    jaccard = len(shared) / len(union) if union else 1.0

    hybrid_ranks = {
        chunk_id: rank
        for rank, chunk_id in enumerate(
            hybrid_ids,
            start=1,
        )
    }

    reranked_ranks = {
        chunk_id: rank
        for rank, chunk_id in enumerate(
            reranked_ids,
            start=1,
        )
    }

    shared_rank_changes = []

    for chunk_id in sorted(shared):
        hybrid_rank = hybrid_ranks[chunk_id]

        reranked_rank = reranked_ranks[chunk_id]

        shared_rank_changes.append(
            {
                "chunk_id": (chunk_id),
                "hybrid_rank": (hybrid_rank),
                "reranked_rank": (reranked_rank),
                "rank_change": (hybrid_rank - reranked_rank),
            }
        )

    shared_rank_changes.sort(
        key=lambda item: (
            -abs(item["rank_change"]),
            item["chunk_id"],
        )
    )

    return {
        "shared_chunk_count": (len(shared)),
        "jaccard": jaccard,
        "same_top_1": (
            bool(hybrid_ids) and bool(reranked_ids) and hybrid_ids[0] == reranked_ids[0]
        ),
        "shared_chunk_ids": sorted(shared),
        "hybrid_only": [
            chunk_id for chunk_id in hybrid_ids if chunk_id not in reranked_set
        ],
        "reranked_only": [
            chunk_id for chunk_id in reranked_ids if chunk_id not in hybrid_set
        ],
        "shared_rank_changes": (shared_rank_changes),
        "rank_change_definition": (
            "hybrid_rank - reranked_rank; "
            "positive means the reranker "
            "promoted the shared chunk"
        ),
    }


def latency_summary(
    samples: list[float],
) -> dict[str, float]:
    return {
        "mean_ms": (statistics.fmean(samples)),
        "median_ms": (statistics.median(samples)),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def run_mode(
    client: httpx.Client,
    *,
    base_url: str,
    query_id: str,
    query: str,
    mode: str,
    top_k: int,
    document_id: str | None,
    repeats: int,
) -> dict[str, Any]:
    elapsed_samples: list[float] = []

    result_runs: list[list[dict[str, Any]]] = []

    for _ in range(repeats):
        (
            elapsed_ms,
            results,
        ) = search(
            client,
            base_url=base_url,
            query=query,
            mode=mode,
            top_k=top_k,
            document_id=(document_id),
        )

        elapsed_samples.append(elapsed_ms)

        result_runs.append(results)

    ensure_stable_results(
        result_runs,
        mode=mode,
        query_id=query_id,
    )

    return {
        "latency": (latency_summary(elapsed_samples)),
        "elapsed_samples_ms": (elapsed_samples),
        "results": (result_runs[0]),
    }


def print_query_summary(
    *,
    query_id: str,
    query: str,
    hybrid: dict[str, Any],
    reranked: dict[str, Any],
    overlap: dict[str, Any],
) -> None:
    hybrid_results = hybrid["results"]

    reranked_results = reranked["results"]

    hybrid_top = hybrid_results[0] if hybrid_results else None

    reranked_top = reranked_results[0] if reranked_results else None

    print()
    print("=" * 72)
    print(f"{query_id}: {query}")
    print("=" * 72)

    if hybrid_top is not None:
        print(
            "Hybrid top-1:   "
            f"{hybrid_top['file_name']} "
            f"p.{hybrid_top['page_number']} "
            f"[{str(hybrid_top['chunk_id'])[:12]}]"
        )

    if reranked_top is not None:
        print(
            "Reranked top-1: "
            f"{reranked_top['file_name']} "
            f"p.{reranked_top['page_number']} "
            f"[{str(reranked_top['chunk_id'])[:12]}]"
        )

    print(f"Top-1 same:      {overlap['same_top_1']}")

    print(f"Top-k Jaccard:   {overlap['jaccard']:.3f}")

    print(f"Hybrid median:   {hybrid['latency']['median_ms']:.1f} ms")

    print(f"Rerank median:   {reranked['latency']['median_ms']:.1f} ms")


def main() -> None:
    args = parse_args()

    queries = load_queries(args.queries)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    started_at = datetime.now(UTC)

    query_reports: list[dict[str, Any]] = []

    with httpx.Client(timeout=120.0) as client:
        if not args.no_warmup:
            warmup_query = queries[0]["query"]

            print("Running unrecorded warm-up requests...")

            for mode in MODES:
                search(
                    client,
                    base_url=(args.base_url),
                    query=(warmup_query),
                    mode=mode,
                    top_k=args.top_k,
                    document_id=(args.document_id),
                )

        for query_item in queries:
            query_id = query_item["id"]

            query = query_item["query"]

            mode_reports: dict[
                str,
                dict[str, Any],
            ] = {}

            for mode in MODES:
                mode_reports[mode] = run_mode(
                    client,
                    base_url=(args.base_url),
                    query_id=query_id,
                    query=query,
                    mode=mode,
                    top_k=args.top_k,
                    document_id=(args.document_id),
                    repeats=(args.repeats),
                )

            overlap = build_overlap(
                mode_reports["hybrid"]["results"],
                mode_reports["hybrid_reranked"]["results"],
            )

            query_report = {
                "id": query_id,
                "query": query,
                "modes": (mode_reports),
                "comparison": (overlap),
            }

            query_reports.append(query_report)

            print_query_summary(
                query_id=query_id,
                query=query,
                hybrid=mode_reports["hybrid"],
                reranked=mode_reports["hybrid_reranked"],
                overlap=overlap,
            )

    finished_at = datetime.now(UTC)

    jaccards = [item["comparison"]["jaccard"] for item in query_reports]

    top_1_agreements = sum(
        1 for item in query_reports if item["comparison"]["same_top_1"]
    )

    hybrid_medians = [
        item["modes"]["hybrid"]["latency"]["median_ms"] for item in query_reports
    ]

    reranked_medians = [
        item["modes"]["hybrid_reranked"]["latency"]["median_ms"]
        for item in query_reports
    ]

    summary = {
        "query_count": (len(query_reports)),
        "mean_top_k_jaccard": (statistics.fmean(jaccards)),
        "top_1_agreement_count": (top_1_agreements),
        "top_1_agreement_rate": (top_1_agreements / len(query_reports)),
        "mean_hybrid_median_latency_ms": (statistics.fmean(hybrid_medians)),
        "mean_reranked_median_latency_ms": (statistics.fmean(reranked_medians)),
        "mean_added_latency_ms": (
            statistics.fmean(reranked_medians) - statistics.fmean(hybrid_medians)
        ),
    }

    report = {
        "experiment": ("phase12_reranking_comparison"),
        "started_at": (started_at.isoformat()),
        "finished_at": (finished_at.isoformat()),
        "base_url": (args.base_url),
        "query_file": str(args.queries),
        "top_k": args.top_k,
        "repeats": args.repeats,
        "warmup": (not args.no_warmup),
        "document_id": (args.document_id),
        "modes": list(MODES),
        "score_warning": (
            "Hybrid scores are RRF scores; "
            "hybrid_reranked scores are "
            "cross-encoder relevance scores. "
            "They must not be compared "
            "numerically."
        ),
        "evaluation_warning": (
            "This experiment measures "
            "ranking/candidate changes and "
            "latency only. It does not establish "
            "retrieval-quality improvement because "
            "the query set has no relevance labels."
        ),
        "configuration": {
            "hybrid": {
                "rrf_rank_constant": 60,
                "candidate_multiplier": 2,
            },
            "reranking": {
                "model": ("cross-encoder/ms-marco-MiniLM-L6-v2"),
                "candidate_pool_size": 20,
            },
        },
        "summary": summary,
        "queries": query_reports,
    }

    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")

    output_path = args.output_dir / (f"reranking_comparison_{timestamp}.json")

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("PHASE 12 SUMMARY")
    print("=" * 72)

    print(f"Queries:                {summary['query_count']}")

    print(f"Mean top-k Jaccard:     {summary['mean_top_k_jaccard']:.3f}")

    print(
        "Top-1 agreement:        "
        f"{summary['top_1_agreement_count']}"
        "/"
        f"{summary['query_count']}"
    )

    print(f"Hybrid median latency:  {summary['mean_hybrid_median_latency_ms']:.1f} ms")

    print(
        f"Rerank median latency:  {summary['mean_reranked_median_latency_ms']:.1f} ms"
    )

    print(f"Added median latency:   {summary['mean_added_latency_ms']:.1f} ms")

    print()
    print(
        "Report:",
        output_path,
    )


if __name__ == "__main__":
    main()

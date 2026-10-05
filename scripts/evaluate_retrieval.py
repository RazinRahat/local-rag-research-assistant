from __future__ import annotations

import argparse
import json
import time
from datetime import (
    UTC,
    datetime,
)
from pathlib import Path
from typing import Any

import httpx

from research_assistant.evaluation.aggregate import (
    aggregate_metrics,
)
from research_assistant.evaluation.comparison import (
    compare_metrics,
)
from research_assistant.evaluation.io import (
    load_evaluation_queries,
)
from research_assistant.evaluation.metrics import (
    evaluate_ranking,
)
from research_assistant.evaluation.models import (
    EvaluationQuery,
    RetrievalMetrics,
)
from research_assistant.evaluation.splits import (
    EvaluationSplit,
    select_queries,
)

DEFAULT_BASE_URL = "http://127.0.0.1:8000"

DEFAULT_QRELS = Path("experiments/relevance_judgments_phase13_expansion.json")

DEFAULT_OUTPUT_DIR = Path("experiments/results")

MODES = (
    "dense",
    "lexical",
    "hybrid",
    "hybrid_reranked",
)

COMPARISONS = (
    (
        "dense",
        "hybrid",
    ),
    (
        "hybrid",
        "hybrid_reranked",
    ),
    (
        "dense",
        "hybrid_reranked",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run labelled retrieval evaluation on one frozen benchmark split.")
    )

    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
    )

    parser.add_argument(
        "--qrels",
        type=Path,
        default=DEFAULT_QRELS,
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
        "--relevance-threshold",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--split",
        choices=(
            EvaluationSplit.DEVELOPMENT.value,
            EvaluationSplit.HOLDOUT.value,
        ),
        default=(EvaluationSplit.DEVELOPMENT.value),
        help=("Benchmark split to evaluate. Defaults to development."),
    )

    parser.add_argument(
        "--confirm-holdout",
        action="store_true",
        help=("Required when evaluating the frozen holdout split."),
    )

    args = parser.parse_args()

    if not 1 <= args.top_k <= 20:
        parser.error("--top-k must be between 1 and 20")

    if args.relevance_threshold not in {
        1,
        2,
        3,
    }:
        parser.error("--relevance-threshold must be 1, 2, or 3")

    if args.split == EvaluationSplit.HOLDOUT.value and not args.confirm_holdout:
        parser.error(
            "Holdout evaluation is locked. "
            "Use --confirm-holdout only "
            "after retrieval changes "
            "have been frozen."
        )

    return args


def search(
    client: httpx.Client,
    *,
    base_url: str,
    query: EvaluationQuery,
    mode: str,
    top_k: int,
) -> tuple[
    float,
    list[dict[str, Any]],
]:
    payload: dict[
        str,
        Any,
    ] = {
        "query": query.query,
        "top_k": top_k,
        "mode": mode,
    }

    if query.document_id is not None:
        payload["document_id"] = query.document_id

    started = time.perf_counter()

    response = client.post(
        (f"{base_url.rstrip('/')}/search"),
        json=payload,
    )

    elapsed_ms = (time.perf_counter() - started) * 1000.0

    response.raise_for_status()

    body = response.json()

    results = body.get("results")

    if not isinstance(
        results,
        list,
    ):
        raise ValueError("Search response does not contain a valid results list")

    return (
        elapsed_ms,
        results,
    )


def extract_chunk(
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
        raise ValueError("Search result has no valid chunk_id")

    return {
        "rank": result.get("rank"),
        "score": result.get("score"),
        "chunk_id": chunk_id,
        "document_id": (chunk.get("document_id")),
        "file_name": (chunk.get("file_name")),
        "page_number": (chunk.get("page_number")),
        "chunk_index": (chunk.get("chunk_index")),
    }


def evaluate_mode(
    client: httpx.Client,
    *,
    base_url: str,
    query: EvaluationQuery,
    mode: str,
    top_k: int,
    relevance_threshold: int,
) -> tuple[
    RetrievalMetrics,
    dict[str, Any],
]:
    (
        elapsed_ms,
        raw_results,
    ) = search(
        client,
        base_url=base_url,
        query=query,
        mode=mode,
        top_k=top_k,
    )

    results = [extract_chunk(result) for result in raw_results]

    chunk_ids = [result["chunk_id"] for result in results]

    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError(f"{query.id}/{mode}: retrieval returned duplicate chunk IDs")

    judgments = {judgment.chunk_id: (judgment) for judgment in query.judgments}

    unjudged = [chunk_id for chunk_id in chunk_ids if chunk_id not in judgments]

    if unjudged:
        raise RuntimeError(
            f"{query.id}/{mode}: "
            "retrieved unjudged chunks. "
            "The relevance pool may be "
            "stale; rebuild the pool "
            "before evaluation. "
            f"Unjudged: {unjudged}"
        )

    metrics = evaluate_ranking(
        chunk_ids,
        query,
        k=top_k,
        relevance_threshold=(relevance_threshold),
    )

    annotated_results = []

    for result in results:
        judgment = judgments[result["chunk_id"]]

        annotated_results.append(
            {
                **result,
                "judged_relevance": (judgment.relevance),
                "judgment_rationale": (judgment.rationale),
            }
        )

    details = {
        "elapsed_ms": (elapsed_ms),
        "metrics": (metrics.model_dump(mode="json")),
        "results": (annotated_results),
    }

    return (
        metrics,
        details,
    )


def main() -> None:
    args = parse_args()

    all_queries = load_evaluation_queries(args.qrels)

    split = EvaluationSplit(args.split)

    queries = select_queries(
        all_queries,
        split=split,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Evaluation split:",
        split.value,
    )

    print(
        "Selected queries:",
        len(queries),
    )

    if split == EvaluationSplit.DEVELOPMENT:
        print("Holdout queries are not being evaluated.")
    else:
        print("HOLDOUT EVALUATION CONFIRMED.")

    started_at = datetime.now(UTC)

    query_reports: list[dict[str, Any]] = []

    aggregate_inputs: dict[
        str,
        list[RetrievalMetrics],
    ] = {mode: [] for mode in MODES}

    with httpx.Client(timeout=120.0) as client:
        for query in queries:
            mode_metrics: dict[
                str,
                RetrievalMetrics,
            ] = {}

            mode_reports: dict[
                str,
                dict[str, Any],
            ] = {}

            print()
            print("=" * 72)

            print(
                query.id,
                "—",
                query.query,
            )

            print("=" * 72)

            for mode in MODES:
                (
                    metrics,
                    details,
                ) = evaluate_mode(
                    client,
                    base_url=(args.base_url),
                    query=query,
                    mode=mode,
                    top_k=args.top_k,
                    relevance_threshold=(args.relevance_threshold),
                )

                mode_metrics[mode] = metrics

                mode_reports[mode] = details

                if query.include_in_aggregate:
                    aggregate_inputs[mode].append(metrics)

                print(
                    f"{mode:17} "
                    f"P@{args.top_k}="
                    f"{metrics.precision_at_k:.3f} "
                    f"R@{args.top_k}="
                    f"{metrics.recall_at_k:.3f} "
                    "MRR="
                    f"{metrics.reciprocal_rank:.3f} "
                    f"nDCG@{args.top_k}="
                    f"{metrics.ndcg_at_k:.3f}"
                )

            comparisons: dict[
                str,
                Any,
            ] = {}

            for (
                baseline_mode,
                candidate_mode,
            ) in COMPARISONS:
                comparison = compare_metrics(
                    baseline_mode=(baseline_mode),
                    candidate_mode=(candidate_mode),
                    baseline=(mode_metrics[baseline_mode]),
                    candidate=(mode_metrics[candidate_mode]),
                )

                key = f"{baseline_mode}_to_{candidate_mode}"

                comparisons[key] = comparison.model_dump(mode="json")

            rerank_comparison = comparisons["hybrid_to_hybrid_reranked"]

            print(
                "Hybrid → Reranked:",
                rerank_comparison["outcome"],
            )

            query_reports.append(
                {
                    "id": (query.id),
                    "query": (query.query),
                    "document_id": (query.document_id),
                    "include_in_aggregate": (query.include_in_aggregate),
                    "evaluation_split": (split.value),
                    "modes": (mode_reports),
                    "comparisons": (comparisons),
                }
            )

    aggregates = {
        mode: (aggregate_metrics(aggregate_inputs[mode]).model_dump(mode="json"))
        for mode in MODES
    }

    finished_at = datetime.now(UTC)

    report = {
        "schema_version": 2,
        "experiment": ("phase13_split_retrieval_evaluation"),
        "evaluation_split": (split.value),
        "started_at": (started_at.isoformat()),
        "finished_at": (finished_at.isoformat()),
        "base_url": (args.base_url),
        "qrels_file": str(args.qrels),
        "query_count": (len(queries)),
        "top_k": (args.top_k),
        "binary_relevance_threshold": (args.relevance_threshold),
        "graded_relevance_scale": ("0-3"),
        "aggregate_policy": (
            "Macro-average only selected split queries where include_in_aggregate=true"
        ),
        "pooling_warning": (
            "Relevance judgments come "
            "from pooled top-20 candidate "
            "sets across the four "
            "retrieval modes. Recall is "
            "pooled recall rather than "
            "exhaustive corpus recall."
        ),
        "score_warning": (
            "Dense, BM25, RRF, and "
            "cross-encoder scores use "
            "different semantics and "
            "must not be compared "
            "numerically."
        ),
        "modes": list(MODES),
        "aggregate": (aggregates),
        "queries": (query_reports),
    }

    timestamp = started_at.strftime("%Y%m%dT%H%M%SZ")

    output_path = args.output_dir / (
        f"retrieval_evaluation_{split.value}_{timestamp}.json"
    )

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

    print(f"{split.value.upper()} AGGREGATE")

    print("=" * 72)

    for mode in MODES:
        metrics = aggregates[mode]

        print(
            f"{mode:17} "
            f"P@{args.top_k}="
            f"{metrics['mean_precision_at_k']:.3f} "
            f"R@{args.top_k}="
            f"{metrics['mean_recall_at_k']:.3f} "
            "MRR="
            f"{metrics['mean_reciprocal_rank']:.3f} "
            f"nDCG@{args.top_k}="
            f"{metrics['mean_ndcg_at_k']:.3f}"
        )

    print()
    print(
        "Report:",
        output_path,
    )


if __name__ == "__main__":
    main()

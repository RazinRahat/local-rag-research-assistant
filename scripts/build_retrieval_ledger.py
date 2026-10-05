from __future__ import annotations

import argparse
import json
from datetime import (
    UTC,
    datetime,
)
from pathlib import Path
from typing import Any

from research_assistant.evaluation.diagnostics import (
    classify_top1_change,
    combine_outcomes,
)
from research_assistant.evaluation.models import (
    ComparisonOutcome,
)

DEFAULT_RESULTS_DIR = Path("experiments/results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Build a per-query retrieval improvement/degradation ledger.")
    )

    parser.add_argument(
        "--evaluation",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
    )

    return parser.parse_args()


def latest_evaluation() -> Path:
    candidates = sorted(
        DEFAULT_RESULTS_DIR.glob("retrieval_evaluation_*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if not candidates:
        raise FileNotFoundError("No retrieval evaluation report was found")

    return candidates[0]


def top_result(
    mode_report: dict[str, Any],
) -> dict[str, Any]:
    results = mode_report.get("results")

    if (
        not isinstance(
            results,
            list,
        )
        or not results
    ):
        raise ValueError("Mode report contains no ranked results")

    result = results[0]

    if not isinstance(
        result,
        dict,
    ):
        raise ValueError("Top result must be an object")

    return result


def comparison_entry(
    *,
    query: dict[str, Any],
    comparison_key: str,
) -> dict[str, Any]:
    comparisons = query["comparisons"]

    comparison = comparisons[comparison_key]

    baseline_mode = comparison["baseline_mode"]

    candidate_mode = comparison["candidate_mode"]

    baseline_top = top_result(query["modes"][baseline_mode])

    candidate_top = top_result(query["modes"][candidate_mode])

    baseline_relevance = int(baseline_top["judged_relevance"])

    candidate_relevance = int(candidate_top["judged_relevance"])

    metric_outcome = ComparisonOutcome(comparison["outcome"])

    top1_outcome = classify_top1_change(
        baseline_relevance,
        candidate_relevance,
    )

    diagnostic_outcome = combine_outcomes(
        metric_outcome,
        top1_outcome,
    )

    return {
        "baseline_mode": (baseline_mode),
        "candidate_mode": (candidate_mode),
        "metric_outcome": (metric_outcome.value),
        "top1_outcome": (top1_outcome.value),
        "diagnostic_outcome": (diagnostic_outcome.value),
        "metric_delta": (comparison["delta"]),
        "baseline_top1": {
            "chunk_id": (baseline_top["chunk_id"]),
            "file_name": (baseline_top["file_name"]),
            "page_number": (baseline_top["page_number"]),
            "relevance": (baseline_relevance),
            "rationale": (baseline_top.get("judgment_rationale")),
        },
        "candidate_top1": {
            "chunk_id": (candidate_top["chunk_id"]),
            "file_name": (candidate_top["file_name"]),
            "page_number": (candidate_top["page_number"]),
            "relevance": (candidate_relevance),
            "rationale": (candidate_top.get("judgment_rationale")),
        },
        "top1_relevance_delta": (candidate_relevance - baseline_relevance),
    }


def main() -> None:
    args = parse_args()

    evaluation_path = args.evaluation or latest_evaluation()

    data = json.loads(evaluation_path.read_text(encoding="utf-8"))

    query_reports = data.get("queries")

    if not isinstance(
        query_reports,
        list,
    ):
        raise ValueError("Evaluation report contains no queries list")

    ledger_queries = []

    comparison_keys = (
        "dense_to_hybrid",
        "hybrid_to_hybrid_reranked",
        "dense_to_hybrid_reranked",
    )

    for query in query_reports:
        entries = {
            key: comparison_entry(
                query=query,
                comparison_key=key,
            )
            for key in comparison_keys
        }

        ledger_queries.append(
            {
                "id": query["id"],
                "query": (query["query"]),
                "include_in_aggregate": (query["include_in_aggregate"]),
                "comparisons": (entries),
            }
        )

    primary = [query for query in ledger_queries if query["include_in_aggregate"]]

    outcome_summary: dict[
        str,
        dict[str, int],
    ] = {}

    for key in comparison_keys:
        counts = {outcome.value: 0 for outcome in ComparisonOutcome}

        for query in primary:
            outcome = query["comparisons"][key]["diagnostic_outcome"]

            counts[outcome] += 1

        outcome_summary[key] = counts

    report = {
        "schema_version": 1,
        "experiment": ("phase13_retrieval_regression_ledger"),
        "created_at": datetime.now(UTC).isoformat(),
        "source_evaluation": str(evaluation_path),
        "diagnostic_policy": (
            "Metric outcome and top-1 "
            "graded relevance are tracked "
            "separately. Opposing changes "
            "are classified as mixed."
        ),
        "summary": (outcome_summary),
        "queries": (ledger_queries),
    }

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    output_path = args.output_dir / (f"retrieval_ledger_{timestamp}.json")

    output_path.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print("PRIMARY QUERY OUTCOMES")

    print("=" * 72)

    for query in primary:
        comparison = query["comparisons"]["hybrid_to_hybrid_reranked"]

        print(
            f"{query['id']:22} "
            f"metric="
            f"{comparison['metric_outcome']:12} "
            f"top1="
            f"{comparison['top1_outcome']:12} "
            f"overall="
            f"{comparison['diagnostic_outcome']}"
        )

    print()
    print(
        "Ledger:",
        output_path,
    )


if __name__ == "__main__":
    main()

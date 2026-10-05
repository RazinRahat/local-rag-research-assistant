from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite

from research_assistant.chunking.models import DocumentChunk
from research_assistant.retrieval.models import RetrievalResult


@dataclass(frozen=True, slots=True)
class RRFConfig:
    """Configuration for reciprocal rank fusion."""

    rank_constant: int = 60

    def __post_init__(self) -> None:
        if self.rank_constant < 1:
            raise ValueError("rank_constant must be at least 1")


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[RetrievalResult]],
    *,
    top_k: int,
    config: RRFConfig | None = None,
    weights: Sequence[float] | None = None,
) -> tuple[RetrievalResult, ...]:
    """Fuse ranked retrieval results using weighted RRF."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    if not rankings:
        return ()

    fusion_config = config or RRFConfig()

    ranking_weights = (
        tuple(weights) if weights is not None else tuple(1.0 for _ in rankings)
    )

    if len(ranking_weights) != len(rankings):
        raise ValueError("weights must contain one value per ranking")

    for weight in ranking_weights:
        if not isfinite(weight) or weight <= 0.0:
            raise ValueError("RRF weights must be finite and greater than 0")

    scores: dict[
        str,
        float,
    ] = {}

    chunks: dict[
        str,
        DocumentChunk,
    ] = {}

    best_ranks: dict[
        str,
        int,
    ] = {}

    for ranking, weight in zip(
        rankings,
        ranking_weights,
        strict=True,
    ):
        seen_chunk_ids: set[str] = set()

        for position, result in enumerate(
            ranking,
            start=1,
        ):
            if result.rank != position:
                raise ValueError(
                    "Retrieval ranking must use sequential ranks starting at 1"
                )

            chunk = result.chunk
            chunk_id = chunk.chunk_id

            if chunk_id in seen_chunk_ids:
                raise ValueError(
                    f"Retrieval ranking contains duplicate chunk_id: {chunk_id}"
                )

            seen_chunk_ids.add(chunk_id)

            existing_chunk = chunks.get(chunk_id)

            if existing_chunk is not None and existing_chunk != chunk:
                raise ValueError(
                    f"Conflicting chunk provenance for chunk_id: {chunk_id}"
                )

            chunks[chunk_id] = chunk

            contribution = weight / (fusion_config.rank_constant + result.rank)

            scores[chunk_id] = (
                scores.get(
                    chunk_id,
                    0.0,
                )
                + contribution
            )

            previous_best_rank = best_ranks.get(chunk_id)

            if previous_best_rank is None or result.rank < previous_best_rank:
                best_ranks[chunk_id] = result.rank

    ordered_chunk_ids = sorted(
        scores,
        key=lambda chunk_id: (
            -scores[chunk_id],
            best_ranks[chunk_id],
            chunk_id,
        ),
    )

    selected_chunk_ids = ordered_chunk_ids[:top_k]

    return tuple(
        RetrievalResult(
            rank=rank,
            score=scores[chunk_id],
            chunk=chunks[chunk_id],
        )
        for rank, chunk_id in enumerate(
            selected_chunk_ids,
            start=1,
        )
    )

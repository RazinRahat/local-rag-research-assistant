from collections.abc import Sequence
from typing import Protocol

import numpy as np
from sentence_transformers import (
    CrossEncoder,
)

from research_assistant.reranking.models import (
    CrossEncoderConfig,
)
from research_assistant.retrieval.models import (
    RetrievalResult,
)


class PairScorer(Protocol):
    """Score query-passage pairs."""

    def score(
        self,
        query: str,
        passages: Sequence[str],
    ) -> tuple[float, ...]:
        """Return one relevance score per passage."""
        ...


class SentenceTransformersCrossEncoder:
    """Sentence Transformers cross-encoder scoring adapter."""

    def __init__(
        self,
        config: CrossEncoderConfig | None = None,
    ) -> None:
        self._config = config or CrossEncoderConfig()

        self._model = CrossEncoder(
            self._config.model_name,
            device=self._config.device,
            max_length=(self._config.max_length),
        )

    @property
    def model_name(self) -> str:
        """Return the configured model identifier."""

        return self._config.model_name

    def score(
        self,
        query: str,
        passages: Sequence[str],
    ) -> tuple[float, ...]:
        """Score query-passage pairs."""

        if not passages:
            return ()

        pairs = [
            (
                query,
                passage,
            )
            for passage in passages
        ]

        raw_scores = self._model.predict(
            pairs,
            batch_size=(self._config.batch_size),
            show_progress_bar=False,
            convert_to_numpy=True,
        )

        scores = np.asarray(
            raw_scores,
            dtype=np.float64,
        ).reshape(-1)

        if scores.size != len(passages):
            raise ValueError("Cross-encoder returned an unexpected number of scores")

        if not np.all(np.isfinite(scores)):
            raise ValueError("Cross-encoder returned a non-finite score")

        return tuple(float(score) for score in scores)


class CrossEncoderReranker:
    """Rerank retrieval candidates using pairwise relevance."""

    def __init__(
        self,
        scorer: PairScorer,
    ) -> None:
        self._scorer = scorer

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievalResult],
        *,
        top_k: int,
    ) -> tuple[
        RetrievalResult,
        ...,
    ]:
        """Return candidates ordered by cross-encoder score."""

        clean_query = query.strip()

        if not clean_query:
            raise ValueError("Reranking query cannot be empty")

        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        if not candidates:
            return ()

        seen_chunk_ids: set[str] = set()

        for position, candidate in enumerate(
            candidates,
            start=1,
        ):
            if candidate.rank != position:
                raise ValueError(
                    "Reranking candidates must use sequential ranks starting at 1"
                )

            chunk_id = candidate.chunk.chunk_id

            if chunk_id in seen_chunk_ids:
                raise ValueError(
                    f"Reranking candidates contain duplicate chunk_id: {chunk_id}"
                )

            seen_chunk_ids.add(chunk_id)

        passages = tuple(candidate.chunk.text for candidate in candidates)

        scores = self._scorer.score(
            clean_query,
            passages,
        )

        if len(scores) != len(candidates):
            raise ValueError("Reranker scorer returned an unexpected number of scores")

        if not all(np.isfinite(score) for score in scores):
            raise ValueError("Reranker scorer returned a non-finite score")

        scored_candidates = [
            (
                float(score),
                candidate,
            )
            for score, candidate in zip(
                scores,
                candidates,
                strict=True,
            )
        ]

        ordered = sorted(
            scored_candidates,
            key=lambda item: (
                -item[0],
                item[1].rank,
                item[1].chunk.chunk_id,
            ),
        )

        selected = ordered[:top_k]

        return tuple(
            RetrievalResult(
                rank=rank,
                score=score,
                chunk=candidate.chunk,
            )
            for rank, (
                score,
                candidate,
            ) in enumerate(
                selected,
                start=1,
            )
        )

from __future__ import annotations

from dataclasses import (
    dataclass,
    field,
)
from math import isfinite

from research_assistant.retrieval.base import (
    Retriever,
)
from research_assistant.retrieval.fusion import (
    RRFConfig,
    reciprocal_rank_fusion,
)
from research_assistant.retrieval.models import (
    RetrievalConfig,
    RetrievalResult,
)


def _default_hybrid_rrf_config() -> RRFConfig:
    return RRFConfig(
        rank_constant=5,
    )


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """Configuration for hybrid dense + lexical retrieval."""

    candidate_multiplier: int = 2

    rrf_config: RRFConfig = field(default_factory=(_default_hybrid_rrf_config))

    dense_weight: float = 1.0
    lexical_weight: float = 1.25

    def __post_init__(self) -> None:
        if self.candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be at least 1")

        if not isfinite(self.dense_weight) or self.dense_weight <= 0.0:
            raise ValueError("dense_weight must be finite and greater than 0")

        if not isfinite(self.lexical_weight) or self.lexical_weight <= 0.0:
            raise ValueError("lexical_weight must be finite and greater than 0")


class HybridRetriever:
    """Fuse dense and lexical retrieval using weighted RRF."""

    def __init__(
        self,
        dense_retriever: Retriever,
        lexical_retriever: Retriever,
        *,
        config: HybridConfig | None = None,
    ) -> None:
        self._dense_retriever = dense_retriever

        self._lexical_retriever = lexical_retriever

        self._config = config or HybridConfig()

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> tuple[
        RetrievalResult,
        ...,
    ]:
        """Retrieve and fuse dense and lexical candidates."""

        clean_query = query.strip()

        if not clean_query:
            raise ValueError("Retrieval query cannot be empty")

        retrieval_config = config or RetrievalConfig()

        if retrieval_config.score_threshold is not None:
            raise ValueError(
                "score_threshold is not "
                "supported for hybrid "
                "retrieval because RRF "
                "scores are rank-based "
                "and are not comparable "
                "to dense cosine "
                "similarity scores."
            )

        candidate_top_k = retrieval_config.top_k * self._config.candidate_multiplier

        candidate_config = RetrievalConfig(
            top_k=(candidate_top_k),
            document_id=(retrieval_config.document_id),
        )

        dense_results = self._dense_retriever.retrieve(
            clean_query,
            candidate_config,
        )

        lexical_results = self._lexical_retriever.retrieve(
            clean_query,
            candidate_config,
        )

        return reciprocal_rank_fusion(
            (
                dense_results,
                lexical_results,
            ),
            top_k=(retrieval_config.top_k),
            config=(self._config.rrf_config),
            weights=(
                self._config.dense_weight,
                self._config.lexical_weight,
            ),
        )

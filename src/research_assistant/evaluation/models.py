from enum import StrEnum

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


class ComparisonOutcome(StrEnum):
    """Direction of a retrieval change."""

    IMPROVEMENT = "improvement"
    DEGRADATION = "degradation"
    NO_CHANGE = "no_change"
    MIXED = "mixed"


class RelevanceJudgment(BaseModel):
    """Human relevance judgment for one chunk."""

    model_config = ConfigDict(
        frozen=True,
    )

    chunk_id: str = Field(
        min_length=1,
    )

    relevance: int = Field(
        ge=0,
        le=3,
    )

    rationale: str | None = None


class EvaluationQuery(BaseModel):
    """One labelled retrieval-evaluation query."""

    model_config = ConfigDict(
        frozen=True,
    )

    id: str = Field(
        min_length=1,
    )

    query: str = Field(
        min_length=1,
    )

    document_id: str | None = None

    include_in_aggregate: bool = True

    judgments: tuple[
        RelevanceJudgment,
        ...,
    ]

    tags: tuple[
        str,
        ...,
    ] = ()

    notes: str | None = None

    @model_validator(
        mode="after",
    )
    def validate_judgments(
        self,
    ) -> "EvaluationQuery":
        chunk_ids = [judgment.chunk_id for judgment in self.judgments]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Evaluation judgments contain duplicate chunk IDs")

        if not any(judgment.relevance > 0 for judgment in self.judgments):
            raise ValueError(
                "Evaluation query must contain at least one relevant judgment"
            )

        return self


class RetrievalMetrics(BaseModel):
    """Retrieval metrics for one query."""

    model_config = ConfigDict(
        frozen=True,
    )

    k: int = Field(
        ge=1,
    )

    relevance_threshold: int = Field(
        default=2,
        ge=1,
        le=3,
    )

    retrieved_count: int = Field(
        ge=0,
    )

    relevant_total: int = Field(
        ge=1,
    )

    relevant_retrieved: int = Field(
        ge=0,
    )

    precision_at_k: float = Field(
        ge=0.0,
        le=1.0,
    )

    recall_at_k: float = Field(
        ge=0.0,
        le=1.0,
    )

    reciprocal_rank: float = Field(
        ge=0.0,
        le=1.0,
    )

    ndcg_at_k: float = Field(
        ge=0.0,
        le=1.0,
    )


class AggregateRetrievalMetrics(BaseModel):
    """Mean retrieval metrics across evaluation queries."""

    model_config = ConfigDict(
        frozen=True,
    )

    query_count: int = Field(
        ge=1,
    )

    k: int = Field(
        ge=1,
    )

    relevance_threshold: int = Field(
        ge=1,
        le=3,
    )

    mean_precision_at_k: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_recall_at_k: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_reciprocal_rank: float = Field(
        ge=0.0,
        le=1.0,
    )

    mean_ndcg_at_k: float = Field(
        ge=0.0,
        le=1.0,
    )


class MetricDelta(BaseModel):
    """Candidate-minus-baseline metric differences."""

    model_config = ConfigDict(
        frozen=True,
    )

    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float


class RetrievalComparison(BaseModel):
    """Comparison between two retrieval strategies."""

    model_config = ConfigDict(
        frozen=True,
    )

    baseline_mode: str
    candidate_mode: str

    baseline: RetrievalMetrics
    candidate: RetrievalMetrics

    delta: MetricDelta

    outcome: ComparisonOutcome

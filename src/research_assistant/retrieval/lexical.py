from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from research_assistant.chunking.models import DocumentChunk

from .models import RetrievalConfig, RetrievalResult

_TOKEN_PATTERN = re.compile(
    r"\b\w+(?:[-']\w+)*\b",
    flags=re.UNICODE,
)


def tokenize_lexically(text: str) -> tuple[str, ...]:
    """Tokenize text for lexical retrieval."""

    return tuple(match.group(0).casefold() for match in _TOKEN_PATTERN.finditer(text))


class ChunkCorpus(Protocol):
    """Source of persisted chunks for lexical retrieval."""

    def list_chunks(
        self,
        document_id: str | None = None,
    ) -> Sequence[DocumentChunk]:
        """Return chunks, optionally restricted to one document."""
        ...


@dataclass(frozen=True, slots=True)
class BM25Config:
    """BM25 scoring configuration."""

    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError("k1 must be greater than 0.")

        if not 0 <= self.b <= 1:
            raise ValueError("b must be between 0 and 1.")


@dataclass(frozen=True, slots=True)
class _IndexedChunk:
    chunk: DocumentChunk
    term_frequencies: Counter[str]
    length: int


class BM25Retriever:
    """Retrieve chunks using BM25 lexical relevance."""

    def __init__(
        self,
        corpus: ChunkCorpus,
        *,
        config: BM25Config | None = None,
    ) -> None:
        self._corpus = corpus
        self._config = config or BM25Config()

    def retrieve(
        self,
        query: str,
        config: RetrievalConfig | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Retrieve lexically relevant document chunks."""

        clean_query = query.strip()

        if not clean_query:
            raise ValueError("Retrieval query cannot be empty")

        retrieval_config = config or RetrievalConfig()

        if retrieval_config.score_threshold is not None:
            raise ValueError(
                "score_threshold is not supported "
                "for BM25 retrieval because BM25 "
                "scores are not comparable to dense "
                "cosine similarity scores."
            )

        chunks = tuple(self._corpus.list_chunks(retrieval_config.document_id))

        if not chunks:
            return ()

        query_terms = tuple(dict.fromkeys(tokenize_lexically(clean_query)))

        if not query_terms:
            return ()

        indexed_chunks = tuple(self._index_chunk(chunk) for chunk in chunks)

        average_length = sum(indexed.length for indexed in indexed_chunks) / len(
            indexed_chunks
        )

        if average_length <= 0:
            return ()

        document_frequencies = self._document_frequencies(indexed_chunks)

        scored = tuple(
            (
                self._score_chunk(
                    indexed=indexed,
                    query_terms=query_terms,
                    document_frequencies=(document_frequencies),
                    document_count=len(indexed_chunks),
                    average_length=(average_length),
                ),
                indexed.chunk,
            )
            for indexed in indexed_chunks
        )

        ranked = sorted(
            ((score, chunk) for score, chunk in scored if score > 0),
            key=lambda item: (
                -item[0],
                item[1].chunk_id,
            ),
        )

        return tuple(
            RetrievalResult(
                rank=rank,
                score=score,
                chunk=chunk,
            )
            for rank, (
                score,
                chunk,
            ) in enumerate(
                ranked[: retrieval_config.top_k],
                start=1,
            )
        )

    @staticmethod
    def _index_chunk(
        chunk: DocumentChunk,
    ) -> _IndexedChunk:
        tokens = tokenize_lexically(chunk.text)

        return _IndexedChunk(
            chunk=chunk,
            term_frequencies=Counter(tokens),
            length=len(tokens),
        )

    @staticmethod
    def _document_frequencies(
        indexed_chunks: Sequence[_IndexedChunk],
    ) -> Counter[str]:
        frequencies: Counter[str] = Counter()

        for indexed in indexed_chunks:
            frequencies.update(indexed.term_frequencies.keys())

        return frequencies

    def _score_chunk(
        self,
        *,
        indexed: _IndexedChunk,
        query_terms: Sequence[str],
        document_frequencies: Counter[str],
        document_count: int,
        average_length: float,
    ) -> float:
        if indexed.length == 0:
            return 0.0

        score = 0.0

        for term in query_terms:
            term_frequency = indexed.term_frequencies[term]

            if term_frequency == 0:
                continue

            document_frequency = document_frequencies[term]

            inverse_document_frequency = math.log(
                1
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )

            length_normalisation = (
                1 - self._config.b + self._config.b * indexed.length / average_length
            )

            numerator = term_frequency * (self._config.k1 + 1)

            denominator = term_frequency + self._config.k1 * length_normalisation

            score += inverse_document_frequency * numerator / denominator

        return score

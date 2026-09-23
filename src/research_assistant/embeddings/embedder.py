from collections.abc import Sequence
from typing import Protocol

import numpy as np
from sentence_transformers import SentenceTransformer

from research_assistant.embeddings.models import EmbeddingConfig

EmbeddingVector = tuple[float, ...]


class EmbeddingProvider(Protocol):
    """Interface required by embedding consumers."""

    @property
    def model_name(self) -> str:
        """Return the embedding model identifier."""
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding vector dimension."""
        ...

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[EmbeddingVector, ...]:
        """Embed document passages."""
        ...

    def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        """Embed a retrieval query."""
        ...


class SentenceTransformerEmbedder:
    """Local Sentence Transformers embedding provider."""

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        device: str | None = None,
    ) -> None:
        self._config = config or EmbeddingConfig()

        self._model = SentenceTransformer(
            self._config.model_name,
            device=device,
        )

        dimension = self._model.get_sentence_embedding_dimension()

        if dimension is None:
            raise ValueError("Embedding model did not report an embedding dimension")

        self._dimension = dimension

    @property
    def model_name(self) -> str:
        return self._config.model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_sequence_length(self) -> int | None:
        return self._model.max_seq_length

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> tuple[EmbeddingVector, ...]:
        if not texts:
            return ()

        embeddings = self._model.encode_document(
            list(texts),
            batch_size=self._config.batch_size,
            normalize_embeddings=(self._config.normalize_embeddings),
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        array = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if array.ndim != 2:
            raise ValueError("Expected two-dimensional document embeddings")

        return tuple(tuple(float(value) for value in vector) for vector in array)

    def embed_query(
        self,
        query: str,
    ) -> EmbeddingVector:
        if not query.strip():
            raise ValueError("Query cannot be empty")

        embedding = self._model.encode_query(
            query,
            prompt=self._config.query_instruction,
            normalize_embeddings=(self._config.normalize_embeddings),
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        array = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if array.ndim != 1:
            raise ValueError("Expected one-dimensional query embedding")

        return tuple(float(value) for value in array)

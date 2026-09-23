class VectorStoreError(Exception):
    """Base exception for vector-store failures."""


class CollectionConfigurationError(VectorStoreError):
    """Raised when a collection has incompatible settings."""


class EmbeddingBatchError(VectorStoreError):
    """Raised when an embedding batch is inconsistent."""
